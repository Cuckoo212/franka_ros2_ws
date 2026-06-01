#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


DEFAULT_GRASP_ROOT = Path("/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint")
DEFAULT_CALIB_FILE = Path("/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib")


def normalize_cable_id(cable_id: str) -> str:
    normalized = cable_id.strip()
    if normalized.isdigit():
        normalized = f"{int(normalized):03d}"
    return normalized


def build_grasp_plan_path(cable_id: str, grasp_root: Path) -> Path:
    stripped = cable_id.strip()
    if "/" in stripped or "\\" in stripped:
        relative = Path(stripped)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid relative grasp plan id: {cable_id}")
        return grasp_root / relative
    normalized = normalize_cable_id(cable_id)
    return grasp_root / f"cable_{normalized}" / f"grasp_point_cable_{normalized}"


def load_skeleton(grasp_plan_path: Path, y_offset: float) -> np.ndarray:
    with grasp_plan_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "skeleton" not in data:
        raise ValueError(f"Missing 'skeleton' field in {grasp_plan_path}")
    skeleton = np.asarray(data["skeleton"], dtype=float)
    if skeleton.ndim != 2 or skeleton.shape[1] != 3 or len(skeleton) < 2:
        raise ValueError(f"Invalid skeleton in {grasp_plan_path}")
    skeleton = skeleton.copy()
    skeleton[:, 1] += y_offset
    return skeleton


def remove_duplicate_points(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if len(points) <= 1:
        return points
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > eps:
            kept.append(point)
    return np.asarray(kept, dtype=float)


def polyline_arc_lengths(points: np.ndarray) -> np.ndarray:
    points = remove_duplicate_points(points)
    if len(points) < 2:
        return np.zeros(len(points))
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def resample_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
    points = remove_duplicate_points(points)
    if len(points) < 2:
        raise ValueError("At least two points are required for resampling.")
    arc_lengths = polyline_arc_lengths(points)
    total_length = float(arc_lengths[-1])
    if total_length <= 1e-9:
        raise ValueError("Polyline length is too small.")

    distances = np.linspace(0.0, total_length, sample_count)
    result = np.empty((sample_count, 3), dtype=float)
    segment_idx = 1
    for out_idx, distance in enumerate(distances):
        while segment_idx < len(arc_lengths) - 1 and distance > arc_lengths[segment_idx]:
            segment_idx += 1
        start_distance = arc_lengths[segment_idx - 1]
        end_distance = arc_lengths[segment_idx]
        segment_length = end_distance - start_distance
        alpha = 0.0 if segment_length <= 1e-12 else (distance - start_distance) / segment_length
        result[out_idx] = points[segment_idx - 1] + alpha * (points[segment_idx] - points[segment_idx - 1])
    return result


def rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have shape Nx3.")
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def quaternion_to_matrix(quaternion: dict[str, float]) -> np.ndarray:
    x = float(quaternion["x"])
    y = float(quaternion["y"])
    z = float(quaternion["z"])
    w = float(quaternion["w"])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("Invalid zero-norm quaternion.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> dict[str, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}


def pose_quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    return quaternion_to_matrix({"x": x, "y": y, "z": z, "w": w})


def transform_calibration(
    calib_path: Path,
    delta_rotation: np.ndarray,
    delta_translation: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with calib_path.open("r", encoding="utf-8") as handle:
        calib = yaml.safe_load(handle)
    transform = calib["transform"]
    old_translation = np.asarray(
        [
            float(transform["translation"]["x"]),
            float(transform["translation"]["y"]),
            float(transform["translation"]["z"]),
        ],
        dtype=float,
    )
    old_rotation = quaternion_to_matrix(transform["rotation"])
    new_rotation = delta_rotation @ old_rotation
    new_translation = delta_rotation @ old_translation + delta_translation
    updated = json.loads(json.dumps(calib))
    updated["transform"]["translation"] = {
        "x": float(new_translation[0]),
        "y": float(new_translation[1]),
        "z": float(new_translation[2]),
    }
    updated["transform"]["rotation"] = matrix_to_quaternion(new_rotation)
    return calib, updated


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cosine)))


def compute_metrics(errors: np.ndarray) -> dict[str, Any]:
    distances = np.linalg.norm(errors, axis=1)
    return {
        "mean_error_xyz_m": errors.mean(axis=0).tolist(),
        "mean_abs_error_xyz_m": np.abs(errors).mean(axis=0).tolist(),
        "rms_error_m": float(math.sqrt(np.mean(distances**2))),
        "mean_distance_m": float(np.mean(distances)),
        "median_distance_m": float(np.median(distances)),
        "p95_distance_m": float(np.percentile(distances, 95.0)),
        "max_distance_m": float(np.max(distances)),
    }


class ManualCableTrajectoryRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("manual_cable_trajectory_recorder")
        self._args = args
        self._records: list[dict[str, Any]] = []
        self._last_position: np.ndarray | None = None
        self._subscription = self.create_subscription(PoseStamped, args.pose_topic, self._on_pose, 50)
        self.get_logger().info(
            f"Recording TCP trajectory from {args.pose_topic}. "
            "Hand-guide the TCP along the real cable path, then press Ctrl+C in this terminal."
        )

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def _on_pose(self, msg: PoseStamped) -> None:
        position = np.asarray(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        if self._args.tcp_offset is not None:
            rotation = pose_quaternion_to_matrix(
                (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                )
            )
            position = position + rotation @ np.asarray(self._args.tcp_offset, dtype=float)

        if self._last_position is not None:
            if np.linalg.norm(position - self._last_position) < self._args.min_distance:
                return
        self._last_position = position
        self._records.append(
            {
                "stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec),
                },
                "frame_id": msg.header.frame_id,
                "position": position.tolist(),
                "orientation": {
                    "x": float(msg.pose.orientation.x),
                    "y": float(msg.pose.orientation.y),
                    "z": float(msg.pose.orientation.z),
                    "w": float(msg.pose.orientation.w),
                },
            }
        )
        if len(self._records) % 100 == 0:
            self.get_logger().info(f"Recorded {len(self._records)} trajectory points.")


def save_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sec", "nanosec", "frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for record in records:
            orientation = record["orientation"]
            writer.writerow(
                [
                    record["stamp"]["sec"],
                    record["stamp"]["nanosec"],
                    record["frame_id"],
                    *record["position"],
                    orientation["x"],
                    orientation["y"],
                    orientation["z"],
                    orientation["w"],
                ]
            )


def analyze_and_save(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    if len(records) < 2:
        raise RuntimeError("Need at least two recorded TCP points.")

    target_id = args.cable_id.strip()
    grasp_plan_path = build_grasp_plan_path(target_id, args.grasp_root)
    if "/" in target_id or "\\" in target_id:
        cable_id = grasp_plan_path.name
        output_dir = args.output_dir or grasp_plan_path.parent
    else:
        cable_id = normalize_cable_id(target_id)
        output_dir = args.output_dir or (args.grasp_root / f"cable_{cable_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    recorded_points = np.asarray([record["position"] for record in records], dtype=float)
    skeleton = load_skeleton(grasp_plan_path, args.skeleton_y_offset)

    sample_count = min(args.compare_samples, len(remove_duplicate_points(recorded_points)), len(remove_duplicate_points(skeleton)))
    sample_count = max(2, sample_count)
    recorded_sampled = resample_polyline(recorded_points, sample_count)
    skeleton_sampled = resample_polyline(skeleton, sample_count)
    skeleton_reversed_sampled = resample_polyline(skeleton[::-1], sample_count)

    forward_errors = recorded_sampled - skeleton_sampled
    reverse_errors = recorded_sampled - skeleton_reversed_sampled
    if compute_metrics(reverse_errors)["rms_error_m"] < compute_metrics(forward_errors)["rms_error_m"]:
        matched_skeleton = skeleton_reversed_sampled
        match_direction = "reversed"
        errors = reverse_errors
    else:
        matched_skeleton = skeleton_sampled
        match_direction = "forward"
        errors = forward_errors

    delta_rotation, delta_translation = rigid_transform(matched_skeleton, recorded_sampled)
    delta_angle_rad = rotation_angle(delta_rotation)
    metrics = compute_metrics(errors)

    report = {
        "cable_id": cable_id,
        "pose_topic": args.pose_topic,
        "grasp_plan_path": str(grasp_plan_path),
        "calibration_path": str(args.calib_file) if args.calib_file else None,
        "recorded_point_count": len(records),
        "skeleton_point_count": int(len(skeleton)),
        "comparison_sample_count": int(sample_count),
        "match_direction": match_direction,
        "skeleton_y_offset_m": float(args.skeleton_y_offset),
        "metrics": metrics,
        "delta_transform_current_robot_skeleton_to_recorded_tcp": {
            "translation_m": delta_translation.tolist(),
            "rotation_matrix": delta_rotation.tolist(),
            "rotation_angle_rad": float(delta_angle_rad),
            "rotation_angle_deg": float(math.degrees(delta_angle_rad)),
            "rotation_quaternion": matrix_to_quaternion(delta_rotation),
        },
        "recorded_trajectory": records,
    }

    if args.calib_file is not None and args.calib_file.is_file():
        old_calib, updated_calib = transform_calibration(args.calib_file, delta_rotation, delta_translation)
        report["old_calibration_transform"] = old_calib["transform"]
        report["suggested_calibration_transform"] = updated_calib["transform"]
        if args.write_updated_calibration:
            updated_path = args.updated_calib_path
            if updated_path is None:
                updated_path = args.calib_file.with_name(f"{args.calib_file.stem}_manual_cable_{cable_id}.calib")
            with updated_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(updated_calib, handle, sort_keys=False)
            report["updated_calibration_path"] = str(updated_path)

    prefix = output_dir / f"manual_tcp_trajectory_cable_{cable_id}"
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    report_path = output_dir / f"manual_tcp_trajectory_comparison_cable_{cable_id}.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"cable_id": cable_id, "records": records}, handle, indent=2)
    save_csv(csv_path, records)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved recorded trajectory JSON: {json_path}")
    print(f"Saved recorded trajectory CSV:  {csv_path}")
    print(f"Saved comparison report:        {report_path}")
    print(
        "Comparison: "
        f"rms={metrics['rms_error_m'] * 1000.0:.2f} mm, "
        f"mean={metrics['mean_distance_m'] * 1000.0:.2f} mm, "
        f"p95={metrics['p95_distance_m'] * 1000.0:.2f} mm, "
        f"max={metrics['max_distance_m'] * 1000.0:.2f} mm, "
        f"direction={match_direction}"
    )
    print(
        "Suggested calibration delta: "
        f"translation={delta_translation.tolist()} m, "
        f"rotation={math.degrees(delta_angle_rad):.3f} deg"
    )
    if not args.write_updated_calibration:
        print("Calibration file was not modified. Add --write-updated-calibration to write a suggested copy.")


def build_pose_topic(namespace: str) -> str:
    clean_namespace = namespace.strip("/")
    if clean_namespace:
        return f"/{clean_namespace}/franka_robot_state_broadcaster/current_pose"
    return "/franka_robot_state_broadcaster/current_pose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a hand-guided TCP trajectory during manual calibration and compare it with "
            "the cable skeleton saved in grasp_point_cable_*."
        )
    )
    parser.add_argument("cable_id", help="Cable id such as 031, or relative plan id such as multi_grasp/cable_001/grasp_002.")
    parser.add_argument("--namespace", default="NS_1", help="Robot namespace. Default: NS_1.")
    parser.add_argument("--pose-topic", default=None, help="Override current_pose topic.")
    parser.add_argument("--grasp-root", type=Path, default=DEFAULT_GRASP_ROOT)
    parser.add_argument("--calib-file", type=Path, default=DEFAULT_CALIB_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-distance", type=float, default=0.001, help="Spatial downsample step in meters.")
    parser.add_argument("--compare-samples", type=int, default=200)
    parser.add_argument(
        "--skeleton-y-offset",
        type=float,
        default=0.0,
        help="Apply an extra Y offset to the skeleton before comparison. Use -0.01 if you recorded the intended 1 cm-above-cable path.",
    )
    parser.add_argument(
        "--tcp-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional offset from TCP to the physical traced point, expressed in TCP frame.",
    )
    parser.add_argument("--write-updated-calibration", action="store_true")
    parser.add_argument("--updated-calib-path", type=Path, default=None)
    args = parser.parse_args()
    if args.pose_topic is None:
        args.pose_topic = build_pose_topic(args.namespace)
    if args.compare_samples < 2:
        raise ValueError("--compare-samples must be at least 2.")
    if args.min_distance < 0.0:
        raise ValueError("--min-distance must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ManualCableTrajectoryRecorder(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        records = node.records
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    analyze_and_save(args, records)


if __name__ == "__main__":
    main()
