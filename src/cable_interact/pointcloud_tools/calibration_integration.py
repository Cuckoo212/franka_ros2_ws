#!/usr/bin/env python3
"""Fuse manual cable calibration samples into one hand-eye calibration file."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CABLE_ROOT = Path(
    "/home/flexcycle/franka_ros2_ws/src/cable_interact/"
    "pointcloud_tools/info_for_3Dpoint/samples_calibration/calibration_001"
)
DEFAULT_CALIB_FILE = Path("/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib")
SKELETON_FROM_PC_NAME = "skeleton_from_pc"


def load_skeleton(path: Path, y_offset: float = 0.0) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    skeleton = np.asarray(data.get("skeleton"), dtype=float)
    if skeleton.ndim != 2 or skeleton.shape[1] != 3 or len(skeleton) < 2:
        raise ValueError(f"Invalid skeleton in {path}")
    skeleton = skeleton.copy()
    skeleton[:, 1] += y_offset
    return skeleton


def load_recorded_points(report: dict[str, Any], report_path: Path) -> np.ndarray:
    records = report.get("recorded_trajectory")
    if records is None:
        raise ValueError(f"Missing recorded_trajectory in {report_path}")
    points = np.asarray([record["position"] for record in records], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"Invalid recorded trajectory in {report_path}")
    return points


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
        return np.zeros(len(points), dtype=float)
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


def compute_metrics(errors: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(errors, axis=1)
    return {
        "rms_error_m": float(math.sqrt(np.mean(distances**2))),
        "median_distance_m": float(np.median(distances)),
        "p95_distance_m": float(np.percentile(distances, 95.0)),
        "max_distance_m": float(np.max(distances)),
    }


def weighted_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have shape Nx3.")
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(weights.sum())
    if len(weights) != len(source) or weight_sum <= 1e-12:
        raise ValueError("weights must be positive and match source length.")

    normalized = weights / weight_sum
    source_centroid = normalized @ source
    target_centroid = normalized @ target
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ (target_centered * normalized[:, None])
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def robust_global_transform(
    source: np.ndarray,
    target: np.ndarray,
    base_weights: np.ndarray,
    huber_delta: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = base_weights.copy()
    rotation = np.eye(3)
    translation = np.zeros(3)

    for _ in range(iterations):
        rotation, translation = weighted_rigid_transform(source, target, weights)
        residuals = np.linalg.norm(target - (source @ rotation.T + translation), axis=1)
        robust_weights = np.ones_like(residuals)
        large = residuals > huber_delta
        robust_weights[large] = huber_delta / np.maximum(residuals[large], 1e-12)
        weights = base_weights * robust_weights

    return rotation, translation, weights


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
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return {"x": float(x / norm), "y": float(y / norm), "z": float(z / norm), "w": float(w / norm)}


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cosine)))


def read_calibration_transform(calib_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    text = calib_path.read_text(encoding="utf-8")

    def read_block(name: str) -> dict[str, float]:
        values: dict[str, float] = {}
        in_transform = False
        in_block = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "transform:":
                in_transform = True
                in_block = False
                continue
            if in_transform and re.match(r"^\S", line):
                in_transform = False
                in_block = False
            if not in_transform:
                continue
            if stripped == f"{name}:":
                in_block = True
                continue
            if in_block and line.startswith("  ") and stripped.endswith(":"):
                break
            if in_block:
                match = re.match(r"^\s*(\w):\s*([-+0-9.eE]+)\s*$", line)
                if match is not None:
                    values[match.group(1)] = float(match.group(2))
        if not values:
            raise ValueError(f"Missing calibration transform block '{name}' in {calib_path}")
        return values

    translation_data = read_block("translation")
    rotation_data = read_block("rotation")
    translation = np.asarray(
        [translation_data["x"], translation_data["y"], translation_data["z"]],
        dtype=float,
    )
    rotation = quaternion_to_matrix(rotation_data)
    return text, translation, rotation


def write_calibration_transform(
    template_text: str,
    translation: np.ndarray,
    quaternion: dict[str, float],
) -> str:
    lines = template_text.splitlines()
    out: list[str] = []
    in_transform = False
    in_translation = False
    in_rotation = False

    for line in lines:
        stripped = line.strip()
        if stripped == "transform:":
            in_transform = True
            in_translation = False
            in_rotation = False
        elif in_transform and re.match(r"^\S", line):
            in_transform = False
            in_translation = False
            in_rotation = False
        elif in_transform and stripped == "translation:":
            in_translation = True
            in_rotation = False
        elif in_transform and stripped == "rotation:":
            in_translation = False
            in_rotation = True
        elif in_transform and line.startswith("  ") and stripped.endswith(":"):
            in_translation = False
            in_rotation = False

        if in_translation and stripped.startswith("x:"):
            out.append(f"    x: {translation[0]:.17g}")
        elif in_translation and stripped.startswith("y:"):
            out.append(f"    y: {translation[1]:.17g}")
        elif in_translation and stripped.startswith("z:"):
            out.append(f"    z: {translation[2]:.17g}")
        elif in_rotation and stripped.startswith("x:"):
            out.append(f"    x: {quaternion['x']:.17g}")
        elif in_rotation and stripped.startswith("y:"):
            out.append(f"    y: {quaternion['y']:.17g}")
        elif in_rotation and stripped.startswith("z:"):
            out.append(f"    z: {quaternion['z']:.17g}")
        elif in_rotation and stripped.startswith("w:"):
            out.append(f"    w: {quaternion['w']:.17g}")
        else:
            out.append(line)

    return "\n".join(out) + "\n"


def transform_calibration(
    calib_path: Path,
    delta_rotation: np.ndarray,
    delta_translation: np.ndarray,
) -> str:
    template_text, old_translation, old_rotation = read_calibration_transform(calib_path)
    new_rotation = delta_rotation @ old_rotation
    new_translation = delta_rotation @ old_translation + delta_translation
    return write_calibration_transform(template_text, new_translation, matrix_to_quaternion(new_rotation))


def find_reports(cable_root: Path) -> list[Path]:
    return sorted(cable_root.glob("cable_*/groundtruth_path/calibration_comparison_cable_*.json"))


def build_point_pairs(
    report_path: Path,
    samples_per_cable: int | None,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, float], str]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    cable_dir = report_path.parents[1]
    cable_id = str(report.get("cable_id") or cable_dir.name)
    skeleton_path = Path(report.get("skeleton_path") or cable_dir / SKELETON_FROM_PC_NAME)
    y_offset = float(report.get("skeleton_y_offset_m", 0.0))
    skeleton = load_skeleton(skeleton_path, y_offset)
    recorded = load_recorded_points(report, report_path)

    sample_count = min(len(remove_duplicate_points(recorded)), len(remove_duplicate_points(skeleton)))
    if samples_per_cable is not None:
        sample_count = min(sample_count, samples_per_cable)
    sample_count = max(2, sample_count)

    recorded_sampled = resample_polyline(recorded, sample_count)
    skeleton_forward = resample_polyline(skeleton, sample_count)
    skeleton_reverse = resample_polyline(skeleton[::-1], sample_count)
    forward_metrics = compute_metrics(recorded_sampled - skeleton_forward)
    reverse_metrics = compute_metrics(recorded_sampled - skeleton_reverse)

    if reverse_metrics["rms_error_m"] < forward_metrics["rms_error_m"]:
        return cable_id, skeleton_reverse, recorded_sampled, reverse_metrics, "reversed"
    return cable_id, skeleton_forward, recorded_sampled, forward_metrics, "forward"


def fuse_calibration(args: argparse.Namespace) -> dict[str, Any]:
    reports = find_reports(args.cable_root)
    if not reports:
        raise FileNotFoundError(f"No calibration comparison reports found under {args.cable_root}")

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    cable_summaries: list[dict[str, Any]] = []

    for report_path in reports:
        cable_id, source, target, metrics, direction = build_point_pairs(report_path, args.samples_per_cable)
        per_point_weight = np.full(len(source), 1.0 / len(source), dtype=float)
        sources.append(source)
        targets.append(target)
        weights.append(per_point_weight)
        cable_summaries.append(
            {
                "cable_id": cable_id,
                "report_path": str(report_path),
                "sample_count": int(len(source)),
                "match_direction": direction,
                **metrics,
            }
        )

    source_all = np.vstack(sources)
    target_all = np.vstack(targets)
    base_weights = np.concatenate(weights)
    base_weights /= base_weights.sum()

    initial_rotation, initial_translation = weighted_rigid_transform(source_all, target_all, base_weights)
    initial_errors = target_all - (source_all @ initial_rotation.T + initial_translation)
    initial_metrics = compute_metrics(initial_errors)

    huber_delta = args.huber_delta
    if huber_delta is None:
        residuals = np.linalg.norm(initial_errors, axis=1)
        huber_delta = max(0.005, 1.5 * float(np.median(residuals)))

    rotation, translation, final_weights = robust_global_transform(
        source_all,
        target_all,
        base_weights,
        huber_delta,
        args.iterations,
    )
    final_errors = target_all - (source_all @ rotation.T + translation)
    final_metrics = compute_metrics(final_errors)
    final_calib = transform_calibration(args.calib_file, rotation, translation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(final_calib)

    return {
        "output": str(args.output),
        "cable_count": len(cable_summaries),
        "point_count": int(len(source_all)),
        "huber_delta_m": float(huber_delta),
        "delta_translation_m": translation.tolist(),
        "delta_rotation_angle_deg": math.degrees(rotation_angle(rotation)),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "final_weight_min": float(final_weights.min()),
        "final_weight_max": float(final_weights.max()),
        "cables": cable_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate all manual cable calibration samples into one robust fused .calib file."
    )
    parser.add_argument("--cable-root", type=Path, default=DEFAULT_CABLE_ROOT)
    parser.add_argument("--calib-file", type=Path, default=DEFAULT_CALIB_FILE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CABLE_ROOT / "fr3_calibration_integrated.calib",
    )
    parser.add_argument(
        "--samples-per-cable",
        type=int,
        default=200,
        help="Maximum resampled point pairs per cable. Use 0 to keep each cable's full comparable count.",
    )
    parser.add_argument("--huber-delta", type=float, default=None, help="Huber residual threshold in meters.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    if args.samples_per_cable is not None and args.samples_per_cable <= 0:
        args.samples_per_cable = None
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    summary = fuse_calibration(args)
    if args.print_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Wrote integrated calibration: {summary['output']}")
        print(
            "Final residuals: "
            f"rms={summary['final_metrics']['rms_error_m']:.6f} m, "
            f"median={summary['final_metrics']['median_distance_m']:.6f} m, "
            f"p95={summary['final_metrics']['p95_distance_m']:.6f} m"
        )


if __name__ == "__main__":
    main()
