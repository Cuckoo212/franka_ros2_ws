#!/usr/bin/env python3
"""Run repeated CMCor interactions by sampling new grasp plans after each recording."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue, SetParametersResult
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from std_msgs.msg import Bool

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


DEFAULT_CABLE_INTERACT_ROOT = Path("/home/flexcycle/franka_ros2_ws/src/cable_interact")
DEFAULT_CMCOR_ROOT = Path("/home/flexcycle/Motion_Correlaion/cmcor")
DEFAULT_GRASP_PLAN_ROOT = DEFAULT_CABLE_INTERACT_ROOT / "pointcloud_tools" / "info_for_3Dpoint"
DEFAULT_DATASET_ROOT = Path("/home/flexcycle/Motion_Correlaion/cmcor/datasets/CMCor")
DEFAULT_HANDEYE_CALIBRATION_PATH = Path(
    "/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib"
)


def add_import_path(path: Path) -> None:
    path_text = str(path.expanduser())
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def normalize_number(value: int) -> str:
    return f"{value:03d}"


def normalize_cable_dir_name(cable_id: str) -> str:
    cable_id = str(cable_id).strip()
    if cable_id.startswith("cable_"):
        suffix = cable_id[len("cable_") :]
    else:
        suffix = cable_id
    if suffix.isdigit():
        suffix = normalize_number(int(suffix))
    return f"cable_{suffix}"


def make_sample_grasp_id(sample_index: int) -> str:
    return f"grasp_sample_{sample_index:03d}"


def resolve_plan_path(grasp_plan_root: Path, target_id: str) -> Path:
    target_id = target_id.strip()
    if "/" in target_id or "\\" in target_id:
        relative = Path(target_id)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative grasp plan id: {target_id}")
        return grasp_plan_root / relative
    normalized = normalize_number(int(target_id)) if target_id.isdigit() else target_id
    return grasp_plan_root / f"cable_{normalized}" / f"grasp_point_cable_{normalized}"


def load_grasp_plan(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    return (points @ rotation.T) + translation[None, :]


def transform_vectors(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    return vectors @ rotation.T


def robot_point_to_camera(point_robot: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return rotation.T @ (point_robot - translation)


def build_gripper_directions(axis: np.ndarray, sample_count: int) -> list[list[float]]:
    axis = axis.astype(np.float64)
    axis /= np.linalg.norm(axis)
    reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(axis.dot(reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    directions = []
    for idx in range(sample_count):
        theta = 2.0 * math.pi * idx / sample_count
        direction = math.cos(theta) * basis_u + math.sin(theta) * basis_v
        direction /= np.linalg.norm(direction)
        directions.append(direction.tolist())
    return directions


def load_last_depth(sequence_dir: Path) -> np.ndarray:
    depth_paths = sorted(sequence_dir.glob("depth_*.png"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth_*.png files found in {sequence_dir}")
    depth = cv2.imread(str(depth_paths[-1]), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_paths[-1]}")
    return depth


def latest_sequence_dir(buffer_root: Path, started_after: float) -> Optional[Path]:
    candidates = [path for path in buffer_root.iterdir() if path.is_dir()]
    candidates = [path for path in candidates if path.stat().st_mtime >= started_after]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


class MultiGraspOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__("multigrasp_orchestrator")
        self.declare_parameter("cable_id", "001")
        self.declare_parameter("initial_grasp_index", 1)
        self.declare_parameter("max_grasps", 3)
        self.declare_parameter("controller_node", "/cartesian_cable_air_interact_controller")
        self.declare_parameter("recording_active_topic", "/cable_interaction/recording_active")
        self.declare_parameter("dataset_root", str(DEFAULT_DATASET_ROOT))
        self.declare_parameter("grasp_plan_root", str(DEFAULT_GRASP_PLAN_ROOT))
        self.declare_parameter("cable_interact_root", str(DEFAULT_CABLE_INTERACT_ROOT))
        self.declare_parameter("cmcor_python_root", str(DEFAULT_CMCOR_ROOT))
        self.declare_parameter("handeye_calibration_path", str(DEFAULT_HANDEYE_CALIBRATION_PATH))
        self.declare_parameter("current_grasp_plan_id", "")
        self.declare_parameter("algorithm", "correlation")
        self.declare_parameter("gpu", False)
        self.declare_parameter("use_dummy_flow", False)
        self.declare_parameter("gripper_direction_sample_count", 36)
        self.declare_parameter("processing_delay_sec", 0.7)
        self.declare_parameter("sequence_wait_timeout_sec", 8.0)
        self.declare_parameter("write_debug_masks", True)
        self.declare_parameter("update_multigrasp_sequences_json", True)
        self.declare_parameter("multigrasp_sequences_json_path", "")

        self.cable_id = str(self.get_parameter("cable_id").value)
        self.cable_dir_name = normalize_cable_dir_name(self.cable_id)
        self.current_grasp_index = int(self.get_parameter("initial_grasp_index").value)
        self.max_grasps = int(self.get_parameter("max_grasps").value)
        self.controller_node = str(self.get_parameter("controller_node").value)
        self.dataset_root = Path(str(self.get_parameter("dataset_root").value)).expanduser()
        self.buffer_root = self.dataset_root / "motion_correlation_buffers"
        self.grasp_plan_root = Path(str(self.get_parameter("grasp_plan_root").value)).expanduser()
        self.multi_grasp_dir = self.grasp_plan_root / "multi_grasp" / self.cable_dir_name
        self.cable_interact_root = Path(str(self.get_parameter("cable_interact_root").value)).expanduser()
        self.cmcor_python_root = Path(str(self.get_parameter("cmcor_python_root").value)).expanduser()
        self.handeye_calibration_path = Path(
            str(self.get_parameter("handeye_calibration_path").value)
        ).expanduser()
        self.algorithm = str(self.get_parameter("algorithm").value)
        self.gpu = bool(self.get_parameter("gpu").value)
        self.use_dummy_flow = bool(self.get_parameter("use_dummy_flow").value)
        self.gripper_direction_sample_count = int(
            self.get_parameter("gripper_direction_sample_count").value
        )
        self.processing_delay_sec = float(self.get_parameter("processing_delay_sec").value)
        self.sequence_wait_timeout_sec = float(
            self.get_parameter("sequence_wait_timeout_sec").value
        )
        self.write_debug_masks = bool(self.get_parameter("write_debug_masks").value)
        self.update_multigrasp_sequences_json = bool(
            self.get_parameter("update_multigrasp_sequences_json").value
        )
        multigrasp_json_path = str(self.get_parameter("multigrasp_sequences_json_path").value).strip()
        self.multigrasp_sequences_json_path = (
            Path(multigrasp_json_path).expanduser()
            if multigrasp_json_path
            else self.dataset_root / "multigrasp_sequences.json"
        )
        self.recorded_sequence_group: list[str] = []

        add_import_path(self.cable_interact_root)
        add_import_path(self.cmcor_python_root)

        from pointcloud_tools import build_cable_point_cloud as pointcloud_tools

        self.pointcloud_tools = pointcloud_tools
        handeye = pointcloud_tools.load_handeye_transform(self.handeye_calibration_path)
        self.camera_to_robot_rotation = np.asarray(handeye["rotation_matrix"], dtype=np.float64)
        self.camera_to_robot_translation = np.asarray(
            handeye["translation_vector"], dtype=np.float64
        )
        self.camera_frame = str(handeye["source_frame"])
        self.robot_frame = str(handeye["target_frame"])

        self.previous_grasps_robot: list[np.ndarray] = []
        self.previous_grasps_camera: list[np.ndarray] = []
        self.last_recording_stop_time: Optional[float] = None
        self.processing_requested = False
        self.processing = False
        self.last_recording_active: Optional[bool] = None

        self.get_parameters_client = self.create_client(
            GetParameters, self.parameter_service_name("get_parameters")
        )
        self.set_parameters_client = self.create_client(
            SetParameters, self.parameter_service_name("set_parameters")
        )
        self._initialize_previous_grasp()

        self.create_subscription(
            Bool,
            str(self.get_parameter("recording_active_topic").value),
            self.recording_active_callback,
            10,
        )
        self.create_timer(0.5, self.timer_callback)
        self.get_logger().info(
            f"Multi-grasp orchestrator ready for {self.cable_dir_name}, "
            f"current grasp {self.current_grasp_index}, max grasps {self.max_grasps}."
        )

    def _initialize_previous_grasp(self) -> None:
        plan_id = str(self.get_parameter("current_grasp_plan_id").value).strip()
        if not plan_id:
            plan_id = self._get_controller_target_id()
        if not plan_id:
            self.get_logger().warn(
                "No current_grasp_plan_id was provided and target_cable_id could not be read. "
                "The first sampled grasp will not have a previous-grasp distance constraint."
            )
            return
        try:
            plan = load_grasp_plan(resolve_plan_path(self.grasp_plan_root, plan_id))
            point_robot = np.asarray(plan["grasp_point"], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Could not initialize previous grasp from {plan_id}: {exc}")
            return
        self._remember_robot_grasp(point_robot)
        self.get_logger().info(
            f"Initialized previous grasp from {plan_id}: "
            f"[{point_robot[0]:.4f}, {point_robot[1]:.4f}, {point_robot[2]:.4f}]"
        )

    def parameter_service_name(self, service_name: str) -> str:
        node_name = self.controller_node.rstrip("/")
        if not node_name.startswith("/"):
            node_name = "/" + node_name
        return f"{node_name}/{service_name}"

    def _get_controller_target_id(self) -> str:
        if not self.get_parameters_client.wait_for_service(timeout_sec=1.0):
            return ""
        request = GetParameters.Request()
        request.names = ["target_cable_id"]
        future = self.get_parameters_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return ""
        values = future.result().values
        if not values:
            return ""
        return str(values[0].string_value)

    def _remember_robot_grasp(self, point_robot: np.ndarray) -> None:
        point_camera = robot_point_to_camera(
            point_robot, self.camera_to_robot_rotation, self.camera_to_robot_translation
        )
        self.previous_grasps_robot.append(point_robot)
        self.previous_grasps_camera.append(point_camera)

    def recording_active_callback(self, msg: Bool) -> None:
        active = bool(msg.data)
        if self.last_recording_active is True and not active:
            self.last_recording_stop_time = time.time()
            self.processing_requested = True
            self.get_logger().info("Interaction recording ended; scheduling multi-grasp update.")
        self.last_recording_active = active

    def timer_callback(self) -> None:
        if not self.processing_requested or self.processing:
            return
        if self.last_recording_stop_time is None:
            return
        if time.time() - self.last_recording_stop_time < self.processing_delay_sec:
            return
        self.processing_requested = False
        self.processing = True
        try:
            self.process_finished_sequence(self.last_recording_stop_time)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Multi-grasp update failed: {exc}\n{traceback.format_exc()}"
            )
        finally:
            self.processing = False

    def process_finished_sequence(self, stopped_at: float) -> None:
        sequence_dir = self.wait_for_sequence_dir(stopped_at - 30.0)
        if sequence_dir is None:
            self.get_logger().error("No freshly recorded CMCor sequence was found.")
            return
        self.register_multigrasp_sequence(sequence_dir.name)
        if self.current_grasp_index >= self.max_grasps:
            self.get_logger().info(
                "Maximum grasp count reached; recorded final sequence but not sampling another grasp."
            )
            return
        self.get_logger().info(f"Processing sequence {sequence_dir.name} for next grasp.")

        sample_grasp_index = self.current_grasp_index
        motion_masks, camera_matrix = self.compute_motion_masks(
            sequence_dir, sample_grasp_index
        )
        depth = load_last_depth(sequence_dir)
        sampled = self.sample_next_grasp(depth, camera_matrix, motion_masks)
        if sampled is None:
            self.get_logger().error("No valid next grasp was sampled.")
            return

        grasp, grasp_mask = sampled
        center_camera, axis_camera, points_camera = grasp
        next_grasp_index = self.current_grasp_index + 1
        plan_id, plan_path = self.write_next_grasp_plan(
            sample_grasp_index,
            sequence_dir,
            center_camera,
            axis_camera,
            points_camera,
            grasp_mask,
        )
        self.current_grasp_index = next_grasp_index
        center_robot = np.asarray(load_grasp_plan(plan_path)["grasp_point"], dtype=np.float64)
        self._remember_robot_grasp(center_robot)
        self.set_controller_target(plan_id)

    def register_multigrasp_sequence(self, sequence_name: str) -> None:
        if not self.update_multigrasp_sequences_json:
            return
        if sequence_name in self.recorded_sequence_group:
            return
        self.recorded_sequence_group.append(sequence_name)
        self.multigrasp_sequences_json_path.parent.mkdir(parents=True, exist_ok=True)
        groups = []
        if self.multigrasp_sequences_json_path.is_file():
            try:
                with self.multigrasp_sequences_json_path.open("r", encoding="utf-8") as fp:
                    loaded = json.load(fp)
                if isinstance(loaded, list):
                    groups = loaded
            except (OSError, json.JSONDecodeError) as exc:
                self.get_logger().warn(
                    f"Could not read {self.multigrasp_sequences_json_path}: {exc}; rewriting it."
                )
        if not groups or groups[-1] is not self.recorded_sequence_group:
            if groups and groups[-1] == self.recorded_sequence_group[:-1]:
                groups[-1] = list(self.recorded_sequence_group)
            elif groups and all(name in groups[-1] for name in self.recorded_sequence_group[:-1]):
                groups[-1] = list(self.recorded_sequence_group)
            else:
                groups.append(list(self.recorded_sequence_group))
        with self.multigrasp_sequences_json_path.open("w", encoding="utf-8") as fp:
            json.dump(groups, fp, indent=2)
            fp.write("\n")
        self.get_logger().info(
            f"Updated multigrasp sequence group in {self.multigrasp_sequences_json_path}: "
            f"{self.recorded_sequence_group}"
        )

    def wait_for_sequence_dir(self, started_after: float) -> Optional[Path]:
        deadline = time.time() + self.sequence_wait_timeout_sec
        while time.time() < deadline:
            seq = latest_sequence_dir(self.buffer_root, started_after)
            if seq is not None and (seq / "actions_gripper.json").is_file():
                return seq
            time.sleep(0.2)
        return None

    def compute_motion_masks(
        self, sequence_dir: Path, sample_grasp_index: int
    ) -> tuple[list[np.ndarray], np.ndarray]:
        from cmcor import motion_perception, motion_perception_settings

        mp = motion_perception.MotionPerception(
            gpu=self.gpu, use_dummy_flow=self.use_dummy_flow
        )
        mp.load_buffer_from_disk(str(sequence_dir))
        valid_action_count = sum(1 for action in mp.action_buffer if int(action) >= 0)
        if valid_action_count == 0:
            raise RuntimeError(
                "Recorded sequence has no active action frames; action_buffer is all -1. "
                "Check the recorder action topic parameter. For this controller it should be "
                "action_topic:=/NS_1/cable_interaction/action_index, or the compatible alias "
                "action_index_topic:=/NS_1/cable_interaction/action_index."
            )
        if self.algorithm == "correlation":
            mp.compute_correlation_images()
            if mp.corr_imgs is None:
                raise RuntimeError("Motion correlation produced no correlation images.")
            settings = motion_perception_settings.MotionPerceptionSettings()
            masks = [np.nan_to_num(image, nan=0.0) > settings.motion_correlation_min_cod
                     for image in mp.corr_imgs]
        elif self.algorithm == "segmentation":
            mp.compute_segmentation_images()
            if mp.corr_imgs is None:
                raise RuntimeError("Motion segmentation produced no segmentation images.")
            settings = motion_perception_settings.MotionPerceptionSettings()
            masks = [image > settings.motion_segmentation_vote_thr for image in mp.corr_imgs]
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        if not masks:
            raise RuntimeError("Motion perception produced no masks.")
        with (sequence_dir / "actions_gripper.json").open("r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        if "camera_matrix" not in metadata:
            raise RuntimeError("Recorded sequence has no camera_matrix in actions_gripper.json.")
        camera_matrix = np.asarray(metadata["camera_matrix"], dtype=np.float64)
        if self.write_debug_masks:
            grasp_name = make_sample_grasp_id(sample_grasp_index)
            debug_dir = self.multi_grasp_dir / grasp_name / "debug" / sequence_dir.name
            debug_dir.mkdir(parents=True, exist_ok=True)
            for idx, mask in enumerate(masks):
                cv2.imwrite(str(debug_dir / f"motion_mask_{idx:02d}.png"), 255 * mask.astype(np.uint8))
        return masks, camera_matrix

    def sample_next_grasp(
        self, depth: np.ndarray, camera_matrix: np.ndarray, motion_masks: list[np.ndarray]
    ):
        from cmcor import grasp_segment_sampler

        vote_image = None
        for mask in motion_masks:
            mask = mask > 0
            if vote_image is None:
                vote_image = mask.astype(np.float32)
            else:
                vote_image[mask] += 1.0
        if vote_image is None or not np.any(vote_image > 0):
            return None

        sampler = grasp_segment_sampler.GraspSegmentSampler()
        sampler.clear_previous_grasps()
        for point_camera in self.previous_grasps_camera:
            sampler.add_previous_grasp(point_camera)
        if not self.previous_grasps_camera:
            self.get_logger().warn("Sampling without previous-grasp distance constraints.")
        sampler.set_input_images(depth, camera_matrix, vote_image)
        return sampler.sample_grasp()

    def write_next_grasp_plan(
        self,
        sample_grasp_index: int,
        sequence_dir: Path,
        center_camera: np.ndarray,
        axis_camera: np.ndarray,
        points_camera: np.ndarray,
        grasp_mask: np.ndarray,
    ) -> tuple[str, Path]:
        center_camera = np.asarray(center_camera, dtype=np.float64)
        axis_camera = np.asarray(axis_camera, dtype=np.float64)
        axis_camera /= np.linalg.norm(axis_camera)
        points_camera = np.asarray(points_camera, dtype=np.float64)

        center_robot = transform_points(
            center_camera[None, :], self.camera_to_robot_rotation, self.camera_to_robot_translation
        )[0]
        axis_robot = transform_vectors(axis_camera[None, :], self.camera_to_robot_rotation)[0]
        axis_robot /= np.linalg.norm(axis_robot)
        points_robot = transform_points(
            points_camera, self.camera_to_robot_rotation, self.camera_to_robot_translation
        )
        gripper_directions = build_gripper_directions(
            axis_robot, self.gripper_direction_sample_count
        )

        self.multi_grasp_dir.mkdir(parents=True, exist_ok=True)
        grasp_name = make_sample_grasp_id(sample_grasp_index)
        grasp_dir = self.multi_grasp_dir / grasp_name
        grasp_dir.mkdir(parents=True, exist_ok=True)
        plan_path = grasp_dir / grasp_name
        plan_id = f"multi_grasp/{self.cable_dir_name}/{grasp_name}/{grasp_name}"
        plan = {
            "source": "multigrasp_orchestrator",
            "source_sequence": sequence_dir.name,
            "camera_frame": self.camera_frame,
            "robot_frame": self.robot_frame,
            "grasp_point": center_robot.tolist(),
            "local_tangent": axis_robot.tolist(),
            "gripper_direction_sample_count": len(gripper_directions),
            "gripper_directions": gripper_directions,
            "camera_grasp_point": center_camera.tolist(),
            "camera_local_tangent": axis_camera.tolist(),
            "sampled_segment_point_count": int(points_camera.shape[0]),
        }
        with plan_path.open("w", encoding="utf-8") as fp:
            json.dump(plan, fp, sort_keys=True, indent=2)

        if self.write_debug_masks:
            cv2.imwrite(
                str(grasp_dir / f"{grasp_name}_mask.png"),
                255 * (grasp_mask > 0).astype(np.uint8),
            )
            self.pointcloud_tools.save_point_cloud_ply(
                str(grasp_dir / f"{grasp_name}_camera_segment.ply"), points_camera
            )
            self.pointcloud_tools.save_point_cloud_ply(
                str(grasp_dir / f"{grasp_name}_robot_segment.ply"), points_robot
            )

        self.get_logger().info(
            f"Wrote next grasp plan {plan_id}: robot point "
            f"[{center_robot[0]:.4f}, {center_robot[1]:.4f}, {center_robot[2]:.4f}]"
        )
        return plan_id, plan_path

    def set_controller_target(self, plan_id: str) -> None:
        if not self.set_parameters_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(f"Controller parameter service not available: {self.controller_node}")
            return
        parameter = ParameterMsg()
        parameter.name = "target_cable_id"
        parameter.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=plan_id)
        request = SetParameters.Request()
        request.parameters = [parameter]
        future = self.set_parameters_client.call_async(request)

        def on_done(done_future) -> None:
            try:
                response = done_future.result()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Failed to set target_cable_id={plan_id}: {exc}")
                return
            results: list[SetParametersResult] = response.results
            if results and results[0].successful:
                self.get_logger().info(f"Triggered controller with target_cable_id={plan_id}")
            else:
                reason = results[0].reason if results else "unknown failure"
                self.get_logger().error(f"Controller rejected target_cable_id={plan_id}: {reason}")

        future.add_done_callback(on_done)


def main() -> None:
    rclpy.init()
    node = MultiGraspOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
