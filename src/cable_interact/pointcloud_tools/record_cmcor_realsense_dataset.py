#!/usr/bin/env python3
"""Record RealSense frames and Franka interaction state as a CMCor dataset sequence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32

from pointcloud_tools.remote_transfer import ScpTransferQueue


class CmcorRealsenseDatasetRecorder(Node):
    def __init__(self) -> None:
        super().__init__("cmcor_realsense_dataset_recorder")
        self.declare_parameter(
            "dataset_root",
            "/home/flexcycle/Motion_Correlaion/cmcor/datasets/CMCor",
        )
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("recording_active_topic", "/cable_interaction/recording_active")
        self.declare_parameter("action_topic", "/cable_interaction/action_index")
        self.declare_parameter("action_index_topic", "")
        self.declare_parameter("ee_point_topic", "/cable_interaction/ee_point")
        self.declare_parameter("save_depth", True)
        self.declare_parameter("min_frame_period", 1.0 / 30.0)
        self.declare_parameter("poor_arm_masks", True)
        self.declare_parameter("gripper_depth", 1.1)
        self.declare_parameter("crop_origin_x", -1)
        self.declare_parameter("crop_origin_y", -1)
        self.declare_parameter("crop_width", 0)
        self.declare_parameter("crop_height", 0)
        self.declare_parameter("scale_factor", 1.0)
        self.declare_parameter("robot_base_to_camera_transform_json", "")
        self.declare_parameter("scp_destination", "")
        self.declare_parameter("scp_timeout_sec", 300.0)

        self.dataset_root = Path(self.get_parameter("dataset_root").value).expanduser()
        self.buffer_root = self.dataset_root / "motion_correlation_buffers"
        self.annotation_root = self.dataset_root / "motion_correlation_annotations"
        self.save_depth = bool(self.get_parameter("save_depth").value)
        self.min_frame_period = float(self.get_parameter("min_frame_period").value)
        self.poor_arm_masks = bool(self.get_parameter("poor_arm_masks").value)
        self.gripper_depth = float(self.get_parameter("gripper_depth").value)
        self.crop_origin_x = int(self.get_parameter("crop_origin_x").value)
        self.crop_origin_y = int(self.get_parameter("crop_origin_y").value)
        self.crop_width = int(self.get_parameter("crop_width").value)
        self.crop_height = int(self.get_parameter("crop_height").value)
        self.scale_factor = float(self.get_parameter("scale_factor").value)
        self.robot_base_to_camera_transform = self._parse_transform_parameter()
        scp_destination = str(self.get_parameter("scp_destination").value).strip()
        self.uploader = (
            ScpTransferQueue(
                scp_destination,
                timeout_sec=float(self.get_parameter("scp_timeout_sec").value),
                log_info=self.get_logger().info,
                log_error=self.get_logger().error,
            )
            if scp_destination
            else None
        )
        action_index_topic = str(self.get_parameter("action_index_topic").value).strip()
        self.action_topic = (
            action_index_topic
            if action_index_topic
            else str(self.get_parameter("action_topic").value)
        )

        self.bridge = CvBridge()
        self.recording = False
        self.sequence_dir: Optional[Path] = None
        self.sequence_name: Optional[str] = None
        self.frame_index = 0
        self.last_saved_stamp: Optional[float] = None
        self.image_shape: Optional[tuple[int, int]] = None
        self.action_buffer: list[int] = []
        self.ee_point_buffer: list[list[float]] = []
        self.latest_action = -1
        self.latest_ee_point: Optional[list[float]] = None
        self.latest_depth_msg: Optional[Image] = None
        self.camera_matrix: Optional[list[list[float]]] = None

        self.buffer_root.mkdir(parents=True, exist_ok=True)
        (self.annotation_root / "test").mkdir(parents=True, exist_ok=True)
        (self.annotation_root / "validation").mkdir(parents=True, exist_ok=True)

        self.create_subscription(
            Bool,
            str(self.get_parameter("recording_active_topic").value),
            self.recording_active_callback,
            10,
        )
        self.create_subscription(
            Int32,
            self.action_topic,
            self.action_callback,
            10,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("ee_point_topic").value),
            self.ee_point_callback,
            30,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("color_topic").value),
            self.color_callback,
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self.depth_callback,
            10,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self.camera_info_callback,
            10,
        )
        self.get_logger().info(
            f"CMCor recorder ready. Dataset root: {self.dataset_root}. "
            f"Action topic: {self.action_topic}"
        )

    def _parse_transform_parameter(self) -> Optional[list[list[float]]]:
        raw = str(self.get_parameter("robot_base_to_camera_transform_json").value).strip()
        if not raw:
            return None
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Invalid transform JSON: {exc}")
            return None
        array = np.asarray(values, dtype=float)
        if array.shape == (16,):
            array = array.reshape(4, 4)
        if array.shape != (4, 4):
            self.get_logger().error(
                "robot_base_to_camera_transform_json must be a 4x4 matrix or 16 values."
            )
            return None
        return array.tolist()

    @staticmethod
    def stamp_to_float(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def recording_active_callback(self, msg: Bool) -> None:
        if msg.data and not self.recording:
            self.start_sequence()
        elif not msg.data and self.recording:
            self.finish_sequence()

    def action_callback(self, msg: Int32) -> None:
        self.latest_action = int(msg.data)

    def ee_point_callback(self, msg: PointStamped) -> None:
        self.latest_ee_point = [float(msg.point.x), float(msg.point.y), float(msg.point.z)]

    def depth_callback(self, msg: Image) -> None:
        self.latest_depth_msg = msg

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_matrix = [
            [float(msg.k[0]), float(msg.k[1]), float(msg.k[2])],
            [float(msg.k[3]), float(msg.k[4]), float(msg.k[5])],
            [float(msg.k[6]), float(msg.k[7]), float(msg.k[8])],
        ]

    def start_sequence(self) -> None:
        self.sequence_name = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.sequence_dir = self.buffer_root / self.sequence_name
        self.sequence_dir.mkdir(parents=True, exist_ok=False)
        self.frame_index = 0
        self.last_saved_stamp = None
        self.image_shape = None
        self.action_buffer = []
        self.ee_point_buffer = []
        self.recording = True
        self.get_logger().info(f"Started CMCor sequence: {self.sequence_dir}")

    def finish_sequence(self) -> None:
        if self.sequence_dir is None:
            self.recording = False
            return
        if self.frame_index == 0:
            self.get_logger().warn(
                f"Recording stopped without frames: {self.sequence_dir}"
            )
            self.recording = False
            return

        json_dict = {
            "action_buffer": self.action_buffer,
            "ee_point_buffer": self.ee_point_buffer,
            "poor_arm_masks": self.poor_arm_masks,
            "scale_factor": self.scale_factor,
        }
        if self.camera_matrix is not None:
            json_dict["camera_matrix"] = self.camera_matrix
        if self.robot_base_to_camera_transform is not None:
            json_dict["robot_base_to_camera_transform"] = self.robot_base_to_camera_transform
        else:
            json_dict["gripper_depth"] = self.gripper_depth
        if self.crop_origin_x >= 0 and self.crop_origin_y >= 0 and self.crop_width > 0 and self.crop_height > 0:
            json_dict["crop_origin"] = [self.crop_origin_x, self.crop_origin_y]
            json_dict["crop_size"] = [self.crop_width, self.crop_height]

        with (self.sequence_dir / "actions_gripper.json").open("w", encoding="utf-8") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)

        if self.image_shape is not None:
            grasped_placeholder = np.zeros(self.image_shape, dtype=np.uint8)
            cv2.imwrite(
                str(self.sequence_dir / "grasped_cable_00000000.png"),
                grasped_placeholder,
            )

        self.get_logger().info(
            f"Finished CMCor sequence {self.sequence_name}: {self.frame_index} frames."
        )
        if self.uploader is not None:
            self.uploader.submit(self.sequence_dir)
        self.recording = False
        self.sequence_dir = None
        self.sequence_name = None

    def color_callback(self, msg: Image) -> None:
        if not self.recording or self.sequence_dir is None:
            return
        if self.latest_ee_point is None:
            self.get_logger().warn("Skipping frame until the first EE point is received.")
            return
        stamp = self.stamp_to_float(msg)
        if self.last_saved_stamp is not None and stamp - self.last_saved_stamp < self.min_frame_period:
            return

        try:
            color_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001 - log conversion errors from cv_bridge.
            self.get_logger().error(f"Failed to convert color frame: {exc}")
            return

        if color_bgr.ndim != 3:
            self.get_logger().error("Color frame is not a 3-channel image; skipping.")
            return

        self.image_shape = color_bgr.shape[:2]
        rgb_path = self.sequence_dir / f"rgb_{self.frame_index:08d}.png"
        cv2.imwrite(str(rgb_path), color_bgr)

        if self.save_depth and self.latest_depth_msg is not None:
            self.save_depth_frame(self.latest_depth_msg)

        self.action_buffer.append(int(self.latest_action))
        self.ee_point_buffer.append(list(self.latest_ee_point))
        self.frame_index += 1
        self.last_saved_stamp = stamp

    def save_depth_frame(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to convert depth frame: {exc}")
            return
        if depth.dtype in (np.float32, np.float64):
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
        depth_path = self.sequence_dir / f"depth_{self.frame_index:08d}.png"
        cv2.imwrite(str(depth_path), depth)

    def destroy_node(self) -> bool:
        if self.recording:
            self.finish_sequence()
        if self.uploader is not None:
            self.uploader.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CmcorRealsenseDatasetRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
