#!/usr/bin/python3.10

import argparse
from pathlib import Path
from typing import List, Sequence

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml

DEFAULT_TOPIC = "/ply_cloud"
DEFAULT_FRAME_ID = "fr3_link0"
DEFAULT_RATE_HZ = 2.0
DEFAULT_CALIB_FILE = Path("/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib")
DEFAULT_PLY_ROOT = Path("/home/flexcycle/cv_models/cmcor/info_for_3Dpoint")
DEFAULT_CAMERA_TOPIC = "/ply_cloud_camera_frame"


def parse_ascii_ply_vertices(ply_path: Path) -> List[Sequence[float]]:
    with ply_path.open("r", encoding="utf-8") as ply_file:
        lines = ply_file.readlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{ply_path} is not a valid PLY file.")

    vertex_count = None
    vertex_properties: List[str] = []
    in_vertex_element = False
    header_end_idx = None

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith("format ") and "ascii" not in line:
            raise ValueError("Only ASCII PLY files are supported.")
        if line.startswith("element "):
            tokens = line.split()
            in_vertex_element = len(tokens) >= 3 and tokens[1] == "vertex"
            if in_vertex_element:
                vertex_count = int(tokens[2])
                vertex_properties = []
        elif in_vertex_element and line.startswith("property "):
            tokens = line.split()
            vertex_properties.append(tokens[-1])
        elif line == "end_header":
            header_end_idx = idx
            break

    if vertex_count is None or header_end_idx is None:
        raise ValueError("Failed to parse vertex section from PLY header.")

    try:
        x_idx = vertex_properties.index("x")
        y_idx = vertex_properties.index("y")
        z_idx = vertex_properties.index("z")
    except ValueError as exc:
        raise ValueError("PLY vertex properties must contain x, y and z.") from exc

    vertices: List[Sequence[float]] = []
    vertex_lines = lines[header_end_idx + 1 : header_end_idx + 1 + vertex_count]
    for raw_line in vertex_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        values = stripped.split()
        vertices.append(
            (
                float(values[x_idx]),
                float(values[y_idx]),
                float(values[z_idx]),
            )
        )

    if not vertices:
        raise ValueError("No vertices were found in the PLY file.")

    return vertices


def load_calibration(calib_path: Path):
    with calib_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    parameters = data["parameters"]
    transform = data["transform"]

    return (
        parameters["robot_base_frame"],
        parameters["tracking_base_frame"],
        transform["translation"],
        transform["rotation"],
    )


def build_cable_paths(cable_id: str) -> tuple[Path, Path]:
    normalized_cable_id = cable_id.strip()
    if normalized_cable_id.isdigit():
        normalized_cable_id = f"{int(normalized_cable_id):03d}"

    cable_dir = DEFAULT_PLY_ROOT / f"cable_{normalized_cable_id}"
    robot_ply_path = cable_dir / "cable_robot_frame.ply"
    camera_ply_path = cable_dir / "cable_camera_frame.ply"
    return robot_ply_path, camera_ply_path


class PlyCloudPublisher(Node):
    def __init__(
        self,
        ply_path: Path,
        topic: str,
        frame_id: str,
        rate_hz: float,
        camera_ply_path: Path | None = None,
        camera_topic: str = "/ply_cloud_camera_frame",
        camera_frame_id: str | None = None,
        calib_path: Path | None = None,
    ):
        super().__init__("ply_cloud_publisher")
        self._topic = topic
        self._frame_id = frame_id
        self._points = parse_ascii_ply_vertices(ply_path)
        self._publisher = self.create_publisher(PointCloud2, topic, 10)
        self._camera_points = None
        self._camera_publisher = None
        self._camera_topic = camera_topic
        self._camera_frame_id = camera_frame_id
        self._timer = self.create_timer(1.0 / rate_hz, self.publish_cloud)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        if self._camera_frame_id is None and calib_path is not None:
            _, child_frame, _, _ = load_calibration(calib_path)
            self._camera_frame_id = child_frame

        self.get_logger().info(
            f"Loaded {len(self._points)} points from {ply_path} and publishing on {topic} in frame {frame_id}."
        )
        if camera_ply_path is not None:
            self._camera_points = parse_ascii_ply_vertices(camera_ply_path)
            self._camera_publisher = self.create_publisher(PointCloud2, camera_topic, 10)
            self.get_logger().info(
                f"Loaded {len(self._camera_points)} points from {camera_ply_path} "
                f"and publishing on {camera_topic} in frame {self._camera_frame_id}."
            )
        if calib_path is not None:
            self.publish_camera_tf(calib_path)

    def publish_cloud(self) -> None:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._frame_id
        cloud_msg = point_cloud2.create_cloud_xyz32(header, self._points)
        self._publisher.publish(cloud_msg)
        if self._camera_points is not None and self._camera_publisher is not None:
            camera_header = Header()
            camera_header.stamp = header.stamp
            camera_header.frame_id = self._camera_frame_id
            camera_cloud_msg = point_cloud2.create_cloud_xyz32(camera_header, self._camera_points)
            self._camera_publisher.publish(camera_cloud_msg)

    def publish_camera_tf(self, calib_path: Path) -> None:
        parent_frame, child_frame, translation, rotation = load_calibration(calib_path)
        if self._camera_frame_id is None:
            self._camera_frame_id = child_frame
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(translation["x"])
        transform.transform.translation.y = float(translation["y"])
        transform.transform.translation.z = float(translation["z"])
        transform.transform.rotation.x = float(rotation["x"])
        transform.transform.rotation.y = float(rotation["y"])
        transform.transform.rotation.z = float(rotation["z"])
        transform.transform.rotation.w = float(rotation["w"])
        self._static_tf_broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"Published static TF {parent_frame} -> {child_frame} from calibration file {calib_path}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish an ASCII PLY point cloud as sensor_msgs/PointCloud2 for RViz."
    )
    parser.add_argument(
        "cable_id",
        help="Cable index such as '014' or '14'. The script resolves both robot and camera PLY paths automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_ply_path, camera_ply_path = build_cable_paths(args.cable_id)
    if not robot_ply_path.is_file():
        raise FileNotFoundError(f"Robot-frame PLY file not found: {robot_ply_path}")
    if DEFAULT_RATE_HZ <= 0.0:
        raise ValueError("DEFAULT_RATE_HZ must be greater than 0.")
    if not DEFAULT_CALIB_FILE.is_file():
        raise FileNotFoundError(f"Calibration file not found: {DEFAULT_CALIB_FILE}")
    if not camera_ply_path.is_file():
        raise FileNotFoundError(f"Camera-frame PLY file not found: {camera_ply_path}")

    rclpy.init()
    node = PlyCloudPublisher(
        robot_ply_path,
        DEFAULT_TOPIC,
        DEFAULT_FRAME_ID,
        DEFAULT_RATE_HZ,
        camera_ply_path,
        DEFAULT_CAMERA_TOPIC,
        None,
        DEFAULT_CALIB_FILE,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
