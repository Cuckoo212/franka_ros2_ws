#!/usr/bin/python3.10

import argparse
import json
import re
from pathlib import Path
from typing import List, Sequence

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import yaml

DEFAULT_TOPIC = "/ply_cloud"
DEFAULT_FRAME_ID = "fr3_link0"
DEFAULT_RATE_HZ = 2.0
DEFAULT_CALIB_FILE = Path("/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib")
DEFAULT_PLY_ROOT = Path("/home/flexcycle/cv_models/cmcor/info_for_3Dpoint")
DEFAULT_CAMERA_TOPIC = "/ply_cloud_camera_frame"
DEFAULT_VECTOR_MARKER_TOPIC = "/cable_grasp_vectors"
DEFAULT_DIRECTION_LENGTH = 0.16
DEFAULT_SELECTED_DIRECTION_LENGTH = 0.22
DEFAULT_TARGET_TCP_FRAME_ID = "target_tcp_frame"


def build_grasp_plan_path(cable_id: str) -> Path:
    normalized_cable_id = cable_id.strip()
    if normalized_cable_id.isdigit():
        normalized_cable_id = f"{int(normalized_cable_id):03d}"
    return DEFAULT_PLY_ROOT / f"cable_{normalized_cable_id}" / f"grasp_point_cable_{normalized_cable_id}"


def build_selection_asset_path(cable_id: str) -> Path:
    normalized_cable_id = cable_id.strip()
    if normalized_cable_id.isdigit():
        normalized_cable_id = f"{int(normalized_cable_id):03d}"
    return (
        DEFAULT_PLY_ROOT
        / f"cable_{normalized_cable_id}"
        / f"grasp_curve_selection_cable_{normalized_cable_id}.js"
    )


def load_grasp_vectors(
    grasp_plan_path: Path,
) -> tuple[Sequence[float], list[Sequence[float]], list[Sequence[float]], list[int], int | None]:
    data = json.loads(grasp_plan_path.read_text(encoding="utf-8"))
    grasp_point = data["grasp_point"]
    gripper_directions = data["gripper_directions"]
    pregrasp_points = data.get("pregrasp_points", [])
    pregrasp_segment_labels = data.get("pregrasp_segment_labels", [])
    selected_pregrasp_label = data.get("selected_pregrasp_label")
    if len(grasp_point) != 3:
        raise ValueError(f"Invalid grasp_point in {grasp_plan_path}")
    return (
        grasp_point,
        gripper_directions,
        pregrasp_points,
        pregrasp_segment_labels,
        selected_pregrasp_label,
    )


def load_selected_direction(selection_asset_path: Path) -> dict | None:
    if not selection_asset_path.is_file():
        return None

    text = selection_asset_path.read_text(encoding="utf-8")
    index_match = re.search(r'"selected_gripper_direction_index"\s*:\s*(\d+)', text)
    def extract_vector(key: str) -> list[float] | None:
        match = re.search(
            rf'"{key}"\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]',
            text,
        )
        if match is None:
            return None
        return [float(match.group(1)), float(match.group(2)), float(match.group(3))]

    y_axis = extract_vector("selected_gripper_direction")
    x_axis = extract_vector("selected_tcp_x_axis")
    z_axis = extract_vector("selected_tcp_z_axis")
    if index_match is None or y_axis is None:
        return None

    return {
        "index": int(index_match.group(1)),
        "y_axis": y_axis,
        "x_axis": x_axis,
        "z_axis": z_axis,
    }


def build_point(coords: Sequence[float]) -> Point:
    point = Point()
    point.x = float(coords[0])
    point.y = float(coords[1])
    point.z = float(coords[2])
    return point


def build_vector_marker_array(
    frame_id: str,
    stamp,
    grasp_point: Sequence[float],
    gripper_directions: list[Sequence[float]],
    pregrasp_points: list[Sequence[float]],
    pregrasp_segment_labels: list[int],
    selected_pregrasp_label: int | None,
    selected_direction: dict | None,
) -> MarkerArray:
    marker_array = MarkerArray()
    origin = build_point(grasp_point)

    candidate_marker = Marker()
    candidate_marker.header.frame_id = frame_id
    candidate_marker.header.stamp = stamp
    candidate_marker.ns = "candidate_gripper_directions"
    candidate_marker.id = 0
    candidate_marker.type = Marker.LINE_LIST
    candidate_marker.action = Marker.ADD
    candidate_marker.pose.orientation.w = 1.0
    candidate_marker.scale.x = 0.0025
    candidate_marker.color.r = 0.08
    candidate_marker.color.g = 0.72
    candidate_marker.color.b = 0.65
    candidate_marker.color.a = 0.95
    for direction in gripper_directions:
        endpoint = [
            float(grasp_point[i]) + DEFAULT_DIRECTION_LENGTH * float(direction[i]) for i in range(3)
        ]
        candidate_marker.points.append(origin)
        candidate_marker.points.append(build_point(endpoint))
    marker_array.markers.append(candidate_marker)

    grasp_point_marker = Marker()
    grasp_point_marker.header.frame_id = frame_id
    grasp_point_marker.header.stamp = stamp
    grasp_point_marker.ns = "grasp_point"
    grasp_point_marker.id = 1
    grasp_point_marker.type = Marker.SPHERE
    grasp_point_marker.action = Marker.ADD
    grasp_point_marker.pose.position = origin
    grasp_point_marker.pose.orientation.w = 1.0
    grasp_point_marker.scale.x = 0.010
    grasp_point_marker.scale.y = 0.010
    grasp_point_marker.scale.z = 0.010
    grasp_point_marker.color.r = 0.15
    grasp_point_marker.color.g = 0.39
    grasp_point_marker.color.b = 0.92
    grasp_point_marker.color.a = 1.0
    marker_array.markers.append(grasp_point_marker)

    pregrasp_marker = Marker()
    pregrasp_marker.header.frame_id = frame_id
    pregrasp_marker.header.stamp = stamp
    pregrasp_marker.ns = "pregrasp_points"
    pregrasp_marker.id = 4
    pregrasp_marker.type = Marker.SPHERE_LIST
    pregrasp_marker.action = Marker.ADD
    pregrasp_marker.pose.orientation.w = 1.0
    pregrasp_marker.scale.x = 0.009
    pregrasp_marker.scale.y = 0.009
    pregrasp_marker.scale.z = 0.009
    pregrasp_marker.color.r = 0.86
    pregrasp_marker.color.g = 0.15
    pregrasp_marker.color.b = 0.15
    pregrasp_marker.color.a = 1.0
    for point in pregrasp_points:
        if len(point) != 3:
            continue
        pregrasp_marker.points.append(build_point(point))
    marker_array.markers.append(pregrasp_marker)

    for index, point in enumerate(pregrasp_points):
        if len(point) != 3:
            continue
        label_marker = Marker()
        label_marker.header.frame_id = frame_id
        label_marker.header.stamp = stamp
        label_marker.ns = "pregrasp_point_labels"
        label_marker.id = 100 + index
        label_marker.type = Marker.TEXT_VIEW_FACING
        label_marker.action = Marker.ADD
        label_marker.pose.position = build_point(point)
        label_marker.pose.position.z += 0.016
        label_marker.pose.orientation.w = 1.0
        label_marker.scale.z = 0.020
        label = pregrasp_segment_labels[index] if index < len(pregrasp_segment_labels) else index + 1
        is_selected_pregrasp = selected_pregrasp_label is not None and int(label) == int(selected_pregrasp_label)
        label_marker.scale.z = 0.028 if is_selected_pregrasp else 0.020
        label_marker.color.r = 0.12 if is_selected_pregrasp else 0.80
        label_marker.color.g = 0.39 if is_selected_pregrasp else 0.10
        label_marker.color.b = 0.92 if is_selected_pregrasp else 0.10
        label_marker.color.a = 1.0
        label_marker.text = f"P{label}"
        marker_array.markers.append(label_marker)

    selected_marker = Marker()
    selected_marker.header.frame_id = frame_id
    selected_marker.header.stamp = stamp
    selected_marker.ns = "selected_gripper_direction"
    selected_marker.id = 2
    selected_marker.type = Marker.ARROW
    selected_marker.action = Marker.ADD if selected_direction is not None else Marker.DELETE
    selected_marker.pose.orientation.w = 1.0
    selected_marker.scale.x = 0.006
    selected_marker.scale.y = 0.014
    selected_marker.scale.z = 0.018
    selected_marker.color.r = 0.94
    selected_marker.color.g = 0.27
    selected_marker.color.b = 0.27
    selected_marker.color.a = 1.0
    if selected_direction is not None:
        selected_vector = selected_direction["y_axis"]
        selected_endpoint = [
            float(grasp_point[i]) + DEFAULT_SELECTED_DIRECTION_LENGTH * float(selected_vector[i])
            for i in range(3)
        ]
        selected_marker.points.append(origin)
        selected_marker.points.append(build_point(selected_endpoint))
    marker_array.markers.append(selected_marker)

    tcp_axes_marker = Marker()
    tcp_axes_marker.header.frame_id = frame_id
    tcp_axes_marker.header.stamp = stamp
    tcp_axes_marker.ns = "target_tcp_axes"
    tcp_axes_marker.id = 3
    tcp_axes_marker.type = Marker.LINE_LIST
    tcp_axes_marker.action = Marker.ADD if selected_direction is not None else Marker.DELETE
    tcp_axes_marker.pose.orientation.w = 1.0
    tcp_axes_marker.scale.x = 0.004
    if selected_direction is not None:
        axes = [
            ("x_axis", [0.95, 0.25, 0.25], 0.10),
            ("y_axis", [0.20, 0.80, 0.25], 0.10),
            ("z_axis", [0.20, 0.45, 0.95], 0.10),
        ]
        for axis_name, color, axis_length in axes:
            axis_vector = selected_direction.get(axis_name)
            if axis_vector is None:
                continue
            endpoint = [
                float(grasp_point[i]) + axis_length * float(axis_vector[i]) for i in range(3)
            ]
            tcp_axes_marker.points.append(origin)
            tcp_axes_marker.points.append(build_point(endpoint))
            for _ in range(2):
                tcp_axes_marker.colors.append(type(candidate_marker.color)())
                tcp_axes_marker.colors[-1].r = color[0]
                tcp_axes_marker.colors[-1].g = color[1]
                tcp_axes_marker.colors[-1].b = color[2]
                tcp_axes_marker.colors[-1].a = 1.0
    marker_array.markers.append(tcp_axes_marker)

    return marker_array


def quaternion_from_axes(
    x_axis: Sequence[float],
    y_axis: Sequence[float],
    z_axis: Sequence[float],
) -> tuple[float, float, float, float]:
    matrix = [
        [float(x_axis[0]), float(y_axis[0]), float(z_axis[0])],
        [float(x_axis[1]), float(y_axis[1]), float(z_axis[1])],
        [float(x_axis[2]), float(y_axis[2]), float(z_axis[2])],
    ]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        s = 0.5 / (trace + 1.0) ** 0.5
        w = 0.25 / s
        x = (matrix[2][1] - matrix[1][2]) * s
        y = (matrix[0][2] - matrix[2][0]) * s
        z = (matrix[1][0] - matrix[0][1]) * s
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        s = 2.0 * (1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) ** 0.5
        w = (matrix[2][1] - matrix[1][2]) / s
        x = 0.25 * s
        y = (matrix[0][1] + matrix[1][0]) / s
        z = (matrix[0][2] + matrix[2][0]) / s
    elif matrix[1][1] > matrix[2][2]:
        s = 2.0 * (1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) ** 0.5
        w = (matrix[0][2] - matrix[2][0]) / s
        x = (matrix[0][1] + matrix[1][0]) / s
        y = 0.25 * s
        z = (matrix[1][2] + matrix[2][1]) / s
    else:
        s = 2.0 * (1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) ** 0.5
        w = (matrix[1][0] - matrix[0][1]) / s
        x = (matrix[0][2] + matrix[2][0]) / s
        y = (matrix[1][2] + matrix[2][1]) / s
        z = 0.25 * s
    return x, y, z, w


def build_target_tcp_transform(
    frame_id: str,
    child_frame_id: str,
    stamp,
    grasp_point: Sequence[float],
    selected_direction: dict | None,
) -> TransformStamped | None:
    if selected_direction is None:
        return None
    x_axis = selected_direction.get("x_axis")
    y_axis = selected_direction.get("y_axis")
    z_axis = selected_direction.get("z_axis")
    if x_axis is None or y_axis is None or z_axis is None:
        return None

    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = frame_id
    transform.child_frame_id = child_frame_id
    transform.transform.translation.x = float(grasp_point[0])
    transform.transform.translation.y = float(grasp_point[1])
    transform.transform.translation.z = float(grasp_point[2])
    qx, qy, qz, qw = quaternion_from_axes(x_axis, y_axis, z_axis)
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    return transform


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
        grasp_plan_path: Path | None = None,
        selection_asset_path: Path | None = None,
        vector_marker_topic: str = DEFAULT_VECTOR_MARKER_TOPIC,
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
        self._grasp_point = None
        self._gripper_directions = None
        self._pregrasp_points = None
        self._pregrasp_segment_labels = None
        self._selected_pregrasp_label = None
        self._selection_asset_path = selection_asset_path
        self._vector_marker_publisher = None
        self._target_tcp_frame_id = DEFAULT_TARGET_TCP_FRAME_ID
        self._timer = self.create_timer(1.0 / rate_hz, self.publish_cloud)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._dynamic_tf_broadcaster = TransformBroadcaster(self)

        if self._camera_frame_id is None and calib_path is not None:
            _, child_frame, _, _ = load_calibration(calib_path)
            self._camera_frame_id = child_frame

        if grasp_plan_path is not None and grasp_plan_path.is_file():
            (
                self._grasp_point,
                self._gripper_directions,
                self._pregrasp_points,
                self._pregrasp_segment_labels,
                self._selected_pregrasp_label,
            ) = load_grasp_vectors(grasp_plan_path)
            self._vector_marker_publisher = self.create_publisher(MarkerArray, vector_marker_topic, 10)
            self.get_logger().info(
                f"Loaded grasp vectors from {grasp_plan_path} and publishing on {vector_marker_topic}."
            )
        elif grasp_plan_path is not None:
            self.get_logger().warning(f"Grasp plan file not found, skipping vector markers: {grasp_plan_path}")

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
        if (
            self._grasp_point is not None
            and self._gripper_directions is not None
            and self._pregrasp_points is not None
            and self._pregrasp_segment_labels is not None
            and self._vector_marker_publisher is not None
        ):
            selected_direction = None
            if self._selection_asset_path is not None:
                selected_direction = load_selected_direction(self._selection_asset_path)
            marker_array = build_vector_marker_array(
                self._frame_id,
                header.stamp,
                self._grasp_point,
                self._gripper_directions,
                self._pregrasp_points,
                self._pregrasp_segment_labels,
                self._selected_pregrasp_label,
                selected_direction,
            )
            self._vector_marker_publisher.publish(marker_array)
            target_tcp_transform = build_target_tcp_transform(
                self._frame_id,
                self._target_tcp_frame_id,
                header.stamp,
                self._grasp_point,
                selected_direction,
            )
            if target_tcp_transform is not None:
                self._dynamic_tf_broadcaster.sendTransform(target_tcp_transform)
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
    grasp_plan_path = build_grasp_plan_path(args.cable_id)
    selection_asset_path = build_selection_asset_path(args.cable_id)
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
        grasp_plan_path,
        selection_asset_path,
        DEFAULT_VECTOR_MARKER_TOPIC,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
