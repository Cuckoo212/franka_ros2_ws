#!/usr/bin/python3.10

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import rclpy
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import yaml

DEFAULT_TOPIC = "/ply_cloud"
DEFAULT_FRAME_ID = "fr3_link0"
DEFAULT_RATE_HZ = 2.0
DEFAULT_CALIB_FILE = Path("/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib")
DEFAULT_PLY_ROOT = Path("/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint")
DEFAULT_CALIBRATION_SAMPLE_ROOT = DEFAULT_PLY_ROOT / "samples_calibration/calibration_001"
DEFAULT_CAMERA_TOPIC = "/ply_cloud_camera_frame"
DEFAULT_CAMERA_FRAME_ID = "camera_color_frame"
DEFAULT_VECTOR_MARKER_TOPIC = "/cable_grasp_vectors"
DEFAULT_DIRECTION_LENGTH = 0.16
DEFAULT_SELECTED_DIRECTION_LENGTH = 0.22
DEFAULT_AXIS_LENGTH = 0.10
DEFAULT_TARGET_TCP_FRAME_ID = "target_tcp_frame"
DEFAULT_HOVER_TCP_FRAME_ID = "hover_target_frame"
DEFAULT_CANDIDATE_TCP_FRAME_PREFIX = "candidate_tcp_frame"
DEFAULT_CANDIDATE_HOVER_FRAME_PREFIX = "candidate_hover_frame"
DEFAULT_HOVER_DISTANCE = 0.15
DEFAULT_CANDIDATE_LABEL_PREFIX = "candidate_"
LINK0_TARGET_Z_AXIS = [0.0, -1.0, 0.0]
LINK0_X_AXIS = [1.0, 0.0, 0.0]
LINK0_NEG_Z_AXIS = [0.0, 0.0, -1.0]



def normalize_cable_id(cable_id: str) -> str:
    normalized_cable_id = cable_id.strip()
    if normalized_cable_id.isdigit():
        normalized_cable_id = f"{int(normalized_cable_id):03d}"
    return normalized_cable_id


def resolve_relative_grasp_path(target_id: str) -> Path | None:
    stripped = target_id.strip()
    if "/" not in stripped and "\\" not in stripped:
        return None
    relative = Path(stripped)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid relative grasp plan id: {target_id}")
    return DEFAULT_PLY_ROOT / relative


def default_grasp_plan_for_cable_dir(cable_dir: Path) -> Path | None:
    name = cable_dir.name
    if not name.startswith("cable_"):
        return None
    suffix = name[len("cable_") :]
    return cable_dir / f"grasp_point_cable_{suffix}"


def resolve_cable_dir(cable_id: str) -> Path:
    relative_path = resolve_relative_grasp_path(cable_id)
    if relative_path is not None:
        return relative_path

    normalized_cable_id = normalize_cable_id(cable_id)
    standard_dir = DEFAULT_PLY_ROOT / f"cable_{normalized_cable_id}"
    calibration_dir = DEFAULT_CALIBRATION_SAMPLE_ROOT / f"cable_{normalized_cable_id}"
    if (standard_dir / "cable_robot_frame.ply").is_file():
        return standard_dir
    if (calibration_dir / "cable_robot_frame.ply").is_file():
        return calibration_dir
    return standard_dir


def build_grasp_plan_path(cable_id: str) -> Path:
    cable_dir = resolve_cable_dir(cable_id)
    if resolve_relative_grasp_path(cable_id) is not None:
        default_plan = default_grasp_plan_for_cable_dir(cable_dir)
        if default_plan is not None and (cable_dir.is_dir() or default_plan.is_file()):
            return default_plan
        return cable_dir
    normalized_cable_id = normalize_cable_id(cable_id)
    return cable_dir / f"grasp_point_cable_{normalized_cable_id}"


def vector_norm(vector: Sequence[float]) -> float:
    return sum(float(value) * float(value) for value in vector) ** 0.5


def normalize_vector(vector: Sequence[float], fallback: Sequence[float]) -> list[float]:
    norm = vector_norm(vector)
    if norm <= 1e-12:
        return [float(value) for value in fallback]
    return [float(value) / norm for value in vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def project_onto_plane(vector: Sequence[float], plane_normal: Sequence[float]) -> list[float]:
    scale = dot(vector, plane_normal)
    return [float(vector[i]) - scale * float(plane_normal[i]) for i in range(3)]


def build_tcp_axes_from_x_direction(x_direction: Sequence[float]) -> dict:
    z_axis = normalize_vector(LINK0_TARGET_Z_AXIS, LINK0_TARGET_Z_AXIS)
    x_axis = normalize_vector(project_onto_plane(x_direction, z_axis), LINK0_X_AXIS)
    y_axis = normalize_vector(cross(z_axis, x_axis), LINK0_NEG_Z_AXIS)
    x_axis = normalize_vector(cross(y_axis, z_axis), x_axis)
    return {
        "selected_tcp_x_axis": x_axis,
        "selected_tcp_y_axis": y_axis,
        "selected_tcp_z_axis": z_axis,
        "selected_gripper_direction": x_axis,
    }


def safe_label_equals(left, right) -> bool:
    if left is None or right is None:
        return False
    return str(left) == str(right)


def sanitize_frame_token(value) -> str:
    sanitized = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in str(value)
    ).strip("_")
    return sanitized or "unnamed"


def load_grasp_plan(grasp_plan_path: Path) -> dict:
    data = json.loads(grasp_plan_path.read_text(encoding="utf-8"))
    grasp_point = data["grasp_point"]
    if len(grasp_point) != 3:
        raise ValueError(f"Invalid grasp_point in {grasp_plan_path}")

    if "gripper_directions" in data:
        gripper_directions = data["gripper_directions"]
    else:
        gripper_directions = [
            data["gripper_direction_a"],
            data["gripper_direction_b"],
        ]

    valid_directions = []
    for direction in gripper_directions:
        if len(direction) != 3:
            raise ValueError(f"Invalid gripper direction in {grasp_plan_path}: {direction}")
        valid_directions.append([float(value) for value in direction])

    pregrasp_points = data.get("pregrasp_points", [])
    pregrasp_segment_labels = data.get("pregrasp_segment_labels", [])
    valid_pregrasp_points = []
    valid_pregrasp_labels = []
    for index, point in enumerate(pregrasp_points):
        if len(point) != 3:
            continue
        valid_pregrasp_points.append([float(value) for value in point])
        if index < len(pregrasp_segment_labels):
            valid_pregrasp_labels.append(str(pregrasp_segment_labels[index]))
        else:
            valid_pregrasp_labels.append(str(index + 1))

    grasp_candidates = []
    for index, candidate in enumerate(data.get("grasp_candidates", [])):
        candidate_point = candidate.get("grasp_point")
        direction = candidate.get("selected_tcp_x_axis") or candidate.get("selected_gripper_direction")
        if direction is None and "gripper_direction_a" in candidate:
            direction = candidate["gripper_direction_a"]
        if not candidate_point or not direction or len(candidate_point) != 3 or len(direction) != 3:
            continue
        candidate_axes = {
            "selected_tcp_x_axis": candidate.get("selected_tcp_x_axis"),
            "selected_tcp_y_axis": candidate.get("selected_tcp_y_axis"),
            "selected_tcp_z_axis": candidate.get("selected_tcp_z_axis"),
        }
        if any(candidate_axes[key] is None or len(candidate_axes[key]) != 3 for key in candidate_axes):
            candidate_axes = build_tcp_axes_from_x_direction(direction)
        grasp_candidates.append(
            {
                "label": str(candidate.get("label", f"{DEFAULT_CANDIDATE_LABEL_PREFIX}{index}")),
                "grasp_point": [float(value) for value in candidate_point],
                "selected_tcp_x_axis": [float(value) for value in candidate_axes["selected_tcp_x_axis"]],
                "selected_tcp_y_axis": [float(value) for value in candidate_axes["selected_tcp_y_axis"]],
                "selected_tcp_z_axis": [float(value) for value in candidate_axes["selected_tcp_z_axis"]],
                "selected_gripper_direction": [float(value) for value in direction],
            }
        )

    return {
        "grasp_point": [float(value) for value in grasp_point],
        "gripper_directions": valid_directions,
        "pregrasp_points": valid_pregrasp_points,
        "pregrasp_segment_labels": valid_pregrasp_labels,
        "selected_pregrasp_label": data.get("selected_pregrasp_label"),
        "grasp_candidates": grasp_candidates,
    }


def load_selected_gripper_frame(grasp_plan_path: Path) -> dict | None:
    if not grasp_plan_path.is_file():
        return None

    try:
        data = json.loads(grasp_plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("selected_gripper_frame_status") != "selected":
        return None

    required_keys = [
        "selected_gripper_frame_position",
        "selected_gripper_direction",
        "selected_tcp_x_axis",
        "selected_tcp_y_axis",
        "selected_tcp_z_axis",
    ]
    for key in required_keys:
        if key not in data or len(data[key]) != 3:
            return None
    return {
        "position": data["selected_gripper_frame_position"],
        "selected_gripper_direction": data["selected_gripper_direction"],
        "selected_tcp_x_axis": data["selected_tcp_x_axis"],
        "selected_tcp_y_axis": data["selected_tcp_y_axis"],
        "selected_tcp_z_axis": data["selected_tcp_z_axis"],
        "selected_gripper_direction_label": data.get("selected_gripper_direction_label"),
    }


def build_point(coords: Sequence[float]) -> Point:
    point = Point()
    point.x = float(coords[0])
    point.y = float(coords[1])
    point.z = float(coords[2])
    return point


def build_color(red: float, green: float, blue: float, alpha: float = 1.0) -> ColorRGBA:
    color = ColorRGBA()
    color.r = float(red)
    color.g = float(green)
    color.b = float(blue)
    color.a = float(alpha)
    return color


def endpoint(origin: Sequence[float], direction: Sequence[float], length: float) -> list[float]:
    return [float(origin[i]) + length * float(direction[i]) for i in range(3)]


def hover_position_from_target(
    target_position: Sequence[float], target_z_axis: Sequence[float]
) -> list[float]:
    z_axis = normalize_vector(target_z_axis, LINK0_TARGET_Z_AXIS)
    return [
        float(target_position[index]) - DEFAULT_HOVER_DISTANCE * z_axis[index]
        for index in range(3)
    ]


def add_marker_text_common(marker: Marker, frame_id: str, stamp, ns: str, marker_id: int) -> None:
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.026
    marker.color.r = 1.0
    marker.color.g = 0.95
    marker.color.b = 0.35
    marker.color.a = 1.0


def build_vector_marker_array(
    frame_id: str,
    stamp,
    grasp_plan: dict,
    selected_frame: dict | None,
) -> MarkerArray:
    marker_array = MarkerArray()
    grasp_point = grasp_plan["grasp_point"]
    gripper_directions = grasp_plan["gripper_directions"]
    pregrasp_points = grasp_plan.get("pregrasp_points", [])
    pregrasp_segment_labels = grasp_plan.get("pregrasp_segment_labels", [])
    selected_pregrasp_label = grasp_plan.get("selected_pregrasp_label")
    grasp_origin = build_point(grasp_point)

    candidate_marker = Marker()
    candidate_marker.header.frame_id = frame_id
    candidate_marker.header.stamp = stamp
    candidate_marker.ns = "candidate_gripper_x_axes"
    candidate_marker.id = 0
    candidate_marker.type = Marker.LINE_LIST
    candidate_marker.action = Marker.ADD
    candidate_marker.pose.orientation.w = 1.0
    candidate_marker.scale.x = 0.0035
    candidate_marker.color.r = 0.08
    candidate_marker.color.g = 0.72
    candidate_marker.color.b = 0.65
    candidate_marker.color.a = 0.95
    direction_colors = [
        build_color(1.0, 0.45, 0.05, 1.0),
        build_color(0.05, 0.80, 0.95, 1.0),
    ]
    for index, direction in enumerate(gripper_directions[:2]):
        direction_end = endpoint(grasp_point, direction, DEFAULT_DIRECTION_LENGTH)
        candidate_marker.points.append(grasp_origin)
        candidate_marker.points.append(build_point(direction_end))
        color = direction_colors[index]
        candidate_marker.colors.extend([color, color])

        direction_label = Marker()
        add_marker_text_common(
            direction_label,
            frame_id,
            stamp,
            "candidate_gripper_direction_labels",
            100 + index,
        )
        direction_label.text = "a" if index == 0 else "b"
        direction_label.pose.position = build_point(
            [direction_end[0], direction_end[1], direction_end[2] + 0.018]
        )
        direction_label.color = color
        marker_array.markers.append(direction_label)
    marker_array.markers.append(candidate_marker)

    candidate_hover_axes = Marker()
    candidate_hover_axes.header.frame_id = frame_id
    candidate_hover_axes.header.stamp = stamp
    candidate_hover_axes.ns = "candidate_hover_frames"
    candidate_hover_axes.id = 110
    candidate_hover_axes.type = Marker.LINE_LIST
    candidate_hover_axes.action = Marker.ADD
    candidate_hover_axes.pose.orientation.w = 1.0
    candidate_hover_axes.scale.x = 0.004
    axis_specs = [
        ("selected_tcp_x_axis", build_color(0.95, 0.20, 0.20), 0.08),
        ("selected_tcp_y_axis", build_color(0.20, 0.85, 0.25), 0.08),
        ("selected_tcp_z_axis", build_color(0.20, 0.45, 0.95), 0.10),
    ]
    for index, direction in enumerate(gripper_directions[:2]):
        axes = build_tcp_axes_from_x_direction(direction)
        hover_position = hover_position_from_target(grasp_point, axes["selected_tcp_z_axis"])
        for axis_name, color, length in axis_specs:
            candidate_hover_axes.points.append(build_point(hover_position))
            candidate_hover_axes.points.append(
                build_point(endpoint(hover_position, axes[axis_name], length))
            )
            candidate_hover_axes.colors.extend([color, color])
        hover_label = Marker()
        add_marker_text_common(
            hover_label, frame_id, stamp, "candidate_hover_frame_labels", 120 + index
        )
        hover_label.text = f"hover {'a' if index == 0 else 'b'}"
        hover_label.pose.position = build_point(
            [hover_position[0], hover_position[1], hover_position[2] + 0.025 + 0.025 * index]
        )
        marker_array.markers.append(hover_label)
    marker_array.markers.append(candidate_hover_axes)

    pregrasp_marker = Marker()
    pregrasp_marker.header.frame_id = frame_id
    pregrasp_marker.header.stamp = stamp
    pregrasp_marker.ns = "candidate_grasp_points"
    pregrasp_marker.id = 1
    pregrasp_marker.type = Marker.SPHERE_LIST
    pregrasp_marker.action = Marker.ADD
    pregrasp_marker.pose.orientation.w = 1.0
    pregrasp_marker.scale.x = 0.012
    pregrasp_marker.scale.y = 0.012
    pregrasp_marker.scale.z = 0.012
    for index, point in enumerate(pregrasp_points):
        label = pregrasp_segment_labels[index] if index < len(pregrasp_segment_labels) else index + 1
        pregrasp_marker.points.append(build_point(point))
        if safe_label_equals(label, selected_pregrasp_label):
            pregrasp_marker.colors.append(build_color(0.15, 0.39, 0.92, 1.0))
        else:
            pregrasp_marker.colors.append(build_color(1.0, 0.63, 0.16, 0.95))
    marker_array.markers.append(pregrasp_marker)

    for index, point in enumerate(pregrasp_points):
        label = pregrasp_segment_labels[index] if index < len(pregrasp_segment_labels) else index + 1
        label_marker = Marker()
        add_marker_text_common(label_marker, frame_id, stamp, "candidate_grasp_point_labels", 20 + index)
        label_marker.text = str(label)
        label_marker.pose.position = build_point([point[0], point[1], point[2] + 0.018])
        if safe_label_equals(label, selected_pregrasp_label):
            label_marker.color.r = 0.15
            label_marker.color.g = 0.39
            label_marker.color.b = 0.92
        marker_array.markers.append(label_marker)

    grasp_point_marker = Marker()
    grasp_point_marker.header.frame_id = frame_id
    grasp_point_marker.header.stamp = stamp
    grasp_point_marker.ns = "selected_grasp_point"
    grasp_point_marker.id = 2
    grasp_point_marker.type = Marker.SPHERE
    grasp_point_marker.action = Marker.ADD
    grasp_point_marker.pose.position = grasp_origin
    grasp_point_marker.pose.orientation.w = 1.0
    grasp_point_marker.scale.x = 0.016
    grasp_point_marker.scale.y = 0.016
    grasp_point_marker.scale.z = 0.016
    grasp_point_marker.color.r = 0.15
    grasp_point_marker.color.g = 0.39
    grasp_point_marker.color.b = 0.92
    grasp_point_marker.color.a = 1.0
    marker_array.markers.append(grasp_point_marker)

    selected_marker = Marker()
    selected_marker.header.frame_id = frame_id
    selected_marker.header.stamp = stamp
    selected_marker.ns = "final_selected_tcp_x_axis"
    selected_marker.id = 3
    selected_marker.type = Marker.ARROW
    selected_marker.action = Marker.ADD if selected_frame is not None else Marker.DELETE
    selected_marker.pose.orientation.w = 1.0
    selected_marker.scale.x = 0.007
    selected_marker.scale.y = 0.016
    selected_marker.scale.z = 0.020
    selected_marker.color.r = 0.94
    selected_marker.color.g = 0.27
    selected_marker.color.b = 0.27
    selected_marker.color.a = 1.0
    if selected_frame is not None:
        frame_origin = selected_frame["position"]
        selected_marker.points.append(build_point(frame_origin))
        selected_marker.points.append(
            build_point(endpoint(frame_origin, selected_frame["selected_tcp_x_axis"], DEFAULT_SELECTED_DIRECTION_LENGTH))
        )
    marker_array.markers.append(selected_marker)

    axes_marker = Marker()
    axes_marker.header.frame_id = frame_id
    axes_marker.header.stamp = stamp
    axes_marker.ns = "final_target_tcp_axes"
    axes_marker.id = 4
    axes_marker.type = Marker.LINE_LIST
    axes_marker.action = Marker.ADD if selected_frame is not None else Marker.DELETE
    axes_marker.pose.orientation.w = 1.0
    axes_marker.scale.x = 0.0045
    if selected_frame is not None:
        frame_origin = selected_frame["position"]
        axes = [
            ("selected_tcp_x_axis", build_color(0.95, 0.25, 0.25), DEFAULT_AXIS_LENGTH),
            ("selected_tcp_y_axis", build_color(0.20, 0.80, 0.25), DEFAULT_AXIS_LENGTH),
            ("selected_tcp_z_axis", build_color(0.20, 0.45, 0.95), DEFAULT_AXIS_LENGTH),
        ]
        for axis_name, color, axis_length in axes:
            axes_marker.points.append(build_point(frame_origin))
            axes_marker.points.append(build_point(endpoint(frame_origin, selected_frame[axis_name], axis_length)))
            axes_marker.colors.append(color)
            axes_marker.colors.append(color)
    marker_array.markers.append(axes_marker)

    final_label = Marker()
    add_marker_text_common(final_label, frame_id, stamp, "final_target_tcp_label", 5)
    final_label.action = Marker.ADD if selected_frame is not None else Marker.DELETE
    final_label.text = "final TCP"
    if selected_frame is not None:
        frame_origin = selected_frame["position"]
        final_label.pose.position = build_point([frame_origin[0], frame_origin[1], frame_origin[2] + 0.035])
        final_label.color.r = 0.94
        final_label.color.g = 0.27
        final_label.color.b = 0.27
    marker_array.markers.append(final_label)

    hover_axes_marker = Marker()
    hover_axes_marker.header.frame_id = frame_id
    hover_axes_marker.header.stamp = stamp
    hover_axes_marker.ns = "final_hover_frame"
    hover_axes_marker.id = 130
    hover_axes_marker.type = Marker.LINE_LIST
    hover_axes_marker.action = Marker.ADD if selected_frame is not None else Marker.DELETE
    hover_axes_marker.pose.orientation.w = 1.0
    hover_axes_marker.scale.x = 0.005
    hover_frame_position = None
    if selected_frame is not None:
        hover_frame_position = hover_position_from_target(
            selected_frame["position"], selected_frame["selected_tcp_z_axis"]
        )
        for axis_name, color, length in axis_specs:
            hover_axes_marker.points.append(build_point(hover_frame_position))
            hover_axes_marker.points.append(
                build_point(endpoint(hover_frame_position, selected_frame[axis_name], length))
            )
            hover_axes_marker.colors.extend([color, color])
    marker_array.markers.append(hover_axes_marker)

    hover_frame_label = Marker()
    add_marker_text_common(
        hover_frame_label, frame_id, stamp, "final_hover_frame_label", 131
    )
    hover_frame_label.action = Marker.ADD if hover_frame_position is not None else Marker.DELETE
    hover_frame_label.text = "hover target (-Z 15 cm)"
    if hover_frame_position is not None:
        hover_frame_label.pose.position = build_point(
            [hover_frame_position[0], hover_frame_position[1], hover_frame_position[2] + 0.035]
        )
    marker_array.markers.append(hover_frame_label)

    return marker_array


def build_additional_grasp_point_markers(
    frame_id: str,
    stamp,
    grasp_point: Sequence[float],
    label: str,
) -> list[Marker]:
    point_marker = Marker()
    point_marker.header.frame_id = frame_id
    point_marker.header.stamp = stamp
    point_marker.ns = "additional_grasp_points"
    point_marker.id = 0
    point_marker.type = Marker.SPHERE
    point_marker.action = Marker.ADD
    point_marker.pose.position = build_point(grasp_point)
    point_marker.pose.orientation.w = 1.0
    point_marker.scale.x = 0.016
    point_marker.scale.y = 0.016
    point_marker.scale.z = 0.016
    point_marker.color.r = 0.94
    point_marker.color.g = 0.27
    point_marker.color.b = 0.27
    point_marker.color.a = 1.0

    label_marker = Marker()
    add_marker_text_common(label_marker, frame_id, stamp, "additional_grasp_point_labels", 0)
    label_marker.text = label
    label_marker.pose.position = build_point(
        [grasp_point[0], grasp_point[1], grasp_point[2] + 0.025]
    )
    label_marker.color.r = 0.94
    label_marker.color.g = 0.27
    label_marker.color.b = 0.27
    return [point_marker, label_marker]


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
    selected_frame: dict | None,
) -> TransformStamped | None:
    if selected_frame is None:
        return None

    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = frame_id
    transform.child_frame_id = child_frame_id
    position = selected_frame["position"]
    transform.transform.translation.x = float(position[0])
    transform.transform.translation.y = float(position[1])
    transform.transform.translation.z = float(position[2])
    qx, qy, qz, qw = quaternion_from_axes(
        selected_frame["selected_tcp_x_axis"],
        selected_frame["selected_tcp_y_axis"],
        selected_frame["selected_tcp_z_axis"],
    )
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    return transform


def build_hover_tcp_transform(
    frame_id: str,
    child_frame_id: str,
    stamp,
    selected_frame: dict | None,
) -> TransformStamped | None:
    if selected_frame is None:
        return None
    position = hover_position_from_target(
        selected_frame["position"], selected_frame["selected_tcp_z_axis"]
    )
    return build_tcp_transform_from_axes(
        frame_id,
        child_frame_id,
        stamp,
        position,
        selected_frame["selected_tcp_x_axis"],
        selected_frame["selected_tcp_y_axis"],
        selected_frame["selected_tcp_z_axis"],
    )


def build_tcp_transform_from_axes(
    frame_id: str,
    child_frame_id: str,
    stamp,
    position: Sequence[float],
    x_axis: Sequence[float],
    y_axis: Sequence[float],
    z_axis: Sequence[float],
) -> TransformStamped:
    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = frame_id
    transform.child_frame_id = child_frame_id
    transform.transform.translation.x = float(position[0])
    transform.transform.translation.y = float(position[1])
    transform.transform.translation.z = float(position[2])
    qx, qy, qz, qw = quaternion_from_axes(x_axis, y_axis, z_axis)
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    return transform


def build_candidate_tcp_transforms(
    frame_id: str,
    stamp,
    grasp_point: Sequence[float],
    gripper_directions: list[Sequence[float]],
) -> list[TransformStamped]:
    transforms = []
    for index, direction in enumerate(gripper_directions[:2]):
        axes = build_tcp_axes_from_x_direction(direction)
        transforms.append(
            build_tcp_transform_from_axes(
                frame_id,
                f"{DEFAULT_CANDIDATE_TCP_FRAME_PREFIX}_{index}",
                stamp,
                grasp_point,
                axes["selected_tcp_x_axis"],
                axes["selected_tcp_y_axis"],
                axes["selected_tcp_z_axis"],
            )
        )
    return transforms


def build_candidate_hover_transforms(
    frame_id: str,
    stamp,
    grasp_point: Sequence[float],
    gripper_directions: list[Sequence[float]],
) -> list[TransformStamped]:
    transforms = []
    for index, direction in enumerate(gripper_directions[:2]):
        axes = build_tcp_axes_from_x_direction(direction)
        hover_position = hover_position_from_target(
            grasp_point, axes["selected_tcp_z_axis"]
        )
        transforms.append(
            build_tcp_transform_from_axes(
                frame_id,
                f"{DEFAULT_CANDIDATE_HOVER_FRAME_PREFIX}_{index}",
                stamp,
                hover_position,
                axes["selected_tcp_x_axis"],
                axes["selected_tcp_y_axis"],
                axes["selected_tcp_z_axis"],
            )
        )
    return transforms


def build_grasp_candidate_tcp_transforms(
    frame_id: str,
    stamp,
    grasp_candidates: list[dict],
) -> list[TransformStamped]:
    transforms = []
    for index, candidate in enumerate(grasp_candidates):
        label = sanitize_frame_token(candidate.get("label", index))
        transforms.append(
            build_tcp_transform_from_axes(
                frame_id,
                f"{DEFAULT_CANDIDATE_TCP_FRAME_PREFIX}_{label}",
                stamp,
                candidate["grasp_point"],
                candidate["selected_tcp_x_axis"],
                candidate["selected_tcp_y_axis"],
                candidate["selected_tcp_z_axis"],
            )
        )
    return transforms


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
    cable_dir = resolve_cable_dir(cable_id)

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
        camera_frame_id: str | None = DEFAULT_CAMERA_FRAME_ID,
        calib_path: Path | None = None,
        grasp_plan_path: Path | None = None,
        additional_grasp_plan_path: Path | None = None,
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
        self._grasp_plan_path = grasp_plan_path
        self._grasp_plan = None
        self._additional_grasp_plan_path = additional_grasp_plan_path
        self._additional_grasp_point = None
        self._vector_marker_publisher = None
        self._target_tcp_frame_id = DEFAULT_TARGET_TCP_FRAME_ID
        self._timer = self.create_timer(1.0 / rate_hz, self.publish_cloud)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._dynamic_tf_broadcaster = TransformBroadcaster(self)

        if calib_path is not None:
            self.publish_camera_tf(calib_path)

        if grasp_plan_path is not None and grasp_plan_path.is_file():
            self._grasp_plan = load_grasp_plan(grasp_plan_path)
            self._vector_marker_publisher = self.create_publisher(MarkerArray, vector_marker_topic, 10)
            self.get_logger().info(
                f"Loaded grasp plan from {grasp_plan_path}; publishing candidate points, candidate TFs, and final TCP marker on {vector_marker_topic}."
            )
        elif grasp_plan_path is not None:
            self.get_logger().warning(
                f"Grasp plan file not found, skipping gripper direction markers: {grasp_plan_path}"
            )

        if additional_grasp_plan_path is not None and additional_grasp_plan_path.is_file():
            self._additional_grasp_point = load_grasp_plan(additional_grasp_plan_path)["grasp_point"]
            if self._vector_marker_publisher is None:
                self._vector_marker_publisher = self.create_publisher(MarkerArray, vector_marker_topic, 10)
            self.get_logger().info(
                f"Loaded additional grasp point from {additional_grasp_plan_path}; "
                f"publishing it as 'grasp point 1' on {vector_marker_topic}."
            )
        elif additional_grasp_plan_path is not None:
            self.get_logger().warning(
                f"Additional grasp plan file not found, skipping grasp point 1: {additional_grasp_plan_path}"
            )

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
    def publish_cloud(self) -> None:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._frame_id
        cloud_msg = point_cloud2.create_cloud_xyz32(header, self._points)
        self._publisher.publish(cloud_msg)
        if self._vector_marker_publisher is not None:
            selected_frame = None
            if self._grasp_plan is not None and self._grasp_plan_path is not None:
                try:
                    self._grasp_plan = load_grasp_plan(self._grasp_plan_path)
                    selected_frame = load_selected_gripper_frame(self._grasp_plan_path)
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    self.get_logger().warning(f"Failed to refresh grasp plan {self._grasp_plan_path}: {exc}")
            marker_array = MarkerArray()
            if self._grasp_plan is not None:
                marker_array = build_vector_marker_array(
                    self._frame_id,
                    header.stamp,
                    self._grasp_plan,
                    selected_frame,
                )
            if self._additional_grasp_plan_path is not None:
                try:
                    self._additional_grasp_point = load_grasp_plan(
                        self._additional_grasp_plan_path
                    )["grasp_point"]
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    self.get_logger().warning(
                        f"Failed to refresh additional grasp plan {self._additional_grasp_plan_path}: {exc}"
                    )
            if self._additional_grasp_point is not None:
                marker_array.markers.extend(
                    build_additional_grasp_point_markers(
                        self._frame_id,
                        header.stamp,
                        self._additional_grasp_point,
                        "grasp point 1",
                    )
                )
            self._vector_marker_publisher.publish(marker_array)
            if self._grasp_plan is not None:
                candidate_tcp_transforms = build_candidate_tcp_transforms(
                    self._frame_id,
                    header.stamp,
                    self._grasp_plan["grasp_point"],
                    self._grasp_plan["gripper_directions"],
                )
                candidate_hover_transforms = build_candidate_hover_transforms(
                    self._frame_id,
                    header.stamp,
                    self._grasp_plan["grasp_point"],
                    self._grasp_plan["gripper_directions"],
                )
                candidate_transforms = candidate_tcp_transforms + candidate_hover_transforms
                if candidate_transforms:
                    self._dynamic_tf_broadcaster.sendTransform(candidate_transforms)
                target_tcp_transform = build_target_tcp_transform(
                    self._frame_id,
                    self._target_tcp_frame_id,
                    header.stamp,
                    selected_frame,
                )
                hover_tcp_transform = build_hover_tcp_transform(
                    self._frame_id,
                    DEFAULT_HOVER_TCP_FRAME_ID,
                    header.stamp,
                    selected_frame,
                )
                selected_transforms = [
                    transform
                    for transform in [target_tcp_transform, hover_tcp_transform]
                    if transform is not None
                ]
                if selected_transforms:
                    self._dynamic_tf_broadcaster.sendTransform(selected_transforms)
        if self._camera_points is not None and self._camera_publisher is not None:
            camera_header = Header()
            camera_header.stamp = header.stamp
            camera_header.frame_id = self._camera_frame_id
            camera_cloud_msg = point_cloud2.create_cloud_xyz32(camera_header, self._camera_points)
            self._camera_publisher.publish(camera_cloud_msg)

    def publish_camera_tf(self, calib_path: Path) -> None:
        parent_frame, child_frame, translation, rotation = load_calibration(calib_path)
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
            f"Published static camera TF {parent_frame} -> {child_frame} from {calib_path}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish an ASCII PLY point cloud as sensor_msgs/PointCloud2 for RViz."
    )
    parser.add_argument(
        "cable_id",
        help=(
            "Cable index such as '014' or '14', or a relative cable directory such as "
            "'multi_grasp/cable_000'. "
            "The script resolves both robot and camera PLY paths automatically."
        ),
    )
    parser.add_argument(
        "--additional-grasp-plan",
        help=(
            "Optional relative grasp-plan path whose grasp_point is added to RViz, "
            "for example 'multi_grasp/cable_000/grasp_sample_001/grasp_sample_001'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_ply_path, camera_ply_path = build_cable_paths(args.cable_id)
    grasp_plan_path = build_grasp_plan_path(args.cable_id)
    additional_grasp_plan_path = (
        build_grasp_plan_path(args.additional_grasp_plan)
        if args.additional_grasp_plan
        else None
    )
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
        DEFAULT_CAMERA_FRAME_ID,
        DEFAULT_CALIB_FILE,
        grasp_plan_path,
        additional_grasp_plan_path,
        DEFAULT_VECTOR_MARKER_TOPIC,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
