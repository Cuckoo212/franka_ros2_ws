"""
Build a 3D point cloud of a whole cable from an RGB-D frame and a cable mask.

Example:
python3 -m pointcloud_tools.build_cable_point_cloud \
    --cable-dir /home/flexcycle/franka_ros2_ws/src/pointcloud_tools/info_for_3Dpoint/cable_000 \
    --show
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


RGB_NAME = "rgb.png"
DEPTH_NAME = "depth.png"
MASK_NAME = "mask.png"
PARAMETERS_NAME = "parameters.json"
CAMERA_FRAME_PLY_NAME = "cable_camera_frame.ply"
ROBOT_FRAME_PLY_NAME = "cable_robot_frame.ply"
DEPTH_VIS_NAME = "depth_vis.png"
CAMERA_FIGURE_NAME = "figure_camera_frame.png"
ROBOT_FIGURE_NAME = "figure_robot_frame.png"
DEFAULT_HANDEYE_CALIBRATION_PATH = (
    "/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib"
)


def resolve_rgb_path(cable_dir):
    cable_dir = Path(cable_dir)
    cable_name = cable_dir.name
    if cable_name.startswith("cable_"):
        cable_index = cable_name[len("cable_"):]
        candidate = cable_dir / f"rgb_{cable_index}.png"
        if candidate.exists():
            return candidate
    return cable_dir / RGB_NAME


def resolve_cable_paths(cable_dir):
    cable_dir = Path(cable_dir)
    return {
        "cable_dir": cable_dir,
        "rgb": resolve_rgb_path(cable_dir),
        "depth": cable_dir / DEPTH_NAME,
        "mask": cable_dir / MASK_NAME,
        "camera_parameters": cable_dir / PARAMETERS_NAME,
        "output_camera_ply": cable_dir / CAMERA_FRAME_PLY_NAME,
        "output_robot_ply": cable_dir / ROBOT_FRAME_PLY_NAME,
        "save_depth_vis": cable_dir / DEPTH_VIS_NAME,
        "save_camera_figure": cable_dir / CAMERA_FIGURE_NAME,
        "save_robot_figure": cable_dir / ROBOT_FIGURE_NAME,
    }


def load_camera_matrix(camera_parameters_path):
    with open(camera_parameters_path, encoding="utf-8") as f:
        parameters = json.load(f)
    camera_matrix = np.array(parameters["camera_matrix"], dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3).")
    return camera_matrix


def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    norm = np.linalg.norm([qx, qy, qz, qw])
    if norm == 0:
        raise ValueError("Quaternion must not be zero.")
    qx, qy, qz, qw = np.array([qx, qy, qz, qw], dtype=np.float64) / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def load_handeye_transform(calibration_path):
    with open(calibration_path, encoding="utf-8") as f:
        calibration_text = f.read()

    def read_string(key):
        match = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", calibration_text, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing key '{key}' in calibration file: {calibration_path}")
        value = match.group(1).strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        return value

    def read_float(key):
        value = read_string(key)
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Calibration key '{key}' must be a float, got: {value}"
            ) from exc

    calibration_type = read_string("calibration_type")
    robot_base_frame = read_string("robot_base_frame")
    tracking_base_frame = read_string("tracking_base_frame")

    rotation_matrix = quaternion_to_rotation_matrix(
        [
            read_float("x"),
            read_float("y"),
            read_float("z"),
            read_float("w"),
        ]
    )
    translation_vector = np.array(
        [read_float("x"), read_float("y"), read_float("z")], dtype=np.float64
    )

    translation_match = re.search(
        r"translation:\s+.*?x:\s*([^\n]+)\s+.*?y:\s*([^\n]+)\s+.*?z:\s*([^\n]+)",
        calibration_text,
        re.DOTALL,
    )
    rotation_match = re.search(
        r"rotation:\s+.*?x:\s*([^\n]+)\s+.*?y:\s*([^\n]+)\s+.*?z:\s*([^\n]+)\s+.*?w:\s*([^\n]+)",
        calibration_text,
        re.DOTALL,
    )
    if translation_match is None or rotation_match is None:
        raise ValueError(f"Invalid calibration transform format: {calibration_path}")

    translation_vector = np.array(
        [float(translation_match.group(1)), float(translation_match.group(2)), float(translation_match.group(3))],
        dtype=np.float64,
    )
    rotation_matrix = quaternion_to_rotation_matrix(
        [
            float(rotation_match.group(1)),
            float(rotation_match.group(2)),
            float(rotation_match.group(3)),
            float(rotation_match.group(4)),
        ]
    )

    if calibration_type != "eye_on_base":
        raise ValueError(
            "Only eye_on_base hand-eye calibration is supported here, because the "
            "output point cloud must be in the robot base frame."
        )

    source_frame = tracking_base_frame
    target_frame = robot_base_frame

    return {
        "rotation_matrix": rotation_matrix.astype(np.float32),
        "translation_vector": translation_vector.astype(np.float32),
        "source_frame": source_frame,
        "target_frame": target_frame,
        "calibration_type": calibration_type,
    }


def transform_point_cloud(points, rotation_matrix, translation_vector):
    if points.size == 0:
        return points
    return (points @ rotation_matrix.T) + translation_vector[None, :]


def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask > 0


def load_rgb(rgb_path):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def load_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    if depth.ndim != 2:
        raise ValueError("Depth image must be single-channel.")
    return depth


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component_label = 1 + int(np.argmax(areas))
    return labels == largest_component_label


def filter_depth_range(mask, depth, z_min_mm=None, z_max_mm=None):
    valid = depth > 0
    if z_min_mm is not None:
        valid &= depth >= z_min_mm
    if z_max_mm is not None:
        valid &= depth <= z_max_mm
    return mask & valid


def colorize_depth(depth, z_min_mm=None, z_max_mm=None):
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    depth_valid = depth[valid].astype(np.float32)
    min_depth = float(np.min(depth_valid) if z_min_mm is None else z_min_mm)
    max_depth = float(np.max(depth_valid) if z_max_mm is None else z_max_mm)
    if max_depth <= min_depth:
        max_depth = min_depth + 1.0

    depth_float = depth.astype(np.float32)
    depth_norm = np.clip((depth_float - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    depth_u8 = (255.0 * depth_norm).astype(np.uint8)
    depth_color_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color_bgr[~valid] = 0
    return cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)


def overlay_mask_on_rgb(rgb, mask, alpha=0.45):
    overlay = rgb.copy()
    overlay[mask] = np.array([255, 80, 80], dtype=np.uint8)
    blended = rgb.astype(np.float32)
    blended[mask] = (
        (1.0 - alpha) * rgb[mask].astype(np.float32)
        + alpha * overlay[mask].astype(np.float32)
    )
    return blended.astype(np.uint8)


def depth_mask_to_point_cloud(depth, camera_matrix, mask, rgb=None):
    if depth.shape != mask.shape:
        raise ValueError("depth and mask must have the same shape.")
    if rgb is not None and rgb.shape[:2] != depth.shape:
        raise ValueError("rgb and depth must have matching height and width.")

    pixels = np.argwhere(mask)
    if pixels.size == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    z_mm = depth[pixels[:, 0], pixels[:, 1]].astype(np.float32)
    valid = z_mm > 0
    pixels = pixels[valid]
    z_mm = z_mm[valid]
    if pixels.size == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    z_m = z_mm / 1000.0
    u = pixels[:, 1].astype(np.float32)
    v = pixels[:, 0].astype(np.float32)
    uvn = np.stack((u, v, np.ones_like(u)), axis=0)
    xy1 = np.linalg.inv(camera_matrix).dot(uvn)
    xyz = xy1 * z_m[None, :]
    points = xyz.T.astype(np.float32)

    colors = None
    if rgb is not None:
        colors = rgb[pixels[:, 0], pixels[:, 1], :].astype(np.uint8)
    return points, colors


def save_point_cloud_ply(output_path, points, colors=None):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        has_color = colors is not None and colors.shape[0] == points.shape[0]
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if has_color:
            for point, color in zip(points, colors):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        else:
            for point in points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def set_equal_3d_axes(ax, points):
    if points.shape[0] == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 0.05

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def create_visualization_figure(
    rgb,
    depth_color,
    mask,
    points,
    point_cloud_frame="camera",
):
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    if rgb is not None:
        ax1.imshow(overlay_mask_on_rgb(rgb, mask))
        ax1.set_title("RGB + Cable Mask")
    else:
        ax1.imshow(mask, cmap="gray")
        ax1.set_title("Cable Mask")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(depth_color)
    ax2.set_title("Depth Visualization")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    if points.shape[0] > 0:
        ax3.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    set_equal_3d_axes(ax3, points)
    ax3.set_title(f"Cable Point Cloud ({point_cloud_frame})")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_zlabel("Z [m]")

    plt.tight_layout()
    return fig


def finalize_figures(figures, save_paths=None, show=False):
    save_paths = save_paths or []

    for fig, save_path in zip(figures, save_paths):
        if save_path is None:
            continue
        save_path = os.fspath(save_path)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show(block=True)

    for fig in figures:
        plt.close(fig)


def build_cable_point_cloud(
    depth_path,
    mask_path,
    camera_parameters_path,
    rgb_path=None,
    output_camera_ply=None,
    output_robot_ply=None,
    save_depth_vis=None,
    save_camera_figure=None,
    save_robot_figure=None,
    keep_only_largest_component=False,
    z_min_mm=None,
    z_max_mm=None,
    handeye_calibration_path=None,
    show=False,
):
    depth = load_depth(depth_path)
    mask = load_mask(mask_path)
    camera_matrix = load_camera_matrix(camera_parameters_path)
    rgb = load_rgb(rgb_path) if rgb_path is not None else None

    if keep_only_largest_component:
        mask = keep_largest_component(mask)
    mask = filter_depth_range(mask, depth, z_min_mm=z_min_mm, z_max_mm=z_max_mm)

    camera_points, colors = depth_mask_to_point_cloud(depth, camera_matrix, mask, rgb=rgb)
    camera_frame = "camera"
    robot_points = None
    robot_frame = None
    handeye_transform = None
    if handeye_calibration_path is not None:
        handeye_transform = load_handeye_transform(handeye_calibration_path)
        camera_frame = handeye_transform["source_frame"]
        robot_frame = handeye_transform["target_frame"]
        robot_points = transform_point_cloud(
            camera_points,
            handeye_transform["rotation_matrix"],
            handeye_transform["translation_vector"],
        )
    depth_color = colorize_depth(depth, z_min_mm=z_min_mm, z_max_mm=z_max_mm)

    if save_depth_vis is not None:
        os.makedirs(os.path.dirname(save_depth_vis) or ".", exist_ok=True)
        cv2.imwrite(save_depth_vis, cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))

    if output_camera_ply is not None:
        save_point_cloud_ply(output_camera_ply, camera_points, colors=colors)
    if output_robot_ply is not None and robot_points is not None:
        save_point_cloud_ply(output_robot_ply, robot_points, colors=colors)

    figures = []
    save_paths = []
    if show or save_camera_figure is not None:
        figures.append(
            create_visualization_figure(
                rgb,
                depth_color,
                mask,
                camera_points,
                point_cloud_frame=camera_frame,
            )
        )
        save_paths.append(save_camera_figure)
    if show or save_robot_figure is not None:
        if robot_points is None:
            raise ValueError(
                "Robot-frame visualization requested, but no handeye calibration was provided."
            )
        figures.append(
            create_visualization_figure(
                rgb,
                depth_color,
                mask,
                robot_points,
                point_cloud_frame=robot_frame,
            )
        )
        save_paths.append(save_robot_figure)

    if figures:
        finalize_figures(figures, save_paths=save_paths, show=show)

    return {
        "camera_points": camera_points,
        "robot_points": robot_points,
        "colors": colors,
        "mask": mask,
        "depth_color": depth_color,
        "camera_matrix": camera_matrix,
        "camera_frame": camera_frame,
        "robot_frame": robot_frame,
        "handeye_transform": handeye_transform,
        "output_camera_ply": output_camera_ply,
        "output_robot_ply": output_robot_ply,
        "save_camera_figure": save_camera_figure,
        "save_robot_figure": save_robot_figure,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a 3D point cloud of a whole cable from depth and mask."
    )
    parser.add_argument(
        "--cable-dir",
        help="Folder containing RGB/depth/mask/parameters files, such as rgb_000.png or rgb.png, depth.png, mask.png, and parameters.json.",
    )
    parser.add_argument("--depth", help="Path to a 16-bit depth PNG in millimeters.")
    parser.add_argument("--mask", help="Path to a binary cable mask image.")
    parser.add_argument(
        "--camera-parameters",
        help="Path to a JSON file with key 'camera_matrix'.",
    )
    parser.add_argument("--rgb", help="Optional RGB image path for coloring the point cloud.")
    parser.add_argument(
        "--output-camera-ply",
        help="Optional output PLY path for the camera-frame point cloud.",
    )
    parser.add_argument(
        "--output-robot-ply",
        help="Optional output PLY path for the robot-frame point cloud.",
    )
    parser.add_argument(
        "--output-ply",
        help="Deprecated alias for --output-robot-ply.",
    )
    parser.add_argument("--save-depth-vis", help="Optional path to save a colorized depth image.")
    parser.add_argument(
        "--save-camera-figure",
        help="Optional path to save the camera-frame RGB/depth/point-cloud figure.",
    )
    parser.add_argument(
        "--save-robot-figure",
        help="Optional path to save the robot-frame RGB/depth/point-cloud figure.",
    )
    parser.add_argument(
        "--save-figure",
        help="Deprecated alias for --save-robot-figure.",
    )
    parser.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="Keep only the largest connected mask component.",
    )
    parser.add_argument("--z-min-mm", type=int, help="Optional minimum depth in millimeters.")
    parser.add_argument("--z-max-mm", type=int, help="Optional maximum depth in millimeters.")
    parser.add_argument(
        "--handeye-calibration",
        help="Optional easy_handeye2 eye_on_base .calib file. If provided, points are transformed from camera frame into robot base frame.",
    )
    parser.add_argument("--show", action="store_true", help="Show RGB/mask, depth, and 3D scatter.")
    args = parser.parse_args()

    if args.cable_dir:
        resolved = resolve_cable_paths(args.cable_dir)
        args.depth = args.depth or os.fspath(resolved["depth"])
        args.mask = args.mask or os.fspath(resolved["mask"])
        args.camera_parameters = args.camera_parameters or os.fspath(resolved["camera_parameters"])
        args.rgb = args.rgb or os.fspath(resolved["rgb"])
        args.handeye_calibration = args.handeye_calibration or DEFAULT_HANDEYE_CALIBRATION_PATH
        args.output_camera_ply = args.output_camera_ply or os.fspath(resolved["output_camera_ply"])
        args.output_robot_ply = args.output_robot_ply or os.fspath(resolved["output_robot_ply"])
        args.save_depth_vis = args.save_depth_vis or os.fspath(resolved["save_depth_vis"])
        args.save_camera_figure = args.save_camera_figure or os.fspath(resolved["save_camera_figure"])
        args.save_robot_figure = args.save_robot_figure or os.fspath(resolved["save_robot_figure"])

    if args.output_ply and not args.output_robot_ply:
        args.output_robot_ply = args.output_ply
    if args.save_figure and not args.save_robot_figure:
        args.save_robot_figure = args.save_figure

    required_paths = {
        "--depth": args.depth,
        "--mask": args.mask,
        "--camera-parameters": args.camera_parameters,
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))

    return args


def main():
    args = parse_args()
    result = build_cable_point_cloud(
        depth_path=args.depth,
        mask_path=args.mask,
        camera_parameters_path=args.camera_parameters,
        rgb_path=args.rgb,
        output_camera_ply=args.output_camera_ply,
        output_robot_ply=args.output_robot_ply,
        save_depth_vis=args.save_depth_vis,
        save_camera_figure=args.save_camera_figure,
        save_robot_figure=args.save_robot_figure,
        keep_only_largest_component=args.keep_largest_component,
        z_min_mm=args.z_min_mm,
        z_max_mm=args.z_max_mm,
        handeye_calibration_path=args.handeye_calibration,
        show=args.show,
    )
    print(f"Camera-frame 3D points: {result['camera_points'].shape[0]}")
    print(f"Camera frame: {result['camera_frame']}")
    if result["output_camera_ply"] is not None:
        print(f"Saved camera-frame point cloud to: {result['output_camera_ply']}")
    if result["robot_points"] is not None:
        print(f"Robot-frame 3D points: {result['robot_points'].shape[0]}")
        print(f"Robot frame: {result['robot_frame']}")
    if result["output_robot_ply"] is not None and result["robot_points"] is not None:
        print(f"Saved robot-frame point cloud to: {result['output_robot_ply']}")
    if args.save_depth_vis is not None:
        print(f"Saved depth visualization to: {args.save_depth_vis}")
    if result["save_camera_figure"] is not None:
        print(f"Saved camera-frame figure to: {result['save_camera_figure']}")
    if result["save_robot_figure"] is not None and result["robot_points"] is not None:
        print(f"Saved robot-frame figure to: {result['save_robot_figure']}")


if __name__ == "__main__":
    main()
