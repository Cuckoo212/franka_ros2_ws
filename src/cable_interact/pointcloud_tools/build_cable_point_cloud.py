"""
Build a 3D point cloud of a whole cable from an RGB-D frame and a cable mask.

Example:
python3 -m pointcloud_tools.build_cable_point_cloud \
    --cable-dir /home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint/cable_000 \
    --show
"""

import argparse
import heapq
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree


RGB_NAME = "rgb.png"
DEPTH_NAME = "depth.png"
MASK_NAME = "mask.png"
PARAMETERS_NAME = "parameters.json"
CAMERA_FRAME_PLY_NAME = "cable_camera_frame.ply"
ROBOT_FRAME_PLY_NAME = "cable_robot_frame.ply"
SKELETON_FROM_PC_NAME = "skeleton_from_pc"
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
        "output_skeleton": cable_dir / SKELETON_FROM_PC_NAME,
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


def copy_calibration_next_to_point_clouds(calibration_path, point_cloud_paths):
    """Copy the calibration, unchanged and with its original name, beside each output PLY."""
    source = Path(calibration_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {source}")

    copied_paths = []
    output_directories = {
        Path(path).expanduser().resolve().parent
        for path in point_cloud_paths
        if path is not None
    }
    for output_directory in sorted(output_directories):
        output_directory.mkdir(parents=True, exist_ok=True)
        destination = output_directory / source.name
        if destination != source:
            shutil.copy2(source, destination)
        copied_paths.append(destination)
    return copied_paths


def optical_points_to_camera_frame(points):
    """Convert OpenCV optical axes (right, down, forward) to ROS camera axes."""
    if points.size == 0:
        return points
    converted = np.empty_like(points)
    converted[:, 0] = points[:, 2]
    converted[:, 1] = -points[:, 0]
    converted[:, 2] = -points[:, 1]
    return converted


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0.0 or len(points) <= 1:
        return points

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    voxel_map = {}
    for point, voxel in zip(points, voxel_indices):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        voxel_map.setdefault(key, []).append(point)

    return np.asarray([np.mean(group, axis=0) for group in voxel_map.values()], dtype=np.float64)


def connected_components_from_adjacency(adjacency):
    adjacency_csr = adjacency.tocsr()
    visited = np.zeros(adjacency.shape[0], dtype=bool)
    components = []

    for start_idx in range(adjacency.shape[0]):
        if visited[start_idx]:
            continue
        stack = [start_idx]
        visited[start_idx] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = adjacency_csr[node].indices
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        components.append(component)

    return components


def build_knn_graph_components(points, neighbor_count, max_edge_length, min_component_points=8):
    if len(points) < 2:
        raise ValueError("At least two points are required to build the skeleton graph.")

    n_neighbors = min(max(2, neighbor_count), len(points) - 1)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=n_neighbors + 1)

    edge_weights = {}
    for src in range(len(points)):
        for dst, distance in zip(indices[src, 1:], distances[src, 1:]):
            dst = int(dst)
            edge_distance = float(distance)
            if src == dst:
                continue
            if max_edge_length is not None and edge_distance > max_edge_length:
                continue
            key = (src, dst) if src < dst else (dst, src)
            if key not in edge_weights or edge_distance < edge_weights[key]:
                edge_weights[key] = edge_distance

    if not edge_weights:
        raise ValueError("Failed to build any skeleton graph edges. Increase --skeleton-component-edge-threshold.")

    row_indices = []
    col_indices = []
    weights = []
    for (src, dst), edge_distance in edge_weights.items():
        row_indices.extend([src, dst])
        col_indices.extend([dst, src])
        weights.extend([edge_distance, edge_distance])

    adjacency = coo_matrix((weights, (row_indices, col_indices)), shape=(len(points), len(points)))
    components = connected_components_from_adjacency(adjacency)
    graph_components = []

    for component in components:
        if len(component) < min_component_points:
            continue
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(component)}
        component_points = points[component]
        graph = [[] for _ in range(len(component_points))]
        for (src, dst), edge_distance in edge_weights.items():
            if src in index_map and dst in index_map:
                new_src = index_map[src]
                new_dst = index_map[dst]
                graph[new_src].append((new_dst, edge_distance))
                graph[new_dst].append((new_src, edge_distance))
        if any(graph):
            graph_components.append((graph, component_points))

    if not graph_components:
        raise ValueError("No skeleton graph components survived filtering.")

    graph_components.sort(key=lambda item: len(item[1]), reverse=True)
    return graph_components


def compute_minimum_spanning_tree(graph):
    row_indices = []
    col_indices = []
    weights = []
    for src, neighbors in enumerate(graph):
        for dst, weight in neighbors:
            row_indices.append(src)
            col_indices.append(dst)
            weights.append(weight)

    adjacency = coo_matrix((weights, (row_indices, col_indices)), shape=(len(graph), len(graph)))
    mst_sparse = minimum_spanning_tree(adjacency)
    mst_coo = mst_sparse.tocoo()

    mst_graph = [[] for _ in range(len(graph))]
    for src, dst, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        edge_weight = float(weight)
        mst_graph[int(src)].append((int(dst), edge_weight))
        mst_graph[int(dst)].append((int(src), edge_weight))

    return mst_graph


def tree_dijkstra(graph, start):
    distances = [float("inf")] * len(graph)
    parents = [-1] * len(graph)
    distances[start] = 0.0
    queue = [(0.0, start)]

    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            next_distance = current_distance + weight
            if next_distance < distances[neighbor]:
                distances[neighbor] = next_distance
                parents[neighbor] = node
                heapq.heappush(queue, (next_distance, neighbor))

    farthest_node = max(range(len(distances)), key=lambda idx: distances[idx])
    return farthest_node, distances, parents


def reconstruct_path(parents, end_node):
    path = [end_node]
    current = end_node
    while parents[current] != -1:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


def extract_skeleton_path_from_graph(graph, filtered_points):
    mst_graph = compute_minimum_spanning_tree(graph)
    endpoint_a, _, _ = tree_dijkstra(mst_graph, 0)
    endpoint_b, _, parents = tree_dijkstra(mst_graph, endpoint_a)
    path_indices = reconstruct_path(parents, endpoint_b)

    skeleton = np.asarray([filtered_points[idx] for idx in path_indices], dtype=np.float64)
    if len(skeleton) < 2:
        raise ValueError("Failed to extract a valid skeleton path from the point cloud.")
    return skeleton


def extract_skeleton_paths(points, neighbor_count, max_edge_length):
    skeletons = []
    for graph, component_points in build_knn_graph_components(points, neighbor_count, max_edge_length):
        skeletons.append(extract_skeleton_path_from_graph(graph, component_points))
    return skeletons


def compute_polyline_arc_lengths(polyline):
    if len(polyline) == 0:
        return np.asarray([], dtype=np.float64)
    if len(polyline) == 1:
        return np.asarray([0.0], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def compute_polyline_midpoint(polyline):
    segment_vectors = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 0.0:
        return polyline[len(polyline) // 2].copy()

    target_length = total_length * 0.5
    accumulated = 0.0
    for idx, segment_length in enumerate(segment_lengths):
        next_accumulated = accumulated + float(segment_length)
        if next_accumulated >= target_length and segment_length > 0.0:
            ratio = (target_length - accumulated) / float(segment_length)
            return polyline[idx] + ratio * segment_vectors[idx]
        accumulated = next_accumulated

    return polyline[-1].copy()


def estimate_default_voxel_size(points):
    if len(points) < 2:
        return 0.001

    sample_count = min(len(points), 2000)
    sample = points[:sample_count]
    tree = cKDTree(sample)
    distances, _ = tree.query(sample, k=2)
    nearest_neighbor_distances = distances[:, 1]
    median_spacing = float(np.median(nearest_neighbor_distances))
    return max(median_spacing * 2.0, 1e-4)


def estimate_component_edge_threshold(points, voxel_size):
    if len(points) < 2:
        return max(voxel_size * 3.0, 0.006)

    sample_count = min(len(points), 3000)
    sample = points[:sample_count]
    tree = cKDTree(sample)
    distances, _ = tree.query(sample, k=2)
    median_spacing = float(np.median(distances[:, 1]))
    return max(voxel_size * 3.0, median_spacing * 5.0, 0.006)


def build_skeleton_from_point_cloud(points, voxel_size=None, neighbor_count=8, component_edge_threshold=None):
    if points is None or len(points) < 2:
        raise ValueError("At least two point-cloud points are required to build a skeleton.")

    points = np.asarray(points, dtype=np.float64)
    effective_voxel_size = estimate_default_voxel_size(points) if voxel_size is None else voxel_size
    downsampled_points = voxel_downsample(points, effective_voxel_size)
    effective_component_edge_threshold = (
        estimate_component_edge_threshold(downsampled_points, effective_voxel_size)
        if component_edge_threshold is None
        else component_edge_threshold
    )
    skeletons = extract_skeleton_paths(
        downsampled_points,
        neighbor_count,
        effective_component_edge_threshold,
    )
    arc_lengths = [compute_polyline_arc_lengths(skeleton) for skeleton in skeletons]
    total_lengths = [float(lengths[-1]) for lengths in arc_lengths]
    midpoints = [compute_polyline_midpoint(skeleton).tolist() for skeleton in skeletons]

    return {
        "mode": "point_cloud_skeleton",
        "point_count": int(len(points)),
        "downsampled_point_count": int(len(downsampled_points)),
        "voxel_size": float(effective_voxel_size),
        "neighbor_count": int(neighbor_count),
        "component_edge_threshold": float(effective_component_edge_threshold),
        "skeleton_count": int(len(skeletons)),
        "skeleton_point_count": int(sum(len(skeleton) for skeleton in skeletons)),
        "skeleton": skeletons[0].tolist(),
        "skeletons": [skeleton.tolist() for skeleton in skeletons],
        "skeleton_midpoint": midpoints[0],
        "skeleton_midpoints": midpoints,
        "skeleton_total_length": total_lengths[0],
        "skeleton_total_lengths": total_lengths,
    }


def save_skeleton_from_point_cloud(output_path, points, frame, source_ply_path=None, voxel_size=None, neighbor_count=8, component_edge_threshold=None):
    skeleton_data = build_skeleton_from_point_cloud(
        points,
        voxel_size=voxel_size,
        neighbor_count=neighbor_count,
        component_edge_threshold=component_edge_threshold,
    )
    skeleton_data["frame"] = frame
    if source_ply_path is not None:
        skeleton_data["source_ply_path"] = os.fspath(source_ply_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(skeleton_data, indent=2) + "\n", encoding="utf-8")
    return skeleton_data


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
    output_skeleton=None,
    save_depth_vis=None,
    save_camera_figure=None,
    save_robot_figure=None,
    keep_only_largest_component=False,
    z_min_mm=None,
    z_max_mm=None,
    handeye_calibration_path=None,
    skeleton_voxel_size=None,
    skeleton_neighbors=8,
    skeleton_component_edge_threshold=None,
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
        tracking_frame = handeye_transform["source_frame"]
        camera_frame = (
            tracking_frame
            if tracking_frame.endswith("_optical_frame")
            else tracking_frame.removesuffix("_frame") + "_optical_frame"
        )
        robot_frame = handeye_transform["target_frame"]
        points_in_tracking_frame = camera_points
        if not tracking_frame.endswith("_optical_frame"):
            points_in_tracking_frame = optical_points_to_camera_frame(camera_points)
        robot_points = transform_point_cloud(
            points_in_tracking_frame,
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
    copied_calibration_paths = []
    if handeye_calibration_path is not None:
        copied_calibration_paths = copy_calibration_next_to_point_clouds(
            handeye_calibration_path,
            [output_camera_ply, output_robot_ply if robot_points is not None else None],
        )
    skeleton_data = None
    if output_skeleton is not None:
        skeleton_source_points = robot_points if robot_points is not None else camera_points
        skeleton_frame = robot_frame if robot_points is not None else camera_frame
        skeleton_source_ply = output_robot_ply if robot_points is not None else output_camera_ply
        skeleton_data = save_skeleton_from_point_cloud(
            output_skeleton,
            skeleton_source_points,
            frame=skeleton_frame,
            source_ply_path=skeleton_source_ply,
            voxel_size=skeleton_voxel_size,
            neighbor_count=skeleton_neighbors,
            component_edge_threshold=skeleton_component_edge_threshold,
        )

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
        "copied_calibration_paths": copied_calibration_paths,
        "output_camera_ply": output_camera_ply,
        "output_robot_ply": output_robot_ply,
        "output_skeleton": output_skeleton,
        "skeleton_data": skeleton_data,
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
        "--output-skeleton",
        help="Optional output JSON path for the skeleton extracted from the point cloud.",
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
        "--skeleton-voxel-size",
        type=float,
        default=None,
        help="Optional voxel size in meters for point-cloud skeleton extraction. Defaults to an auto-estimated value.",
    )
    parser.add_argument(
        "--skeleton-neighbors",
        type=int,
        default=8,
        help="k-nearest-neighbor count used for point-cloud skeleton extraction. Default: 8.",
    )
    parser.add_argument(
        "--skeleton-component-edge-threshold",
        type=float,
        default=None,
        help="Optional maximum kNN edge length in meters before splitting skeleton components.",
    )
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
        args.output_skeleton = args.output_skeleton or os.fspath(resolved["output_skeleton"])
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
    if args.skeleton_voxel_size is not None and args.skeleton_voxel_size <= 0.0:
        parser.error("--skeleton-voxel-size must be greater than 0.")
    if args.skeleton_neighbors < 2:
        parser.error("--skeleton-neighbors must be at least 2.")
    if args.skeleton_component_edge_threshold is not None and args.skeleton_component_edge_threshold <= 0.0:
        parser.error("--skeleton-component-edge-threshold must be greater than 0.")

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
        output_skeleton=args.output_skeleton,
        save_camera_figure=args.save_camera_figure,
        save_robot_figure=args.save_robot_figure,
        keep_only_largest_component=args.keep_largest_component,
        z_min_mm=args.z_min_mm,
        z_max_mm=args.z_max_mm,
        handeye_calibration_path=args.handeye_calibration,
        skeleton_voxel_size=args.skeleton_voxel_size,
        skeleton_neighbors=args.skeleton_neighbors,
        skeleton_component_edge_threshold=args.skeleton_component_edge_threshold,
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
    for calibration_copy in result["copied_calibration_paths"]:
        print(f"Copied hand-eye calibration next to point cloud: {calibration_copy}")
    if result["output_skeleton"] is not None and result["skeleton_data"] is not None:
        print(
            "Saved point-cloud skeleton to: "
            f"{result['output_skeleton']} "
            f"({result['skeleton_data']['skeleton_point_count']} skeleton points)"
        )
    if args.save_depth_vis is not None:
        print(f"Saved depth visualization to: {args.save_depth_vis}")
    if result["save_camera_figure"] is not None:
        print(f"Saved camera-frame figure to: {result['save_camera_figure']}")
    if result["save_robot_figure"] is not None and result["robot_points"] is not None:
        print(f"Saved robot-frame figure to: {result['save_robot_figure']}")


if __name__ == "__main__":
    main()
