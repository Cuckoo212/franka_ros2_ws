#!/usr/bin/python3.10

import argparse
import json
import heapq
import html
import re
import webbrowser
from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree


Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
X_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)


def parse_ascii_ply_vertices(ply_path: Path) -> np.ndarray:
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

    return np.asarray(vertices, dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0 or len(points) <= 1:
        return points

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    voxel_map: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for point, voxel in zip(points, voxel_indices):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        voxel_map.setdefault(key, []).append(point)

    return np.asarray([np.mean(group, axis=0) for group in voxel_map.values()], dtype=np.float64)


def build_knn_graph(points: np.ndarray, neighbor_count: int) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
    if len(points) < 2:
        raise ValueError("At least two points are required to build the skeleton graph.")

    n_neighbors = min(max(2, neighbor_count), len(points) - 1)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=n_neighbors + 1)

    edge_weights: dict[tuple[int, int], float] = {}
    for src in range(len(points)):
        for dst, distance in zip(indices[src, 1:], distances[src, 1:]):
            dst = int(dst)
            if src == dst:
                continue
            key = (src, dst) if src < dst else (dst, src)
            edge_distance = float(distance)
            if key not in edge_weights or edge_distance < edge_weights[key]:
                edge_weights[key] = edge_distance

    row_indices = []
    col_indices = []
    weights = []
    for (src, dst), edge_distance in edge_weights.items():
        row_indices.extend([src, dst])
        col_indices.extend([dst, src])
        weights.extend([edge_distance, edge_distance])

    adjacency = coo_matrix((weights, (row_indices, col_indices)), shape=(len(points), len(points)))
    components = connected_components_from_adjacency(adjacency)
    largest_component = max(components, key=len)
    if len(largest_component) < 2:
        raise ValueError("Failed to build a connected point-cloud graph.")

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_component)}
    filtered_points = points[largest_component]
    graph: list[list[tuple[int, float]]] = [[] for _ in range(len(filtered_points))]
    for (src, dst), edge_distance in edge_weights.items():
        if src in index_map and dst in index_map:
            new_src = index_map[src]
            new_dst = index_map[dst]
            graph[new_src].append((new_dst, edge_distance))
            graph[new_dst].append((new_src, edge_distance))

    return graph, filtered_points


def extract_skeleton_path(points: np.ndarray, neighbor_count: int) -> np.ndarray:
    graph, filtered_points = build_knn_graph(points, neighbor_count)
    mst_graph = compute_minimum_spanning_tree(graph)

    endpoint_a, _, _ = tree_dijkstra(mst_graph, 0)
    endpoint_b, _, parents = tree_dijkstra(mst_graph, endpoint_a)
    path_indices = reconstruct_path(parents, endpoint_b)

    skeleton = np.asarray([filtered_points[idx] for idx in path_indices], dtype=np.float64)
    if len(skeleton) < 2:
        raise ValueError("Failed to extract a valid skeleton path from the point cloud.")
    return skeleton


def connected_components_from_adjacency(adjacency: coo_matrix) -> list[list[int]]:
    adjacency_csr = adjacency.tocsr()
    visited = np.zeros(adjacency.shape[0], dtype=bool)
    components: list[list[int]] = []

    for start_idx in range(adjacency.shape[0]):
        if visited[start_idx]:
            continue
        stack = [start_idx]
        visited[start_idx] = True
        component: list[int] = []
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


def compute_minimum_spanning_tree(graph: list[list[tuple[int, float]]]) -> list[list[tuple[int, float]]]:
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

    mst_graph: list[list[tuple[int, float]]] = [[] for _ in range(len(graph))]
    for src, dst, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        edge_weight = float(weight)
        mst_graph[int(src)].append((int(dst), edge_weight))
        mst_graph[int(dst)].append((int(src), edge_weight))

    return mst_graph


def tree_dijkstra(graph: list[list[tuple[int, float]]], start: int) -> tuple[int, list[float], list[int]]:
    distances = [float("inf")] * len(graph)
    parents = [-1] * len(graph)
    distances[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]

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


def reconstruct_path(parents: list[int], end_node: int) -> list[int]:
    path = [end_node]
    current = end_node
    while parents[current] != -1:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


def compute_polyline_midpoint(polyline: np.ndarray) -> np.ndarray:
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


def sample_polyline_points_by_arc_lengths(
    polyline: np.ndarray,
    target_lengths: Sequence[float],
) -> np.ndarray:
    if len(polyline) == 0:
        raise ValueError("Polyline must contain at least one point.")
    if len(polyline) == 1:
        return np.repeat(polyline[:1], len(target_lengths), axis=0)

    segment_vectors = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative_lengths[-1])
    if total_length <= 0.0:
        return np.repeat(polyline[:1], len(target_lengths), axis=0)

    sampled_points: list[np.ndarray] = []
    segment_index = 0
    for target_length in target_lengths:
        clamped_length = float(np.clip(target_length, 0.0, total_length))
        while (
            segment_index + 1 < len(cumulative_lengths)
            and cumulative_lengths[segment_index + 1] < clamped_length
        ):
            segment_index += 1

        if segment_index >= len(segment_vectors):
            sampled_points.append(polyline[-1].copy())
            continue

        start_length = float(cumulative_lengths[segment_index])
        segment_length = float(segment_lengths[segment_index])
        if segment_length <= 1e-12:
            sampled_points.append(polyline[segment_index].copy())
            continue

        ratio = (clamped_length - start_length) / segment_length
        sampled_points.append(polyline[segment_index] + ratio * segment_vectors[segment_index])

    return np.asarray(sampled_points, dtype=np.float64)


def compute_pregrasp_points(
    polyline: np.ndarray,
    segment_count: int = 10,
) -> tuple[np.ndarray, list[int]]:
    if segment_count <= 1:
        raise ValueError("segment_count must be greater than 1.")

    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    target_lengths = [total_length * idx / segment_count for idx in range(1, segment_count)]
    points = sample_polyline_points_by_arc_lengths(polyline, target_lengths)
    return points, [idx for idx in range(1, segment_count)]


def select_pregrasp_point(
    pregrasp_points: np.ndarray,
    pregrasp_segment_labels: Sequence[int],
    selected_label: int,
) -> np.ndarray:
    for index, label in enumerate(pregrasp_segment_labels):
        if int(label) == int(selected_label):
            return pregrasp_points[index]
    raise ValueError(f"selected_pregrasp_label must be one of {list(pregrasp_segment_labels)}.")


def compute_polyline_arc_lengths(polyline: np.ndarray) -> np.ndarray:
    if len(polyline) == 0:
        return np.asarray([], dtype=np.float64)
    if len(polyline) == 1:
        return np.asarray([0.0], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def sample_polyline_segment_between_arc_lengths(
    polyline: np.ndarray,
    start_length: float,
    end_length: float,
) -> np.ndarray:
    if len(polyline) < 2:
        raise ValueError("At least two skeleton points are required to sample a local tangent segment.")

    cumulative_lengths = compute_polyline_arc_lengths(polyline)
    total_length = float(cumulative_lengths[-1])
    if total_length <= 0.0:
        raise ValueError("Skeleton arc length is too small to compute a local tangent segment.")

    start_length = float(np.clip(start_length, 0.0, total_length))
    end_length = float(np.clip(end_length, 0.0, total_length))
    if end_length < start_length:
        start_length, end_length = end_length, start_length
    if end_length - start_length <= 1e-6:
        start_length = max(0.0, start_length - 0.001)
        end_length = min(total_length, end_length + 0.001)

    segment_points: list[np.ndarray] = []
    boundary_points = sample_polyline_points_by_arc_lengths(polyline, [start_length, end_length])
    segment_points.append(boundary_points[0])
    for point, arc_length in zip(polyline, cumulative_lengths):
        if start_length < float(arc_length) < end_length:
            segment_points.append(point)
    segment_points.append(boundary_points[1])

    return np.asarray(segment_points, dtype=np.float64)


def compute_tangent_from_arc_window(
    polyline: np.ndarray,
    center_length: float,
    half_window_length: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    if half_window_length <= 0.0:
        raise ValueError("half_window_length must be positive.")

    local_segment = sample_polyline_segment_between_arc_lengths(
        polyline,
        center_length - half_window_length,
        center_length + half_window_length,
    )
    if len(local_segment) < 2:
        raise ValueError("Local tangent segment must contain at least two points.")

    centered = local_segment - np.mean(local_segment, axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = normalize_vector(eigenvectors[:, int(np.argmax(eigenvalues))])

    reference = local_segment[-1] - local_segment[0]
    if np.linalg.norm(reference) > 1e-12 and float(np.dot(tangent, reference)) < 0.0:
        tangent = -tangent

    return tangent, local_segment, (float(center_length - half_window_length), float(center_length + half_window_length))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def compute_tangent_at_point(polyline: np.ndarray, query_point: np.ndarray, neighbor_count: int = 9) -> np.ndarray:
    if len(polyline) < 2:
        raise ValueError("At least two skeleton points are required to compute a tangent.")

    tree = cKDTree(polyline)
    k = min(max(3, neighbor_count), len(polyline))
    _, indices = tree.query(query_point, k=k)
    neighborhood = polyline[np.atleast_1d(indices)]

    centered = neighborhood - np.mean(neighborhood, axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = normalize_vector(eigenvectors[:, int(np.argmax(eigenvalues))])

    # Keep the tangent direction consistent with the local polyline ordering.
    closest_index = int(np.atleast_1d(indices)[0])
    prev_index = max(0, closest_index - 1)
    next_index = min(len(polyline) - 1, closest_index + 1)
    reference = polyline[next_index] - polyline[prev_index]
    if np.linalg.norm(reference) > 1e-12 and float(np.dot(tangent, reference)) < 0.0:
        tangent = -tangent

    return tangent


def compute_opposite_gripper_directions(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent_xz = tangent - float(np.dot(tangent, Y_AXIS)) * Y_AXIS
    tangent_xz_norm = float(np.linalg.norm(tangent_xz))
    if tangent_xz_norm <= 1e-12:
        direction_a = X_AXIS.copy()
    else:
        direction_a = tangent_xz / tangent_xz_norm

    direction_b = -direction_a
    return direction_a, direction_b


def project_to_point_cloud(query_point: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, float]:
    tree = cKDTree(points)
    distance, index = tree.query(query_point, k=1)
    return points[int(index)], float(distance)


def estimate_default_voxel_size(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.001

    sample_count = min(len(points), 2000)
    sample = points[:sample_count]
    tree = cKDTree(sample)
    distances, _ = tree.query(sample, k=2)
    nearest_neighbor_distances = distances[:, 1]
    median_spacing = float(np.median(nearest_neighbor_distances))
    return max(median_spacing * 2.0, 1e-4)


def project_onto_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    return vector - np.dot(vector, plane_normal) * plane_normal


def normalize_vector_or(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return fallback.astype(np.float64).copy()
    return vector / norm


def quaternion_from_axes_np(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> list[float]:
    matrix = np.column_stack([x_axis, y_axis, z_axis])
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def build_board_target_gripper_frame(grasp_point: np.ndarray, gripper_direction: np.ndarray, label: str) -> dict:
    z_axis = Y_AXIS.copy()
    x_axis = project_onto_plane(normalize_vector_or(gripper_direction, X_AXIS), z_axis)
    x_axis = normalize_vector_or(x_axis, X_AXIS)
    y_axis = normalize_vector_or(np.cross(z_axis, x_axis), np.array([0.0, 0.0, 1.0], dtype=np.float64))
    x_axis = normalize_vector_or(np.cross(y_axis, z_axis), x_axis)
    return {
        "selected_gripper_frame_status": "selected",
        "selected_gripper_direction_label": label,
        "selected_gripper_frame_position": grasp_point.tolist(),
        "selected_gripper_direction": x_axis.tolist(),
        "selected_tcp_x_axis": x_axis.tolist(),
        "selected_tcp_y_axis": y_axis.tolist(),
        "selected_tcp_z_axis": z_axis.tolist(),
        "selected_tcp_quaternion_xyzw": quaternion_from_axes_np(x_axis, y_axis, z_axis),
    }


def compute_grasp_point(
    ply_path: Path,
    voxel_size: float | None,
    neighbor_count: int,
    selected_pregrasp_label: int = 3,
    selected_gripper_direction: str = "a",
    tangent_window_half_length: float = 0.01,
) -> dict:
    points = parse_ascii_ply_vertices(ply_path)
    effective_voxel_size = estimate_default_voxel_size(points) if voxel_size is None else voxel_size
    downsampled_points = voxel_downsample(points, effective_voxel_size)
    skeleton = extract_skeleton_path(downsampled_points, neighbor_count)
    skeleton_midpoint = compute_polyline_midpoint(skeleton)
    pregrasp_segment_count = 10
    pregrasp_points, pregrasp_segment_labels = compute_pregrasp_points(skeleton, segment_count=pregrasp_segment_count)
    selected_pregrasp_point = select_pregrasp_point(
        pregrasp_points,
        pregrasp_segment_labels,
        selected_pregrasp_label,
    )
    skeleton_arc_lengths = compute_polyline_arc_lengths(skeleton)
    skeleton_total_length = float(skeleton_arc_lengths[-1])
    selected_pregrasp_arc_length = skeleton_total_length * float(selected_pregrasp_label) / float(pregrasp_segment_count)
    local_tangent, local_tangent_segment, local_tangent_window = compute_tangent_from_arc_window(
        skeleton,
        selected_pregrasp_arc_length,
        tangent_window_half_length,
    )
    gripper_direction_a, gripper_direction_b = compute_opposite_gripper_directions(local_tangent)
    grasp_point, projection_distance = project_to_point_cloud(selected_pregrasp_point, points)

    selected_label = selected_gripper_direction.lower()
    if selected_label not in {"a", "b"}:
        raise ValueError("selected_gripper_direction must be 'a' or 'b'.")
    selected_direction = gripper_direction_a if selected_label == "a" else gripper_direction_b
    target_frame = build_board_target_gripper_frame(grasp_point, selected_direction, selected_label)

    result = {
        "ply_path": str(ply_path),
        "point_count": int(len(points)),
        "downsampled_point_count": int(len(downsampled_points)),
        "voxel_size": float(effective_voxel_size),
        "neighbor_count": int(neighbor_count),
        "skeleton_point_count": int(len(skeleton)),
        "skeleton": skeleton.tolist(),
        "skeleton_midpoint": skeleton_midpoint.tolist(),
        "skeleton_total_length": skeleton_total_length,
        "local_tangent": local_tangent.tolist(),
        "local_tangent_method": "arc_window_pca",
        "local_tangent_window_half_length": float(tangent_window_half_length),
        "local_tangent_window_arc_lengths": [local_tangent_window[0], local_tangent_window[1]],
        "local_tangent_segment": local_tangent_segment.tolist(),
        "pregrasp_segment_count": pregrasp_segment_count,
        "pregrasp_segment_labels": pregrasp_segment_labels,
        "selected_pregrasp_label": int(selected_pregrasp_label),
        "selected_pregrasp_arc_length": selected_pregrasp_arc_length,
        "pregrasp_points": pregrasp_points.tolist(),
        "gripper_direction_a": gripper_direction_a.tolist(),
        "gripper_direction_b": gripper_direction_b.tolist(),
        "gripper_directions": [gripper_direction_a.tolist(), gripper_direction_b.tolist()],
        "grasp_point": grasp_point.tolist(),
        "projection_distance": float(projection_distance),
    }
    result.update(target_frame)
    return result


def extract_cable_id(ply_path: Path) -> str:
    for part in ply_path.parts:
        match = re.fullmatch(r"cable_(\d+)", part)
        if match is not None:
            return match.group(1)
    raise ValueError(f"Failed to infer cable id from path: {ply_path}")


def build_output_path(ply_path: Path) -> Path:
    cable_id = extract_cable_id(ply_path)
    return ply_path.parent / f"grasp_point_cable_{cable_id}"


def build_viewer_output_path(ply_path: Path) -> Path:
    cable_id = extract_cable_id(ply_path)
    return ply_path.parent / f"grasp_curve_viewer_cable_{cable_id}.html"


def _project_xz(point: Sequence[float]) -> list[float]:
    return [float(point[0]), float(point[2])]


def build_visualization_html(result: dict) -> str:
    points = np.asarray(parse_ascii_ply_vertices(Path(result["ply_path"])), dtype=np.float64)
    skeleton = np.asarray(result.get("skeleton", []), dtype=np.float64)
    step = max(1, len(points) // 6000)
    payload = {
        "points": points[::step].tolist(),
        "skeleton": skeleton.tolist(),
        "grasp_point": result["grasp_point"],
        "gripper_direction_a": result["gripper_direction_a"],
        "gripper_direction_b": result["gripper_direction_b"],
        "local_tangent_segment": result.get("local_tangent_segment", []),
        "pregrasp_points": result.get("pregrasp_points", []),
        "pregrasp_segment_labels": result.get("pregrasp_segment_labels", []),
        "selected_pregrasp_label": result.get("selected_pregrasp_label"),
        "selected_tcp_x_axis": result.get("selected_tcp_x_axis", [1.0, 0.0, 0.0]),
        "selected_tcp_y_axis": result.get("selected_tcp_y_axis", [0.0, 0.0, 1.0]),
        "selected_tcp_z_axis": result.get("selected_tcp_z_axis", [0.0, 1.0, 0.0]),
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    metadata_rows = {
        "ply_path": result["ply_path"],
        "point_count": result["point_count"],
        "downsampled_point_count": result["downsampled_point_count"],
        "voxel_size": f'{result["voxel_size"]:.6f}',
        "skeleton_point_count": result["skeleton_point_count"],
        "projection_distance_m": f'{result["projection_distance"]:.6f}',
        "grasp_point": [round(float(v), 6) for v in result["grasp_point"]],
        "local_tangent": [round(float(v), 6) for v in result["local_tangent"]],
        "local_tangent_method": result.get("local_tangent_method"),
        "local_tangent_window_half_length_m": result.get("local_tangent_window_half_length"),
        "pregrasp_segment_count": result.get("pregrasp_segment_count"),
        "selected_pregrasp_label": result.get("selected_pregrasp_label"),
        "selected_gripper_direction_label": result.get("selected_gripper_direction_label"),
        "selected_tcp_x_axis": [round(float(v), 6) for v in result.get("selected_tcp_x_axis", [])],
        "selected_tcp_y_axis": [round(float(v), 6) for v in result.get("selected_tcp_y_axis", [])],
        "selected_tcp_z_axis": [round(float(v), 6) for v in result.get("selected_tcp_z_axis", [])],
        "selected_tcp_quaternion_xyzw": [round(float(v), 6) for v in result.get("selected_tcp_quaternion_xyzw", [])],
    }
    rows_html = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metadata_rows.items()
    )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cable On Board Grasp Viewer</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7fb; color: #1f2937; }}
    .layout {{ display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; }}
    aside {{ padding: 22px; background: #ffffff; border-right: 1px solid #d8dee9; overflow-wrap: anywhere; }}
    main {{ position: relative; min-height: 100vh; background: #111827; }}
    h1 {{ font-size: 20px; margin: 0 0 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ text-align: left; vertical-align: top; border-bottom: 1px solid #e5e7eb; padding: 8px 4px; }}
    th {{ width: 42%; color: #64748b; font-weight: 600; }}
    canvas {{ display: block; width: 100%; height: 100vh; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    .hud {{ position: absolute; left: 16px; bottom: 16px; color: #d1d5db; background: rgba(17,24,39,0.76); padding: 10px 12px; border: 1px solid #374151; font-size: 13px; }}
    .legend {{ display: grid; gap: 4px; }}
    .chip {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 18px; height: 3px; display: inline-block; }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Cable On Board Grasp Viewer</h1>
      <table>{rows_html}</table>
    </aside>
    <main>
      <canvas id="viewer"></canvas>
      <div class="hud"><div class="legend">
        <span class="chip"><i class="swatch" style="background:#e5e7eb"></i>point cloud</span>
        <span class="chip"><i class="swatch" style="background:#22c55e"></i>skeleton</span>
        <span class="chip"><i class="swatch" style="background:#2563eb"></i>grasp point</span>
        <span class="chip"><i class="swatch" style="background:#f97316"></i>A direction</span>
        <span class="chip"><i class="swatch" style="background:#06b6d4"></i>B direction</span>
        <span class="chip"><i class="swatch" style="background:#ef4444"></i>TCP X</span>
        <span class="chip"><i class="swatch" style="background:#22c55e"></i>TCP Y</span>
        <span class="chip"><i class="swatch" style="background:#3b82f6"></i>TCP Z</span>
      </div></div>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const canvas = document.getElementById('viewer');
    const ctx = canvas.getContext('2d');
    let yaw = -0.85, pitch = 0.35, zoom = 1.0;
    let dragging = false, lastX = 0, lastY = 0;
    const all = payload.points.concat(payload.skeleton, [payload.grasp_point]);
    const center = [0, 0, 0];
    for (const p of all) {{ center[0] += p[0]; center[1] += p[1]; center[2] += p[2]; }}
    center[0] /= all.length; center[1] /= all.length; center[2] /= all.length;
    let radius = 1e-6;
    for (const p of all) {{ radius = Math.max(radius, Math.hypot(p[0]-center[0], p[1]-center[1], p[2]-center[2])); }}
    function resize() {{
      const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr)); canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
    }}
    function rotate(point) {{
      let x = point[0]-center[0], y = point[1]-center[1], z = point[2]-center[2];
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy*x + sy*z, z1 = -sy*x + cy*z;
      return [x1, cp*y - sp*z1, sp*y + cp*z1];
    }}
    function project(point) {{
      const r = rotate(point), rect = canvas.getBoundingClientRect();
      const scale = 0.42 * Math.min(rect.width, rect.height) * zoom / radius;
      return [rect.width*0.5 + r[0]*scale, rect.height*0.5 - r[1]*scale, r[2]];
    }}
    function endpoint(origin, direction, length) {{
      const n = Math.hypot(direction[0], direction[1], direction[2]) || 1;
      return [origin[0]+length*direction[0]/n, origin[1]+length*direction[1]/n, origin[2]+length*direction[2]/n];
    }}
    function drawLine(a, b, color, width = 3, label = null) {{
      const pa = project(a), pb = project(b); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
      if (label) {{ ctx.fillStyle = color; ctx.font = '700 14px Arial'; ctx.fillText(label, pb[0]+7, pb[1]-7); }}
    }}
    function draw() {{
      const rect = canvas.getBoundingClientRect(); ctx.clearRect(0,0,rect.width,rect.height); ctx.fillStyle = '#111827'; ctx.fillRect(0,0,rect.width,rect.height);
      const sorted = payload.points.map(p => [p, project(p)[2]]).sort((a,b) => a[1]-b[1]);
      ctx.fillStyle = 'rgba(229,231,235,0.58)';
      for (const [p] of sorted) {{ const q = project(p); ctx.beginPath(); ctx.arc(q[0], q[1], 1.25, 0, Math.PI*2); ctx.fill(); }}
      if (payload.skeleton.length > 1) {{ ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.beginPath(); const first = project(payload.skeleton[0]); ctx.moveTo(first[0], first[1]); for (const p of payload.skeleton.slice(1)) {{ const q = project(p); ctx.lineTo(q[0], q[1]); }} ctx.stroke(); }}
      if (payload.local_tangent_segment && payload.local_tangent_segment.length > 1) {{ ctx.strokeStyle = '#fde047'; ctx.lineWidth = 5; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.beginPath(); const firstLocal = project(payload.local_tangent_segment[0]); ctx.moveTo(firstLocal[0], firstLocal[1]); for (const p of payload.local_tangent_segment.slice(1)) {{ const q = project(p); ctx.lineTo(q[0], q[1]); }} ctx.stroke(); }}
      const g = payload.grasp_point;
      drawLine(g, endpoint(g, payload.gripper_direction_a, 0.12), '#f97316', 5, 'A');
      drawLine(g, endpoint(g, payload.gripper_direction_b, 0.12), '#06b6d4', 5, 'B');
      drawLine(g, endpoint(g, payload.selected_tcp_x_axis, 0.08), '#ef4444', 4, 'TCP X');
      drawLine(g, endpoint(g, payload.selected_tcp_y_axis, 0.08), '#22c55e', 4, 'TCP Y');
      drawLine(g, endpoint(g, payload.selected_tcp_z_axis, 0.08), '#3b82f6', 4, 'TCP Z');
      if (payload.pregrasp_points && payload.pregrasp_segment_labels) {{
        ctx.font = '700 13px Arial';
        ctx.textBaseline = 'bottom';
        for (let i = 0; i < payload.pregrasp_points.length; i += 1) {{
          const p = payload.pregrasp_points[i];
          const label = payload.pregrasp_segment_labels[i] ?? (i + 1);
          const q = project(p);
          ctx.fillStyle = label === payload.selected_pregrasp_label ? '#fde047' : '#fca5a5';
          ctx.strokeStyle = 'rgba(17,24,39,0.9)';
          ctx.lineWidth = 3;
          ctx.strokeText(`P${{label}}`, q[0] + 8, q[1] - 8);
          ctx.fillText(`P${{label}}`, q[0] + 8, q[1] - 8);
        }}
      }}
      const gp = project(g); ctx.fillStyle = '#2563eb'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(gp[0], gp[1], 7, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    }}
    canvas.addEventListener('pointerdown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); }});
    canvas.addEventListener('pointermove', e => {{ if (!dragging) return; const dx = e.clientX-lastX, dy = e.clientY-lastY; lastX=e.clientX; lastY=e.clientY; yaw += dx*0.008; pitch = Math.max(-1.45, Math.min(1.45, pitch + dy*0.008)); draw(); }});
    canvas.addEventListener('pointerup', () => {{ dragging = false; }});
    canvas.addEventListener('wheel', e => {{ e.preventDefault(); zoom *= Math.exp(-e.deltaY*0.001); zoom = Math.max(0.25, Math.min(8.0, zoom)); draw(); }}, {{ passive: false }});
    window.addEventListener('resize', resize); resize();
  </script>
</body>
</html>
'''


def write_visualization_html(viewer_path: Path, result: dict) -> None:
    viewer_path.write_text(build_visualization_html(result), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a grasp point for a cable-like PLY point cloud by extracting a 3D skeleton "
            "curve and selecting the midpoint along its arc length."
        )
    )
    parser.add_argument("ply_path", type=Path, help="Absolute path to the ASCII .ply file.")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Optional voxel size in meters for downsampling. Defaults to an auto-estimated value.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=8,
        help="k-nearest-neighbor count used to build the point-cloud graph.",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Generate a lightweight HTML viewer and open it in the default browser.",
    )
    parser.add_argument(
        "--viewer-path",
        type=Path,
        default=None,
        help="Optional output path for the viewer HTML file.",
    )
    parser.add_argument(
        "--selected-pregrasp-label",
        type=int,
        default=3,
        choices=range(1, 10),
        metavar="{1..9}",
        help="Which boundary point P1..P9 to use after dividing the cable skeleton into 10 equal arc-length segments.",
    )
    parser.add_argument(
        "--tangent-window-half-length",
        type=float,
        default=0.01,
        help="Half length in meters of the local cable arc window used for tangent PCA. Default is 0.01 m, i.e. 2 cm total.",
    )
    parser.add_argument(
        "--selected-gripper-direction",
        choices=("a", "b"),
        default="a",
        help="Which computed tangent direction to write as the selected target TCP x-axis in the JSON preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ply_path.is_file():
        raise FileNotFoundError(f"PLY file not found: {args.ply_path}")
    if args.voxel_size is not None and args.voxel_size <= 0.0:
        raise ValueError("--voxel-size must be greater than 0.")
    if args.neighbors < 2:
        raise ValueError("--neighbors must be at least 2.")
    if args.tangent_window_half_length <= 0.0:
        raise ValueError("--tangent-window-half-length must be greater than 0.")

    result = compute_grasp_point(
        args.ply_path,
        args.voxel_size,
        args.neighbors,
        args.selected_pregrasp_label,
        args.selected_gripper_direction,
        args.tangent_window_half_length,
    )
    output_path = build_output_path(args.ply_path)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    viewer_path = args.viewer_path if args.viewer_path is not None else build_viewer_output_path(args.ply_path)
    if args.show_viewer:
        write_visualization_html(viewer_path, result)
        webbrowser.open(viewer_path.resolve().as_uri())
    print(json.dumps(result, indent=2))
    print(f"Saved grasp point to: {output_path}")
    if args.show_viewer:
        print(f"Saved viewer to: {viewer_path}")


if __name__ == "__main__":
    main()
