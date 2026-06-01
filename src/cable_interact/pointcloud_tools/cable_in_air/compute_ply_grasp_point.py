#!/usr/bin/python3.10

import argparse
import json
import heapq
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
CONTROLLER_GRASP_PLAN_ROOT = Path("/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint")


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
    if len(components) > 1:
        component_bridge_edges = compute_component_bridge_edges(points, components)
        for src, dst, edge_distance in component_bridge_edges:
            key = (src, dst) if src < dst else (dst, src)
            existing_weight = edge_weights.get(key)
            if existing_weight is None or edge_distance < existing_weight:
                edge_weights[key] = edge_distance

    graph: list[list[tuple[int, float]]] = [[] for _ in range(len(points))]
    for (src, dst), edge_distance in edge_weights.items():
        graph[src].append((dst, edge_distance))
        graph[dst].append((src, edge_distance))

    return graph, points


def compute_component_bridge_edges(
    points: np.ndarray,
    components: list[list[int]],
) -> list[tuple[int, int, float]]:
    if len(components) <= 1:
        return []

    component_edge_candidates: list[tuple[float, int, int, int, int]] = []
    for first_component_idx in range(len(components)):
        first_indices = np.asarray(components[first_component_idx], dtype=np.int64)
        first_points = points[first_indices]
        for second_component_idx in range(first_component_idx + 1, len(components)):
            second_indices = np.asarray(components[second_component_idx], dtype=np.int64)
            second_points = points[second_indices]

            if len(first_points) <= len(second_points):
                query_indices = first_indices
                query_points = first_points
                target_indices = second_indices
                target_points = second_points
            else:
                query_indices = second_indices
                query_points = second_points
                target_indices = first_indices
                target_points = first_points

            tree = cKDTree(target_points)
            distances, nearest_indices = tree.query(query_points, k=1)
            best_local_index = int(np.argmin(distances))
            src_index = int(query_indices[best_local_index])
            dst_index = int(target_indices[int(nearest_indices[best_local_index])])
            best_distance = float(distances[best_local_index])
            component_edge_candidates.append(
                (
                    best_distance,
                    first_component_idx,
                    second_component_idx,
                    src_index,
                    dst_index,
                )
            )

    component_graph: list[list[tuple[int, float, int, int]]] = [[] for _ in range(len(components))]
    for distance, first_idx, second_idx, src_index, dst_index in component_edge_candidates:
        component_graph[first_idx].append((second_idx, distance, src_index, dst_index))
        component_graph[second_idx].append((first_idx, distance, dst_index, src_index))

    bridge_edges: list[tuple[int, int, float]] = []
    visited = [False] * len(components)
    visited[0] = True
    queue: list[tuple[float, int, int, int]] = []
    for neighbor_idx, distance, src_index, dst_index in component_graph[0]:
        heapq.heappush(queue, (distance, neighbor_idx, src_index, dst_index))

    while queue and len(bridge_edges) < len(components) - 1:
        distance, component_idx, src_index, dst_index = heapq.heappop(queue)
        if visited[component_idx]:
            continue

        visited[component_idx] = True
        bridge_edges.append((src_index, dst_index, distance))
        for neighbor_idx, neighbor_distance, neighbor_src, neighbor_dst in component_graph[component_idx]:
            if not visited[neighbor_idx]:
                heapq.heappush(queue, (neighbor_distance, neighbor_idx, neighbor_src, neighbor_dst))

    return bridge_edges


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
        current_segment_length = float(segment_lengths[segment_index])
        if current_segment_length <= 1e-12:
            sampled_points.append(polyline[segment_index].copy())
            continue

        ratio = (clamped_length - start_length) / current_segment_length
        sampled_points.append(polyline[segment_index] + ratio * segment_vectors[segment_index])

    return np.asarray(sampled_points, dtype=np.float64)


def compute_pregrasp_points(
    polyline: np.ndarray,
    segment_count: int = 10,
) -> tuple[np.ndarray, list[int]]:
    if segment_count <= 0:
        raise ValueError("segment_count must be positive.")

    if segment_count == 1:
        return np.empty((0, 3), dtype=np.float64), []

    total_length = float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))
    boundary_ratios = np.asarray(
        [(idx / segment_count) for idx in range(1, segment_count)],
        dtype=np.float64,
    )
    points = sample_polyline_points_by_arc_lengths(
        polyline,
        [ratio * total_length for ratio in boundary_ratios],
    )
    return points, [idx for idx in range(1, segment_count)]


def select_pregrasp_point(
    pregrasp_points: np.ndarray,
    pregrasp_segment_labels: Sequence[int],
    selected_label: int,
) -> np.ndarray:
    if len(pregrasp_points) == 0:
        raise ValueError("No preparatory grasp points are available.")

    for index, label in enumerate(pregrasp_segment_labels):
        if int(label) == int(selected_label):
            return pregrasp_points[index]

    raise ValueError(f"Preparatory grasp point P{selected_label} was not found.")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def estimate_local_polyline_tangent(polyline: np.ndarray, index: int, window: int = 2) -> np.ndarray:
    if len(polyline) < 2:
        raise ValueError("At least two polyline points are required to estimate a tangent.")

    start_index = max(0, index - window)
    end_index = min(len(polyline) - 1, index + window)
    direction = polyline[end_index] - polyline[start_index]

    if np.linalg.norm(direction) <= 1e-12:
        if index > 0:
            direction = polyline[index] - polyline[index - 1]
        elif index + 1 < len(polyline):
            direction = polyline[index + 1] - polyline[index]

    return normalize_vector(direction)


def interpolate_cubic_hermite_gap(
    start_point: np.ndarray,
    end_point: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    inserted_point_count: int,
    tangent_scale: float,
) -> np.ndarray:
    if inserted_point_count <= 0:
        return np.empty((0, 3), dtype=np.float64)

    chord = end_point - start_point
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-12:
        return np.empty((0, 3), dtype=np.float64)

    chord_direction = chord / chord_length
    aligned_start_tangent = start_tangent.copy()
    aligned_end_tangent = end_tangent.copy()
    if float(np.dot(aligned_start_tangent, chord_direction)) < 0.0:
        aligned_start_tangent = -aligned_start_tangent
    if float(np.dot(aligned_end_tangent, chord_direction)) < 0.0:
        aligned_end_tangent = -aligned_end_tangent

    tangent_magnitude = max(chord_length * tangent_scale, 1e-6)
    m0 = aligned_start_tangent * tangent_magnitude
    m1 = aligned_end_tangent * tangent_magnitude

    samples = np.linspace(0.0, 1.0, inserted_point_count + 2, dtype=np.float64)[1:-1]
    interpolated_points = []
    for t in samples:
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        point = h00 * start_point + h10 * m0 + h01 * end_point + h11 * m1
        interpolated_points.append(point)

    return np.asarray(interpolated_points, dtype=np.float64)


def fill_skeleton_gaps(
    skeleton: np.ndarray,
    gap_ratio: float,
    tangent_scale: float,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    if len(skeleton) < 3:
        return skeleton.copy(), []

    segment_vectors = np.diff(skeleton, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    positive_lengths = segment_lengths[segment_lengths > 1e-12]
    if len(positive_lengths) == 0:
        return skeleton.copy(), []

    median_spacing = float(np.median(positive_lengths))
    gap_threshold = median_spacing * gap_ratio
    if gap_threshold <= 0.0:
        return skeleton.copy(), []

    filled_points: list[np.ndarray] = [skeleton[0]]
    gap_records: list[dict[str, float | int]] = []

    for idx, segment_length in enumerate(segment_lengths):
        start_point = skeleton[idx]
        end_point = skeleton[idx + 1]

        if float(segment_length) > gap_threshold:
            inserted_point_count = max(int(np.ceil(float(segment_length) / median_spacing)) - 1, 1)
            start_tangent = estimate_local_polyline_tangent(skeleton, idx)
            end_tangent = estimate_local_polyline_tangent(skeleton, idx + 1)
            interpolated_gap_points = interpolate_cubic_hermite_gap(
                start_point,
                end_point,
                start_tangent,
                end_tangent,
                inserted_point_count,
                tangent_scale,
            )
            if len(interpolated_gap_points) > 0:
                filled_points.extend(interpolated_gap_points)
            gap_records.append(
                {
                    "segment_index": int(idx),
                    "segment_length": float(segment_length),
                    "inserted_point_count": int(inserted_point_count),
                }
            )

        filled_points.append(end_point)

    return np.asarray(filled_points, dtype=np.float64), gap_records


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


def choose_reference_axis(normal: np.ndarray) -> np.ndarray:
    candidate_axes = (X_AXIS, Y_AXIS, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return min(candidate_axes, key=lambda axis: abs(float(np.dot(axis, normal))))


def compute_gripper_direction_plane_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized_tangent = normalize_vector(tangent)
    reference_axis = choose_reference_axis(normalized_tangent)
    basis_u = np.cross(normalized_tangent, reference_axis)
    if np.linalg.norm(basis_u) <= 1e-12:
        fallback_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        basis_u = np.cross(normalized_tangent, fallback_axis)
    basis_u = normalize_vector(basis_u)
    basis_v = normalize_vector(np.cross(normalized_tangent, basis_u))
    return basis_u, basis_v


def sample_gripper_directions_around_tangent(
    tangent: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_u, basis_v = compute_gripper_direction_plane_basis(tangent)
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False, dtype=np.float64)
    directions = np.asarray(
        [np.cos(angle) * basis_u + np.sin(angle) * basis_v for angle in angles],
        dtype=np.float64,
    )
    directions = np.asarray([normalize_vector(direction) for direction in directions], dtype=np.float64)
    return basis_u, basis_v, directions


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


def build_visualization_payload(
    points: np.ndarray,
    downsampled_points: np.ndarray,
    skeleton: np.ndarray,
    filled_skeleton: np.ndarray,
    grasp_point: np.ndarray,
    projected_grasp_point: np.ndarray,
    gripper_directions: np.ndarray,
    pregrasp_points: np.ndarray,
    pregrasp_segment_labels: Sequence[int],
) -> dict:
    return {
        "raw_points": points.tolist(),
        "downsampled_points": downsampled_points.tolist(),
        "skeleton": skeleton.tolist(),
        "filled_skeleton": filled_skeleton.tolist(),
        "grasp_point": grasp_point.tolist(),
        "projected_grasp_point": projected_grasp_point.tolist(),
        "gripper_directions": gripper_directions.tolist(),
        "pregrasp_points": pregrasp_points.tolist(),
        "pregrasp_segment_labels": list(pregrasp_segment_labels),
    }


def compute_grasp_point(
    ply_path: Path,
    voxel_size: float | None,
    neighbor_count: int,
    gap_ratio: float,
    gap_tension: float,
    gripper_direction_sample_count: int,
) -> tuple[dict, dict]:
    selected_pregrasp_label = 3
    points = parse_ascii_ply_vertices(ply_path)
    effective_voxel_size = estimate_default_voxel_size(points) if voxel_size is None else voxel_size
    downsampled_points = voxel_downsample(points, effective_voxel_size)
    skeleton = extract_skeleton_path(downsampled_points, neighbor_count)
    filled_skeleton, gap_segments = fill_skeleton_gaps(skeleton, gap_ratio, gap_tension)
    pregrasp_points, pregrasp_segment_labels = compute_pregrasp_points(filled_skeleton, segment_count=10)
    grasp_point = select_pregrasp_point(pregrasp_points, pregrasp_segment_labels, selected_pregrasp_label)
    local_tangent = compute_tangent_at_point(filled_skeleton, grasp_point)
    plane_basis_u, plane_basis_v, gripper_directions = sample_gripper_directions_around_tangent(
        local_tangent,
        gripper_direction_sample_count,
    )
    projected_grasp_point, projection_distance = project_to_point_cloud(grasp_point, points)
    skeleton_midpoint = compute_polyline_midpoint(filled_skeleton)

    result = {
        "ply_path": str(ply_path),
        "point_count": int(len(points)),
        "downsampled_point_count": int(len(downsampled_points)),打开控制器就已经看到了目标orientation的tf
        "voxel_size": float(effective_voxel_size),
        "neighbor_count": int(neighbor_count),
        "skeleton_point_count": int(len(skeleton)),
        "filled_skeleton_point_count": int(len(filled_skeleton)),
        "gap_fill_segment_count": int(len(gap_segments)),
        "gap_fill_segments": gap_segments,
        "gap_ratio": float(gap_ratio),
        "gap_tension": float(gap_tension),
        "skeleton_midpoint": skeleton_midpoint.tolist(),
        "local_tangent": local_tangent.tolist(),
        "gripper_direction_plane_basis_u": plane_basis_u.tolist(),
        "gripper_direction_plane_basis_v": plane_basis_v.tolist(),
        "gripper_direction_sample_count": int(gripper_direction_sample_count),
        "gripper_directions": gripper_directions.tolist(),
        "pregrasp_segment_count": 10,
        "pregrasp_segment_labels": pregrasp_segment_labels,
        "selected_pregrasp_label": int(selected_pregrasp_label),
        "pregrasp_points": pregrasp_points.tolist(),
        "grasp_point": grasp_point.tolist(),
        "projected_grasp_point": projected_grasp_point.tolist(),
        "projection_distance": float(projection_distance),
    }
    visualization_payload = build_visualization_payload(
        points,
        downsampled_points,
        skeleton,
        filled_skeleton,
        grasp_point,
        projected_grasp_point,
        gripper_directions,
        pregrasp_points,
        pregrasp_segment_labels,
    )
    return result, visualization_payload


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


def build_viewer_selection_asset_name(ply_path: Path) -> str:
    cable_id = extract_cable_id(ply_path)
    return f"grasp_curve_selection_cable_{cable_id}.js"


def build_viewer_selection_asset_fallback_uri(ply_path: Path) -> str:
    cable_id = extract_cable_id(ply_path)
    selection_path = (
        CONTROLLER_GRASP_PLAN_ROOT
        / f"cable_{cable_id}"
        / f"grasp_curve_selection_cable_{cable_id}.js"
    )
    return selection_path.resolve().as_uri()


def build_visualization_html(visualization_payload: dict, metadata: dict) -> str:
    payload_json = json.dumps(visualization_payload, separators=(",", ":"))
    metadata_json = json.dumps(metadata, separators=(",", ":"))
    metadata_ply_path = Path(metadata["ply_path"])
    selection_asset_name = build_viewer_selection_asset_name(metadata_ply_path)
    selection_asset_fallback_uri = build_viewer_selection_asset_fallback_uri(metadata_ply_path)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cable Grasp Curve Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe5;
      --panel: rgba(255, 250, 242, 0.92);
      --ink: #1f2430;
      --muted: #5b6270;
      --accent: #0f766e;
      --accent-2: #ea580c;
      --accent-3: #2563eb;
      --line: rgba(31, 36, 48, 0.12);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 32%),
        radial-gradient(circle at bottom right, rgba(234, 88, 12, 0.14), transparent 28%),
        var(--bg);
      color: var(--ink);
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(280px, 340px) 1fr;
    }}
    aside {{
      padding: 24px 20px;
      border-right: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(10px);
    }}
    main {{
      position: relative;
      min-height: 100vh;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 1.4rem;
    }}
    p {{
      margin: 0 0 14px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .meta {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin: 18px 0 22px;
    }}
    .card {{
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.68);
    }}
    .card strong {{
      display: block;
      font-size: 0.9rem;
      margin-bottom: 4px;
    }}
    .legend {{
      display: grid;
      gap: 8px;
      margin-top: 18px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.95rem;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      flex: 0 0 auto;
    }}
    .controls {{
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }}
    canvas {{
      width: 100%;
      height: 100vh;
      display: block;
      cursor: grab;
    }}
    canvas.dragging {{
      cursor: grabbing;
    }}
    .overlay {{
      position: absolute;
      right: 18px;
      bottom: 18px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255, 250, 242, 0.9);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.9rem;
    }}
    @media (max-width: 920px) {{
      body {{
        grid-template-columns: 1fr;
      }}
      aside {{
        border-right: none;
        border-bottom: 1px solid var(--line);
      }}
      canvas {{
        height: 72vh;
      }}
    }}
  </style>
</head>
<body>
  <aside>
    <h1>Cable Curve Viewer</h1>
    <p>Interactive 3D preview of the point cloud, the raw skeleton, the gap-filled curve, the preparatory grasp points, and the grasp point.</p>
    <div class="meta">
      <div class="card">
        <strong>PLY</strong>
        <span id="ply-path"></span>
      </div>
      <div class="card">
        <strong>Gap Fill</strong>
        <span id="gap-summary"></span>
      </div>
      <div class="card">
        <strong>Projection Distance</strong>
        <span id="projection-distance"></span>
      </div>
      <div class="card">
        <strong>Selected Direction</strong>
        <span id="selected-direction">Awaiting controller selection file.</span>
      </div>
      <div class="card">
        <strong>True Grasp Point</strong>
        <span id="selected-pregrasp-point"></span>
      </div>
    </div>
    <div class="legend">
      <div class="legend-item"><span class="swatch" style="background:#94a3b8"></span><span>Raw point cloud</span></div>
      <div class="legend-item"><span class="swatch" style="background:#64748b"></span><span>Downsampled cloud</span></div>
      <div class="legend-item"><span class="swatch" style="background:#ea580c"></span><span>Original skeleton</span></div>
      <div class="legend-item"><span class="swatch" style="background:#0f766e"></span><span>Gap-filled curve</span></div>
      <div class="legend-item"><span class="swatch" style="background:#14b8a6"></span><span>36 candidate gripper directions</span></div>
      <div class="legend-item"><span class="swatch" style="background:#eab308"></span><span>Controller-selected direction</span></div>
      <div class="legend-item"><span class="swatch" style="background:#dc2626"></span><span>Preparatory grasp points</span></div>
      <div class="legend-item"><span class="swatch" style="background:#f59e0b"></span><span>Selected preparatory point (P3)</span></div>
      <div class="legend-item"><span class="swatch" style="background:#2563eb"></span><span>Grasp point</span></div>
      <div class="legend-item"><span class="swatch" style="background:#7c3aed"></span><span>Projected point on cloud</span></div>
    </div>
    <div class="controls">
      Drag to rotate.<br>
      Mouse wheel to zoom.<br>
      Double-click to reset view.
    </div>
  </aside>
  <main>
    <canvas id="viewer"></canvas>
    <div class="overlay" id="view-state">yaw 0.00 | pitch 0.00 | zoom 1.00</div>
  </main>
  <script>
    const payload = {payload_json};
    const metadata = {metadata_json};
    const selectionAssetName = {json.dumps(selection_asset_name)};
    const selectionAssetFallbackUri = {json.dumps(selection_asset_fallback_uri)};

    const canvas = document.getElementById("viewer");
    const ctx = canvas.getContext("2d");
    const stateLabel = document.getElementById("view-state");
    const selectedDirectionLabel = document.getElementById("selected-direction");
    const selectedPregraspPointLabel = document.getElementById("selected-pregrasp-point");
    document.getElementById("ply-path").textContent = metadata.ply_path;
    document.getElementById("gap-summary").textContent = `${{metadata.gap_fill_segment_count}} segment(s), filled curve points: ${{metadata.filled_skeleton_point_count}}`;
    document.getElementById("projection-distance").textContent = `${{metadata.projection_distance.toFixed(6)}} m`;
    selectedPregraspPointLabel.textContent = `P${{metadata.selected_pregrasp_label}}`;
    let selection = null;

    const yawDefault = -0.72;
    const pitchDefault = 0.48;
    let yaw = yawDefault;
    let pitch = pitchDefault;
    let zoom = 1.0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    const candidateDirectionLength = 0.16;

    function updateSelectionLabel() {{
      selectedDirectionLabel.textContent = selection
        ? `candidate ${{selection.selected_gripper_direction_index}}, rot +Y ${{selection.rotation_about_positive_y.toFixed(4)}} rad, joint7 ${{selection.estimated_joint7.toFixed(4)}} rad`
        : "Awaiting controller selection file.";
    }}

    function buildDirectionSegments(origin, directions, scale) {{
      return directions.map((direction) => [
        origin,
        origin.map((value, i) => value + direction[i] * scale),
      ]);
    }}

    function getCandidateDirectionSegments() {{
      return buildDirectionSegments(
        payload.grasp_point,
        payload.gripper_directions,
        candidateDirectionLength,
      );
    }}

    function getSelectedDirectionSegments() {{
      return selection
        ? buildDirectionSegments(
            payload.grasp_point,
            [selection.selected_gripper_direction],
            candidateDirectionLength * 1.2,
          )
        : [];
    }}

    const series = [
      {{ points: payload.raw_points, color: "#94a3b8", radius: 2.0, type: "points" }},
      {{ points: payload.downsampled_points, color: "#64748b", radius: 2.6, type: "points" }},
      {{ points: payload.skeleton, color: "#ea580c", radius: 3.4, type: "line" }},
      {{ points: payload.filled_skeleton, color: "#0f766e", radius: 3.8, type: "line" }},
      {{ get segments() {{ return getCandidateDirectionSegments(); }}, color: "#14b8a6", radius: 1.8, type: "segments" }},
      {{ get segments() {{ return getSelectedDirectionSegments(); }}, color: "#eab308", radius: 4.8, type: "segments" }},
      {{ points: payload.pregrasp_points, color: "#dc2626", radius: 6.5, type: "points" }},
      {{ points: [payload.grasp_point], color: "#f59e0b", radius: 11.0, type: "points" }},
      {{ points: [payload.projected_grasp_point], color: "#7c3aed", radius: 7.0, type: "points" }},
      {{ points: [payload.grasp_point], color: "#2563eb", radius: 8.0, type: "points" }},
    ];

    function flattenPoints() {{
      return series.flatMap((entry) => {{
        if (entry.points) {{
          return entry.points;
        }}
        if (entry.segments) {{
          return entry.segments.flat();
        }}
        return [];
      }});
    }}

    const allPoints = flattenPoints();
    const mins = [Infinity, Infinity, Infinity];
    const maxs = [-Infinity, -Infinity, -Infinity];
    for (const point of allPoints) {{
      for (let i = 0; i < 3; i += 1) {{
        mins[i] = Math.min(mins[i], point[i]);
        maxs[i] = Math.max(maxs[i], point[i]);
      }}
    }}

    const center = mins.map((value, i) => 0.5 * (value + maxs[i]));
    const extent = Math.max(...maxs.map((value, i) => value - mins[i]), 1e-6);
    const normalizedSeries = series.map((entry) => {{
      if (entry.points) {{
        return {{
          ...entry,
          points: entry.points.map((point) => point.map((value, i) => (value - center[i]) / extent)),
        }};
      }}
      if (entry.segments) {{
        return {{
          ...entry,
          segments: entry.segments.map((segment) =>
            segment.map((point) => point.map((value, i) => (value - center[i]) / extent)),
          ),
        }};
      }}
      return entry;
    }});

    function resize() {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      render();
    }}

    function rotatePoint(point) {{
      const [x, y, z] = point;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);

      const x1 = cy * x + sy * z;
      const z1 = -sy * x + cy * z;
      const y2 = cp * y - sp * z1;
      const z2 = sp * y + cp * z1;
      return [x1, y2, z2];
    }}

    function projectPoint(point, width, height) {{
      const [x, y, z] = rotatePoint(point);
      const cameraDistance = 3.0 / zoom;
      const perspective = 1.0 / Math.max(0.2, cameraDistance - z);
      const scale = Math.min(width, height) * 0.42;
      return {{
        x: width * 0.5 + x * scale * perspective,
        y: height * 0.5 - y * scale * perspective,
        depth: z,
        perspective,
      }};
    }}

    function drawPoint(projected, color, radius) {{
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(projected.x, projected.y, radius * projected.perspective, 0, Math.PI * 2);
      ctx.fill();
    }}

    function render() {{
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      ctx.clearRect(0, 0, width, height);

      const drawQueue = [];
      for (const entry of normalizedSeries) {{
        if (entry.type === "points") {{
          for (const point of entry.points) {{
            const projected = projectPoint(point, width, height);
            drawQueue.push({{ depth: projected.depth, kind: "point", projected, color: entry.color, radius: entry.radius }});
          }}
        }} else if (entry.type === "segments") {{
          for (const segment of entry.segments) {{
            const p0 = projectPoint(segment[0], width, height);
            const p1 = projectPoint(segment[1], width, height);
            drawQueue.push({{
              depth: 0.5 * (p0.depth + p1.depth),
              kind: "line",
              p0,
              p1,
              color: entry.color,
              radius: entry.radius,
            }});
          }}
        }} else {{
          for (let i = 0; i < entry.points.length - 1; i += 1) {{
            const p0 = projectPoint(entry.points[i], width, height);
            const p1 = projectPoint(entry.points[i + 1], width, height);
            drawQueue.push({{
              depth: 0.5 * (p0.depth + p1.depth),
              kind: "line",
              p0,
              p1,
              color: entry.color,
              radius: entry.radius,
            }});
          }}
          for (const point of entry.points) {{
            const projected = projectPoint(point, width, height);
            drawQueue.push({{ depth: projected.depth, kind: "point", projected, color: entry.color, radius: entry.radius * 0.75 }});
          }}
        }}
      }}

      drawQueue.sort((a, b) => a.depth - b.depth);
      for (const item of drawQueue) {{
        if (item.kind === "line") {{
          ctx.beginPath();
          ctx.strokeStyle = item.color;
          ctx.lineWidth = Math.max(1.2, item.radius * 0.45 * (item.p0.perspective + item.p1.perspective));
          ctx.moveTo(item.p0.x, item.p0.y);
          ctx.lineTo(item.p1.x, item.p1.y);
          ctx.stroke();
        }} else {{
          drawPoint(item.projected, item.color, item.radius);
        }}
      }}

      if (payload.pregrasp_points && payload.pregrasp_segment_labels) {{
        ctx.save();
        ctx.font = '12px "IBM Plex Sans", "Segoe UI", sans-serif';
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        payload.pregrasp_points.forEach((point, index) => {{
          const normalizedPoint = point.map((value, i) => (value - center[i]) / extent);
          const projected = projectPoint(normalizedPoint, width, height);
          const label = payload.pregrasp_segment_labels[index] ?? (index + 1);
          if (label === metadata.selected_pregrasp_label) {{
            ctx.save();
            ctx.font = 'bold 14px "IBM Plex Sans", "Segoe UI", sans-serif';
            ctx.fillStyle = "#1d4ed8";
            ctx.strokeStyle = "rgba(255, 250, 242, 0.95)";
            ctx.lineWidth = 3.5;
            ctx.strokeText(`P${{label}}`, projected.x + 9, projected.y - 9);
            ctx.fillText(`P${{label}}`, projected.x + 9, projected.y - 9);
            ctx.restore();
          }} else {{
            ctx.fillStyle = "#991b1b";
            ctx.fillText(`P${{label}}`, projected.x + 8, projected.y - 8);
          }}
        }});
        ctx.restore();
      }}

      stateLabel.textContent = `yaw ${{yaw.toFixed(2)}} | pitch ${{pitch.toFixed(2)}} | zoom ${{zoom.toFixed(2)}}`;
    }}

    canvas.addEventListener("mousedown", (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.classList.add("dragging");
    }});

    window.addEventListener("mouseup", () => {{
      dragging = false;
      canvas.classList.remove("dragging");
    }});

    window.addEventListener("mousemove", (event) => {{
      if (!dragging) {{
        return;
      }}
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      yaw += dx * 0.01;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.01));
      render();
    }});

    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const delta = Math.sign(event.deltaY);
      zoom = Math.max(0.35, Math.min(4.0, zoom * (delta > 0 ? 0.92 : 1.08)));
      render();
    }}, {{ passive: false }});

    canvas.addEventListener("dblclick", () => {{
      yaw = yawDefault;
      pitch = pitchDefault;
      zoom = 1.0;
      render();
    }});

    function refreshSelection() {{
      const candidateSources = [
        `${{selectionAssetName}}?t=${{Date.now()}}`,
        `${{selectionAssetFallbackUri}}?t=${{Date.now()}}`,
      ];

      function trySource(index) {{
        if (index >= candidateSources.length) {{
          selection = null;
          updateSelectionLabel();
          render();
          return;
        }}

        window.viewerSelection = null;
        const script = document.createElement("script");
        script.src = candidateSources[index];
        script.onload = () => {{
          selection = window.viewerSelection;
          updateSelectionLabel();
          render();
          script.remove();
        }};
        script.onerror = () => {{
          script.remove();
          trySource(index + 1);
        }};
        document.head.appendChild(script);
      }}

      trySource(0);
    }}

    updateSelectionLabel();
    refreshSelection();
    window.setInterval(refreshSelection, 1500);
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""


def write_visualization_html(viewer_path: Path, visualization_payload: dict, metadata: dict) -> None:
    viewer_path.write_text(build_visualization_html(visualization_payload, metadata), encoding="utf-8")


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
        "--gap-ratio",
        type=float,
        default=3.0,
        help="Treat skeleton segments longer than median_spacing * gap_ratio as missing gaps to fill.",
    )
    parser.add_argument(
        "--gap-tension",
        type=float,
        default=0.6,
        help="Scale factor controlling the curvature of the cubic Hermite gap interpolation.",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Generate an interactive HTML viewer and open it in the default browser.",
    )
    parser.add_argument(
        "--gripper-direction-samples",
        type=int,
        default=36,
        help="Number of uniformly sampled unit directions on the plane perpendicular to the cable tangent.",
    )
    parser.add_argument(
        "--viewer-path",
        type=Path,
        default=None,
        help="Optional output path for the interactive viewer HTML file.",
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
    if args.gap_ratio <= 1.0:
        raise ValueError("--gap-ratio must be greater than 1.")
    if args.gap_tension <= 0.0:
        raise ValueError("--gap-tension must be greater than 0.")
    if args.gripper_direction_samples < 4:
        raise ValueError("--gripper-direction-samples must be at least 4.")

    result, visualization_payload = compute_grasp_point(
        args.ply_path,
        args.voxel_size,
        args.neighbors,
        args.gap_ratio,
        args.gap_tension,
        args.gripper_direction_samples,
    )
    output_path = build_output_path(args.ply_path)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    viewer_path = args.viewer_path if args.viewer_path is not None else build_viewer_output_path(args.ply_path)
    if args.show_viewer:
        write_visualization_html(viewer_path, visualization_payload, result)
        webbrowser.open(viewer_path.resolve().as_uri())
    print(json.dumps(result, indent=2))
    print(f"Saved grasp point to: {output_path}")
    if args.show_viewer:
        print(f"Saved interactive viewer to: {viewer_path}")


if __name__ == "__main__":
    main()
