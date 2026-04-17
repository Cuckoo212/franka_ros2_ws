#!/usr/bin/python3.10

import argparse
import json
import heapq
import re
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
        tangent_xz = tangent_xz / tangent_xz_norm
        direction_a = normalize_vector(np.cross(Y_AXIS, tangent_xz))

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


def compute_grasp_point(
    ply_path: Path,
    voxel_size: float | None,
    neighbor_count: int,
) -> dict:
    points = parse_ascii_ply_vertices(ply_path)
    effective_voxel_size = estimate_default_voxel_size(points) if voxel_size is None else voxel_size
    downsampled_points = voxel_downsample(points, effective_voxel_size)
    skeleton = extract_skeleton_path(downsampled_points, neighbor_count)
    skeleton_midpoint = compute_polyline_midpoint(skeleton)
    local_tangent = compute_tangent_at_point(skeleton, skeleton_midpoint)
    gripper_direction_a, gripper_direction_b = compute_opposite_gripper_directions(local_tangent)
    grasp_point, projection_distance = project_to_point_cloud(skeleton_midpoint, points)

    return {
        "ply_path": str(ply_path),
        "point_count": int(len(points)),
        "downsampled_point_count": int(len(downsampled_points)),
        "voxel_size": float(effective_voxel_size),
        "neighbor_count": int(neighbor_count),
        "skeleton_point_count": int(len(skeleton)),
        "skeleton_midpoint": skeleton_midpoint.tolist(),
        "local_tangent": local_tangent.tolist(),
        "gripper_direction_a": gripper_direction_a.tolist(),
        "gripper_direction_b": gripper_direction_b.tolist(),
        "grasp_point": grasp_point.tolist(),
        "projection_distance": float(projection_distance),
    }


def extract_cable_id(ply_path: Path) -> str:
    for part in ply_path.parts:
        match = re.fullmatch(r"cable_(\d+)", part)
        if match is not None:
            return match.group(1)
    raise ValueError(f"Failed to infer cable id from path: {ply_path}")


def build_output_path(ply_path: Path) -> Path:
    cable_id = extract_cable_id(ply_path)
    return ply_path.parent / f"grasp_point_cable_{cable_id}"


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ply_path.is_file():
        raise FileNotFoundError(f"PLY file not found: {args.ply_path}")
    if args.voxel_size is not None and args.voxel_size <= 0.0:
        raise ValueError("--voxel-size must be greater than 0.")
    if args.neighbors < 2:
        raise ValueError("--neighbors must be at least 2.")

    result = compute_grasp_point(args.ply_path, args.voxel_size, args.neighbors)
    output_path = build_output_path(args.ply_path)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved grasp point to: {output_path}")


if __name__ == "__main__":
    main()
