#!/usr/bin/env python3
"""Process Mini PC multi-grasp jobs on a GPU PC and return grasp plans."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


DEFAULT_DATASET_ROOT = Path(
    "/home/flexcycle/Motion_Correlation/cmcor/datasets/CMCor_multigrasp_board"
)
DEFAULT_CMCOR_ROOT = Path("/home/flexcycle/Motion_Correlation/cmcor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch Mini PC jobs, run CMCor on this GPU PC, and return grasp plans."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cmcor-python-root", type=Path, default=DEFAULT_CMCOR_ROOT)
    parser.add_argument("--job-inbox", type=Path)
    parser.add_argument("--result-staging-root", type=Path)
    parser.add_argument("--result-outbox", type=Path)
    parser.add_argument("--poll-period-sec", type=float, default=1.0)
    parser.add_argument("--sequence-stable-sec", type=float, default=2.0)
    parser.add_argument("--sequence-wait-timeout-sec", type=float, default=300.0)
    return parser.parse_args()


def add_import_path(path: Path) -> None:
    path_text = str(path.expanduser())
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def make_sample_grasp_id(sample_index: int) -> str:
    return f"grasp_sample_{sample_index:03d}"


def transform_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    if points.size == 0:
        return points
    return (points @ rotation.T) + translation[None, :]


def transform_vectors(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    return vectors @ rotation.T


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
        directions.append((direction / np.linalg.norm(direction)).tolist())
    return directions


def load_last_depth(sequence_dir: Path) -> np.ndarray:
    depth_paths = sorted(sequence_dir.glob("depth_*.png"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth_*.png files found in {sequence_dir}")
    depth = cv2.imread(str(depth_paths[-1]), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_paths[-1]}")
    return depth


def directory_signature(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def wait_for_stable_sequence(
    sequence_dir: Path, timeout_sec: float, stable_sec: float
) -> None:
    deadline = time.time() + timeout_sec
    previous_signature = None
    stable_since = None
    while time.time() < deadline:
        if (sequence_dir / "actions_gripper.json").is_file():
            signature = directory_signature(sequence_dir)
            if signature == previous_signature:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_sec:
                    return
            else:
                previous_signature = signature
                stable_since = None
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for stable sequence directory: {sequence_dir}")


def save_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {points.shape[0]}\n")
        fp.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for point in points:
            fp.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def compute_motion_masks(sequence_dir: Path, job: dict) -> tuple[list[np.ndarray], np.ndarray]:
    from cmcor import motion_perception, motion_perception_settings

    mp = motion_perception.MotionPerception(
        gpu=bool(job["gpu"]), use_dummy_flow=bool(job["use_dummy_flow"])
    )
    mp.load_buffer_from_disk(str(sequence_dir))
    if not any(int(action) >= 0 for action in mp.action_buffer):
        raise RuntimeError("Recorded sequence has no active action frames")
    algorithm = str(job["algorithm"])
    settings = motion_perception_settings.MotionPerceptionSettings()
    if algorithm == "correlation":
        mp.compute_correlation_images()
        if mp.corr_imgs is None:
            raise RuntimeError("Motion correlation produced no correlation images")
        masks = [
            np.nan_to_num(image, nan=0.0) > settings.motion_correlation_min_cod
            for image in mp.corr_imgs
        ]
    elif algorithm == "segmentation":
        mp.compute_segmentation_images()
        if mp.corr_imgs is None:
            raise RuntimeError("Motion segmentation produced no segmentation images")
        masks = [image > settings.motion_segmentation_vote_thr for image in mp.corr_imgs]
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if not masks:
        raise RuntimeError("Motion perception produced no masks")
    with (sequence_dir / "actions_gripper.json").open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    if "camera_matrix" not in metadata:
        raise RuntimeError("Recorded sequence has no camera_matrix in actions_gripper.json")
    return masks, np.asarray(metadata["camera_matrix"], dtype=np.float64)


def sample_next_grasp(
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    motion_masks: list[np.ndarray],
    previous_grasps_camera: list[list[float]],
):
    from cmcor import grasp_segment_sampler

    vote_image = np.zeros_like(motion_masks[0], dtype=np.float32)
    for mask in motion_masks:
        vote_image[mask > 0] += 1.0
    if not np.any(vote_image > 0):
        raise RuntimeError("Motion perception vote image is empty")
    sampler = grasp_segment_sampler.GraspSegmentSampler()
    sampler.clear_previous_grasps()
    for point_camera in previous_grasps_camera:
        sampler.add_previous_grasp(np.asarray(point_camera, dtype=np.float64))
    sampler.set_input_images(depth, camera_matrix, vote_image)
    sampled = sampler.sample_grasp()
    if sampled is None:
        raise RuntimeError("No valid next grasp was sampled")
    return sampled, vote_image


def write_result(job: dict, sequence_dir: Path, output_root: Path) -> Path:
    masks, camera_matrix = compute_motion_masks(sequence_dir, job)
    depth = load_last_depth(sequence_dir)
    sampled, vote_image = sample_next_grasp(
        depth, camera_matrix, masks, list(job["previous_grasps_camera"])
    )
    grasp, grasp_mask = sampled
    center_camera, axis_camera, points_camera = grasp
    center_camera = np.asarray(center_camera, dtype=np.float64)
    axis_camera = np.asarray(axis_camera, dtype=np.float64)
    axis_camera /= np.linalg.norm(axis_camera)
    points_camera = np.asarray(points_camera, dtype=np.float64)
    rotation = np.asarray(job["camera_to_robot_rotation"], dtype=np.float64)
    translation = np.asarray(job["camera_to_robot_translation"], dtype=np.float64)
    center_robot = transform_points(center_camera[None, :], rotation, translation)[0]
    axis_robot = transform_vectors(axis_camera[None, :], rotation)[0]
    axis_robot /= np.linalg.norm(axis_robot)
    points_robot = transform_points(points_camera, rotation, translation)
    gripper_directions = build_gripper_directions(
        axis_robot, int(job["gripper_direction_sample_count"])
    )

    grasp_name = make_sample_grasp_id(int(job["sample_grasp_index"]))
    grasp_dir = output_root / grasp_name
    grasp_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "source": "multigrasp_3090_worker",
        "source_sequence": sequence_dir.name,
        "camera_frame": str(job["camera_frame"]),
        "robot_frame": str(job["robot_frame"]),
        "grasp_point": center_robot.tolist(),
        "local_tangent": axis_robot.tolist(),
        "gripper_direction_sample_count": len(gripper_directions),
        "gripper_directions": gripper_directions,
        "camera_grasp_point": center_camera.tolist(),
        "camera_local_tangent": axis_camera.tolist(),
        "sampled_segment_point_count": int(points_camera.shape[0]),
    }
    with (grasp_dir / grasp_name).open("w", encoding="utf-8") as fp:
        json.dump(plan, fp, sort_keys=True, indent=2)
        fp.write("\n")

    if bool(job["write_debug_masks"]):
        cv2.imwrite(str(grasp_dir / f"{grasp_name}_mask.png"), 255 * (grasp_mask > 0).astype(np.uint8))
        cv2.imwrite(
            str(grasp_dir / f"{grasp_name}_vote.png"),
            np.clip(255.0 * vote_image / max(1.0, float(vote_image.max())), 0, 255).astype(np.uint8),
        )
        save_point_cloud_ply(grasp_dir / f"{grasp_name}_camera_segment.ply", points_camera)
        save_point_cloud_ply(grasp_dir / f"{grasp_name}_robot_segment.ply", points_robot)
        debug_dir = grasp_dir / "debug" / sequence_dir.name
        debug_dir.mkdir(parents=True, exist_ok=True)
        for idx, mask in enumerate(masks):
            cv2.imwrite(str(debug_dir / f"motion_mask_{idx:02d}.png"), 255 * mask.astype(np.uint8))
    return grasp_dir


def publish_file(path: Path, result_outbox: Path) -> None:
    result_outbox.mkdir(parents=True, exist_ok=True)
    final_path = result_outbox / path.name
    tmp_path = result_outbox / f"{path.name}.tmp"
    tmp_path.unlink(missing_ok=True)
    final_path.unlink(missing_ok=True)
    os.replace(path, tmp_path)
    os.replace(tmp_path, final_path)


def process_job(job_path: Path, args: argparse.Namespace) -> None:
    claimed_path = job_path.with_suffix(".processing")
    os.replace(job_path, claimed_path)
    try:
        with claimed_path.open("r", encoding="utf-8") as fp:
            job = json.load(fp)
        job_id = str(job["job_id"])
        sequence_dir = args.dataset_root / "motion_correlation_buffers" / str(job["sequence_name"])
        print(f"Processing {job_id}; waiting for {sequence_dir}", flush=True)
        wait_for_stable_sequence(
            sequence_dir, args.sequence_wait_timeout_sec, args.sequence_stable_sec
        )
        with tempfile.TemporaryDirectory(prefix=f"{job_id}_", dir=args.result_staging_root) as tmp_dir:
            tmp_root = Path(tmp_dir)
            grasp_dir = write_result(job, sequence_dir, tmp_root)
            archive_path = args.result_staging_root / f"{job_id}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(grasp_dir, arcname=grasp_dir.name)
            ready_path = args.result_staging_root / f"{job_id}.ready"
            ready_path.write_text("ready\n", encoding="utf-8")
            publish_file(archive_path, args.result_outbox)
            publish_file(ready_path, args.result_outbox)
        os.replace(claimed_path, claimed_path.with_suffix(".done"))
        print(f"Completed {job_id}", flush=True)
    except Exception as exc:  # noqa: BLE001
        error = {"error": str(exc), "traceback": traceback.format_exc()}
        error_path = args.result_staging_root / f"{claimed_path.stem}.error.json"
        error_path.write_text(json.dumps(error, indent=2) + "\n", encoding="utf-8")
        publish_file(error_path, args.result_outbox)
        os.replace(claimed_path, claimed_path.with_suffix(".error"))
        print(f"Failed {claimed_path.stem}: {exc}", file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser()
    args.cmcor_python_root = args.cmcor_python_root.expanduser()
    args.job_inbox = (
        args.job_inbox.expanduser()
        if args.job_inbox
        else args.dataset_root / "multigrasp_jobs"
    )
    args.result_staging_root = (
        args.result_staging_root.expanduser()
        if args.result_staging_root
        else args.dataset_root / "multigrasp_worker_results"
    )
    args.result_outbox = (
        args.result_outbox.expanduser()
        if args.result_outbox
        else args.dataset_root / "multigrasp_result_outbox"
    )
    args.job_inbox.mkdir(parents=True, exist_ok=True)
    args.result_staging_root.mkdir(parents=True, exist_ok=True)
    args.result_outbox.mkdir(parents=True, exist_ok=True)
    add_import_path(args.cmcor_python_root)
    print(f"Watching multi-grasp jobs in {args.job_inbox}", flush=True)
    while True:
        jobs = sorted(args.job_inbox.glob("*.json"))
        if not jobs:
            time.sleep(args.poll_period_sec)
            continue
        for job_path in jobs:
            process_job(job_path, args)


if __name__ == "__main__":
    main()
