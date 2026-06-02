"""
Capture aligned RealSense RGB-D frames for cable point-cloud reconstruction.

Example:
python3 -m pointcloud_tools.capture_cable_rgbd
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from pointcloud_tools.remote_transfer import ScpTransferQueue


DEFAULT_INFO_ROOT = Path("/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint")
DEFAULT_OUTPUT_ROOT = DEFAULT_INFO_ROOT / "multi_grasp"
CABLE_DIR_PREFIX = "cable_"
DEPTH_NAME = "depth.png"
MASK_NAME = "mask.png"
PARAMETERS_NAME = "parameters.json"


def next_cable_dir(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    existing_indices = []
    for path in output_root.iterdir():
        if not path.is_dir() or not path.name.startswith(CABLE_DIR_PREFIX):
            continue
        suffix = path.name[len(CABLE_DIR_PREFIX):]
        if suffix.isdigit():
            existing_indices.append(int(suffix))

    next_index = max(existing_indices, default=-1) + 1
    return output_root / f"{CABLE_DIR_PREFIX}{next_index:03d}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture aligned RealSense RGB, depth, mask placeholder, and camera parameters."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root folder for multi-grasp cable_XXX captures. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )
    parser.add_argument(
        "--scp-destination",
        default="",
        help=(
            "Optional remote folder for background uploads, for example "
            "flexcycle@10.157.175.101:~/Desktop/cable_rgbd/. "
            "Passwordless SSH must already be configured."
        ),
    )
    parser.add_argument(
        "--scp-timeout-sec",
        type=float,
        default=300.0,
        help="Timeout for each background SCP upload. Default: 300",
    )
    return parser.parse_args()


def run_capture(output_root, scp_destination="", scp_timeout_sec=300.0):
    uploader = (
        ScpTransferQueue(scp_destination, timeout_sec=scp_timeout_sec)
        if scp_destination
        else None
    )
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align = rs.align(rs.stream.color)

    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        camera_matrix = [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ]

        print("K =", camera_matrix)
        print(f"Saving captures under: {output_root}")
        print("Default multi-grasp layout: info_for_3Dpoint/multi_grasp/cable_XXX")
        print("Press 's' to save a cable frame, or Esc to exit.")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            depth_mm = (depth * depth_scale * 1000).astype(np.uint16)

            cv2.imshow("color", color)
            cv2.imshow("depth", cv2.convertScaleAbs(depth, alpha=0.03))

            key = cv2.waitKey(1)

            if key == ord("s"):
                cable_dir = next_cable_dir(output_root)
                cable_dir.mkdir(parents=True, exist_ok=True)
                cable_index = cable_dir.name[len(CABLE_DIR_PREFIX):]

                rgb_path = cable_dir / f"rgb_{cable_index}.png"
                depth_path = cable_dir / DEPTH_NAME
                mask_path = cable_dir / MASK_NAME
                parameters_path = cable_dir / PARAMETERS_NAME

                # Downstream tools expect a mask file next to RGB, depth, and parameters.
                mask = np.zeros(depth_mm.shape, dtype=np.uint8)

                cv2.imwrite(str(rgb_path), color)
                cv2.imwrite(str(depth_path), depth_mm)
                cv2.imwrite(str(mask_path), mask)

                with parameters_path.open("w", encoding="utf-8") as handle:
                    json.dump({"camera_matrix": camera_matrix}, handle, indent=2)

                print(
                    "Saved "
                    f"{rgb_path.name}, {depth_path.name}, {mask_path.name}, {parameters_path.name} "
                    f"to {cable_dir}"
                )
                if uploader is not None:
                    uploader.submit(cable_dir)

            elif key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if uploader is not None:
            uploader.close()


def main():
    args = parse_args()
    run_capture(args.output_root, args.scp_destination, args.scp_timeout_sec)


if __name__ == "__main__":
    main()
