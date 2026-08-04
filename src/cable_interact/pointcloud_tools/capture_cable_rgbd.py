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


def validate_crop_rect(crop_rect, image_width, image_height):
    if crop_rect is None:
        return None

    left, top, width, height = crop_rect
    if width <= 0 or height <= 0:
        raise ValueError("Crop width and height must be positive.")
    if left < 0 or top < 0:
        raise ValueError("Crop left and top must be non-negative.")
    if left + width > image_width or top + height > image_height:
        raise ValueError(
            "Crop rectangle must fit inside the captured image: "
            f"left={left}, top={top}, width={width}, height={height}, "
            f"image={image_width}x{image_height}"
        )
    return left, top, width, height


def crop_frame(image, crop_rect):
    if crop_rect is None:
        return image
    left, top, width, height = crop_rect
    return image[top:top + height, left:left + width]


def crop_camera_matrix(camera_matrix, crop_rect):
    if crop_rect is None:
        return camera_matrix

    left, top, _, _ = crop_rect
    cropped = [row[:] for row in camera_matrix]
    cropped[0][2] -= left
    cropped[1][2] -= top
    return cropped


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
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help=(
            "Save only this ROI from the aligned RGB/depth frames. "
            "The saved camera_matrix principal point is shifted accordingly."
        ),
    )
    return parser.parse_args()


def run_capture(output_root, scp_destination="", scp_timeout_sec=300.0, crop_rect=None):
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
        image_width = intr.width
        image_height = intr.height
        crop_rect = validate_crop_rect(crop_rect, image_width, image_height)
        camera_matrix = [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ]
        saved_camera_matrix = crop_camera_matrix(camera_matrix, crop_rect)

        print("K =", saved_camera_matrix)
        if crop_rect is not None:
            left, top, width, height = crop_rect
            print(f"Saving crop: left={left}, top={top}, width={width}, height={height}")
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
            saved_color = crop_frame(color, crop_rect)
            saved_depth_mm = crop_frame(depth_mm, crop_rect)

            cv2.imshow("color", saved_color)
            cv2.imshow("depth", cv2.convertScaleAbs(saved_depth_mm, alpha=0.03))

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
                mask = np.zeros(saved_depth_mm.shape, dtype=np.uint8)

                cv2.imwrite(str(rgb_path), saved_color)
                cv2.imwrite(str(depth_path), saved_depth_mm)
                cv2.imwrite(str(mask_path), mask)

                with parameters_path.open("w", encoding="utf-8") as handle:
                    parameters = {"camera_matrix": saved_camera_matrix}
                    if crop_rect is not None:
                        parameters["crop"] = {
                            "left": crop_rect[0],
                            "top": crop_rect[1],
                            "width": crop_rect[2],
                            "height": crop_rect[3],
                        }
                    json.dump(parameters, handle, indent=2)

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
    run_capture(args.output_root, args.scp_destination, args.scp_timeout_sec, args.crop)


if __name__ == "__main__":
    main()
