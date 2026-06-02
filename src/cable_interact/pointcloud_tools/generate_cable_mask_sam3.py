#!/usr/bin/env python3
"""Generate a binary cable foreground mask for a captured RGB image with SAM3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM3_PROJECT_ROOT = PROJECT_ROOT / "sam3_project"
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "pointcloud_tools" / "info_for_3Dpoint" / "multi_grasp"
)
DEFAULT_CHECKPOINT = SAM3_PROJECT_ROOT / "checkpoints" / "sam3.pt"

# Prefer the vendored SAM3 source copied into cable_interact.
sys.path.insert(0, str(SAM3_PROJECT_ROOT))

import torch  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM3 to replace a captured cable sample's mask.png."
    )
    parser.add_argument(
        "--cable-id",
        default="cable_000",
        help="Sample directory name under multi_grasp (default: cable_000).",
    )
    parser.add_argument(
        "--rgb-id",
        help="RGB filename or stem (default: rgb_<suffix of cable-id>.png).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing cable_XXX sample directories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Local SAM3 checkpoint path.",
    )
    parser.add_argument("--prompt", default="cable", help="SAM3 text prompt.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum SAM3 candidate score (default: 0.5).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0,
        help="Discard candidates smaller than this fraction of the image.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=1.0,
        help="Discard candidates larger than this fraction of the image.",
    )
    parser.add_argument(
        "--selection",
        choices=("union", "best"),
        default="union",
        help="Merge all retained masks or use only the highest-score mask.",
    )
    parser.add_argument(
        "--overlay-name",
        default="mask_overlay.png",
        help="Debug overlay filename written beside mask.png.",
    )
    return parser.parse_args()


def rgb_filename(cable_id: str, rgb_id: str | None) -> str:
    if rgb_id:
        return rgb_id if Path(rgb_id).suffix else f"{rgb_id}.png"
    suffix = cable_id.removeprefix("cable_")
    return f"rgb_{suffix}.png"


def save_overlay(image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    overlay = image.astype(np.float32)
    foreground = mask > 0
    overlay[foreground] = overlay[foreground] * 0.55 + np.array(
        [255.0, 0.0, 0.0]
    ) * 0.45
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(output_path)


def main() -> None:
    args = parse_args()
    cable_dir = args.dataset_root.expanduser().resolve() / args.cable_id
    image_path = cable_dir / rgb_filename(args.cable_id, args.rgb_id)
    mask_path = cable_dir / "mask.png"
    overlay_path = cable_dir / args.overlay_name
    checkpoint_path = args.checkpoint.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
    if not 0.0 <= args.min_area_ratio <= args.max_area_ratio <= 1.0:
        raise ValueError("Area ratios must satisfy 0 <= min <= max <= 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM3 on {device}: {checkpoint_path}")
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        device=device,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=args.prompt)

    image_area = image_array.shape[0] * image_array.shape[1]
    candidates: list[tuple[float, np.ndarray]] = []
    for index, (mask, score) in enumerate(zip(output["masks"], output["scores"])):
        mask_array = np.squeeze(mask.detach().cpu().numpy()).astype(bool)
        score_value = float(score.detach().cpu().item())
        area_ratio = float(mask_array.sum()) / image_area
        retained = args.min_area_ratio <= area_ratio <= args.max_area_ratio
        print(
            f"candidate {index}: score={score_value:.4f}, "
            f"area_ratio={area_ratio:.6f}, retained={retained}"
        )
        if retained:
            candidates.append((score_value, mask_array))

    if not candidates:
        raise RuntimeError(
            "SAM3 produced no retained cable mask. Adjust --confidence-threshold "
            "or the area-ratio limits."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    if args.selection == "best":
        foreground = candidates[0][1]
    else:
        foreground = np.logical_or.reduce([item[1] for item in candidates])

    binary_mask = foreground.astype(np.uint8) * 255
    Image.fromarray(binary_mask, mode="L").save(mask_path)
    save_overlay(image_array, binary_mask, overlay_path)
    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")


if __name__ == "__main__":
    main()
