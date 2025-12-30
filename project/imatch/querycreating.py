"""
Utilities to create rotated + center-cropped reference images for robustness tests.

The functions here are kept lightweight so they can be reused both from scripts
and interactive notebooks.  They operate on plain filesystem paths and rely on
Pillow for the heavy lifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

from PIL import Image, ImageOps


SUPPORTED_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ReferenceSummary:
    """Simple record describing a generated reference image."""

    source: Path
    output: Path
    angle: float
    crop_ratio: float
    resized_to_original: bool


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_angle(angle: float) -> str:
    if float(angle).is_integer():
        return f"{int(angle):03d}"
    return str(angle).replace(".", "p").replace("-", "m")


def _center_crop(image: Image.Image, crop_ratio: float) -> Image.Image:
    if not 0 < crop_ratio <= 1:
        raise ValueError(f"crop_ratio must be within (0, 1]; received {crop_ratio}")

    width, height = image.size
    new_w = max(1, int(round(width * crop_ratio)))
    new_h = max(1, int(round(height * crop_ratio)))

    left = (width - new_w) // 2
    top = (height - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def build_reference_image(
    image: Image.Image,
    angle: float,
    crop_ratio: float,
    resize_to_original: bool = False,
) -> Image.Image:
    """
    Rotate and crop the provided image to build a reference variant.

    Parameters
    ----------
    image:
        Original PIL image.
    angle:
        Degrees to rotate (counter-clockwise). Any float is accepted.
    crop_ratio:
        Center-crop ratio within (0, 1]. For example 0.5 keeps the central 50%.
    resize_to_original:
        If True, resize the cropped patch back to the original resolution. When
        False (default) the output keeps the cropped resolution, which mirrors
        the intended “zoomed-in” viewpoint difference.
    """

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    orig_size = image.size
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    aligned = ImageOps.fit(
        rotated,
        orig_size,
        method=Image.BICUBIC,
        centering=(0.5, 0.5),
    )
    cropped = _center_crop(aligned, crop_ratio)

    if resize_to_original and cropped.size != orig_size:
        cropped = cropped.resize(orig_size, resample=Image.BICUBIC)
    return cropped


def iter_source_images(source_dir: Path, suffixes: Sequence[str] | None = None) -> Iterable[Path]:
    """Yield every image path under `source_dir` matching the allowed suffixes."""
    suffixes = tuple(suffixes or SUPPORTED_SUFFIXES)
    for suffix in suffixes:
        yield from source_dir.glob(f"*{suffix}")


def generate_references_for_directory(
    source_dir: Path,
    destination_dir: Path,
    angles: Sequence[float],
    crop_ratio: float = 0.5,
    overwrite: bool = True,
    suffixes: Sequence[str] | None = None,
    resize_to_original: bool = False,
    image_format: str = "JPEG",
    progress_callback: Callable[[ReferenceSummary], None] | None = None,
) -> List[ReferenceSummary]:
    """
    Generate rotated + cropped reference images for every file in `source_dir`.

    Parameters
    ----------
    progress_callback:
        Optional callable invoked every time a reference image is written. Receives
        the ReferenceSummary for that output.
    """
    if not angles:
        raise ValueError("\033[31m\tangles must contain at least one rotation value.\033[0m")

    source_dir = source_dir.expanduser().resolve()
    destination_dir = destination_dir.expanduser().resolve()
    _ensure_directory(destination_dir)

    results: List[ReferenceSummary] = []
    files = list(iter_source_images(source_dir, suffixes))
    for idx, img_path in enumerate(sorted(files)):
        try:
            with Image.open(img_path) as img:
                for angle in angles:
                    angle_token = _sanitize_angle(angle)
                    crop_token = int(round(crop_ratio * 100))
                    output_name = f"{img_path.stem}_rot{angle_token}_crop{crop_token}.jpg"
                    output_path = destination_dir / output_name

                    if output_path.exists() and not overwrite:
                        continue

                    reference_img = build_reference_image(
                        img,
                        angle=angle,
                        crop_ratio=crop_ratio,
                        resize_to_original=resize_to_original,
                    )
                    reference_img.save(output_path, format=image_format, quality=95)

                    summary = ReferenceSummary(
                        source=img_path,
                        output=output_path,
                        angle=angle,
                        crop_ratio=crop_ratio,
                        resized_to_original=resize_to_original,
                    )
                    results.append(summary)
                    if progress_callback is not None:
                        progress_callback(summary)
        except OSError as err:
            print(f"\033[31m\t[WARN] Failed to process {img_path}: {err}\033[0m")

    return results
