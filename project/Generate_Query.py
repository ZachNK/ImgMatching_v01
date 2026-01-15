"""
Generate rotated + center-cropped reference images for rotation-robustness tests.

The script relies on the helpers in imatch.querycreating so future automation
can import and reuse the same logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from imatch.querycreating import generate_references_for_directory, iter_source_images
from imatch.utils import create_progress
from imatch.loading import (
    IMG_ROOT,
    REFERENCE_ROOT,
    REFERENCE_PREFIX,
    REFERENCE_DATASET_PREFIX,
    DATASET_KEY as DEFAULT_DATASET_KEY,
)

ANGLES: Sequence[float] = (45.0, 90.0, 135.0, 180.0)
CROP_RATIO: float = 0.5
DATASET_KEY = os.getenv("DATASET_KEY", DEFAULT_DATASET_KEY)
SOURCE_ROOT = IMG_ROOT / DATASET_KEY
REFERENCE_BASE = REFERENCE_ROOT


@dataclass(frozen=True)
class ReferenceTask:
    source: Path
    destination: Path


FOLDERS: Sequence[str] = (
    "251030172146_300",
)

TASKS: Iterable[ReferenceTask] = tuple(
    ReferenceTask(
        source=SOURCE_ROOT / folder,
        destination=REFERENCE_BASE / folder,
    )
    for folder in FOLDERS
)


def _count_source_images(path: Path) -> int:
    expanded = path.expanduser()
    if not expanded.exists():
        return 0
    return sum(1 for _ in iter_source_images(expanded))


def _estimate_total_outputs(tasks: Iterable[ReferenceTask]) -> int:
    image_count = sum(_count_source_images(task.source) for task in tasks)
    return image_count * len(ANGLES)


def main() -> None:
    total_outputs = 0
    planned_total = _estimate_total_outputs(TASKS)
    dynamic_total = planned_total
    completed = 0

    with create_progress() as progress:
        progress_task = progress.add_task(
            "[cyan]Reference generation[/cyan]", total=dynamic_total or None
        )

        def _advance_progress(_: object) -> None:
            nonlocal completed, dynamic_total
            completed += 1
            if dynamic_total and completed > dynamic_total:
                dynamic_total = completed
                progress.update(progress_task, total=dynamic_total)
            progress.advance(progress_task)

        for task in TASKS:
            src = task.source.expanduser()
            dst = task.destination.expanduser()

            if not src.exists():
                print(f"\033[31m\t[WARN] Source directory missing, skipping: {src}\033[0m")
                continue

            print(f"\033[36m[INFO] Generating references for {src} -> {dst}\033[0m")
            results = generate_references_for_directory(
                source_dir=src,
                destination_dir=dst,
                angles=ANGLES,
                crop_ratio=CROP_RATIO,
                overwrite=True,
                resize_to_original=False,
                progress_callback=_advance_progress,
            )
            total_outputs += len(results)
            print(f"\t\033[36m\tGenerated {len(results)} reference images.\033[0m")

    print(f"\033[32m\t[DONE] Total reference images generated: {total_outputs}\033[0m")


if __name__ == "__main__":
    main()
