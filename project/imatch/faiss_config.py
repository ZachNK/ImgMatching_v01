from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


WEIGHT_KEYS = ("vits16+", "vitb16", "vith16+", "vitl16", "vitl16sat", "vits16")


@dataclass
class RootConfig:
    raw_db_roots: Dict[str, Path]
    raw_reference_roots: Dict[str, Path]
    variant_db_base: Path
    variant_reference_base: Path


def _make_raw_roots(db_base: Path, reference_base: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    db_roots = {w: db_base / w for w in WEIGHT_KEYS}
    reference_roots = {w: reference_base / w for w in WEIGHT_KEYS}
    return db_roots, reference_roots


def _default_raw_bases(dataset: str) -> Tuple[Path, Path]:
    """
    Resolve raw base paths for a dataset.
    - env overrides (RAW_EMBED_ROOT / RAW_REFERENCE_ROOT) take highest priority.
    - For shinsung on Windows, keep the historical /exports default.
    - For jamshill on Windows, default to D:\dinov3_exports\... (reference uses jamshill_data as provided).
    """
    env_db = os.getenv("RAW_EMBED_ROOT")
    env_reference = os.getenv("RAW_REFERENCE_ROOT")

    if dataset == "jamshill":
        posix_db = Path("/exports/dinov3_embeds/jamshill_data")
        posix_reference = Path("/exports/dinov3_reference_embeds/jamshill_data")
        win_db = Path(r"D:\dinov3_exports\dinov3_embeds\jamshill_data")
        # Reference base follows the user-provided jamshill_data naming on Windows.
        win_reference = Path(r"D:\dinov3_exports\dinov3_reference_embeds\jamshill_data")
    else:
        posix_db = Path("/exports/dinov3_embeds/shinsung_data")
        posix_reference = Path("/exports/dinov3_reference_embeds/shinsung_data")
        win_db = posix_db
        win_reference = posix_reference

    if os.name == "nt":
        default_db, default_reference = win_db, win_reference
    else:
        default_db, default_reference = posix_db, posix_reference

    db_base = Path(env_db) if env_db else default_db
    reference_base = Path(env_reference) if env_reference else default_reference
    return db_base, reference_base


def _default_variant_db_base(dataset: str = "shinsung") -> Path:
    env_override = os.getenv("VARIANT_DB_BASE")
    if env_override:
        return Path(env_override)
    base_name = f"{dataset}_data"
    if os.name == "nt":
        return Path(rf"H:\dinov3_exports\dinov3_embeds\{base_name}")
    return Path(f"/exports/dinov3_embeds/{base_name}")


def _default_variant_reference_base(dataset: str = "shinsung") -> Path:
    env_override = os.getenv("VARIANT_REFERENCE_BASE")
    if env_override:
        return Path(env_override)
    base_name = f"{dataset}_data"
    if os.name == "nt" and dataset == "jamshill":
        base_name = "jamshill_data"
    if os.name == "nt":
        return Path(rf"H:\dinov3_exports\dinov3_reference_embeds\{base_name}")
    return Path(f"/exports/dinov3_reference_embeds/{base_name}")


def _build_root_config(dataset: str) -> RootConfig:
    db_base, reference_base = _default_raw_bases(dataset)
    variant_db_base = _default_variant_db_base(dataset)
    variant_reference_base = _default_variant_reference_base(dataset)
    raw_db_roots, raw_reference_roots = _make_raw_roots(db_base, reference_base)
    return RootConfig(
        raw_db_roots=raw_db_roots,
        raw_reference_roots=raw_reference_roots,
        variant_db_base=variant_db_base,
        variant_reference_base=variant_reference_base,
    )


DEFAULT_DATASET = "shinsung"
ROOT_CONFIG = _build_root_config(DEFAULT_DATASET)

# Backward-compatible aliases for default dataset config.
RAW_DB_BASE, RAW_REFERENCE_BASE = _default_raw_bases(DEFAULT_DATASET)
RAW_DB_ROOTS = ROOT_CONFIG.raw_db_roots
RAW_REFERENCE_ROOTS = ROOT_CONFIG.raw_reference_roots
VARIANT_DB_BASE = ROOT_CONFIG.variant_db_base
VARIANT_REFERENCE_BASE = ROOT_CONFIG.variant_reference_base


def _default_out_root() -> Path:
    """
    Pick an absolute output root that works on the current OS.
    - Respect FAISS_OUT_ROOT when provided.
    - Use D: drive when running on Windows.
    - Use /exports (typical container mount for the D: drive) on POSIX.
    """
    env_override = os.getenv("FAISS_OUT_ROOT")
    if env_override:
        return Path(env_override)
    if os.name == "nt":
        return Path(r"D:\dinov3_exports\dinov3_faiss_match")
    return Path("/exports/dinov3_faiss_match")


DEFAULT_OUT_ROOT = _default_out_root()

DIRECTION_CHOICES = ("reference-data", "data-reference", "reference-reference", "data-data")
LEGACY_DIRECTION_ALIASES = {
    "query-data": "reference-data",
    "data-query": "data-reference",
    "query-query": "reference-reference",
}


def _build_roots(weight: str, use_variant: bool, roots: RootConfig = ROOT_CONFIG) -> Tuple[Optional[Path], Optional[Path]]:
    if use_variant:
        return roots.variant_db_base / f"_{weight}", roots.variant_reference_base / f"_{weight}"
    return roots.raw_db_roots.get(weight), roots.raw_reference_roots.get(weight)
