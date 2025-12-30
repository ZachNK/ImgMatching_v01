"""
Utility helpers for building runtime patch-token variant configs.

Only two variants are supported:
- raw: no filtering
- subsample: stride-based downsampling of the patch grid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from imatch.postprocess import format_variant_label


@dataclass(frozen=True)
class RuntimeVariant:
    patch_variant: str
    patch_params: Dict[str, object]
    label: str


def _normalize_variant_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in ("", "raw", "none"):
        return "raw"
    if key in ("sub", "subsample"):
        return "subsample"
    return key


def build_runtime_variant(
    base_variant: str,
    subsample_stride: Optional[int] = None,
) -> RuntimeVariant:
    """
    Convert a manifest-specified variant into a normalized runtime variant.

    Args:
        base_variant: "raw" or "sub"/"subsample".
        subsample_stride: stride to use when base_variant is subsample (defaults to 1).
    """
    variant_key = _normalize_variant_name(base_variant)

    if variant_key == "raw":
        label, params = format_variant_label("raw", None)
        return RuntimeVariant(patch_variant="raw", patch_params=params, label=label)

    if variant_key == "subsample":
        stride = 1 if subsample_stride is None else int(subsample_stride)
        if stride <= 0:
            raise ValueError("\033[91m[Error] subsample stride must be positive.\033[0m")
        label, params = format_variant_label("subsample", {"stride": stride})
        return RuntimeVariant(patch_variant="subsample", patch_params=params, label=label)

    raise ValueError(
        f"\033[91m[Error] Unsupported variant '{base_variant}'. Use 'raw' or 'subsample'.\033[0m"
    )


__all__: Tuple[str, ...] = ("RuntimeVariant", "build_runtime_variant")
