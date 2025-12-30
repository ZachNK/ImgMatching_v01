"""
Patch token post-processing helpers.

The project needs to experiment with multiple filtering strategies (raw tokens,
subsampling, etc.). This module centralises the logic so
Test_Embedding and other pipelines can invoke the same API regardless of the
selected variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch

from imatch.extracting import patch2grid


ProcessorFn = Callable[[torch.Tensor, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, Any]]]


@dataclass(frozen=True)
class VariantProcessor:
    """Container describing a patch-token post-processing strategy."""

    name: str
    handler: ProcessorFn
    defaults: Dict[str, Any]
    description: str


def _ensure_tensor(tokens: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tokens):
        raise TypeError(f"tokens must be torch.Tensor, got {type(tokens)}")
    if tokens.ndim != 2:
        raise ValueError(f"patch tokens expected shape (N, C); received {tokens.shape}")
    return tokens


def _variant_raw(tokens: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tokens = _ensure_tensor(tokens)
    info = {
        "kept_tokens": int(tokens.shape[0]),
        "keep_ratio": 1.0,
        "params": params,
    }
    return tokens.contiguous(), info


def _variant_subsample(tokens: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tokens = _ensure_tensor(tokens)
    stride = int(params.get("stride", 2))
    if stride <= 0:
        raise ValueError(f"stride must be positive; received {stride}")

    grid = patch2grid(tokens)
    if grid.ndim != 3:
        raise ValueError(f"expected reshaped grid to be 3D, got {grid.shape}")

    subsampled = grid[::stride, ::stride, :]
    flattened = subsampled.reshape(-1, subsampled.shape[-1])
    keep_ratio = float(flattened.shape[0] / max(tokens.shape[0], 1))

    info = {
        "kept_tokens": int(flattened.shape[0]),
        "keep_ratio": keep_ratio,
        "params": {"stride": stride},
        "grid_shape": tuple(subsampled.shape),
        "grid": subsampled.contiguous(),
    }
    return flattened.contiguous(), info

def _normalize_variant_params_for_variant(
    variant: str,
    overrides: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Validate the combination of variant and variant_params and return a
    handler-ready parameter dictionary.

    Rules:
      - raw:
          * norm_threshold / stride must all be None
      - subsample:
          * stride is required
          * norm_threshold is not allowed
    """
    overrides = dict(overrides or {})

    allowed_keys = {"norm_threshold", "stride"}
    unknown = [k for k in overrides.keys() if k not in allowed_keys]
    if unknown:
        raise ValueError(
            f"[91m[Error] Unsupported keys in variant_params for variant '{variant}': "
            f"{unknown}[0m"
        )

    norm_threshold = overrides.get("norm_threshold")
    stride = overrides.get("stride")

    non_null = {
        key: value
        for key, value in (
            ("norm_threshold", norm_threshold),
            ("stride", stride),
        )
        if value is not None
    }

    if variant == "raw":
        if non_null:
            raise ValueError(
                "[91m[Error] variant='raw' must not specify any non-null "
                f"variant_params; got {non_null}[0m"
            )
        return {}

    if variant == "subsample":
        if stride is None:
            raise ValueError(
                "[91m[Error] variant='subsample' requires "
                "variant_params['stride'] to be set.[0m"
            )
        extra = {k: v for k, v in non_null.items() if k != "stride"}
        if extra:
            raise ValueError(
                "[91m[Error] variant='subsample' does not accept keys "
                f"{list(extra.keys())} in variant_params.[0m"
            )
        return {"stride": int(stride)}

    # Should never reach here (resolve_variant filters variants)
    raise ValueError(
        f"[91m[Error] Unknown variant '{variant}' in "
        "_normalize_variant_params_for_variant.[0m"
    )


def format_variant_label(
    variant: str,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a human/file-friendly variant label and normalised parameters.

    Returns:
        variant_label: string appended to filenames (e.g. sub2)
        normalized_params: dict validated via _normalize_variant_params_for_variant
    """
    variant_key = str(variant).strip().lower() or "raw"
    normalized_params = _normalize_variant_params_for_variant(variant_key, overrides)

    if variant_key == "subsample":
        label = f"sub{int(normalized_params.get('stride', 0))}"
    else:
        label = variant_key

    return label, normalized_params



VARIANT_REGISTRY: Dict[str, VariantProcessor] = {
    "raw": VariantProcessor(
        name="raw",
        handler=_variant_raw,
        defaults={},
        description="No additional filtering. Save all patch tokens.",
    ),
    "subsample": VariantProcessor(
        name="subsample",
        handler=_variant_subsample,
        defaults={"stride": 2},
        description="Downsample the patch grid by a strided subsampling.",
    ),
}

PROCESSING_VARIANTS = {"raw", "subsample"}


def available_variants() -> Dict[str, VariantProcessor]:
    """Return the registry so other modules can list supported variants."""
    return VARIANT_REGISTRY


def resolve_variant(variant: str) -> VariantProcessor:
    """
    Resolve a variant label to a VariantProcessor.
    variant:
    - Only support 'raw', 'subsample'
    """
    if variant not in VARIANT_REGISTRY:
        raise ValueError(
            f"\033[91m\t[Error] Unknown patch-token variant '{variant}'. "
            f"Available variants: {list(VARIANT_REGISTRY.keys())}\033[0m"
        )
    return VARIANT_REGISTRY[variant]


def process_patch_tokens(
    tokens: torch.Tensor,
    variant: str,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Process patch tokens according to the specified variant and parameters.
    -----------
    inputs:
        tokens: (N, C) patch tokens
        variant: one of 'raw', 'subsample'
        overrides: optional dict of parameters to override defaults for the variant
    -----------
    returns:
        processed_tokens: (M, C) filtered patch tokens
        info: dict with processing details

        e.g. {'kept_tokens': 200, 'keep_ratio': 0.2, 'params': {'stride': 2}}
    -----------
    """
    processor = resolve_variant(variant)

    # manifest에서 온 variant_params를 검증/정규화
    params = _normalize_variant_params_for_variant(variant, overrides)

    processed_tokens, info = processor.handler(tokens, params)

    if "params" not in info:
        info["params"] = params
    return processed_tokens, info
