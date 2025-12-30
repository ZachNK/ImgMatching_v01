"""
Generate dense feature visualisations from the exported patch grids.

The logic is wrapped in generate_dense_feature so it can be reused in batch
runs while keeping the original single-run behaviour.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from imatch.postprocess import format_variant_label
from imatch.loading import (
    EMBED_ROOT,
    dataset_embed_root,
    weights_path,
    file_prefix,
    sanitize_group_token,
    normalize_group_value,
)

# dataset relocation
reloc_prefix = ""

varAltitude = 450
varIndex = 1
varWeight = "vit7b16"


def _build_context(
    altitude: int | str,
    index: int,
    weight: str,
    variant: str,
    embedding_cfg: str,
    variant_params: dict[str, object] | None = None,
    variant_label: str | None = None,
    dataset_key: str | None = None,
) -> dict[str, Path | str]:
    """Prepare the shared paths used throughout the dense feature pipeline."""
    hub_entry, _, dataset_type = weights_path(weight) # e.g. dinov3_vitl16
    label_str = normalize_group_value(altitude)
    label_token = sanitize_group_token(altitude)
    index_str = f"{int(index):04d}"
    prefix = file_prefix(label_str, index) # e.g. 200_0150
    base_variant_label, _ = format_variant_label(variant, variant_params)
    resolved_variant_label = variant_label or base_variant_label

    grid_name = f"PatchGrid_{embedding_cfg}_{resolved_variant_label}_{hub_entry}_{dataset_type}_{label_token}_{index_str}"
    dense_name = f"DenseFT_{embedding_cfg}_{resolved_variant_label}_{hub_entry}_{dataset_type}_{label_token}_{index_str}"

    altitude_dir = dataset_embed_root(dataset_key) / f"{reloc_prefix}{weight}" / f"{reloc_prefix}{label_token}"
    grid_dir = altitude_dir / f"{reloc_prefix}PatchGrid"
    dense_dir = altitude_dir / f"{reloc_prefix}DenseFT"

    return {
        "hub_entry": hub_entry,
        "prefix": prefix,
        "grid_path": grid_dir / f"{grid_name}.npy",
        "dense_path": dense_dir / f"{dense_name}.png",
        "dense_dir": dense_dir,
        "variant_label": variant_label,
    }


def generate_dense_feature(
    altitude: int | str,
    index: int,
    weight: str,
    target_res: int = 1024,
    variant: str = "raw",
    embedding_cfg: str | None = None,
    variant_params: dict[str, object] | None = None,
    variant_label: str | None = None,
    pca_basis_path: str | None = None,
    dataset_key: str | None = None,
) -> None:
    """
    Load the patch grid exported by Test_global_embedding and save a PNG.

    variant_params are parsed to mirror the filenames produced during embedding.
    """
    resolved_embedding_cfg = embedding_cfg or f"res{int(target_res)}"
    ctx = _build_context(
        altitude,
        index,
        weight,
        variant,
        resolved_embedding_cfg,
        variant_params,
        variant_label,
        dataset_key,
    )
    grid_path = ctx["grid_path"]
    dense_path = ctx["dense_path"]
    dense_dir = ctx["dense_dir"]

    if not grid_path.exists():
        raise FileNotFoundError(
            f"\033[91mPatch grid not found for group={altitude}, index={index}, weight={weight}: {grid_path}\033[0m"
        )

    Path(dense_dir).mkdir(parents=True, exist_ok=True)

    grid = torch.from_numpy(np.load(grid_path))  # (H, W, C)
    if grid.ndim != 3:
        raise ValueError(f"\033[91m[Error] Patch grid must be 3D, got {grid.shape}\033[0m")
    H, W, C = grid.shape
    score_grid_path = grid_path.with_name(grid_path.stem + "_scores.npy")
    score_weights = None
    if score_grid_path.exists():
        scores = np.load(score_grid_path)  # (H, W)
        scores = torch.from_numpy(scores.astype("float32"))
        # 0~1濡??뺢퇋??
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        score_weights = scores.reshape(-1, 1)  # (H*W, 1)

    flat = grid.reshape(-1, grid.shape[-1])  # (H*W, C)
    feat = flat - flat.mean(dim=0, keepdim=True)

    if score_weights is not None:
        feat_weighted = feat * score_weights  # 以묒슂 ?⑥튂???ш쾶, ?섎㉧吏???묎쾶
    else:
        feat_weighted = feat

    if C > 3:
        _, _, v = torch.pca_lowrank(feat_weighted, q=3)   # v: (C, 3)
        proj = feat_weighted @ v[:, :3]                   # (H*W, 3)
    else:
        proj = feat_weighted
        if C < 3:
            pad = torch.zeros((proj.shape[0], 3 - C), dtype=proj.dtype, device=proj.device)
            proj = torch.cat([proj, pad], dim=1)

    rgb = proj.reshape(grid.shape[0], grid.shape[1], 3).numpy()
    rgb -= rgb.min()
    rgb /= (rgb.max() + 1e-6)

    rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    rgb_up = F.interpolate(rgb, size=(1024, 1024), mode="bilinear", align_corners=False)
    rgb_up = rgb_up.squeeze(0).permute(1, 2, 0).numpy()

    img = Image.fromarray((rgb_up * 255).astype("uint8"))
    img.save(dense_path)
    print(f"\033[32m[saved] Dense feature image -> {dense_path}\033[0m")


def main() -> None:
    generate_dense_feature(
        altitude=varAltitude,
        index=varIndex,
        weight=varWeight,
    )


if __name__ == "__main__":
    main()
