"""
Extract embeddings for reference images generated via rotations/crops.

This script mirrors Test_Embedding but reads images directly from the reference
directories (e.g. /opt/references/references_<dataset>/R...) and stores outputs under
/exports/dinov3_reference_embeds/<dataset>/<weight_key>/....
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from imatch.extracting import global_embedding, patch_embedding, patch2grid
from imatch.preprocess import build_transform
from imatch.loading import (
    weights_path,
    load_image,
    REFERENCE_ROOT,
    REFERENCE_PREFIX,
    REFERENCE_DATASET_PREFIX,
    DATASET_KEY as DEFAULT_DATASET_KEY,
    REFERENCE_EMBED_ROOT as BASE_REFERENCE_EMBED_ROOT,
    dataset_reference_embed_root,
)
from imatch.postprocess import format_variant_label, process_patch_tokens
from imatch.pretrained import pretrained_model
from imatch.utils import progress_bar, token_preview



ACTIVE_DATASET_KEY = os.getenv("DATASET_KEY", DEFAULT_DATASET_KEY)

# dataset relocation
reloc_prefix = ""

def _default_reference_dirs() -> Sequence[Path]:
    base_dir = REFERENCE_ROOT 
    pattern = "*"
    if not base_dir.exists():
        return ()
    return tuple(sorted(p for p in base_dir.glob(pattern) if p.is_dir()))


REFERENCE_DIRS: Sequence[Path] = _default_reference_dirs()

VAR_WEIGHT_KEYS: Sequence[str] = ("vitb16", "vits16+")
VAR_TARGET_RES = 1024
VARIANT = "raw"
VARIANT_PARAMS: Dict[str, object] = {}

REFERENCE_EMBED_ROOT = dataset_reference_embed_root(ACTIVE_DATASET_KEY, BASE_REFERENCE_EMBED_ROOT)
REPO_DIR = Path("/workspace/dinov3")

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TOKEN_OUTPUT_KEYS = ("global", "patch", "grid")


@dataclass(frozen=True)
class ReferenceInfo:
    scene: str
    altitude: int
    index: int
    rotation: int
    tag: str
    identifier: str
    source: Path
    reference_dir: Path


@dataclass(frozen=True)
class ReferenceEmbeddingResult:
    info: ReferenceInfo
    weight_key: str
    target_res: int
    variant: str
    embedding_cfg: str
    global_path: Optional[Path]
    patch_path: Optional[Path]
    grid_path: Optional[Path]
    rotation: int


def _normalize_output_plan(
    plan: Optional[Dict[str, Dict[str, bool]]],
) -> Dict[str, Dict[str, bool]]:
    if plan is None:
        return {key: {"npy": True, "json": True} for key in TOKEN_OUTPUT_KEYS}
    normalized = {key: {"npy": False, "json": False} for key in TOKEN_OUTPUT_KEYS}
    for key in TOKEN_OUTPUT_KEYS:
        entry = plan.get(key) if isinstance(plan, dict) else None
        if isinstance(entry, dict):
            normalized[key]["npy"] = bool(entry.get("npy"))
            normalized[key]["json"] = bool(entry.get("json"))
    if not any(normalized[key]["npy"] or normalized[key]["json"] for key in TOKEN_OUTPUT_KEYS):
        return {key: {"npy": True, "json": True} for key in TOKEN_OUTPUT_KEYS}
    return normalized


def _should_emit(entry: Dict[str, bool]) -> bool:
    return bool(entry.get("npy") or entry.get("json"))


def _build_output_dirs(
    weight_key: str,
    altitude: int,
    rotation: int,
    root: Path = REFERENCE_EMBED_ROOT,
    dataset_key: Optional[str] = None,
) -> Dict[str, Path]:
    rotation_dir = f"{int(rotation):03d}"
    dataset_root = dataset_reference_embed_root(dataset_key, root)
    altitude_dir = dataset_root / f"{reloc_prefix}{weight_key}" / f"{reloc_prefix}{int(altitude)}" / f"{reloc_prefix}{rotation_dir}"
    return {
        "altitude": altitude_dir,
        "global": altitude_dir / f"{reloc_prefix}GlobalToken",
        "patch": altitude_dir / f"{reloc_prefix}PatchToken",
        "grid": altitude_dir / f"{reloc_prefix}PatchGrid",
        "denseft": altitude_dir / f"{reloc_prefix}DenseFT",
    }


def _parse_reference_filename(path: Path) -> ReferenceInfo:
    stem = path.stem  # e.g. 250912150549_400_0001_rot045_crop50
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"\033[91mUnexpected reference filename format: {path.name}\033[0m")

    scene = parts[0]
    altitude = int(parts[1])
    index = int(parts[2])
    tag = "_".join(parts[3:])
    rotation = 0
    for piece in parts[3:]:
        if piece.startswith("rot") and len(piece) >= 4:
            try:
                rotation = int(piece[3:])
            except ValueError:
                rotation = 0
            break
    identifier = f"{altitude}_{index:04d}_{tag}"
    return ReferenceInfo(
        scene=scene,
        altitude=altitude,
        index=index,
        rotation=rotation,
        tag=tag,
        identifier=identifier,
        source=path,
        reference_dir=path.parent,
    )


def _file_entry(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    stat = path.stat()
    return {"path": path.name, "size_bytes": stat.st_size}


def _write_meta(meta_path: Path, payload: Dict[str, object]) -> None:
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _gather_gpu_stats(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _resolve_patch_size(model: torch.nn.Module) -> Tuple[int, int]:
    patch = None

    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "patch_size"):
        patch = patch_embed.patch_size

    if patch is None and hasattr(model, "patch_size"):
        patch = model.patch_size

    if patch is None and hasattr(model, "stem"):
        stem = getattr(model.stem, "0", model.stem[0] if len(model.stem) > 0 else None)
        if stem is not None and hasattr(stem, "kernel_size"):
            patch = stem.kernel_size

    if patch is None and hasattr(model, "downsample_layers"):
        first = model.downsample_layers[0] if len(model.downsample_layers) > 0 else None
        if first is not None:
            conv = getattr(first, "0", first[0] if len(first) > 0 else None)
            if conv is not None and hasattr(conv, "kernel_size"):
                patch = conv.kernel_size

    if patch is None:
        raise AttributeError("\033[91mUnable to infer patch size from model.\033[0m")

    if isinstance(patch, torch.Size):
        patch = tuple(int(p) for p in patch)
    elif isinstance(patch, (list, tuple)):
        patch = tuple(int(p) for p in patch)
    elif isinstance(patch, int):
        patch = (patch, patch)
    else:
        patch = tuple(int(p) for p in patch)  # type: ignore[arg-type]

    if len(patch) == 1:
        patch = (patch[0], patch[0])
    elif len(patch) < 2:
        raise ValueError(f"\033[91mResolved patch size invalid: {patch}\033[0m")

    return patch[0], patch[1]


def process_reference_image(
    model: torch.nn.Module,
    device: torch.device,
    hub_entry: str,
    dataset_type: str,
    weight_key: str,
    info: ReferenceInfo,
    target_res: int,
    embedding_cfg: Optional[str],
    variant: str,
    variant_params: Dict[str, object],
    output_plan: Optional[Dict[str, Dict[str, bool]]] = None,
    reference_embed_root: Path = REFERENCE_EMBED_ROOT,
    variant_label: Optional[str] = None,
    dataset_key: Optional[str] = None,
) -> ReferenceEmbeddingResult:
    resolved_embedding_cfg = embedding_cfg or f"res{target_res}"

    plan = _normalize_output_plan(output_plan)
    global_plan = plan["global"]
    patch_plan = plan["patch"]
    grid_plan = plan["grid"]
    variant_key = str(variant).strip().lower() or "raw"
    base_variant_label, normalized_variant_params = format_variant_label(variant_key, variant_params)
    resolved_variant_label = variant_label or base_variant_label
    resolved_variant_params = dict(normalized_variant_params)

    emit_global = _should_emit(global_plan)
    emit_patch = _should_emit(patch_plan)
    emit_grid = _should_emit(grid_plan)
    need_patch = emit_patch or emit_grid
    if not (emit_global or need_patch):
        raise ValueError("\033[91m[Error] Reference job must enable at least one token output.\033[0m")

    resolved_dataset_key = dataset_key or ACTIVE_DATASET_KEY
    dirs = _build_output_dirs(
        weight_key,
        info.altitude,
        info.rotation,
        reference_embed_root,
        resolved_dataset_key,
    )
    altitude_dir = dirs["altitude"]
    global_dir = dirs["global"]
    patch_dir = dirs["patch"]
    grid_dir = dirs["grid"]

    global_base = f"ReferenceGlobal_{resolved_embedding_cfg}_{resolved_variant_label}_{hub_entry}_{dataset_type}_{info.identifier}"
    patch_base = f"ReferencePatchToken_{resolved_embedding_cfg}_{resolved_variant_label}_{hub_entry}_{dataset_type}_{info.identifier}"
    grid_base = f"ReferencePatchGrid_{resolved_embedding_cfg}_{resolved_variant_label}_{hub_entry}_{dataset_type}_{info.identifier}"

    npy_path = global_dir / f"{global_base}.npy"
    patch_path = patch_dir / f"{patch_base}.npy"
    grid_path = grid_dir / f"{grid_base}.npy"
    global_meta_path = global_dir / f"{global_base}_meta.json"
    patch_meta_path = patch_dir / f"{patch_base}_meta.json"
    grid_meta_path = grid_dir / f"{grid_base}_meta.json"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # print(
    #     "\n",
    #     "================= Debug: Embedding (Reference) =================\n",
    #     "INPUT: \n",
    #     f"[Reference] REPO_DIR: \033[33m{REPO_DIR}\033[0m\n", # e.g. /workspace/dinov3
    #     f"[Reference] IMAGE_PATH: \033[33m{info.source}\033[0m\n",
    #     f"[Reference] HUB_ENTRY: \033[33m{hub_entry}\033[0m\n", # e.g. dinov3_vits16
    #     f"[Reference] weight_path: \033[33m{weight_key}\033[0m\n", # e.g. /opt/weights/03_ViT_SAT-493M/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
    #     f"[Reference] device: \033[33m{device}\033[0m\n", # e.g. "cuda"
    #     "OUTPUT: \n",
    #     f"[config] embedding_cfg: \033[33m{resolved_embedding_cfg}\033[0m\n",
    #     f"[config] variant: \033[33m{variant_key}\033[0m (label=\033[33m{resolved_variant_label}\033[0m)\n",
    #     f"[config] altitude/index: \033[33m{info.altitude}/{info.rotation}\033[0m\n",
    #     f"Reference Global embedding DINOv3 numpy array -> \033[34m{npy_path}\033[0m\n",
    #     f"Reference Patch token numpy array             -> \033[34m{patch_path}\033[0m\n",
    #     f"Reference Patch grid numpy array              -> \033[34m{grid_path}\033[0m\n",
    #     "================= Debug: Embedding (Reference) =================\n",
    #     )

    img_tensor = progress_bar(load_image, info.source.as_posix())

    patch_h, patch_w = _resolve_patch_size(model)
    patch_multiple = max(1, math.floor(target_res / patch_h))

    transform = progress_bar(
        build_transform,
        patch_size=patch_h,
        patch_multiple=patch_multiple,
        interpolation="bicubic",
        normalize=dataset_type,
    )
    input_tensor = progress_bar(transform, img_tensor).unsqueeze(0)

    timings: Dict[str, Optional[float]] = {
        "global_forward": None,
        "patch_forward": None,
        "postprocess": None,
        "index_build": None,
        "reference": None,
        "pipeline_total": None,
    }

    pipeline_start = time.perf_counter()
    global_tokens: Optional[torch.Tensor] = None
    patch_tokens: Optional[torch.Tensor] = None

    with torch.inference_mode():
        if emit_global:
            g_start = time.perf_counter()
            global_tokens = progress_bar(global_embedding, model, input_tensor, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["global_forward"] = (time.perf_counter() - g_start) * 1000.0

        if need_patch:
            p_start = time.perf_counter()
            patch_tokens = progress_bar(patch_embedding, model, input_tensor, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["patch_forward"] = (time.perf_counter() - p_start) * 1000.0

    patch_grid = None
    grid_array = None
    patch_numpy = None
    patch_post_info = None

    if global_tokens is not None:
        global_tokens = global_tokens.detach().cpu()
    if patch_tokens is not None:
        patch_tokens = patch_tokens.detach().cpu()
        post_start = time.perf_counter()
        processed_tokens, base_post_info = process_patch_tokens(
            patch_tokens,
            variant_key,
            resolved_variant_params,
        )
        grid_from_info = None
        if base_post_info is not None and "grid" in base_post_info:
            grid_from_info = base_post_info.pop("grid")
        if base_post_info is not None and "grid_shape" in base_post_info:
            base_post_info.pop("grid_shape")

        patch_post_info = base_post_info or {}
        if "keep_ratio" not in patch_post_info:
            patch_post_info["keep_ratio"] = base_post_info.get("keep_ratio", 1.0) if base_post_info else 1.0
        patch_post_info["params"] = {
            "patch_variant": variant_key,
            "patch_params": dict(resolved_variant_params),
        }
        timings["postprocess"] = (time.perf_counter() - post_start) * 1000.0
        patch_tokens = processed_tokens
        if patch_plan["npy"]:
            patch_numpy = patch_tokens.numpy()
        if emit_grid:
            try:
                if grid_from_info is not None:
                    patch_grid = (
                        grid_from_info.detach().cpu()
                        if isinstance(grid_from_info, torch.Tensor)
                        else torch.as_tensor(grid_from_info)
                    )
                else:
                    patch_grid = patch2grid(patch_tokens)
            except ValueError as err:
                print(f"\033[91m[WARN] Reference patch grid reshape failed: {err}\033[0m")
            if patch_grid is not None:
                grid_array = (
                    patch_grid.detach().cpu().numpy()
                    if isinstance(patch_grid, torch.Tensor)
                    else np.asarray(patch_grid)
                )
    elif need_patch:
        print("\033[91m[WARN] Patch tokens could not be extracted.\033[0m")

    global_arr = global_tokens.numpy() if global_tokens is not None else None
    global_saved = False
    if global_plan["npy"] and global_arr is not None:
        global_dir.mkdir(parents=True, exist_ok=True)
        progress_bar(np.save, npy_path, global_arr)
        print(f"\033[32m[saved] Reference image Global embedding DINOv3 numpy array -> {npy_path}\033[0m")
        global_saved = True
    elif global_plan["npy"]:
        print("\033[91m[WARN] Global npy requested but tokens unavailable.\033[0m")

    if patch_plan["npy"] and patch_numpy is not None:
        patch_dir.mkdir(parents=True, exist_ok=True)
        progress_bar(np.save, patch_path, patch_numpy)
        print(f"\033[32m[saved] Reference image Patch embedding DINOv3 numpy array -> {patch_path}\033[0m")
        if patch_post_info is not None and "scores" in patch_post_info:
            scores = np.asarray(patch_post_info["scores"])
            score_path = patch_dir / f"{patch_base}_scores.npy"
            progress_bar(np.save, score_path, scores)
            print(f"\033[32m[saved] Reference Patch token score numpy array -> {score_path}\033[0m")

    elif patch_plan["npy"] and patch_numpy is None:
        print("\033[91m[WARN] Patch npy requested but tokens unavailable.\033[0m")

    grid_saved = False
    if grid_plan["npy"]:
        if grid_array is not None:
            grid_dir.mkdir(parents=True, exist_ok=True)
            np.save(grid_path, grid_array)
            print(f"\033[32m[saved] Reference image Patch Grid DINOv3 numpy array -> {grid_path}\033[0m")
            grid_saved = True
            
            if (
                patch_post_info is not None
                and "scores" in patch_post_info
                and patch_grid is not None
            ):
                scores = np.asarray(patch_post_info["scores"])
                H, W = patch_grid.shape[0], patch_grid.shape[1]
                if scores.size == H * W:
                    score_grid = scores.reshape(H, W)
                    score_grid_path = grid_dir / f"{grid_base}_scores.npy"
                    np.save(score_grid_path, score_grid)
                    print(f"\033[32m[saved] Reference Patch grid score numpy array -> {score_grid_path}\033[0m")
                else:
                    print("\033[91m[WARN] Reference score size does not match grid; skip score grid.\033[0m")

        else:
            print("\033[91m[WARN] Patch grid npy requested but grid unavailable.\033[0m")

    timings["pipeline_total"] = (time.perf_counter() - pipeline_start) * 1000.0
    gpu_peak = _gather_gpu_stats(device)
    if gpu_peak is not None:
        print(
            f"[GPU] [Reference] weight={weight_key} altitude={info.altitude} "
            f"index={info.index} rotation={info.rotation} peak_mem={gpu_peak:.2f} MB"
        )
    else:
        print(
            f"[GPU] [Reference] weight={weight_key} altitude={info.altitude} "
            f"index={info.index} rotation={info.rotation} peak_mem=N/A"
        )

    def _sum_sizes(entries: Dict[str, Optional[Dict[str, object]]]) -> Optional[int]:
        total = 0
        has = False
        for entry in entries.values():
            if entry and "size_bytes" in entry:
                total += int(entry["size_bytes"])
                has = True
        return total if has else None

    global_files = {
        "vector": _file_entry(npy_path) if global_saved else None,
        "patch_tokens": None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }
    patch_files = {
        "vector": _file_entry(patch_path) if patch_plan["npy"] and patch_numpy is not None else None,
        "patch_tokens": None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }
    grid_files = {
        "vector": _file_entry(grid_path) if grid_saved else None,
        "patch_tokens": None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }

    default_variant_params = {
        "patch_variant": variant_key,
        "patch_params": dict(resolved_variant_params),
    }

    used_variant_params = dict(default_variant_params)
    if patch_post_info and "params" in patch_post_info:
        params = patch_post_info["params"]
        if isinstance(params, dict):
            used_variant_params.update(params)

    reference_config = {
        "embedding_cfg": resolved_embedding_cfg,
        "variant": variant_key,
        "experiment_variant": resolved_variant_label,
        "variant_label": resolved_variant_label,
        "variant_params": used_variant_params,
        "weight_id": hub_entry,
        "dataset_type": dataset_type,
        "altitude": info.altitude,
        "index": info.index,
        "prefix": info.identifier,
        "target_res": target_res,
        "reference": {
            "source_file": info.source.as_posix(),
            "tag": info.tag,
            "reference_dir": info.reference_dir.as_posix(),
        },
    }

    if global_plan["json"] and global_arr is not None:
        global_dir.mkdir(parents=True, exist_ok=True)
        global_meta = {
            "run_id": global_base,
            "token_type": "GlobalToken",
            "config": reference_config,
            "files": global_files,
            "metrics": {
                "token_count": 1,
                "embedding_dim": int(global_arr.shape[0]),
                "matching_count": None,
                "mutual_knn_tokens": None,
                "keep_ratio": None,
                "recall@1": None,
                "recall@5": None,
                "recall@10": None,
                "mAP": None,
                "top1_precision": None,
            },
            "timing_ms": dict(timings),
            "resources": {
                "gpu_peak_mem_mb": gpu_peak,
                "embedding_storage_bytes": _sum_sizes(global_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(global_meta_path, global_meta)
    elif global_plan["json"]:
        print("\033[91m[WARN] Global meta requested but tokens unavailable.\033[0m")

    if patch_plan["json"] and patch_tokens is not None:
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_meta = {
            "run_id": patch_base,
            "token_type": "PatchToken",
            "config": reference_config,
            "files": patch_files,
            "metrics": {
                "token_count": int(patch_tokens.shape[0]),
                "embedding_dim": int(patch_tokens.shape[1]) if patch_tokens.ndim == 2 else None,
                "matching_count": patch_post_info.get("kept_tokens") if patch_post_info else int(patch_tokens.shape[0]),
                "mutual_knn_tokens": None,
                "keep_ratio": patch_post_info.get("keep_ratio") if patch_post_info else 1.0,
                "recall@1": None,
                "recall@5": None,
                "recall@10": None,
                "mAP": None,
                "top1_precision": None,
            },
            "timing_ms": dict(timings),
            "resources": {
                "gpu_peak_mem_mb": gpu_peak,
                "embedding_storage_bytes": _sum_sizes(patch_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(patch_meta_path, patch_meta)
    elif patch_plan["json"]:
        print("\033[91m[WARN] Patch meta requested but tokens unavailable.\033[0m")

    if grid_plan["json"] and grid_array is not None:
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_h = int(grid_array.shape[0]) if grid_array.ndim >= 1 else None
        grid_w = int(grid_array.shape[1]) if grid_array.ndim >= 2 else None
        grid_dim = int(grid_array.shape[2]) if grid_array.ndim >= 3 else None
        grid_meta = {
            "run_id": grid_base,
            "token_type": "PatchGrid",
            "config": reference_config,
            "files": grid_files,
            "metrics": {
                "token_count": int(grid_h * grid_w) if grid_h is not None and grid_w is not None else None,
                "grid_shape": [grid_h, grid_w] if grid_h is not None and grid_w is not None else None,
                "embedding_dim": grid_dim,
                "matching_count": None,
                "mutual_knn_tokens": None,
                "keep_ratio": patch_post_info.get("keep_ratio") if patch_post_info else None,
                "recall@1": None,
                "recall@5": None,
                "recall@10": None,
                "mAP": None,
                "top1_precision": None,
            },
            "timing_ms": dict(timings),
            "resources": {
                "gpu_peak_mem_mb": gpu_peak,
                "embedding_storage_bytes": _sum_sizes(grid_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(grid_meta_path, grid_meta)
    elif grid_plan["json"]:
        print("\033[91m[WARN] Patch grid meta requested but grid unavailable.\033[0m")

    return ReferenceEmbeddingResult(
        info=info,
        weight_key=weight_key,
        target_res=target_res,
        variant=variant_key,
        embedding_cfg=resolved_embedding_cfg,
        global_path=npy_path if global_saved else None,
        patch_path=patch_path if patch_plan["npy"] and patch_numpy is not None else None,
        grid_path=grid_path if grid_saved else None,
        rotation=info.rotation,
    )


def iter_reference_files(
    directories: Sequence[Path] | None = None,
    patterns: Sequence[str] | None = None,
    recursive: bool = False,
) -> Iterable[Path]:
    target_dirs = list(directories) if directories else list(REFERENCE_DIRS)
    if not target_dirs:
        return

    resolved_patterns: List[str] = []
    raw_patterns = list(patterns) if patterns else [f"*{suffix}" for suffix in SUPPORTED_SUFFIXES]
    for pattern in raw_patterns:
        pat = pattern.strip()
        if not pat:
            continue
        if pat.startswith("."):
            pat = f"*{pat}"
        if not any(ch in pat for ch in "*?[]"):
            pat = f"*{pat}"
        resolved_patterns.append(pat)

    for qdir in target_dirs:
        qdir = Path(qdir)
        if not qdir.exists():
            print(f"\033[91m[WARN] Reference directory missing, skipping: {qdir}\033[0m")
            continue
        for pattern in resolved_patterns:
            iterator = qdir.rglob(pattern) if recursive else qdir.glob(pattern)
            for path in sorted(iterator):
                if path.is_file():
                    yield path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_plan = _normalize_output_plan(None)
    total = 0
    for weight_key in VAR_WEIGHT_KEYS:
        hub_entry, weight_path, dataset_type = weights_path(weight_key)
        print(f"[INFO] Loading model {hub_entry} ({weight_key}) on {device}")
        model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
        model.eval()

        for reference_path in iter_reference_files():
            info = _parse_reference_filename(reference_path)
            process_reference_image(
                model=model,
                device=device,
                hub_entry=hub_entry,
                dataset_type=dataset_type,
                weight_key=weight_key,
                info=info,
                target_res=VAR_TARGET_RES,
                embedding_cfg=None,
                variant=VARIANT,
                variant_params=VARIANT_PARAMS,
                output_plan=default_plan,
                dataset_key=ACTIVE_DATASET_KEY,
            )
            total += 1


if __name__ == "__main__":
    main()
