# project/Test_Embedding.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
import numpy as np
import torch
import os
from imatch.pretrained import pretrained_model
from imatch.preprocess import build_transform
from imatch.extracting import (
    global_embedding,
    patch_embedding,
    patch2grid
)
from imatch.loading import (
    EMBED_ROOT,
    dataset_embed_root,
    weights_path,
    file_prefix,
    img_path,
    load_image,
    sanitize_group_token,
    normalize_group_value,
)
from imatch.utils import (
    progress_bar,
    token_preview
)
from imatch.postprocess import format_variant_label, process_patch_tokens

# dataset relocation
reloc_prefix = ""

# 
varAltitude = 450
varIndex = 1
varWeight = "vitb16"
varTargetRes = 1024

## 
REPO_DIR = Path("/workspace/dinov3")
TOKEN_OUTPUT_KEYS = ("global", "patch", "grid")

SESSION_STATS: Dict[str, Dict[str, int]] = {}


def _session_stat_entry(weight: str) -> Dict[str, int]:
    entry = SESSION_STATS.get(weight)
    if entry is None:
        entry = {"session_loads": 0, "direct_loads": 0, "reuses": 0, "runs": 0}
        SESSION_STATS[weight] = entry
    return entry


def collect_embedding_session_stats(reset: bool = False) -> Dict[str, Dict[str, int]]:
    snapshot = {weight: dict(stats) for weight, stats in SESSION_STATS.items()}
    if reset:
        SESSION_STATS.clear()
    return snapshot

def _normalize_output_plan(
    plan: Optional[Dict[str, Dict[str, bool]]]
) -> Dict[str, Dict[str, bool]]:
    """
    Normalize the output plan into a full dictionary with all token types and formats.
    """
    
    if plan is None:
        return {key: {"npy": True, "json": True} for key in TOKEN_OUTPUT_KEYS}
    normalized = {key: {"npy": False, "json": False} for key in TOKEN_OUTPUT_KEYS}
    for key in TOKEN_OUTPUT_KEYS:
        entry = plan.get(key) if isinstance(plan, dict) else None
        if isinstance(entry, dict):
            normalized[key]["npy"] = bool(entry.get("npy"))
            normalized[key]["json"] = bool(entry.get("json"))
    return normalized


def _build_context(
    altitude: int | str,
    index: int,
    weight: str,
    dataset_key: str | None,
    target_res: int,
    variant: str,
    embedding_cfg: Optional[str],
    variant_params: Optional[Dict[str, object]],
    variant_label: Optional[str] = None,
) -> Dict[str, object]:
    """Assemble frequently reused values for a single inference run."""
    hub_entry, key, dt_type = weights_path(weight) # e.g. "dinov3_vit7b16", "/opt/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth", "SAT"
    image_spec = img_path(altitude, index, dataset_key=dataset_key)
    prefix = file_prefix(image_spec.label, index) # e.g. "200_0150"

    # Derive embedding configuration when not explicitly provided.
    default_embedding_cfg = f"res{target_res}"
    resolved_embedding_cfg = embedding_cfg or default_embedding_cfg
    variant_key = str(variant).strip().lower() or "raw"
    base_variant_label, normalized_variant_params = format_variant_label(variant_key, variant_params)
    resolved_variant_label = variant_label or base_variant_label

    # Token naming follows: token_type ??embedding_cfg ??variant ??weight_id ??dataset_type ??altitude ??index
    label_display = normalize_group_value(altitude)
    label_token = sanitize_group_token(altitude)
    index_str = f"{int(index):04d}"
    global_base = f"GlobalToken_{resolved_embedding_cfg}_{resolved_variant_label}_{hub_entry}_{dt_type}_{label_token}_{index_str}"
    patch_base = f"PatchToken_{resolved_embedding_cfg}_{resolved_variant_label}_{hub_entry}_{dt_type}_{label_token}_{index_str}"
    patch_grid_base = f"PatchGrid_{resolved_embedding_cfg}_{resolved_variant_label}_{hub_entry}_{dt_type}_{label_token}_{index_str}"

    export_root = dataset_embed_root(image_spec.dataset_key) / f"{reloc_prefix}{weight}"
    altitude_dir = export_root / f"{reloc_prefix}{label_token}"
    global_dir = altitude_dir / f"{reloc_prefix}GlobalToken"
    patch_dir = altitude_dir / f"{reloc_prefix}PatchToken"
    grid_dir = altitude_dir / f"{reloc_prefix}PatchGrid"
    return {
        "hub_entry": hub_entry, # e.g. "dinov3_vit7b16"
        "key_path": key, # e.g. "/opt/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
        "dataset_type": dt_type,
        "image_path": image_spec.path, # resolved absolute path
        "dataset_key": image_spec.dataset_key,
        "label_display": label_display,
        "label_token": label_token,
        "file_name": global_base,
        "patch_name": patch_base,
        "grid_name": patch_grid_base,
        "export_root": export_root,
        "altitude_dir": altitude_dir,
        "global_dir": global_dir,
        "patch_dir": patch_dir,
        "grid_dir": grid_dir,
        "embedding_cfg": resolved_embedding_cfg,
        "variant": variant_key,
        "variant_label": resolved_variant_label,
        "variant_params": dict(normalized_variant_params),
        "index_str": index_str,
        "prefix": prefix,
    }


def _file_entry(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": path.name,
        "size_bytes": stat.st_size,
    }


def _write_meta(meta_path: Path, payload: Dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _gather_gpu_stats(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _resolve_patch_size(model: torch.nn.Module) -> tuple[int, int]:
    """
    DINOv3 ConvNeXt checkpoints do not expose `patch_embed`, so gracefully try
    the common variants and always return a 2D patch size tuple.
    """
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
        raise AttributeError(
            "\033[91mAttributeError: Unable to infer patch size: model lacks patch_embed/patch_size/stem/downsample_layers metadata.\033[0m"
        )

    if isinstance(patch, torch.Size):
        patch = tuple(int(p) for p in patch)
    elif isinstance(patch, (list, tuple)):
        patch = tuple(int(p) for p in patch)
    elif isinstance(patch, int):
        patch = (patch, patch)
    else:
        # Fall back to trying to iterate, otherwise bail.
        try:
            patch = tuple(int(p) for p in patch)  # type: ignore[arg-type]
        except TypeError as err:  # pragma: no cover - defensive
            raise TypeError(f"\033[91mTypeError: Unsupported patch size type = {type(patch)}\033[0m") from err

    if len(patch) == 1:
        patch = (patch[0], patch[0])
    elif len(patch) < 2:
        raise ValueError(f"\033[91mValueError: Resolved patch size has insufficient dimensions = {patch}\033[0m")

    return patch[0], patch[1]


class EmbeddingSession:
    """Cache a model loaded via torch.hub.load so jobs can reuse it."""

    def __init__(self, weight_key: str, device: Optional[torch.device] = None) -> None:
        self.weight_key = weight_key
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hub_entry, weight_path, dataset_type = weights_path(weight_key)
        self.hub_entry = hub_entry
        self.weight_path = weight_path
        self.dataset_type = dataset_type
        self.model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, self.device)
        self.model.eval()
        self.session_id = f"{weight_key}:{id(self):x}"
        stats = _session_stat_entry(weight_key)
        stats["session_loads"] += 1
        print(
            f"[SESSION] [LOAD] weight={weight_key} session={self.session_id} device={self.device} "
            f"hub_entry={hub_entry}"
        )

    def close(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def run_global_embedding(
    altitude: int | str,
    index: int,
    weight: str,
    target_res: int = 1024,
    variant: str = "raw",
    embedding_cfg: Optional[str] = None,
    variant_params: Optional[Dict[str, object]] = None,
    output_plan: Optional[Dict[str, Dict[str, bool]]] = None,
    dataset_key: Optional[str] = None,
    session: Optional["EmbeddingSession"] = None,
    variant_label: Optional[str] = None,
) -> None:
    """Execute the full embedding pipeline for the given parameters."""
    if session is not None and session.weight_key != weight:
        raise ValueError("\033[91m[Error] EmbeddingSession weight mismatch.\033[0m")

    ctx = _build_context(
        altitude,
        index,
        weight,
        dataset_key,
        target_res,
        variant,
        embedding_cfg,
        variant_params,
        variant_label,
    )
    hub_entry = ctx["hub_entry"]
    weight_path = ctx["key_path"]
    dataset_type = session.dataset_type if session is not None else ctx["dataset_type"]
    image_path = ctx["image_path"]
    file_name = ctx["file_name"]
    patch_name = ctx["patch_name"]
    grid_name = ctx["grid_name"]
    global_dir = ctx["global_dir"]
    patch_dir = ctx["patch_dir"]
    grid_dir = ctx["grid_dir"]
    resolved_embedding_cfg = ctx["embedding_cfg"]
    resolved_variant = ctx["variant"]
    resolved_variant_label = ctx["variant_label"]
    resolved_variant_params = dict(ctx["variant_params"])
    label_display = ctx["label_display"]
    index_str = ctx["index_str"]
    prefix = ctx["prefix"]

    stats = _session_stat_entry(weight)
    stats["runs"] += 1
    if session is None:
        stats["direct_loads"] += 1
        print(
            f"[SESSION] [DIRECT LOAD] weight={weight} altitude={label_display} index={index} "
            f"hub_entry={hub_entry}"
        )
    else:
        stats["reuses"] += 1
        print(
            f"[SESSION] [REUSE] weight={weight} session={session.session_id} altitude={label_display} index={index}"
        )

    plan = _normalize_output_plan(output_plan)
    global_plan = plan["global"]
    patch_plan = plan["patch"]
    grid_plan = plan["grid"]

    emit_global = bool(global_plan["npy"] or global_plan["json"])
    emit_patch = bool(patch_plan["npy"] or patch_plan["json"])
    emit_grid = bool(grid_plan["npy"] or grid_plan["json"])
    need_global = emit_global
    need_patch = emit_patch or emit_grid

    npy_path = global_dir / f"{file_name}.npy"
    patch_path = patch_dir / f"{patch_name}.npy"
    grid_path = grid_dir / f"{grid_name}.npy"
    global_meta_path = global_dir / f"{file_name}_meta.json"
    patch_meta_path = patch_dir / f"{patch_name}_meta.json"
    grid_meta_path = grid_dir / f"{grid_name}_meta.json"

    device = session.device if session is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if session is None:
        model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
    else:
        model = session.model

    img_tensor = progress_bar(load_image, image_path.as_posix())
    print(f"[Global Embedding 1] Input image shape: {img_tensor.shape}")

    patch_h, patch_w = _resolve_patch_size(model)
    patch_multiple = max(1, math.floor(target_res / patch_h))
    print(
        f"[Global Embedding 2] Model patch size: {(patch_h, patch_w)}\n"
        f"[Global Embedding 3] Image resized to: {patch_multiple * patch_h}x{patch_multiple * patch_w}\n"
    )

    transform = progress_bar(
        build_transform,
        patch_size=patch_h,
        patch_multiple=patch_multiple,
        interpolation="bicubic",
        normalize=dataset_type,
    )

    input_tensor = progress_bar(transform, img_tensor).unsqueeze(0)
    print(f"[Global Embedding 5] Input tensor shape: {input_tensor.shape}")

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
        if need_global:
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
    patch_numpy = None
    patch_post_info = None
    base_post_info = None
    if global_tokens is not None:
        global_tokens = global_tokens.detach().cpu()
    if patch_tokens is not None:
        patch_tokens = patch_tokens.detach().cpu()
        post_start = time.perf_counter()
        processed_tokens, base_post_info = process_patch_tokens(
            patch_tokens,
            resolved_variant,
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
            "patch_variant": resolved_variant,
            "patch_params": dict(resolved_variant_params),
        }
        timings["postprocess"] = (time.perf_counter() - post_start) * 1000.0
        patch_tokens = processed_tokens

        if emit_patch:
            patch_numpy = patch_tokens.numpy()
        if emit_grid:
            try:
                if grid_from_info is not None:
                    patch_grid = grid_from_info.detach().cpu() if isinstance(grid_from_info, torch.Tensor) else torch.as_tensor(grid_from_info)
                else:
                    patch_grid = patch2grid(patch_tokens)
            except ValueError as err:
                print(f"\033[91m[WARN 1: Global Embedding]  patch grid reshape failed: {err}\033[0m")
    elif need_patch:
        print("\033[91m[WARN 2: Global Embedding] patch tokens could not be extracted.\033[0m")

    if global_tokens is not None:
        print("[Global Embedding 6] Global feature shape:", tuple(global_tokens.shape))
        print("[Global Embedding 7] Global feature:", token_preview(global_tokens))
    elif need_global:
        print("\033[91m[WARN: Global Embedding] Global tokens were requested but not produced.\033[0m")

    if patch_tokens is not None:
        print("[Global Embedding 8] Patch tokens shape:", tuple(patch_tokens.shape))
        if patch_post_info is not None:
            kept = patch_post_info.get("kept_tokens", patch_tokens.shape[0])
            keep_ratio = patch_post_info.get("keep_ratio", 1.0)
            print(f"[Global Embedding 8A] Patch variant '{resolved_variant}' kept {kept} tokens ({keep_ratio:.3f} ratio)")
        if patch_grid is not None:
            print("[Global Embedding 9] Patch grid shape:", tuple(patch_grid.shape))
            print("[Global Embedding 10] Patch grid preview:", token_preview(patch_grid))
    elif need_patch:
        print("\033[91m[WARN: Global Embedding] Patch tokens were requested but not produced.\033[0m")

    global_array = None
    # Export Global tokens
    if global_plan["npy"]:
        if global_tokens is not None:
            global_array = global_tokens.numpy()
            global_dir.mkdir(parents=True, exist_ok=True)
            progress_bar(np.save, npy_path, global_array)
            print(f"\033[32m[saved] Test Global embedding DINOv3 numpy array -> {npy_path}\033[0m")
        else:
            print("\033[91m[warn] Global npy requested but tokens unavailable.\033[0m")
    else:
        print("\033[91m[skip] Global npy disabled by configuration.\033[0m")

    # Export Patch tokens
    if patch_plan["npy"]:
        if patch_numpy is not None:
            patch_dir.mkdir(parents=True, exist_ok=True)
            progress_bar(np.save, patch_path, patch_numpy)
            print(f"\033[32m[saved] Test Patch token numpy array       -> {patch_path}\033[0m")
            
            if patch_post_info is not None and "scores" in patch_post_info:
                scores = np.asarray(patch_post_info["scores"])
                score_path = patch_dir / f"{patch_name}_scores.npy"
                progress_bar(np.save, score_path, scores)
                print(f"\033[32m[saved] Test Patch token scores array      -> {score_path}\033[0m")
        else:
            print("\033[91m[warn] Patch npy requested but tokens unavailable.\033[0m")
    else:
        print("\033[91m[skip] Patch npy disabled by configuration.\033[0m")
    
    grid_array = None
    # Export Patch grid
    if emit_grid and patch_grid is not None:
        if isinstance(patch_grid, torch.Tensor):
            grid_array = patch_grid.detach().cpu().numpy()
        else:
            grid_array = np.asarray(patch_grid)

    if grid_plan["npy"]:
        if grid_array is not None:
            grid_dir.mkdir(parents=True, exist_ok=True)
            np.save(grid_path, grid_array)
            print(f"\033[32m[saved] Test Patch grid numpy array         -> {grid_path}\033[0m")

            score_grid = None
            if (
                patch_post_info is not None
                and "scores" in patch_post_info
                and patch_grid is not None
            ):
                scores = np.asarray(patch_post_info["scores"])
                H, W = patch_grid.shape[0], patch_grid.shape[1]
                if scores.size == H * W:
                    score_grid = scores.reshape((H, W)) 
                    score_grid_path = grid_dir / f"{grid_name}_scores.npy"
                    np.save(score_grid_path, score_grid)
                    print(f"\033[32m[saved] Test Patch grid scores array        -> {score_grid_path}\033[0m")
                else:
                    print("\033[91m[warn] Score size does not match grid shape, skip saving score grid.\033[0m")

        else:
            print("\033[91m[warn] Patch grid npy requested but grid unavailable.\033[0m")
    else:
        print("\033[91m[skip] Patch grid npy disabled by configuration.\033[0m")

    timings["pipeline_total"] = (time.perf_counter() - pipeline_start) * 1000.0
    gpu_peak_mem_mb = _gather_gpu_stats(device)
    if gpu_peak_mem_mb is not None:
        print(
            f"[GPU] weight={weight} altitude={label_display} index={index_str} "
            f"peak_mem={gpu_peak_mem_mb:.2f} MB"
        )
    else:
        print(f"[GPU] weight={weight} altitude={label_display} index={index_str} peak_mem=N/A")

    def _sum_sizes(entries: Dict[str, Optional[Dict[str, Any]]]) -> Optional[int]:
        total = 0
        has_file = False
        for entry in entries.values():
            if entry and "size_bytes" in entry:
                total += int(entry["size_bytes"])
                has_file = True
        return total if has_file else None

    global_files = {
        "vector": _file_entry(npy_path) if global_plan["npy"] and global_array is not None else None,
        "patch_tokens": None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }
    patch_files = {
        "vector": None,
        "patch_tokens": _file_entry(patch_path) if patch_plan["npy"] and patch_numpy is not None else None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }
    grid_files = {
        "vector": _file_entry(grid_path) if grid_plan["npy"] and grid_array is not None else None,
        "patch_tokens": None,
        "patch_grid": None,
        "dense_vis": None,
        "index": None,
    }

    default_variant_params = {
        "patch_variant": resolved_variant,
        "patch_params": dict(resolved_variant_params),
    }

    used_variant_params = dict(default_variant_params)
    if patch_post_info and "params" in patch_post_info:
        params = patch_post_info["params"]
        if isinstance(params, dict):
            used_variant_params.update(params)

    rotations_config = used_variant_params.get("rotations") if isinstance(used_variant_params, dict) else None
    aggregation_config = used_variant_params.get("aggregation") if isinstance(used_variant_params, dict) else None

    common_config = {
        "embedding_cfg": resolved_embedding_cfg,
        "variant": resolved_variant,
        "experiment_variant": resolved_variant_label,
        "variant_label": resolved_variant_label,
        "variant_params": used_variant_params,
        "weight_id": hub_entry,
        "dataset_type": dataset_type,
        "altitude": altitude,
        "index": index,
        "prefix": prefix,
        "target_res": target_res,
        "rotations": rotations_config if isinstance(rotations_config, (list, tuple)) else [0],
        "aggregation": aggregation_config if isinstance(aggregation_config, str) else "single",
    }

    if global_plan["json"] and global_tokens is not None:
        global_dir.mkdir(parents=True, exist_ok=True)
        global_meta = {
            "run_id": file_name,
            "token_type": "GlobalToken",
            "config": dict(common_config),
            "files": global_files,
            "metrics": {
                "token_count": 1,
                "embedding_dim": int(global_tokens.shape[0]) if global_tokens is not None else None,
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
                "gpu_peak_mem_mb": gpu_peak_mem_mb,
                "embedding_storage_bytes": _sum_sizes(global_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(global_meta_path, global_meta)
    elif global_plan["json"]:
        print("\033[91m[warn] Global meta requested but tokens unavailable.\033[0m")

    if patch_plan["json"] and patch_tokens is not None:
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_metrics = {
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
        }

        patch_meta = {
            "run_id": patch_name,
            "token_type": "PatchToken",
            "config": dict(common_config),
            "files": patch_files,
            "metrics": patch_metrics,
            "timing_ms": dict(timings),
            "resources": {
                "gpu_peak_mem_mb": gpu_peak_mem_mb,
                "embedding_storage_bytes": _sum_sizes(patch_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(patch_meta_path, patch_meta)
    elif patch_plan["json"]:
        print("\033[91m[warn] Patch meta requested but tokens unavailable.\033[0m")

    if grid_plan["json"] and grid_array is not None:
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_h = int(grid_array.shape[0]) if grid_array.ndim >= 1 else None
        grid_w = int(grid_array.shape[1]) if grid_array.ndim >= 2 else None
        grid_dim = int(grid_array.shape[2]) if grid_array.ndim >= 3 else None
        grid_metrics = {
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
        }

        grid_meta = {
            "run_id": grid_name,
            "token_type": "PatchGrid",
            "config": dict(common_config),
            "files": grid_files,
            "metrics": grid_metrics,
            "timing_ms": dict(timings),
            "resources": {
                "gpu_peak_mem_mb": gpu_peak_mem_mb,
                "embedding_storage_bytes": _sum_sizes(grid_files),
                "index_size_bytes": None,
            },
        }
        _write_meta(grid_meta_path, grid_meta)
    elif grid_plan["json"]:
        print("\033[91m[warn] Patch grid meta requested but grid unavailable.\033[0m")


def main() -> None:
    run_global_embedding(
        altitude=varAltitude,
        index=varIndex,
        weight=varWeight,
        target_res=varTargetRes,
    )


if __name__ == "__main__":
    main()
