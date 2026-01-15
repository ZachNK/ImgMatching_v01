"""
Manifest-driven runner (reference embeddings) using JSON-only configuration.

Expected manifest schema (example):
{
  "dataset_key": "shinsung_data",
  "reference_embed_root": "/exports/dinov3_reference_embeds",
  "experiment": {
    "variant": "raw"
  },
  "models": [
    {
      "weights": ["vitb16"],
      "target_res": 1024,
      "embedding_cfg": null,
      "image_groups": [
        { "folder": ["R200_0001"], "indices": [1], "rotation": [45, 90, 135, 180] }
      ],
      "outputs": { "global": {"npy": true,"json": true}, "patch": {...}, "grid": {...} },
      "run": { "test_embedding": true, "generate_denseft": true }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from Test_Embedding4Query import (
    REFERENCE_EMBED_ROOT as DEFAULT_REFERENCE_EMBED_ROOT,
    REPO_DIR,
    SUPPORTED_SUFFIXES,
    process_reference_image,
    _parse_reference_filename,
)
from Generate_DenseFT4Query import generate_reference_dense_feature
from imatch.loading import (
    weights_path,
    set_dataset_key,
    normalize_group_value,
    sanitize_group_token,
    dataset_reference_embed_root,
)
from imatch.pretrained import pretrained_model
from imatch.utils import progress_bar, create_progress
from imatch.variants import build_runtime_variant

TOKEN_KINDS = ("global", "patch", "grid")

BASE_DIR = Path(__file__).resolve().parent
DATA_KEY_PATH = BASE_DIR / "json/data_key.json"
REFERENCE_ROOT_ENV = Path(os.getenv("REFERENCE_ROOT", "/opt/references"))
REFERENCE_PREFIX_ENV = os.getenv("REFERENCE_PREFIX", "")
REFERENCE_DATASET_PREFIX_ENV = os.getenv("REFERENCE_DATASET_PREFIX", "")


@dataclass
class ReferenceModelSession:
    model: torch.nn.Module
    hub_entry: str
    dataset_type: str
    device: torch.device
    session_id: str


_REFERENCE_SESSION_CACHE: Dict[Tuple[str, str, int], ReferenceModelSession] = {}
REFERENCE_SESSION_STATS: Dict[str, Dict[str, int]] = {}


def _session_cache_key(weight_key: str, device: torch.device) -> Tuple[str, str, int]:
    index = device.index if device.index is not None else -1
    return (weight_key, device.type, index)


def _reference_stats_entry(weight_key: str) -> Dict[str, int]:
    entry = REFERENCE_SESSION_STATS.get(weight_key)
    if entry is None:
        entry = {"session_loads": 0, "direct_loads": 0, "reuses": 0}
        REFERENCE_SESSION_STATS[weight_key] = entry
    return entry


def collect_reference_session_stats(reset: bool = False) -> Dict[str, Dict[str, int]]:
    snapshot = {weight: dict(stats) for weight, stats in REFERENCE_SESSION_STATS.items()}
    if reset:
        REFERENCE_SESSION_STATS.clear()
    return snapshot


def _clear_reference_sessions() -> None:
    for session in _REFERENCE_SESSION_CACHE.values():
        if session.device.type == "cuda":
            torch.cuda.empty_cache()
    _REFERENCE_SESSION_CACHE.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reference embedding jobs from a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("json/manifestReference.json"),
        help="Path to the reference manifest JSON file.",
    )
    parser.add_argument(
        "--reload-each",
        action="store_true",
        help="Reload reference models for every embedding instead of reusing cached sessions.",
    )
    return parser.parse_args()


def _load_data_registry() -> Dict[str, Any]:
    if not DATA_KEY_PATH.exists():
        raise FileNotFoundError(f"\033[91m[Error] data_key.json not found: {DATA_KEY_PATH}\033[0m")
    return json.loads(DATA_KEY_PATH.read_text(encoding="utf-8"))


DATA_REGISTRY = _load_data_registry()
DATASETS = DATA_REGISTRY.get("datasets", {})


def _first_key(mapping: Dict[str, Any]) -> str:
    return next(iter(mapping)) if mapping else ""

def _build_folder_map(dataset_key: str, dataset_cfg: Dict[str, Any]) -> Dict[str, Tuple[str, str, str]]:
    """
    folder name -> (capture_id, label_display, label_token)
    Uses folder override when provided; otherwise falls back to folder_template.
    """
    images = dataset_cfg.get("images") or dataset_cfg.get("captures") or {}
    folder_template = dataset_cfg.get("folder_template", "{capture_id}_{label}")
    mapping: Dict[str, Tuple[str, str, str]] = {}
    for capture_id, raw_value in images.items():
        override = raw_value if isinstance(raw_value, dict) else {}
        label_value = override.get("label")
        if label_value is None:
            label_value = override.get("altitude", raw_value)
        label_display = normalize_group_value(label_value)
        label_token = sanitize_group_token(label_display)
        folder_override = override.get("folder")
        folder = folder_override or folder_template.format(
            capture_id=capture_id,
            label=label_display,
            label_token=label_token,
            dataset_key=dataset_key,
        )
        mapping[str(folder)] = (str(capture_id), label_display, label_token)
    return mapping

def _normalize_folder_list(field: Any) -> List[str]:
    if field is None:
        raise ValueError("\033[91m[Error] Image group must define 'folder'.\033[0m")
    def _coerce(item: Any) -> str:
        if isinstance(item, dict) and "folder" in item:
            return str(item.get("folder")).strip()
        return str(item).strip()
    if isinstance(field, list):
        return [_coerce(v) for v in field if _coerce(v)]
    coerced = _coerce(field)
    return [coerced] if coerced else []


def _normalize_indices(field: Any, expected: int) -> List[List[int]]:
    if not field:
        raise ValueError("\033[91m[Error] Image group must define 'indices'.\033[0m")
    if isinstance(field, list) and field and isinstance(field[0], list):
        if len(field) != expected:
            raise ValueError("\033[91m[Error] indices length must match folders when using list-of-lists.\033[0m")
        result = []
        for lst in field:
            if len(lst) == 2 and all(isinstance(v, (int, float)) for v in lst):
                start, end = int(lst[0]), int(lst[1])
                result.append(list(range(start, end + 1)))
            else:
                result.append([int(i) for i in lst])
        return result
    if isinstance(field, list) and len(field) == 2 and all(isinstance(v, (int, float)) for v in field):
        start, end = int(field[0]), int(field[1])
        shared = list(range(start, end + 1))
    else:
        shared = [int(i) for i in field]
    return [shared for _ in range(expected)]


def _normalize_rotations(field: Any, expected: int) -> List[List[int]]:
    if not field:
        return [[0] for _ in range(expected)]
    if isinstance(field, list) and field and isinstance(field[0], list):
        if len(field) != expected:
            raise ValueError(
                "\033[91m[Error] rotation length must match folders when using list-of-lists.\033[0m"
            )
        return [[int(float(val)) for val in lst] for lst in field]
    shared = [int(float(val)) for val in field]
    return [shared for _ in range(expected)]


def _resolve_capture_id(altitude: Any, altitude_map: Dict[str, List[str]], dataset_key: str) -> str:
    raise ValueError("\033[91m[Error] Altitude-based selection is no longer supported; use folder in image_groups.\033[0m")


def _resolve_dataset_context(manifest: Dict[str, Any]) -> tuple[str, Dict[str, List[str]], Path, str, str]:
    dataset_key = manifest.get("dataset_key") or os.getenv("DATASET_KEY") or _first_key(DATASETS)
    if not dataset_key or dataset_key not in DATASETS:
        raise ValueError(
            f"\033[91m[Error] Dataset key '{dataset_key or 'undefined'}' is not registered in data_key.json.\033[0m"
        )

    dataset_cfg = DATASETS[dataset_key]
    images = dataset_cfg.get("images") or dataset_cfg.get("captures")
    if not isinstance(images, dict) or not images:
        raise ValueError(
            f"\033[91m[Error] Dataset '{dataset_key}' must define an 'images' mapping.\033[0m"
        )

    folder_map = _build_folder_map(dataset_key, dataset_cfg)
    reference_root = REFERENCE_ROOT_ENV
    reference_prefix = REFERENCE_PREFIX_ENV
    dataset_prefix = REFERENCE_DATASET_PREFIX_ENV
    return dataset_key, folder_map, reference_root, reference_prefix, dataset_prefix


def expand_reference_entries(
    group: Dict[str, Any],
    folder_map: Dict[str, Tuple[str, str, str]],
    dataset_key: str,
    reference_root: Path,
    reference_prefix: str,
    dataset_prefix: str,
) -> List[Dict[str, Any]]:
    folders = _normalize_folder_list(group.get("folder"))
    indices_per_folder = _normalize_indices(group.get("indices"), len(folders))
    rotations_per_folder = _normalize_rotations(group.get("rotation"), len(folders))

    dataset_dir = reference_root 
    expanded: List[Dict[str, Any]] = []
    for folder_name, idx_list, rot_list in zip(folders, indices_per_folder, rotations_per_folder):
        info = folder_map.get(folder_name)
        if info is None:
            raise ValueError(
                f"\033[91m[Error] Folder '{folder_name}' is not registered under dataset '{dataset_key}'.\033[0m"
            )
        capture_id, label_display, label_token = info
        reference_dir = dataset_dir / {folder_name}
        altitude_value = folder_name
        label_token = sanitize_group_token(folder_name)
        expanded.append(
            {
                "name": folder_name,
                "capture_id": capture_id,
                "altitude": altitude_value,
                "label_token": label_token,
                "reference_dir": reference_dir,
                "indices": idx_list,
                "rotations": rot_list,
            }
        )
    return expanded


def _expand_weights(raw_entry: Any) -> List[str]:
    if raw_entry is None:
        raise ValueError("\033[91m[Error] Each model entry must define 'weights'.\033[0m")
    raw_list = raw_entry if isinstance(raw_entry, list) else [raw_entry]
    keys: List[str] = []
    for item in raw_list:
        if not isinstance(item, str):
            raise TypeError(f"\033[91m[Error] weights entries must be strings, got {type(item)}\033[0m")
        for part in item.split(","):
            key = part.strip()
            if key:
                keys.append(key)
    if not keys:
        raise ValueError("\033[91m[Error] No valid weights entries resolved.\033[0m")
    return keys


def _blank_output_plan() -> Dict[str, Dict[str, bool]]:
    return {key: {"npy": False, "json": False} for key in TOKEN_KINDS}


def _parse_output_entry(raw: Any) -> Dict[str, bool]:
    if isinstance(raw, dict):
        npy = bool(raw.get("npy"))
        json_enabled = bool(raw.get("json"))
        enable = bool(raw.get("enable")) or bool(raw.get("enabled"))
        if not (npy or json_enabled) and enable:
            npy = True
            json_enabled = True
        return {"npy": npy, "json": json_enabled}
    flag = bool(raw)
    return {"npy": flag, "json": flag}


def _normalize_outputs(outputs: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
    plan = _blank_output_plan()
    if isinstance(outputs, dict):
        for key in TOKEN_KINDS:
            if key in outputs:
                plan[key] = _parse_output_entry(outputs[key])
    if not any(plan[k]["npy"] or plan[k]["json"] for k in TOKEN_KINDS):
        for key in TOKEN_KINDS:
            plan[key] = {"npy": True, "json": True}
    return plan


def _load_weighted_model(
    weight_key: str,
    device: torch.device,
    reload_each: bool = False,
) -> tuple[torch.nn.Module, str, str]:
    stats = _reference_stats_entry(weight_key)
    if reload_each:
        stats["direct_loads"] += 1
        hub_entry, weight_path, dataset_type = weights_path(weight_key)
        print(
            f"[QSESSION] [DIRECT LOAD] weight={weight_key} hub_entry={hub_entry} "
            f"device={device}"
        )
        model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
        model.eval()
        return model, hub_entry, dataset_type

    cache_key = _session_cache_key(weight_key, device)
    cached = _REFERENCE_SESSION_CACHE.get(cache_key)
    if cached is not None:
        stats["reuses"] += 1
        print(
            f"[QSESSION] [REUSE] weight={weight_key} session={cached.session_id} device={device}"
        )
        return cached.model, cached.hub_entry, cached.dataset_type

    hub_entry, weight_path, dataset_type = weights_path(weight_key)
    print(f"[INFO] Loading model {hub_entry} ({weight_key}) on {device}")
    model, _ = progress_bar(pretrained_model, REPO_DIR, hub_entry, weight_path, device)
    model.eval()
    session = ReferenceModelSession(
        model=model,
        hub_entry=hub_entry,
        dataset_type=dataset_type,
        device=device,
        session_id=f"{weight_key}:{id(model):x}",
    )
    _REFERENCE_SESSION_CACHE[cache_key] = session
    stats["session_loads"] += 1
    return session.model, session.hub_entry, session.dataset_type


def _resolve_reference_matches(
    reference_dir: Path,
    capture_id: str,
    label_token: str,
    index_val: int,
    rotation: int,
) -> List[Path]:
    reference_dir = Path(reference_dir)
    if not reference_dir.exists():
        print(f"\033[91m[WARN] Reference directory missing, skipping: {reference_dir}\033[0m")
        return []

    idx_int = int(index_val)
    rotation_tag = f"rot{int(rotation):03d}"
    regex = re.compile(
        rf"^{re.escape(capture_id)}_{re.escape(label_token)}_(\d+)_rot{int(rotation):03d}(?:_|$)",
        re.IGNORECASE,
    )
    matches: List[Path] = []
    for suffix in SUPPORTED_SUFFIXES:
        pattern = f"*{rotation_tag}*{suffix}"
        for path in sorted(reference_dir.glob(pattern)):
            m = regex.match(path.stem)
            if m:
                try:
                    if int(m.group(1)) == idx_int:
                        matches.append(path)
                except ValueError:
                    continue

    # Deduplicate while preserving order.
    seen = {}
    for path in matches:
        seen[path] = None
    return list(seen.keys())


def execute_manifest(manifest_path: Path, reload_each: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_key, folder_map, reference_root, reference_prefix, dataset_prefix = _resolve_dataset_context(manifest)
    set_dataset_key(dataset_key)
    dataset_cfg = DATASETS[dataset_key]
    base_reference_root = Path(manifest.get("reference_embed_root") or DEFAULT_REFERENCE_EMBED_ROOT)
    reference_embed_root = dataset_reference_embed_root(dataset_key, base_reference_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = manifest.get("experiment", {}) if isinstance(manifest, dict) else {}
    variant_cfg = experiment.get("variant", {}) if isinstance(experiment, dict) else {}

    def _as_list(val: Any) -> List[Any]:
        if val is None:
            return [None]
        if isinstance(val, list):
            return list(val)
        return [val]

    raw_on = bool(variant_cfg.get("raw", True)) if isinstance(variant_cfg, dict) else True
    sub_cfg = variant_cfg.get("sub", variant_cfg.get("subsample", {})) if isinstance(variant_cfg, dict) else {}
    sub_on = bool(sub_cfg.get("use", False))
    sub_stride_list = _as_list(sub_cfg.get("stride")) if sub_on else [None]
    base_variants: List[tuple[str, Optional[int]]] = []
    if raw_on:
        base_variants.append(("raw", None))
    if sub_on:
        for stride in sub_stride_list:
            base_variants.append(("subsample", stride if stride is not None else 1))

    runtime_variants = []
    for base_name, stride in base_variants:
        rv = build_runtime_variant(
            base_name,
            subsample_stride=stride,
        )
        runtime_variants.append(rv)

    if not runtime_variants:
        raise ValueError(
            "\033[91m[Error] No variants enabled. Set experiment.variant.raw=true or "
            "experiment.variant.sub.use=true (with stride) in manifest.\033[0m"
        )

    failures: List[Dict[str, Any]] = []
    job_counter = 0
    processed = 0
    planned_total = 0
    try:
        with create_progress() as reference_progress:
            progress_task = reference_progress.add_task("[cyan]Reference Embedding...[/cyan]", total=0)

            for runtime_variant in runtime_variants:
                for model_entry in manifest.get("models", []):
                    weight_keys = _expand_weights(model_entry.get("weights") or model_entry.get("weight_key"))
                    image_groups = model_entry.get("image_groups", [])
                    outputs_cfg = _normalize_outputs(model_entry.get("outputs"))
                    run_cfg = model_entry.get("run", {})
                    target_res = int(model_entry.get("target_res", 1024))
                    embedding_cfg = model_entry.get("embedding_cfg")
                    denseft_active = bool(run_cfg.get("generate_denseft", False))
                    if denseft_active:
                        outputs_cfg["grid"]["npy"] = True
                    test_embedding_enabled = bool(run_cfg.get("test_embedding", True))

                    if not image_groups or not test_embedding_enabled:
                        print("\033[93m[WARN] Skipping model entry without runnable image_groups/test_embedding.\033[0m")
                        continue

                    expanded_groups: List[Dict[str, Any]] = []
                    for group in image_groups:
                        expanded_groups.extend(
                            expand_reference_entries(
                                group,
                                folder_map,
                                dataset_key,
                                reference_root,
                                reference_prefix,
                                dataset_prefix,
                            )
                        )

                    for weight_key in weight_keys:
                        model, hub_entry, dataset_type = _load_weighted_model(weight_key, device, reload_each=reload_each)
                        job_counter += 1
                        print(
                            f"[JOB {job_counter}] dataset={dataset_key} weight={weight_key} variant={runtime_variant.patch_variant} "
                            f"(label={runtime_variant.label}) target_res={target_res} embedding_cfg={embedding_cfg} "
                            f"stride={runtime_variant.patch_params.get('stride')} "
                            f"outputs(global/patch/grid)={[(outputs_cfg[k]['npy'], outputs_cfg[k]['json']) for k in TOKEN_KINDS]} "
                            f"denseft={int(denseft_active)}"
                        )

                        for combo in expanded_groups:
                            print(
                                f"  [Group] {combo['name']} alt={combo['altitude']} "
                                f"indices={len(combo['indices'])} rotations={combo['rotations']}"
                            )
                            for index_val in combo["indices"]:
                                for rotation in combo["rotations"]:
                                    matches = _resolve_reference_matches(
                                        combo["reference_dir"],
                                        combo["capture_id"],
                                        combo["label_token"],
                                        index_val,
                                        rotation,
                                    )
                                    if not matches:
                                        print(
                                            f"    [WARN] Missing files for index={index_val} rotation={rotation} "
                                            f"in {combo['reference_dir']}"
                                        )
                                        continue

                                    planned_total += len(matches)
                                    reference_progress.update(progress_task, total=planned_total)

                                    for reference_path in matches:
                                        try:
                                            try:
                                                info = _parse_reference_filename(reference_path)
                                            except ValueError as err:
                                                print(f"    [WARN] Skipping unexpected file format: {reference_path} -> {err}")
                                                continue

                                            print(
                                                f"    -> {reference_path.name} "
                                                f"(alt={info.altitude}, idx={info.index}, rot={rotation})"
                                            )
                                            try:
                                                result = process_reference_image(
                                                    model=model,
                                                    device=device,
                                                    hub_entry=hub_entry,
                                                    dataset_type=dataset_type,
                                                    weight_key=weight_key,
                                                    info=info,
                                                    target_res=target_res,
                                                    embedding_cfg=embedding_cfg,
                                                    variant=runtime_variant.patch_variant,
                                                    variant_params=dict(runtime_variant.patch_params),
                                                    output_plan=outputs_cfg,
                                                    reference_embed_root=reference_embed_root,
                                                    variant_label=runtime_variant.label,
                                                    dataset_key=dataset_key,
                                                )
                                                processed += 1
                                                if denseft_active and result.grid_path is not None:
                                                    generate_reference_dense_feature(result.grid_path)
                                                elif denseft_active:
                                                    print("      [WARN] DenseFT requested but PatchGrid npy missing; skipped.")
                                            except Exception:
                                                failure = traceback.format_exc()
                                                failures.append(
                                                    {
                                                        "weight": weight_key,
                                                        "file": reference_path.as_posix(),
                                                        "rotation": rotation,
                                                        "index": index_val,
                                                        "traceback": failure,
                                                    }
                                                )
                                                print(f"      [WARN] Failed to process {reference_path}")
                                                print(failure)
                                        finally:
                                            reference_progress.advance(progress_task)
                        # Model stays cached for subsequent jobs; GPU memory released at shutdown.

        print(f"\n\033[32mProcessed {processed} reference images across {job_counter} weight jobs.\033[0m")
        if failures:
            print("\n=== Failures ===")
            for entry in failures:
                print(
                    f"* weight={entry['weight']} file={entry['file']} "
                    f"index={entry['index']} rotation={entry['rotation']}"
                )
                print(entry["traceback"])

    finally:
        _clear_reference_sessions()
        stats = collect_reference_session_stats(reset=True)
        if stats:
            print("\n[QSESSION] Reference weight usage summary:")
            for weight_key, data in stats.items():
                print(
                    f"  - weight={weight_key} session_loads={data.get('session_loads', 0)} "
                    f"reuses={data.get('reuses', 0)} direct_loads={data.get('direct_loads', 0)}"
                    "\n\n"
                )


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"\033[91m[Error] Manifest not found: {manifest_path}\033[0m")
    execute_manifest(manifest_path, reload_each=args.reload_each)


if __name__ == "__main__":
    main()
