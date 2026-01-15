from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from imatch.faiss_runtime import faiss, tqdm


@dataclass
class EmbeddingRecord:
    vec_path: Path
    meta_path: Optional[Path]
    image_path: Optional[str]


def _is_global_token(path: Path) -> bool:
    """Check filename to include only GlobalToken/ReferenceGlobal vectors."""
    name = path.name
    return name.startswith("GlobalToken") or name.startswith("ReferenceGlobal") or name.startswith("QueryGlobal")


def _read_meta(meta_path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_image_path(meta: Optional[Dict[str, object]]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    cfg = meta.get("config") or {}
    if isinstance(cfg, dict):
        reference_info = cfg.get("reference")
        if isinstance(reference_info, dict) and "source_file" in reference_info:
            return reference_info.get("source_file")
        query_info = cfg.get("query")
        if isinstance(query_info, dict) and "source_file" in query_info:
            return query_info.get("source_file")
    return None


def _find_vectors(root: Path, show_progress: bool, role: str) -> List[EmbeddingRecord]:
    """Recursively find GlobalToken npy files under root."""
    records: List[EmbeddingRecord] = []
    iter_paths = root.rglob("*.npy")
    if show_progress and tqdm:
        iter_paths = tqdm(iter_paths, desc=f"{role}:scan", unit="file")
    elif show_progress and not tqdm:
        print("[INFO] tqdm is not installed; scan progress bar disabled. pip install tqdm to enable.")

    for npy_path in iter_paths:
        if not _is_global_token(npy_path):
            continue
        meta_path = npy_path.with_name(npy_path.stem + "_meta.json")
        meta = _read_meta(meta_path) if meta_path.exists() else None
        img_path = _extract_image_path(meta)
        records.append(
            EmbeddingRecord(
                vec_path=npy_path, meta_path=meta_path if meta_path.exists() else None, image_path=img_path
            )
        )
    return records


def _load_vec(path: Path) -> np.ndarray:
    arr = np.load(path)
    vec = np.asarray(arr, dtype=np.float32).reshape(-1)
    if vec.ndim != 1:
        raise ValueError(f"Vector at {path} is not 1-D after reshape; got shape {vec.shape}")
    return vec


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize rows in-place for cosine/IP search."""
    faiss.normalize_L2(vecs)
    return vecs


def _load_records(
    root: Optional[Path], role: str, show_progress: bool = False
) -> Tuple[np.ndarray, List[EmbeddingRecord], Dict[str, float]]:
    if root is None or not root.exists():
        raise FileNotFoundError(f"[{role}] root not found: {root}")
    t0 = time.perf_counter()
    records = _find_vectors(root, show_progress=show_progress, role=role)
    find_ms = (time.perf_counter() - t0) * 1000.0
    if not records:
        raise RuntimeError(f"[{role}] No GlobalToken npy files found under {root}")
    t1 = time.perf_counter()
    if show_progress and tqdm:
        vecs = []
        for rec in tqdm(records, desc=f"{role}:load", unit="vec"):
            vecs.append(_load_vec(rec.vec_path))
    else:
        vecs = [_load_vec(rec.vec_path) for rec in records]
    load_ms = (time.perf_counter() - t1) * 1000.0
    mat = np.stack(vecs, axis=0)
    t2 = time.perf_counter()
    _normalize(mat)
    norm_ms = (time.perf_counter() - t2) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[DEBUG] {role}: files={len(records)}, find_ms={find_ms:.2f}, load_ms={load_ms:.2f}, "
        f"normalize_ms={norm_ms:.2f}, total_ms={total_ms:.2f}"
    )
    timing = {
        "find": find_ms,
        "load": load_ms,
        "normalize": norm_ms,
        "total": total_ms,
    }
    return mat, records, timing


def _ensure_out_dir(base: Path, weight: str, mode: str, direction_key: str) -> Path:
    out_dir = base / f"{weight}_{mode}_{direction_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _embed_time_ms(meta_path: Optional[Path]) -> float:
    """
    Try to pull embedding pipeline time from *_meta.json.
    Prefers timing_ms.pipeline_total, then timing_ms.global_forward, else 0.
    """
    if meta_path is None or not meta_path.exists():
        return 0.0
    meta = _read_meta(meta_path)
    if not isinstance(meta, dict):
        return 0.0
    timing = meta.get("timing_ms") or meta.get("timing") or {}
    if not isinstance(timing, dict):
        return 0.0
    for key in ("pipeline_total", "global_forward", "total", "embed"):
        val = timing.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0
