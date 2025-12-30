"""
Build per-weight FAISS indexes from precomputed DINOv3 GlobalToken embeddings
and run TopK retrieval.
Mode keywords: reference-data (reference->DB), data-reference (DB->reference), reference-reference (reference->reference), data-data (DB->DB).
`--aggregate` saves a single JSON per weight when enabled.

Data layout (host paths as provided):
  Raw DB embeddings (no variant):       D:\dinov3_exports\dinov3_embeds\shinsung_data\{weight}
  Raw reference embeddings (no variant):    D:\dinov3_exports\dinov3_reference_embeds\shinsung_data\{weight}
  Variant DB embeddings (with variant): H:\dinov3_exports\dinov3_embeds\shinsung_data\_{weight}
  Variant reference embeddings:             H:\dinov3_exports\dinov3_reference_embeds\shinsung_data\_{weight}

Only GlobalToken vectors are used. Index is built per weight and searched
with the matching weight's reference embeddings. Results are saved to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as err:  # pragma: no cover
    print(
        "[FATAL] faiss is not installed. Install faiss-cpu or faiss-gpu before running.\n"
        "  pip install faiss-cpu  # or faiss-gpu\n"
        f"  import error: {err}"
    )
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


# -----------------------------------------------------------------------------
# Config: paths for raw vs variant embeddings
# -----------------------------------------------------------------------------
# Defaults point to in-container mount (/exports/...) but can be overridden via env.
RAW_DB_BASE = Path(os.getenv("RAW_EMBED_ROOT", "/exports/dinov3_embeds/shinsung_data"))
RAW_REFERENCE_BASE = Path(os.getenv("RAW_REFERENCE_ROOT", "/exports/dinov3_reference_embeds/shinsung_data"))

RAW_DB_ROOTS: Dict[str, Path] = {
    "vits16+": RAW_DB_BASE / "vits16+",
    "vitb16": RAW_DB_BASE / "vitb16",
    "vith16+": RAW_DB_BASE / "vith16+",
    "vitl16": RAW_DB_BASE / "vitl16",
    "vitl16sat": RAW_DB_BASE / "vitl16sat",
    "vits16": RAW_DB_BASE / "vits16",
}

RAW_REFERENCE_ROOTS: Dict[str, Path] = {
    "vits16+": RAW_REFERENCE_BASE / "vits16+",
    "vitb16": RAW_REFERENCE_BASE / "vitb16",
    "vith16+": RAW_REFERENCE_BASE / "vith16+",
    "vitl16": RAW_REFERENCE_BASE / "vitl16",
    "vitl16sat": RAW_REFERENCE_BASE / "vitl16sat",
    "vits16": RAW_REFERENCE_BASE / "vits16",
}

# Variant roots: folder names are prefixed with an underscore.
def _default_variant_db_base() -> Path:
    env_override = os.getenv("VARIANT_DB_BASE")
    if env_override:
        return Path(env_override)
    if os.name == "nt":
        return Path(r"H:\dinov3_exports\dinov3_embeds\shinsung_data")
    return Path("/exports/dinov3_embeds/shinsung_data")


def _default_variant_reference_base() -> Path:
    env_override = os.getenv("VARIANT_REFERENCE_BASE")
    if env_override:
        return Path(env_override)
    if os.name == "nt":
        return Path(r"H:\dinov3_exports\dinov3_reference_embeds\shinsung_data")
    return Path("/exports/dinov3_reference_embeds/shinsung_data")


VARIANT_DB_BASE = _default_variant_db_base()
VARIANT_REFERENCE_BASE = _default_variant_reference_base()


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


# Default output root for retrieval results
DEFAULT_OUT_ROOT = _default_out_root()

DIRECTION_CHOICES = ("reference-data", "data-reference", "reference-reference", "data-data")
LEGACY_DIRECTION_ALIASES = {
    "query-data": "reference-data",
    "data-query": "data-reference",
    "query-query": "reference-reference",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@dataclass
class EmbeddingRecord:
    vec_path: Path
    meta_path: Optional[Path]
    image_path: Optional[str]


def _is_global_token(path: Path) -> bool:
    """Check filename to include only GlobalToken/ReferenceGlobal vectors."""
    name = path.name
    return name.startswith("GlobalToken") or name.startswith("ReferenceGlobal") or name.startswith("QueryGlobal")


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
        meta = None
        image_path = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None
        if isinstance(meta, dict):
            cfg = meta.get("config") or {}
            if isinstance(cfg, dict):
                reference_info = cfg.get("reference")
                if isinstance(reference_info, dict) and "source_file" in reference_info:
                    image_path = reference_info.get("source_file")
                query_info = cfg.get("query")
                if image_path is None and isinstance(query_info, dict) and "source_file" in query_info:
                    image_path = query_info.get("source_file")
        records.append(
            EmbeddingRecord(
                vec_path=npy_path, meta_path=meta_path if meta_path.exists() else None, image_path=image_path
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


def _build_index(db_vectors: np.ndarray, use_gpu: bool) -> faiss.Index:
    dim = db_vectors.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    gpu_ms = None
    if use_gpu:
        # Move to all available GPUs. Falls back to CPU if no GPU is present.
        try:
            tgpu = time.perf_counter()
            index = faiss.index_cpu_to_all_gpus(index)
            gpu_ms = (time.perf_counter() - tgpu) * 1000.0
        except Exception as exc:  # pragma: no cover - GPU init errors
            print(f"[WARN] GPU init failed ({exc}), falling back to CPU index.")
    index.add(db_vectors)
    add_ms = (time.perf_counter() - t0) * 1000.0
    if gpu_ms is not None:
        print(
            f"[DEBUG] build_index: dim={dim}, cpu_init_ms={cpu_ms:.2f}, "
            f"gpu_xfer_ms={gpu_ms:.2f}, add_ms={add_ms:.2f}"
        )
    else:
        print(f"[DEBUG] build_index: dim={dim}, cpu_init_ms={cpu_ms:.2f}, add_ms={add_ms:.2f}")
    return index


def _build_roots(weight: str, use_variant: bool) -> Tuple[Optional[Path], Optional[Path]]:
    if use_variant:
        return VARIANT_DB_BASE / f"_{weight}", VARIANT_REFERENCE_BASE / f"_{weight}"
    return RAW_DB_ROOTS.get(weight), RAW_REFERENCE_ROOTS.get(weight)


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


def _ensure_out_dir(base: Path, weight: str, mode: str) -> Path:
    out_dir = base / f"{weight}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_single_weight(
    weight: str,
    use_variant: bool,
    k: int,
    out_root: Path,
    use_gpu: bool,
    direction: str,
    aggregate: bool,
    show_progress: bool,
) -> None:
    """
    direction:
      - "reference-data": reference embeddings search DB embeddings
      - "data-reference": DB embeddings search reference embeddings
      - "reference-reference": reference embeddings search reference embeddings
      - "data-data": DB embeddings search DB embeddings
    """
    direction = LEGACY_DIRECTION_ALIASES.get(direction, direction)
    mode = "variant" if use_variant else "raw"
    if direction not in DIRECTION_CHOICES:
        raise ValueError("direction must be one of 'reference-data', 'data-reference', 'reference-reference', 'data-data'")
    direction_key = {
        "reference-data": "reference2db",
        "data-reference": "db2reference",
        "reference-reference": "reference2reference",
        "data-data": "db2db",
    }[direction]

    db_root, reference_root = _build_roots(weight, use_variant)
    if direction == "reference-data":
        index_root, search_root = db_root, reference_root
        index_role, search_role = "db", "reference"
    elif direction == "data-reference":
        index_root, search_root = reference_root, db_root
        index_role, search_role = "reference", "db"
    elif direction == "reference-reference":
        index_root = search_root = reference_root
        index_role = search_role = "reference"
    else:  # data-data
        index_root = search_root = db_root
        index_role = search_role = "db"

    direction_label = {
        "reference-data": "reference embeddings search DB embeddings",
        "data-reference": "DB embeddings search reference embeddings",
        "reference-reference": "reference embeddings search reference embeddings",
        "data-data": "DB embeddings search DB embeddings",
    }[direction]
    print(f"\n[INFO] Processing weight={weight} mode={mode} direction={direction_label}")

    idx_start = time.perf_counter()
    index_vecs, index_records, index_timing = _load_records(
        index_root, f"{weight}:{index_role}", show_progress=show_progress
    )
    index = _build_index(index_vecs, use_gpu=use_gpu)
    index_build_ms = (time.perf_counter() - idx_start) * 1000.0

    search_vecs, search_records, search_timing = _load_records(
        search_root, f"{weight}:{search_role}", show_progress=show_progress
    )

    out_dir = _ensure_out_dir(out_root, weight, mode)
    print(
        f"[INFO] Index vectors: {len(index_records)} ({index_role}), "
        f"Search vectors: {len(search_records)} ({search_role}), "
        f"index_build_ms={index_build_ms:.2f}"
    )

    aggregate_results: List[Dict[str, object]] = []
    total_search_ms = 0.0

    iter_records = enumerate(search_records, start=1)
    using_tqdm = bool(tqdm) and bool(show_progress)
    if show_progress and not tqdm:
        print("[INFO] tqdm is not installed; progress bar disabled. pip install tqdm to enable.")
    if using_tqdm:
        iter_records = tqdm(iter_records, total=len(search_records), desc=f"{weight}-{mode}-{direction}", unit="vec")

    for si, srec in iter_records:
        qvec = np.asarray(search_vecs[si - 1 : si], dtype=np.float32)  # single row view
        search_start = time.perf_counter()
        k_eff = min(k, len(index_records))
        scores, idxs = index.search(qvec, k_eff)
        search_ms = (time.perf_counter() - search_start) * 1000.0
        total_search_ms += search_ms

        top_scores = scores[0].tolist()
        top_idxs = idxs[0].tolist()

        hits = []
        for rank, (db_idx, score) in enumerate(zip(top_idxs, top_scores), start=1):
            rec = index_records[db_idx]
            hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "vector": rec.vec_path.as_posix(),
                    "meta": rec.meta_path.as_posix() if rec.meta_path else None,
                    "image": rec.image_path,
                }
            )

        result = {
            "weight": weight,
            "mode": mode,
            "direction": direction_key,
            "k": k_eff,
            "index_count": len(index_records),
            "search_count": len(search_records),
            "search": {
                "vector": srec.vec_path.as_posix(),
                "meta": srec.meta_path.as_posix() if srec.meta_path else None,
                "image": srec.image_path,
            },
            "hits": hits,
            "timing_ms": {
                "search": search_ms,
                "index_load": index_timing,
                "search_load": search_timing,
            },
        }

        if aggregate:
            aggregate_results.append(result)
        else:
            out_path = out_dir / f"{srec.vec_path.stem}_top{k_eff}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        if not using_tqdm and (si % 25 == 0 or si == len(search_records)):
            print(f"[INFO] {weight} {mode} {direction}: processed {si}/{len(search_records)}")

    if aggregate:
        agg_payload = {
            "weight": weight,
            "mode": mode,
            "direction": direction_key,
            "k": k,
            "index": {
                "role": index_role,
                "root": str(index_root) if index_root else None,
                "count": len(index_records),
            },
            "search": {
                "role": search_role,
                "root": str(search_root) if search_root else None,
                "count": len(search_records),
            },
            "timing_ms": {
                "index_load": index_timing,
                "search_load": search_timing,
                "index_build": index_build_ms,
                "total_search": total_search_ms,
                "avg_search": total_search_ms / len(search_records) if search_records else None,
            },
            "results": aggregate_results,
        }
        out_path = out_dir / f"{weight}_{mode}_{direction}_top{k}.json"
        out_path.write_text(json.dumps(agg_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] saved aggregate: {out_path}")


def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Per-weight FAISS TopK retrieval for DINOv3 embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-w",
        "--weights",
        nargs="+",
        default=list(RAW_DB_ROOTS.keys()),
        help="Weight keys to process (default: all known raw weights).",
    )
    parser.add_argument("--variant", action="store_true", help="Use variant roots (H: with leading underscore).")
    parser.add_argument("--k", type=int, default=10, help="TopK size (default: 10).")
    parser.add_argument("--gpu", action="store_true", help="Use faiss-gpu (index_cpu_to_all_gpus).")
    parser.add_argument(
        "-m",
        "--match",
        choices=[
            *DIRECTION_CHOICES,
            *LEGACY_DIRECTION_ALIASES.keys(),
        ],
        default="data-reference",
        help=(
            "reference-data: reference embeddings search DB embeddings; "
            "data-reference: DB embeddings search reference embeddings; "
            "reference-reference: reference embeddings search reference embeddings; "
            "data-data: DB embeddings search DB embeddings. "
            "(Legacy: query-data/data-query/query-query are accepted aliases.)"
        ),
    )
    parser.add_argument(
        "-a",
        "--aggregate",
        action="store_true",
        help="Save a single aggregated JSON per weight (instead of per-search files).",
    )
    parser.add_argument(
        "-o",
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=(
            f"Output directory for retrieval JSON (default: {DEFAULT_OUT_ROOT}; "
            "override with FAISS_OUT_ROOT env)"
        ),
    )
    parser.add_argument(
        "--list-cli",
        action="store_true",
        help="Print available CLI options and exit.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars during load/search.",
    )
    args = parser.parse_args()
    return parser, args


def main() -> None:
    parser, args = parse_args()
    if args.list_cli:
        parser.print_help()
        return
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Normalize match flag aliases
    direction = args.match

    for weight in args.weights:
        try:
            run_single_weight(
                weight,
                args.variant,
                args.k,
                out_root,
                use_gpu=args.gpu,
                direction=direction,
                aggregate=bool(args.aggregate),
                show_progress=bool(args.progress),
            )
        except Exception as exc:
            print(f"[ERROR] weight={weight} failed: {exc}")


if __name__ == "__main__":
    main()
