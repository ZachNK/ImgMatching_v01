"""
Run FAISS TopK search and emit per-embedding JSON rows shaped for the
CoarseLocalizationInfo DTO (Redis interface).

- Mirrors project/match_faiss_gpu.py search behavior (IndexFlatIP + L2 norm).
- Saves one JSON per search embedding (no per-weight aggregation).
- Output schema matches DTO.CoarseLocalizationInfo:
    encoder, backend, dim, metric, topk, timing_ms{embed,search,total}, results[{rank,path,filename,score}]
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

# Make DTO import work when executing from repo root or project/ subdir.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from DTO import CoarseLocalizationInfo as _DTOCoarse  # type: ignore
except Exception:
    _DTOCoarse = None
    print("[WARN] DTO.CoarseLocalizationInfo import failed; falling back to local shape.")


# -----------------------------------------------------------------------------
# DTO mirror (used when DTO import fails)
# -----------------------------------------------------------------------------
@dataclass
class _SearchHit:
    rank: int
    path: str
    filename: str
    score: float


@dataclass
class _SearchTiming:
    embed: float
    search: float
    total: float


@dataclass
class _CoarseLocalizationInfo:
    encoder: str
    backend: str
    dim: int
    metric: str
    topk: int
    timing_ms: _SearchTiming
    results: List[_SearchHit]

    def to_dict(self) -> Dict[str, object]:
        return {
            "encoder": self.encoder,
            "backend": self.backend,
            "dim": self.dim,
            "metric": self.metric,
            "topk": self.topk,
            "timing_ms": {
                "embed": self.timing_ms.embed,
                "search": self.timing_ms.search,
                "total": self.timing_ms.total,
            },
            "results": [vars(h) for h in self.results],
        }


# Prefer DTO class when import succeeded; otherwise use mirror.
CoarseLocalizationInfo = _DTOCoarse or _CoarseLocalizationInfo
SearchTiming = getattr(CoarseLocalizationInfo, "SearchTiming", _SearchTiming)
SearchHit = getattr(CoarseLocalizationInfo, "SearchHit", _SearchHit)


# -----------------------------------------------------------------------------
# Config: paths for raw vs variant embeddings
# -----------------------------------------------------------------------------
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
    - For jamshill on Windows, default to D:\\dinov3_exports\\... (reference uses jamshill_data as provided).
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


def _build_index(db_vectors: np.ndarray, use_gpu: bool) -> Tuple[faiss.Index, bool]:
    dim = db_vectors.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    gpu_ms = None
    used_gpu = False
    if use_gpu:
        try:
            tgpu = time.perf_counter()
            index = faiss.index_cpu_to_all_gpus(index)
            gpu_ms = (time.perf_counter() - tgpu) * 1000.0
            used_gpu = True
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
    return index, used_gpu


def _build_roots(weight: str, use_variant: bool, roots: RootConfig = ROOT_CONFIG) -> Tuple[Optional[Path], Optional[Path]]:
    if use_variant:
        return roots.variant_db_base / f"_{weight}", roots.variant_reference_base / f"_{weight}"
    return roots.raw_db_roots.get(weight), roots.raw_reference_roots.get(weight)


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


# -----------------------------------------------------------------------------
# Core search
# -----------------------------------------------------------------------------
def run_single_weight(
    weight: str,
    use_variant: bool,
    k: int,
    out_root: Path,
    use_gpu: bool,
    direction: str,
    show_progress: bool,
    roots: RootConfig = ROOT_CONFIG,
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
        raise ValueError(
            "direction must be one of 'reference-data', 'data-reference', 'reference-reference', 'data-data'"
        )
    direction_key = {
        "reference-data": "reference2db",
        "data-reference": "db2reference",
        "reference-reference": "reference2reference",
        "data-data": "db2db",
    }[direction]

    db_root, reference_root = _build_roots(weight, use_variant, roots)
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
    index, used_gpu = _build_index(index_vecs, use_gpu=use_gpu)
    index_build_ms = (time.perf_counter() - idx_start) * 1000.0

    search_vecs, search_records, search_timing = _load_records(
        search_root, f"{weight}:{search_role}", show_progress=show_progress
    )

    embed_times = [_embed_time_ms(rec.meta_path) for rec in search_records]

    out_dir = _ensure_out_dir(out_root, weight, mode, direction_key)
    dim = int(index_vecs.shape[1])
    backend_label = "faiss-gpu" if used_gpu else "faiss-cpu"
    metric_label = "ip"  # IndexFlatIP + L2 norm -> cosine/IP

    print(
        f"[INFO] Index vectors: {len(index_records)} ({index_role}), "
        f"Search vectors: {len(search_records)} ({search_role}), "
        f"index_build_ms={index_build_ms:.2f}"
    )

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

        top_scores = scores[0].tolist()
        top_idxs = idxs[0].tolist()

        hits: List[SearchHit] = []
        for rank, (db_idx, score) in enumerate(zip(top_idxs, top_scores), start=1):
            rec = index_records[db_idx]
            img_path = rec.image_path or rec.vec_path.as_posix()
            hits.append(
                SearchHit(
                    rank=rank,
                    path=img_path,
                    filename=Path(img_path).name,
                    score=float(score),
                )
            )

        embed_ms = float(embed_times[si - 1]) if si - 1 < len(embed_times) else 0.0
        total_ms = embed_ms + search_ms
        timing = SearchTiming(embed=embed_ms, search=search_ms, total=total_ms)

        payload = CoarseLocalizationInfo(
            encoder=weight,
            backend=backend_label,
            dim=dim,
            metric=metric_label,
            topk=k_eff,
            timing_ms=timing,
            results=hits,
        )

        if hasattr(payload, "to_dict"):
            body = payload.to_dict()  # type: ignore[attr-defined]
        elif hasattr(payload, "to_json"):
            body = json.loads(payload.to_json())  # type: ignore[attr-defined]
        else:
            body = {
                "encoder": payload.encoder,
                "backend": payload.backend,
                "dim": payload.dim,
                "metric": payload.metric,
                "topk": payload.topk,
                "timing_ms": {
                    "embed": timing.embed,
                    "search": timing.search,
                    "total": timing.total,
                },
                "results": [vars(h) for h in hits],
            }

        out_path = out_dir / f"{srec.vec_path.stem}_{direction_key}_top{k_eff}_redis.json"
        out_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

        if not using_tqdm and (si % 25 == 0 or si == len(search_records)):
            print(f"[INFO] {weight} {mode} {direction}: processed {si}/{len(search_records)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="FAISS TopK retrieval emitting CoarseLocalizationInfo JSON rows (one per embedding).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-w",
        "--weights",
        nargs="+",
        default=list(RAW_DB_ROOTS.keys()),
        help="Weight keys to process (default: all known raw weights).",
    )
    parser.add_argument(
        "--dataset",
        choices=["shinsung", "jamshill"],
        default=DEFAULT_DATASET,
        help="Select which dataset roots to use (env RAW_EMBED_ROOT/RAW_REFERENCE_ROOT override).",
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
        "-o",
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=(f"Output directory for retrieval JSON (default: {DEFAULT_OUT_ROOT}; override with FAISS_OUT_ROOT env)"),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars during scan/load/search.",
    )
    parser.add_argument(
        "--list-cli",
        action="store_true",
        help="Print available CLI options and exit.",
    )
    args = parser.parse_args()
    return parser, args


def main() -> None:
    parser, args = parse_args()
    if args.list_cli:
        parser.print_help()
        return
    root_config = _build_root_config(args.dataset)
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

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
                show_progress=bool(args.progress),
                roots=root_config,
            )
        except Exception as exc:
            print(f"[ERROR] weight={weight} failed: {exc}")


if __name__ == "__main__":
    main()
