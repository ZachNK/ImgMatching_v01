"""
FAISS TopK retrieval with selectable index types/params.

This script intentionally leaves project/faiss_4Redis.py unchanged and provides
an extended CLI for index benchmarking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from imatch.faiss_config import (
    DEFAULT_DATASET,
    DEFAULT_OUT_ROOT,
    DIRECTION_CHOICES,
    LEGACY_DIRECTION_ALIASES,
    RAW_DB_ROOTS,
    ROOT_CONFIG,
    RootConfig,
    _build_root_config,
    _build_roots,
)
from imatch.faiss_index import IndexSpec, _build_index_with_spec
from imatch.faiss_io import _embed_time_ms, _load_records
from imatch.faiss_runtime import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from DTO import CoarseLocalizationInfo as _DTOCoarse  # type: ignore
except Exception:
    _DTOCoarse = None
    print("[WARN] DTO.CoarseLocalizationInfo import failed; falling back to local shape.")


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


CoarseLocalizationInfo = _DTOCoarse or _CoarseLocalizationInfo
SearchTiming = getattr(CoarseLocalizationInfo, "SearchTiming", _SearchTiming)
SearchHit = getattr(CoarseLocalizationInfo, "SearchHit", _SearchHit)


def _index_tag(spec: IndexSpec, normalize_vectors: bool) -> str:
    parts = [spec.index_type.replace("-", "")]
    if spec.index_type in ("ivf-flat", "ivf-pq"):
        parts.append(f"nlist{spec.nlist}")
        parts.append(f"nprobe{spec.nprobe}")
    if spec.index_type == "ivf-pq":
        parts.append(f"m{spec.pq_m}")
        parts.append(f"nbits{spec.pq_nbits}")
    parts.append(spec.metric.lower())
    if not normalize_vectors:
        parts.append("nonorm")
    return "_".join(parts)


def _compact_index_info(info: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in info.items() if v is not None}


def run_single_weight(
    weight: str,
    use_variant: bool,
    k: int,
    out_root: Path,
    use_gpu: bool,
    direction: str,
    show_progress: bool,
    index_spec: IndexSpec,
    normalize_vectors: bool,
    roots: RootConfig = ROOT_CONFIG,
) -> None:
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
    else:
        index_root = search_root = db_root
        index_role = search_role = "db"

    index_tag = _index_tag(index_spec, normalize_vectors)
    print(f"\n[INFO] Processing weight={weight} mode={mode} direction={direction} index={index_tag}")

    load_start = time.perf_counter()
    index_vecs, index_records, _ = _load_records(
        index_root, f"{weight}:{index_role}", show_progress=show_progress, normalize=normalize_vectors
    )
    load_ms = (time.perf_counter() - load_start) * 1000.0

    build_start = time.perf_counter()
    index, used_gpu, index_meta = _build_index_with_spec(index_vecs, use_gpu=use_gpu, spec=index_spec)
    index_build_ms = (time.perf_counter() - build_start) * 1000.0

    search_vecs, search_records, _ = _load_records(
        search_root, f"{weight}:{search_role}", show_progress=show_progress, normalize=normalize_vectors
    )
    embed_times = [_embed_time_ms(rec.meta_path) for rec in search_records]

    out_dir = out_root / f"{weight}_{mode}_{direction_key}__{index_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dim = int(index_vecs.shape[1])
    backend_label = "faiss-gpu" if used_gpu else "faiss-cpu"
    metric_label = index_spec.metric.lower()
    compact_index_meta = _compact_index_info(index_meta)

    print(
        f"[INFO] Index vectors: {len(index_records)} ({index_role}), "
        f"Search vectors: {len(search_records)} ({search_role}), "
        f"index_build_ms={index_build_ms:.2f}, index_load_ms={load_ms:.2f}"
    )

    iter_records = enumerate(search_records, start=1)
    using_tqdm = bool(tqdm) and bool(show_progress)
    if show_progress and not tqdm:
        print("[INFO] tqdm is not installed; progress bar disabled. pip install tqdm to enable.")
    if using_tqdm:
        iter_records = tqdm(iter_records, total=len(search_records), desc=f"{weight}-{mode}-{direction}", unit="vec")

    for si, srec in iter_records:
        qvec = np.asarray(search_vecs[si - 1 : si], dtype=np.float32)
        search_start = time.perf_counter()
        k_eff = min(k, len(index_records))
        scores, idxs = index.search(qvec, k_eff)
        search_ms = (time.perf_counter() - search_start) * 1000.0

        top_scores = scores[0].tolist()
        top_idxs = idxs[0].tolist()

        hits: List[SearchHit] = []
        for rank, (db_idx, score) in enumerate(zip(top_idxs, top_scores), start=1):
            rec = index_records[db_idx]
            vector_path = rec.vec_path.as_posix()
            image_path = rec.image_path
            hits.append(
                SearchHit(
                    rank=rank,
                    path=vector_path,
                    filename=image_path or Path(vector_path).name,
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

        body["index_build_ms"] = index_build_ms
        body["index_tag"] = index_tag
        body["index"] = compact_index_meta
        body["normalize"] = bool(normalize_vectors)

        out_path = out_dir / f"{srec.vec_path.stem}_{direction_key}_top{k_eff}_redis.json"
        out_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

        if not using_tqdm and (si % 25 == 0 or si == len(search_records)):
            print(f"[INFO] {weight} {mode} {direction}: processed {si}/{len(search_records)}")


def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="FAISS TopK retrieval (index-selectable) emitting CoarseLocalizationInfo JSON rows.",
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
    parser.add_argument("--k", type=int, default=10, help="TopK size.")
    parser.add_argument("--gpu", action="store_true", help="Use faiss-gpu (index_cpu_to_all_gpus).")
    parser.add_argument(
        "--index",
        choices=["flat", "ivf-flat", "ivf-pq"],
        default="flat",
        help="Index type.",
    )
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip", help="Distance/similarity metric.")
    parser.add_argument("--nlist", type=int, default=4096, help="IVF clusters.")
    parser.add_argument("--nprobe", type=int, default=16, help="IVF probe count at search time.")
    parser.add_argument("--pq-m", type=int, default=16, help="IVFPQ sub-vector count (m).")
    parser.add_argument("--pq-nbits", type=int, default=8, help="IVFPQ bits per sub-vector.")
    parser.add_argument("--train-size", type=int, default=100000, help="Training sample size for IVF/IVFPQ.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before indexing/search.",
    )
    parser.add_argument(
        "-m",
        "--match",
        choices=[*DIRECTION_CHOICES, *LEGACY_DIRECTION_ALIASES.keys()],
        default="data-reference",
        help=(
            "reference-data: reference embeddings search DB embeddings; "
            "data-reference: DB embeddings search reference embeddings; "
            "reference-reference: reference embeddings search reference embeddings; "
            "data-data: DB embeddings search DB embeddings."
        ),
    )
    parser.add_argument(
        "-o",
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=(f"Output directory for retrieval JSON (default: {DEFAULT_OUT_ROOT}; override with FAISS_OUT_ROOT env)."),
    )
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars during scan/load/search.")
    parser.add_argument("--list-cli", action="store_true", help="Print available CLI options and exit.")
    args = parser.parse_args()
    return parser, args


def main() -> None:
    parser, args = parse_args()
    if args.list_cli:
        parser.print_help()
        return

    if args.index == "flat" and args.metric not in {"ip", "l2"}:
        raise ValueError("flat index supports only ip/l2 metric")
    if args.index in {"ivf-flat", "ivf-pq"} and args.nlist <= 0:
        raise ValueError("nlist must be > 0")
    if args.index in {"ivf-flat", "ivf-pq"} and args.nprobe <= 0:
        raise ValueError("nprobe must be > 0")
    if args.index == "ivf-pq" and args.pq_m <= 0:
        raise ValueError("pq-m must be > 0")
    if args.index == "ivf-pq" and args.pq_nbits <= 0:
        raise ValueError("pq-nbits must be > 0")

    root_config = _build_root_config(args.dataset)
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    index_spec = IndexSpec(
        index_type=args.index,
        metric=args.metric,
        nlist=int(args.nlist),
        nprobe=int(args.nprobe),
        pq_m=int(args.pq_m),
        pq_nbits=int(args.pq_nbits),
        train_size=int(args.train_size),
    )
    normalize_vectors = (args.metric.lower() == "ip") and (not bool(args.no_normalize))

    for weight in args.weights:
        try:
            run_single_weight(
                weight=weight,
                use_variant=bool(args.variant),
                k=int(args.k),
                out_root=out_root,
                use_gpu=bool(args.gpu),
                direction=args.match,
                show_progress=bool(args.progress),
                index_spec=index_spec,
                normalize_vectors=normalize_vectors,
                roots=root_config,
            )
        except Exception as exc:
            print(f"[ERROR] weight={weight} failed: {exc}")


if __name__ == "__main__":
    main()

