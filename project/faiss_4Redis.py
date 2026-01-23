"""
Run FAISS TopK search and emit per-embedding JSON rows shaped for the
CoarseLocalizationInfo DTO (Redis interface).

- Mirrors project/match_faiss_gpu.py search behavior (IndexFlatIP + L2 norm).
- Saves one JSON per search embedding (no per-weight aggregation).
- Output schema matches DTO.CoarseLocalizationInfo with extra index timing:
    encoder, backend, dim, metric, topk, timing_ms{embed,search,total}, index_build_ms,
    results[{rank,path,filename,score}]

e.g.
1) 경로 환경변수 세팅
root@3a7731a832ee:/workspace/project# export RAW_EMBED_ROOT=/exports/dinov3_embeds/jamshill_flight
root@3a7731a832ee:/workspace/project# export RAW_REFERENCE_ROOT=/exports/dinov3_embeds/jamshill_reference
root@3a7731a832ee:/workspace/project# export FAISS_OUT_ROOT=/exports/dinov3_faiss_match   # 결과 저장 위치

2) 실행 예시 (TopK=10, data→reference 검색)
root@3a7731a832ee:/workspace/project# python faiss_4Redis.py --dataset jamshill --weights vitb16 vith16+ vitl16 vitl16sat vits16 vits16+ --k 10 --match data-reference --progress --gpu
# GPU를 쓰고 싶으면 --gpu 추가

3) 결과 위치/파일명
출력 루트: FAISS_OUT_ROOT (위 예시: D:\ImgMatching_export\dinov3_faiss_match)
하위 폴더: <weight>_raw_db2reference
파일명: <GlobalToken파일명>_db2reference_top10_redis.json
각 JSON은 해당 쿼리(GlobalToken)별로 TopK 결과와 점수·원본 경로를 담습니다.
이렇게 하면 jamshill_flight의 GlobalToken들이 jamshill_reference 전체를 탐색해 Top 10 후보가 JSON으로 저장됨.
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
from imatch.faiss_index import _build_index
from imatch.faiss_io import _embed_time_ms, _ensure_out_dir, _load_records
from imatch.faiss_runtime import tqdm


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

    load_start = time.perf_counter()
    index_vecs, index_records, index_timing = _load_records(
        index_root, f"{weight}:{index_role}", show_progress=show_progress
    )
    load_ms = (time.perf_counter() - load_start) * 1000.0

    build_start = time.perf_counter()
    index, used_gpu = _build_index(index_vecs, use_gpu=use_gpu)
    index_build_ms = (time.perf_counter() - build_start) * 1000.0

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
        f"index_build_ms={index_build_ms:.2f}, index_load_ms={load_ms:.2f}"
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
