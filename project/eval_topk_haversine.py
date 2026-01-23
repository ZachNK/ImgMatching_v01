"""
python eval_topk_haversine.py --faiss-root /exports/dinov3_faiss_match --query-index /exports/jamshill_flight_index.json --reference-index /exports/jamshill_reference_index.json --topk 10 --haversine-hit-m 10 20 30 40 50 60 70 80 90 100

"""


from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            if enc == "utf-8" and "BOM" in str(exc):
                continue
            raise
    raise ValueError(f"Failed to parse JSON: {path}")


def _numeric_tokens_after_backend(name: str) -> List[str]:
    tokens = Path(name).stem.split("_")
    idx = None
    for key in ("LVD", "SAT"):
        if key in tokens:
            idx = tokens.index(key)
            break
    if idx is None:
        raise ValueError("LVD/SAT token not found")
    numeric: List[str] = []
    for tok in tokens[idx + 1 :]:
        if tok.isdigit():
            numeric.append(tok)
        else:
            break
    if not numeric:
        raise ValueError("No numeric tokens after backend")
    return numeric


def _infer_query_imageref(json_path: Path) -> Optional[str]:
    try:
        tokens = _numeric_tokens_after_backend(json_path.name)
    except ValueError:
        return None
    if len(tokens) < 2:
        return None
    capture = tokens[0]
    frame = f"{int(tokens[1]):05d}"
    return f"{capture}_{frame}.jpg"


def _infer_hit_imageref(raw_name: str) -> Optional[str]:
    if not raw_name:
        return None
    base = Path(raw_name).name
    suffix = base.lower().split(".")[-1] if "." in base else ""
    if suffix in ("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"):
        return base
    try:
        tokens = _numeric_tokens_after_backend(base)
        return "_".join(tokens) + ".jpg"
    except ValueError:
        pass
    stem = Path(base).stem
    if stem.replace("_", "").isdigit():
        return stem + ".jpg"
    return None


def _resolve_imageref(ref: Optional[str], index: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not ref:
        return None
    if ref in index:
        return ref
    stem = Path(ref).stem
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        padded = f"{int(parts[-1]):05d}"
        candidate = "_".join(parts[:-1] + [padded]) + ".jpg"
        if candidate in index:
            return candidate
    return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _haversine_distance(q: Dict[str, Any], r: Dict[str, Any]) -> Optional[float]:
    lat1 = q.get("latitude")
    lon1 = q.get("longitude")
    lat2 = r.get("latitude")
    lon2 = r.get("longitude")
    if None in (lat1, lon1, lat2, lon2):
        return None
    return _haversine_m(float(lat1), float(lon1), float(lat2), float(lon2))


def _xy_distance(q: Dict[str, Any], r: Dict[str, Any]) -> Optional[float]:
    x1 = q.get("x")
    y1 = q.get("y")
    x2 = r.get("x")
    y2 = r.get("y")
    if None in (x1, y1, x2, y2):
        return None
    dx = float(x1) - float(x2)
    dy = float(y1) - float(y2)
    return math.hypot(dx, dy)


def _min_distance_to_refs(
    query: Dict[str, Any],
    refs: Iterable[Tuple[str, Dict[str, Any]]],
    mode: str,
) -> Tuple[Optional[str], Optional[float]]:
    best_ref: Optional[str] = None
    best_dist: Optional[float] = None
    for ref_name, ref_meta in refs:
        dist = _haversine_distance(query, ref_meta) if mode == "haversine" else _xy_distance(query, ref_meta)
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_ref = ref_name
    return best_ref, best_dist


def _safe_mean(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def _safe_median(values: List[float]) -> Optional[float]:
    return median(values) if values else None


def _threshold_key(threshold_m: float) -> str:
    if float(threshold_m).is_integer():
        return f"{int(threshold_m)}m"
    return f"{str(threshold_m).replace('.', 'p')}m"


def _iter_result_files(root: Path, pattern: str) -> List[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def _summarize_group(details: List[Dict[str, Any]], haversine_hit_keys: List[str]) -> Dict[str, Any]:
    haversine_recall_flags = [d["haversine_recall"] for d in details if d["haversine_recall"] is not None]
    xy_recall_flags = [d["xy_recall"] for d in details if d["xy_recall"] is not None]

    haversine_ranks = [d["haversine_rank"] for d in details if d["haversine_rank"] is not None]
    xy_ranks = [d["xy_rank"] for d in details if d["xy_rank"] is not None]

    haversine_top1 = [d["haversine_top1_m"] for d in details if d["haversine_top1_m"] is not None]
    xy_top1 = [d["xy_top1_m"] for d in details if d["xy_top1_m"] is not None]

    haversine_min = [d["haversine_min_topk_m"] for d in details if d["haversine_min_topk_m"] is not None]
    xy_min = [d["xy_min_topk_m"] for d in details if d["xy_min_topk_m"] is not None]

    embed_ms = [d["embed_ms"] for d in details if d.get("embed_ms") is not None]
    search_ms = [d["search_ms"] for d in details if d.get("search_ms") is not None]
    total_ms = [d["total_ms"] for d in details if d.get("total_ms") is not None]
    index_build_ms = [d["index_build_ms"] for d in details if d.get("index_build_ms") is not None]

    extra: Dict[str, Any] = {}
    for key in haversine_hit_keys:
        values = [d[key] for d in details if d.get(key) is not None]
        extra[key] = _safe_mean(values)

    return {
        "queries": len(details),
        "haversine_recall_at_k": _safe_mean(haversine_recall_flags),
        "xy_recall_at_k": _safe_mean(xy_recall_flags),
        "haversine_avg_rank": _safe_mean(haversine_ranks),
        "xy_avg_rank": _safe_mean(xy_ranks),
        "haversine_top1_mean_m": _safe_mean(haversine_top1),
        "haversine_top1_median_m": _safe_median(haversine_top1),
        "xy_top1_mean_m": _safe_mean(xy_top1),
        "xy_top1_median_m": _safe_median(xy_top1),
        "haversine_min_topk_mean_m": _safe_mean(haversine_min),
        "haversine_min_topk_median_m": _safe_median(haversine_min),
        "xy_min_topk_mean_m": _safe_mean(xy_min),
        "xy_min_topk_median_m": _safe_median(xy_min),
        "embed_ms_mean": _safe_mean(embed_ms),
        "embed_ms_median": _safe_median(embed_ms),
        "search_ms_mean": _safe_mean(search_ms),
        "search_ms_median": _safe_median(search_ms),
        "total_ms_mean": _safe_mean(total_ms),
        "total_ms_median": _safe_median(total_ms),
        "index_build_ms_mean": _safe_mean(index_build_ms),
        "index_build_ms_median": _safe_median(index_build_ms),
        **extra,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: "" if row.get(col) is None else row.get(col) for col in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAISS TopK results using haversine and x/y distances.",
    )
    parser.add_argument(
        "--faiss-root",
        type=Path,
        default=Path("/exports/dinov3_faiss_match"),
        help="Root folder containing faiss_4Redis JSON outputs.",
    )
    parser.add_argument(
        "--query-index",
        type=Path,
        default=Path("/exports/dinov3_faiss_match/jamshill_flight_index.json"),
        help="Imageref index JSON for query (flight) images.",
    )
    parser.add_argument(
        "--reference-index",
        type=Path,
        default=Path("/exports/dinov3_faiss_match/jamshill_reference_index.json"),
        help="Imageref index JSON for reference images.",
    )
    parser.add_argument("--topk", type=int, default=10, help="TopK threshold for evaluation.")
    parser.add_argument(
        "--haversine-hit-m",
        type=float,
        nargs="+",
        default=[30.0],
        help="Haversine Hit@K distance thresholds in meters (e.g. 10 30 50).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_top*_redis.json",
        help="Glob pattern for result JSON files under faiss-root.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("/exports/evaluation/eval_haversine_summary.json"),
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--out-details",
        type=Path,
        default=Path("/exports/evaluation/eval_haversine_details.json"),
        help="Per-query detail JSON output path.",
    )
    parser.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("/exports/evaluation/eval_haversine_summary.csv"),
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--out-details-csv",
        type=Path,
        default=Path("/exports/evaluation/eval_haversine_details.csv"),
        help="Per-query detail CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.faiss_root.exists():
        raise FileNotFoundError(f"faiss root not found: {args.faiss_root}")
    if not args.query_index.exists():
        raise FileNotFoundError(f"query index not found: {args.query_index}")
    if not args.reference_index.exists():
        raise FileNotFoundError(f"reference index not found: {args.reference_index}")

    query_index: Dict[str, Dict[str, Any]] = _read_json(args.query_index)
    ref_index: Dict[str, Dict[str, Any]] = _read_json(args.reference_index)
    ref_items = list(ref_index.items())
    haversine_hit_thresholds = [float(t) for t in args.haversine_hit_m]
    haversine_hit_keys = [f"haversine_hit_{_threshold_key(t)}" for t in haversine_hit_thresholds]

    result_files = _iter_result_files(args.faiss_root, args.pattern)
    if not result_files:
        raise FileNotFoundError(f"No result JSON files found under {args.faiss_root} with pattern {args.pattern}")

    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for json_path in result_files:
        payload = _read_json(json_path)
        hits = payload.get("results")
        if not isinstance(hits, list) or not hits:
            continue
        timing = payload.get("timing_ms") if isinstance(payload.get("timing_ms"), dict) else {}
        embed_ms = timing.get("embed") if isinstance(timing, dict) else None
        search_ms = timing.get("search") if isinstance(timing, dict) else None
        total_ms = timing.get("total") if isinstance(timing, dict) else None
        index_build_ms = payload.get("index_build_ms")

        query_imageref = _infer_query_imageref(json_path)
        query_imageref = _resolve_imageref(query_imageref, query_index)
        if not query_imageref:
            print(f"[WARN] query imageref not resolved for {json_path.name}")
            continue
        query_meta = query_index.get(query_imageref)
        if not isinstance(query_meta, dict):
            print(f"[WARN] query metadata missing for {query_imageref}")
            continue

        gt_haversine_ref, gt_haversine_dist = _min_distance_to_refs(query_meta, ref_items, mode="haversine")
        gt_xy_ref, gt_xy_dist = _min_distance_to_refs(query_meta, ref_items, mode="xy")

        top_hits = sorted(hits, key=lambda h: int(h.get("rank", 1)))[: args.topk]
        hit_imagerefs: List[Optional[str]] = []
        hit_scores: List[Optional[float]] = []
        for hit in top_hits:
            raw = hit.get("filename") or hit.get("path") or ""
            imageref = _infer_hit_imageref(str(raw))
            imageref = _resolve_imageref(imageref, ref_index)
            hit_imagerefs.append(imageref)
            score = hit.get("score")
            hit_scores.append(float(score) if isinstance(score, (int, float)) else None)

        def _rank_of(target: Optional[str]) -> Optional[int]:
            if not target:
                return None
            for idx, name in enumerate(hit_imagerefs, start=1):
                if name == target:
                    return idx
            return None

        haversine_dists: List[float] = []
        xy_dists: List[float] = []
        for name in hit_imagerefs:
            if not name:
                continue
            ref_meta = ref_index.get(name)
            if not isinstance(ref_meta, dict):
                continue
            haversine = _haversine_distance(query_meta, ref_meta)
            xy = _xy_distance(query_meta, ref_meta)
            if haversine is not None:
                haversine_dists.append(haversine)
            if xy is not None:
                xy_dists.append(xy)

        haversine_top1 = None
        xy_top1 = None
        if hit_imagerefs:
            top1_name = hit_imagerefs[0]
            if top1_name and top1_name in ref_index:
                haversine_top1 = _haversine_distance(query_meta, ref_index[top1_name])
                xy_top1 = _xy_distance(query_meta, ref_index[top1_name])

        detail = {
            "query": query_imageref,
            "group": json_path.parent.name,
            "json": str(json_path),
            "topk": args.topk,
            "gt_haversine": gt_haversine_ref,
            "gt_haversine_m": gt_haversine_dist,
            "gt_xy": gt_xy_ref,
            "gt_xy_m": gt_xy_dist,
            "haversine_rank": _rank_of(gt_haversine_ref),
            "xy_rank": _rank_of(gt_xy_ref),
            "haversine_recall": 1 if _rank_of(gt_haversine_ref) is not None else 0 if gt_haversine_ref else None,
            "xy_recall": 1 if _rank_of(gt_xy_ref) is not None else 0 if gt_xy_ref else None,
            "haversine_top1_m": haversine_top1,
            "xy_top1_m": xy_top1,
            "haversine_min_topk_m": min(haversine_dists) if haversine_dists else None,
            "xy_min_topk_m": min(xy_dists) if xy_dists else None,
            "top1_score": hit_scores[0] if hit_scores else None,
            "embed_ms": float(embed_ms) if isinstance(embed_ms, (int, float)) else None,
            "search_ms": float(search_ms) if isinstance(search_ms, (int, float)) else None,
            "total_ms": float(total_ms) if isinstance(total_ms, (int, float)) else None,
            "index_build_ms": float(index_build_ms) if isinstance(index_build_ms, (int, float)) else None,
        }
        haversine_min_val = detail["haversine_min_topk_m"]
        for th, key in zip(haversine_hit_thresholds, haversine_hit_keys):
            if haversine_min_val is None:
                detail[key] = None
            else:
                detail[key] = 1 if haversine_min_val <= th else 0

        grouped.setdefault(json_path.parent.name, []).append(detail)

    summaries: Dict[str, Any] = {
        group: _summarize_group(items, haversine_hit_keys) for group, items in grouped.items()
    }
    output = {
        "summary": summaries,
        "details": grouped,
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_details.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(output["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_details.write_text(json.dumps(output["details"], ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows: List[Dict[str, Any]] = []
    for group, stats in summaries.items():
        row = {"group": group}
        row.update(stats)
        summary_rows.append(row)

    detail_rows: List[Dict[str, Any]] = []
    for group, items in grouped.items():
        for item in items:
            row = dict(item)
            row["group"] = group
            detail_rows.append(row)

    summary_cols = [
        "group",
        "queries",
        "haversine_recall_at_k",
        "xy_recall_at_k",
        "haversine_avg_rank",
        "xy_avg_rank",
        "haversine_top1_mean_m",
        "haversine_top1_median_m",
        "xy_top1_mean_m",
        "xy_top1_median_m",
        "haversine_min_topk_mean_m",
        "haversine_min_topk_median_m",
        "xy_min_topk_mean_m",
        "xy_min_topk_median_m",
        "embed_ms_mean",
        "embed_ms_median",
        "search_ms_mean",
        "search_ms_median",
        "total_ms_mean",
        "total_ms_median",
        "index_build_ms_mean",
        "index_build_ms_median",
    ]
    summary_cols += haversine_hit_keys
    detail_cols = [
        "group",
        "query",
        "json",
        "topk",
        "gt_haversine",
        "gt_haversine_m",
        "gt_xy",
        "gt_xy_m",
        "haversine_rank",
        "xy_rank",
        "haversine_recall",
        "xy_recall",
        "haversine_top1_m",
        "xy_top1_m",
        "haversine_min_topk_m",
        "xy_min_topk_m",
        "top1_score",
        "embed_ms",
        "search_ms",
        "total_ms",
        "index_build_ms",
    ]
    detail_cols += haversine_hit_keys

    _write_csv(args.out_summary_csv, summary_rows, summary_cols)
    _write_csv(args.out_details_csv, detail_rows, detail_cols)

    print(f"[INFO] summary -> {args.out_summary}")
    print(f"[INFO] details -> {args.out_details}")
    print(f"[INFO] summary csv -> {args.out_summary_csv}")
    print(f"[INFO] details csv -> {args.out_details_csv}")


if __name__ == "__main__":
    main()
