"""
Display TopK results with per-hit geo/xyz distances and GT markers.

실행 예시: 
쿼리 이미지 (jamshill_data_flight 251124160703_00045.jpg) 로, 참조이미지 (e.g jamshill_data_reference)에 Top10까지 추출.
faiss_4Redis.py로 추출한 json파일을 기반으로 score를 보여주고, 해당 이미지를 디스플레이.
python verify_display_topk.py --query 251124160703_00045 --match data-reference --model vitb16 vith16+ vitl16 vitl16sat vits16 vits16+ --show-gps

"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Hardcoded configuration (CLI overrides where available)
# -----------------------------------------------------------------------------
QUERY_ROOT = Path(r"D:/Datasets/01_01_jamshill_data_flight")
TOPK_ROOT = Path(r"D:/Datasets/01_03_jamshill_data_reference")
FAISS_ROOT = Path(r"D:/ImgMatching_export/dinov3_faiss_match")
FAISS_ROOT_FALLBACK = Path("/exports/dinov3_faiss_match")
DEFAULT_QUERY = "251124160703_00130"
TOP_K: int = 10
SHOW_SCORE = True

for _candidate in (
    QUERY_ROOT,
    Path("/opt/datasets/01_01_jamshill_data_flight"),
):
    if _candidate.exists():
        QUERY_ROOT = _candidate
        break

for _candidate in (
    TOPK_ROOT,
    Path("/opt/datasets/01_03_jamshill_data_reference"),
):
    if _candidate.exists():
        TOPK_ROOT = _candidate
        break

MODEL_ENCODER = {
    "vitb16": "vitb16",
    "vith16+": "vith16plus",
    "vitl16": "vitl16",
    "vitl16sat": "vitl16",
    "vits16": "vits16",
    "vits16+": "vits16plus",
}
MODEL_KEYS = sorted(MODEL_ENCODER.keys())

MATCH_CHOICES = {
    "data-reference": "db2reference",
    "reference-data": "reference2db",
    "reference-reference": "reference2reference",
    "data-data": "db2db",
}

_METADATA_CACHE: Dict[Path, Dict[str, Dict[str, object]]] = {}
_MISSING_META_REPORTED: set[Path] = set()


def _numeric_tokens_after_backend(name: str) -> List[str]:
    tokens = Path(name).stem.split("_")
    for key in ("LVD", "SAT"):
        if key in tokens:
            idx = tokens.index(key)
            break
    else:
        raise ValueError(f"LVD/SAT token not found in {name}")

    numeric: List[str] = []
    for tok in tokens[idx + 1 :]:
        if re.fullmatch(r"\d+", tok):
            numeric.append(tok)
        else:
            break
    if not numeric:
        raise ValueError(f"No numeric tokens after backend in {name}")
    return numeric


def _numeric_tokens_from_image_path(name: str) -> List[str]:
    p = Path(name)
    stem_tokens = p.stem.split("_")
    numeric = [tok for tok in stem_tokens if re.fullmatch(r"\d+", tok)]
    if len(numeric) >= 2:
        return numeric
    folder_tokens = p.parent.name.split("_")
    folder_numeric = [tok for tok in folder_tokens if re.fullmatch(r"\d+", tok)]
    if folder_numeric:
        tail = stem_tokens[-1] if stem_tokens else ""
        if re.fullmatch(r"\d+", tail):
            return folder_numeric + [tail]
    raise ValueError(f"No numeric tokens in image path {name}")


def _numeric_tokens_any(name: str) -> List[str]:
    try:
        return _numeric_tokens_after_backend(name)
    except ValueError:
        return _numeric_tokens_from_image_path(name)


def _find_with_ext(base: Path, exts: Sequence[str] = ("jpg", "jpeg", "png")) -> Optional[Path]:
    for ext in exts:
        candidate = base.with_suffix(f".{ext}")
        if candidate.exists():
            return candidate
    return None


def _resolve_faiss_root(cli_root: Optional[Path]) -> Path:
    if cli_root is not None:
        return cli_root
    if FAISS_ROOT.exists():
        return FAISS_ROOT
    return FAISS_ROOT_FALLBACK


def _split_query_id(query_id: str) -> Tuple[str, str]:
    stem = Path(query_id).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Query id must be like capture_frame (got: {query_id})")
    capture = parts[0]
    frame = parts[-1]
    if frame.isdigit():
        frame = f"{int(frame):05d}"
    return capture, frame


def resolve_query_image_from_id(query_id: str, query_root: Path) -> Path:
    candidate = Path(query_id)
    if candidate.exists():
        return candidate
    capture, frame = _split_query_id(query_id)
    img_base = query_root / capture / f"{capture}_{frame}"
    found = _find_with_ext(img_base)
    return found or img_base.with_suffix(".jpg")


def _model_backend(model: str) -> str:
    return "SAT" if "sat" in model.lower() else "LVD"


def _model_encoder(model: str) -> str:
    return MODEL_ENCODER.get(model, model)


def _find_model_dir(faiss_root: Path, model: str, direction_key: str) -> Optional[Path]:
    pattern = f"{model}_*_{direction_key}"
    candidates = [p for p in faiss_root.rglob(pattern) if p.is_dir()]
    if not candidates:
        return None
    raw_candidates = [p for p in candidates if "_raw_" in p.name]
    chosen = sorted(raw_candidates or candidates, key=lambda p: (len(p.parts), str(p)))[0]
    return chosen


def _find_faiss_json(
    model_dir: Path,
    encoder: str,
    backend: str,
    capture: str,
    frame: str,
    direction_key: str,
    top_k: int,
) -> Optional[Path]:
    frame_candidates = {frame}
    if frame.isdigit():
        frame_candidates.add(f"{int(frame):05d}")
    matches: List[Path] = []
    for path in model_dir.rglob("*.json"):
        name = path.name
        if "GlobalToken" not in name:
            continue
        if f"_{encoder}_" not in name:
            continue
        if f"_{backend}_" not in name:
            continue
        if f"_{capture}_" not in name:
            continue
        if not any(f"_{fr}_" in name for fr in frame_candidates):
            continue
        if f"_{direction_key}_top" not in name:
            continue
        matches.append(path)
    if not matches:
        return None
    topk_matches = [p for p in matches if f"_top{top_k}_" in p.name]
    return sorted(topk_matches or matches, key=lambda p: str(p))[0]


def resolve_query_image(json_path: Path, query_root: Path) -> Path:
    tokens = _numeric_tokens_after_backend(json_path.name)
    if len(tokens) < 2:
        raise ValueError("Query tokens missing capture/frame.")
    capture = tokens[0]
    frame = f"{int(tokens[1]):05d}"
    folder = query_root / capture
    img_base = folder / f"{capture}_{frame}"
    found = _find_with_ext(img_base)
    return found or img_base.with_suffix(".jpg")


def resolve_hit_image(filename: str, topk_root: Path) -> Path:
    tokens = _numeric_tokens_any(filename)
    if len(tokens) < 2:
        raise ValueError("Hit tokens missing folder/frame.")
    frame = f"{int(tokens[-1]):05d}"
    folder = "_".join(tokens[:-1])
    img_base = topk_root / folder / f"{folder}_{frame}"
    found = _find_with_ext(img_base)
    return found or img_base.with_suffix(".jpg")


def _label_json_path(img_path: Path, dataset_root: Path) -> Path:
    folder = img_path.parent
    base = folder.name
    root_name = dataset_root.name.lower()
    if "reference" in root_name:
        candidates = [folder / f"{base}.json"]
    else:
        candidates = [
            folder / f"{base}_label.json",
            folder / f"{base}_labels.json",
            folder / f"{base}.json",
        ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _lookup_metadata(img_path: Path, dataset_root: Path) -> Optional[Dict[str, object]]:
    json_path = _label_json_path(img_path, dataset_root)
    mapping = _METADATA_CACHE.get(json_path)
    if mapping is None:
        if not json_path.exists():
            if json_path not in _MISSING_META_REPORTED:
                print(f"[WARN] metadata json not found: {json_path}")
                _MISSING_META_REPORTED.add(json_path)
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                if json_path not in _MISSING_META_REPORTED:
                    print(f"[WARN] metadata json parse failed: {json_path} ({exc})")
                    _MISSING_META_REPORTED.add(json_path)
                return None
        images = data.get("images")
        mapping = {}
        if isinstance(images, list):
            def _add_key(key: str, item: Dict[str, object]) -> None:
                mapping[key] = item
                mapping[key.lower()] = item

            for item in images:
                if not isinstance(item, dict):
                    continue
                ref = item.get("imageref")
                if isinstance(ref, str):
                    _add_key(ref, item)
                    ref_path = Path(ref)
                    _add_key(ref_path.name, item)
                    _add_key(ref_path.stem, item)
                    _add_key(ref_path.stem.lower(), item)
        _METADATA_CACHE[json_path] = mapping
    name = img_path.name
    stem = img_path.stem
    result = mapping.get(name) or mapping.get(name.lower()) or mapping.get(stem) or mapping.get(stem.lower())
    if result:
        return result
    target = name.lower()
    for key, item in mapping.items():
        try:
            if Path(key).name.lower() == target:
                return item
        except Exception:
            if isinstance(key, str) and key.lower() == target:
                return item
    return None


def _num(meta: Dict[str, object], key: str) -> Optional[float]:
    val = meta.get(key) if isinstance(meta, dict) else None
    return float(val) if isinstance(val, (int, float)) else None


def _pose_from_meta(meta: Optional[Dict[str, object]]) -> Dict[str, Optional[float]]:
    if not isinstance(meta, dict):
        return {}
    pos = meta.get("position") if isinstance(meta.get("position"), dict) else {}
    att = meta.get("attitude") if isinstance(meta.get("attitude"), dict) else {}
    return {
        "x": _num(pos, "x"),
        "y": _num(pos, "y"),
        "z": _num(pos, "z"),
        "latitude": _num(pos, "latitude"),
        "longitude": _num(pos, "longitude"),
        "altitude": _num(pos, "altitude"),
        "roll": _num(att, "roll"),
        "pitch": _num(att, "pitch"),
        "yaw": _num(att, "yaw"),
    }


def _format_id_index(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"id: {parts[0]}\nindex: {parts[-1]}"
    return f"id: {stem}"


def _format_pose(meta: Optional[Dict[str, object]]) -> Optional[str]:
    pose = _pose_from_meta(meta)
    if not pose:
        return None
    lines: List[str] = []
    if pose.get("latitude") is not None:
        lines.append(f"lat(DD): {pose['latitude']:.6f}")
    if pose.get("longitude") is not None:
        lines.append(f"long(DD): {pose['longitude']:.6f}")
    if pose.get("altitude") is not None:
        lines.append(f"alt(m): {pose['altitude']:.1f}")
    return "\n".join(lines) if lines else None


def _format_timing(timing: Optional[Dict[str, object]]) -> List[str]:
    if not isinstance(timing, dict):
        return []
    lines: List[str] = []
    for key, label in (("embed", "embed(ms)"), ("search", "search(ms)"), ("total", "total(ms)")):
        val = timing.get(key)
        if isinstance(val, (int, float)):
            lines.append(f"{label}: {float(val):.2f}")
    return lines


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _dist_geo_alt(query: Dict[str, Optional[float]], ref: Dict[str, Optional[float]]) -> Optional[float]:
    lat1, lon1 = query.get("latitude"), query.get("longitude")
    lat2, lon2 = ref.get("latitude"), ref.get("longitude")
    if None in (lat1, lon1, lat2, lon2):
        return None
    d_hav = _haversine_m(float(lat1), float(lon1), float(lat2), float(lon2))
    alt1, alt2 = query.get("altitude"), ref.get("altitude")
    if alt1 is None or alt2 is None:
        return d_hav
    return math.hypot(d_hav, float(alt1) - float(alt2))


def _dist_xyz(query: Dict[str, Optional[float]], ref: Dict[str, Optional[float]]) -> Optional[float]:
    x1, y1, z1 = query.get("x"), query.get("y"), query.get("z")
    x2, y2, z2 = ref.get("x"), ref.get("y"), ref.get("z")
    if None in (x1, y1, z1, x2, y2, z2):
        return None
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def _discover_reference_labels(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.rglob("*.json"):
        if path.parent.name == path.stem:
            candidates.append(path)
    return sorted(candidates)


def _build_reference_index(paths: Sequence[Path]) -> Dict[str, Dict[str, Optional[float]]]:
    mapping: Dict[str, Dict[str, Optional[float]]] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                print(f"[WARN] reference json parse failed: {path} ({exc})")
                continue
        images = data.get("images")
        if not isinstance(images, list):
            continue
        for item in images:
            if not isinstance(item, dict):
                continue
            ref = item.get("imageref")
            if not isinstance(ref, str) or not ref:
                continue
            pose = _pose_from_meta(item)
            mapping[ref] = pose
    return mapping


def _normalize_imageref(ref: Optional[str], index: Dict[str, Dict[str, Optional[float]]]) -> Optional[str]:
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


def _infer_imageref_from_hit(raw: str) -> Optional[str]:
    if not raw:
        return None
    base = Path(raw).name
    if base.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
        return base
    try:
        tokens = _numeric_tokens_any(base)
        return "_".join(tokens) + ".jpg"
    except ValueError:
        pass
    stem = Path(base).stem
    if stem:
        return stem + ".jpg"
    return None


def _min_dist_ref(
    query_pose: Dict[str, Optional[float]],
    ref_index: Dict[str, Dict[str, Optional[float]]],
    mode: str,
) -> Tuple[Optional[str], Optional[float]]:
    best_ref = None
    best_dist = None
    for name, ref_pose in ref_index.items():
        dist = _dist_geo_alt(query_pose, ref_pose) if mode == "geo" else _dist_xyz(query_pose, ref_pose)
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best_ref = name
            best_dist = dist
    return best_ref, best_dist


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def plot_models_gallery(
    query_img: Path,
    rows: Sequence[Dict[str, object]],
    top_k: int,
    show_gps: bool,
    query_pose: Optional[str],
    gt_geo: Optional[Tuple[str, Optional[float]]],
    gt_xyz: Optional[Tuple[str, Optional[float]]],
) -> plt.Figure:
    row_count = 2 + len(rows)
    cols = max(top_k, 1) + 1
    fig, axes = plt.subplots(row_count, cols, figsize=(3.2 * cols, 2.2 * row_count))
    axes = np.atleast_2d(axes)
    img_ratio = 0.78

    # Header row for TopK labels.
    for ci in range(cols):
        ax = axes[0, ci]
        ax.axis("off")
        if ci == 0:
            ax.text(0.5, 0.25, "Query", ha="center", va="center", fontsize=10)
        else:
            ax.text(0.5, 0.25, f"Top{ci}", ha="center", va="center", fontsize=10)

    def _split_cell_axes(base_ax: plt.Axes) -> Tuple[plt.Axes, plt.Axes]:
        base_ax.axis("off")
        img_ax = base_ax.inset_axes([0.0, 0.0, img_ratio, 1.0])
        txt_ax = base_ax.inset_axes([img_ratio, 0.0, 1.0 - img_ratio, 1.0])
        for sub_ax in (img_ax, txt_ax):
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            sub_ax.set_frame_on(False)
        return img_ax, txt_ax

    # Query row.
    query_timing_lines = _format_timing(rows[0].get("timing_ms") if rows else None)
    for ci in range(cols):
        ax = axes[1, ci]
        if ci == 0:
            img_ax, txt_ax = _split_cell_axes(ax)
            try:
                img = load_image(query_img)
                img_ax.imshow(img)
            except Exception:
                img_ax.imshow(np.full((512, 512, 3), 220, dtype=np.uint8))
                img_ax.text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                    transform=img_ax.transAxes,
                )
            text_lines = [_format_id_index(query_img.name)]
            if show_gps and query_pose:
                text_lines.append(query_pose)
                text_lines.extend(query_timing_lines)
            txt_ax.text(
                0.0,
                1.0,
                "\n".join(text_lines),
                ha="left",
                va="top",
                fontsize=9,
                transform=txt_ax.transAxes,
            )
        else:
            ax.axis("off")

    for r_idx, row in enumerate(rows, start=2):
        model = row.get("model", f"model{r_idx}")
        backend = row.get("backend")
        total_ms = float(row.get("total_ms", 0.0) or 0.0)
        hits = row.get("hits", [])
        timing_lines = _format_timing(row.get("timing_ms"))

        label_ax = axes[r_idx, 0]
        label_ax.axis("off")
        label_text = f"{model}"
        label_ax.text(0.5, 0.5, label_text, ha="center", va="center", fontsize=9)

        for ci in range(1, cols):
            ax = axes[r_idx, ci]
            hit_idx = ci - 1
            if hit_idx >= len(hits) or hit_idx >= top_k:
                ax.axis("off")
                continue
            img_ax, txt_ax = _split_cell_axes(ax)
            hit = hits[hit_idx]
            rank = hit.get("rank", hit_idx + 1)
            label = ""
            if SHOW_SCORE and "score" in hit:
                label = f"score={float(hit['score']):.3f}"
            filename = hit.get("filename", "") or hit.get("path", "")
            img_path = hit.get("resolved_path") or resolve_hit_image(filename, TOPK_ROOT)
            try:
                img = load_image(img_path)
                img_ax.imshow(img)
            except Exception:
                img_ax.imshow(np.full((512, 512, 3), 220, dtype=np.uint8))
                img_ax.text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="red",
                    transform=img_ax.transAxes,
                )
            display_name = _format_id_index(Path(filename).name or img_path.name)

            text_lines: List[str] = []
            if label:
                text_lines.append(label)
            text_lines.append(display_name)

            flags = []
            if hit.get("is_gt_geo"):
                flags.append("GTgeo")
            if hit.get("is_gt_xyz"):
                flags.append("GTxyz")
            if flags:
                text_lines.append(",".join(flags))

            if show_gps:
                pose = hit.get("pose")
                if not pose:
                    pose = _format_pose(_lookup_metadata(img_path, TOPK_ROOT))
                    hit["pose"] = pose
                if pose:
                    text_lines.append(pose)
                    text_lines.extend(timing_lines)

            txt_ax.text(
                0.0,
                1.0,
                "\n".join(text_lines),
                ha="left",
                va="top",
                fontsize=8,
                transform=txt_ax.transAxes,
            )

    plt.tight_layout()
    return fig


def _total_ms_from_payload(payload: Dict[str, object]) -> float:
    timing = payload.get("timing_ms")
    if isinstance(timing, dict):
        for key in ("total", "search", "embed"):
            val = timing.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display TopK with geo/xyz verification.")
    parser.add_argument(
        "--model",
        nargs="+",
        default=MODEL_KEYS,
        choices=MODEL_KEYS,
        help="Model names to load (default: all).",
    )
    parser.add_argument(
        "--query",
        help="Query image id (capture_frame) or full path; CLI overrides hardcoded default.",
    )
    parser.add_argument(
        "--match",
        choices=sorted(MATCH_CHOICES.keys()),
        default="data-reference",
        help="Select which FAISS direction to load (filters output filenames).",
    )
    parser.add_argument(
        "--faiss-root",
        type=Path,
        help="FAISS output root (default: D:/ImgMatching_export/dinov3_faiss_match or /exports).",
    )
    parser.add_argument("--show-gps", action="store_true", help="Show lat/long/alt and attitude text.")
    parser.add_argument(
        "--reference-label",
        action="append",
        type=Path,
        help="Reference label JSON path (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    show_gps = bool(args.show_gps)
    models = args.model
    if not models:
        raise ValueError("No models selected.")

    query_id = args.query or DEFAULT_QUERY
    faiss_root = _resolve_faiss_root(args.faiss_root)
    direction_key = MATCH_CHOICES[args.match]
    capture, frame = _split_query_id(query_id)

    selected_paths: List[Path] = []
    for m in models:
        model_dir = _find_model_dir(faiss_root, m, direction_key)
        if model_dir is None:
            raise FileNotFoundError(f"{m} directory not found under {faiss_root} (match={direction_key})")
        encoder = _model_encoder(m)
        backend = _model_backend(m)
        json_path = _find_faiss_json(model_dir, encoder, backend, capture, frame, direction_key, TOP_K)
        if json_path is None:
            raise FileNotFoundError(
                f"{m} JSON not found for query={capture}_{frame} under {model_dir} (match={direction_key})"
            )
        selected_paths.append(json_path)

    query_img = resolve_query_image_from_id(query_id, QUERY_ROOT)
    query_meta = _lookup_metadata(query_img, QUERY_ROOT)
    query_pose_text = _format_pose(query_meta) if show_gps else None
    query_pose = _pose_from_meta(query_meta)

    if args.reference_label:
        ref_label_paths = args.reference_label
    else:
        ref_label_paths = _discover_reference_labels(TOPK_ROOT)
    if not ref_label_paths:
        print("[WARN] reference label JSONs not found; distance metrics will be missing.")
    else:
        print(f"[INFO] reference label JSONs: {len(ref_label_paths)}")

    ref_index = _build_reference_index(ref_label_paths) if ref_label_paths else {}
    gt_geo = _min_dist_ref(query_pose, ref_index, "geo") if ref_index else (None, None)
    gt_xyz = _min_dist_ref(query_pose, ref_index, "xyz") if ref_index else (None, None)

    rows: List[Dict[str, object]] = []
    for m, json_path in zip(models, selected_paths):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        hits = payload.get("results", [])
        total_ms = _total_ms_from_payload(payload)
        backend = payload.get("backend")
        timing = payload.get("timing_ms") if isinstance(payload.get("timing_ms"), dict) else {}
        processed_hits: List[Dict[str, object]] = []
        for hit in hits:
            filename = hit.get("filename", "") or hit.get("path", "")
            img_path = resolve_hit_image(filename, TOPK_ROOT)
            imageref = _infer_imageref_from_hit(str(filename))
            imageref = _normalize_imageref(imageref, ref_index)
            ref_pose = ref_index.get(imageref) if imageref else None
            dist_geo = _dist_geo_alt(query_pose, ref_pose) if ref_pose else None
            dist_xyz = _dist_xyz(query_pose, ref_pose) if ref_pose else None
            processed_hit = dict(hit)
            processed_hit["resolved_path"] = img_path
            processed_hit["dist_geo_m"] = dist_geo
            processed_hit["dist_xyz_m"] = dist_xyz
            processed_hit["is_gt_geo"] = bool(imageref and gt_geo[0] and imageref == gt_geo[0])
            processed_hit["is_gt_xyz"] = bool(imageref and gt_xyz[0] and imageref == gt_xyz[0])
            if show_gps:
                processed_hit["pose"] = _format_pose(_lookup_metadata(img_path, TOPK_ROOT))
            processed_hits.append(processed_hit)
        rows.append(
            {
                "model": m,
                "backend": backend,
                "total_ms": total_ms,
                "timing_ms": timing,
                "hits": processed_hits,
            }
        )

    fig = plot_models_gallery(
        query_img,
        rows,
        top_k=TOP_K,
        show_gps=show_gps,
        query_pose=query_pose_text,
        gt_geo=gt_geo,
        gt_xyz=gt_xyz,
    )
    plt.show()


if __name__ == "__main__":
    main()
