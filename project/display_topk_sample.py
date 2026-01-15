"""
Multi-model TopK viewer (hardcoded paths, faiss_4Redis JSON).

- 쿼리/TopK 이미지 루트(QUERY_ROOT, TOPK_ROOT)는 코드 상단에서 하드코딩.
- 모델별 결과 JSON 경로는 MODEL_JSONS에서 하드코딩(필요시 여기만 수정).

사용 예시:
  python project/display_topk_sample.py --model vitb16 vith16+ vitl16 vitl16sat

동작:
- 좌상단: 쿼리 이미지 1장.
- 이후 행: 지정한 모델별로 Top@1~Top@10 이미지를 좌→우로 나열.
  (타이틀에 TopK, score, 파일명; 행 왼쪽에 총 시간 ms 표시)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Hardcoded configuration: A, B, C 경로만 수정하면 됩니다.
# -----------------------------------------------------------------------------
QUERY_ROOT = Path(r"D:/Datasets/01_01_jamshill_data_flight")  # A
TOPK_ROOT = Path(r"D:/Datasets/01_03_jamshill_data_reference")  # B
TOP_K: int = 10  # TopK 고정
SHOW_SCORE = True

# 모델별 JSON 경로를 하드코딩합니다. 필요하면 여기만 수정하세요.
MODEL_JSONS: Dict[str, Path] = {
    "vitb16": Path(
        r"D:/ImgMatching_export/dinov3_faiss_match/vitb16_raw_db2reference/"
        r"GlobalToken_res1024_raw_dinov3_vitb16_LVD_251124160703_00042_db2reference_top10_redis.json"
    ),
    "vith16+": Path(
        r"D:/ImgMatching_export/dinov3_faiss_match/vith16+_raw_db2reference/"
        r"GlobalToken_res1024_raw_dinov3_vith16plus_LVD_251124160703_00042_db2reference_top10_redis.json"
    ),
    "vitl16": Path(
        r"D:/ImgMatching_export/dinov3_faiss_match/vitl16_raw_db2reference/"
        r"GlobalToken_res1024_raw_dinov3_vitl16_LVD_251124160703_00042_db2reference_top10_redis.json"
    ),
    "vitl16sat": Path(
        r"D:/ImgMatching_export/dinov3_faiss_match/vitl16sat_raw_db2reference/"
        r"GlobalToken_res1024_raw_dinov3_vitl16_SAT_251124160703_00042_db2reference_top10_redis.json"
    ),
    "vits16": Path(
        r"D:/ImgMatching_export/dinov3_faiss_match/vits16_raw_db2reference/"
        r"GlobalToken_res1024_raw_dinov3_vits16_LVD_251124160703_00042_db2reference_top10_redis.json"
    ),
    "vits16+": Path(
        r"D:/ImgMatching_export/dinov3_faiss_match/vits16+_raw_db2reference/"
        r"GlobalToken_res1024_raw_dinov3_vits16plus_LVD_251124160703_00042_db2reference_top10_redis.json"
    ),
}

# 이미지별 메타데이터(JSON) 캐시 및 미발견 로그
_METADATA_CACHE: Dict[Path, Dict[str, Dict[str, object]]] = {}
_MISSING_META_REPORTED: set[Path] = set()


def _numeric_tokens_after_backend(name: str) -> List[str]:
    """
    파일명에서 LVD/SAT 다음 연속된 숫자 토큰을 추출합니다.
    예) GlobalToken..._LVD_251216104934_300_50_0_270_1862.npy
        -> ['251216104934', '300', '50', '0', '270', '1862']
    """
    tokens = Path(name).stem.split("_")
    try:
        idx = tokens.index("LVD")
    except ValueError:
        try:
            idx = tokens.index("SAT")
        except ValueError:
            raise ValueError(f"LVD/SAT 토큰을 찾지 못했습니다: {name}")

    numeric: List[str] = []
    for tok in tokens[idx + 1 :]:
        if re.fullmatch(r"\d+", tok):
            numeric.append(tok)
        else:
            break
    if not numeric:
        raise ValueError(f"LVD/SAT 뒤 숫자 토큰이 없습니다: {name}")
    return numeric


def _numeric_tokens_from_image_path(name: str) -> List[str]:
    """
    Image path (no LVD/SAT) -> digits from stem and parent folder.
    예) /opt/.../251216104934_300_50_0_270/251216104934_300_50_0_270_01862.jpg
        -> ['251216104934', '300', '50', '0', '270', '01862']
    """
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
    raise ValueError(f"LVD/SAT 토큰이 없을 때 이미지 경로에서 계층 통정을 찾지 못했습니다: {name}")


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


def resolve_query_image(json_path: Path, query_root: Path) -> Path:
    """
    JSON 파일명에서 캡처ID/프레임을 추출해 쿼리 이미지 경로를 만듭니다.
    예) ..._LVD_251124155712_0001_... -> folder=251124155712, frame=00001
    """
    tokens = _numeric_tokens_after_backend(json_path.name)
    if len(tokens) < 2:
        raise ValueError("쿼리 파일명에서 캡처ID/프레임을 구할 수 없습니다.")

    capture = tokens[0]
    frame = f"{int(tokens[1]):05d}"
    folder = query_root / capture
    img_base = folder / f"{capture}_{frame}"
    found = _find_with_ext(img_base)
    return found or img_base.with_suffix(".jpg")


def resolve_hit_image(filename: str, topk_root: Path) -> Path:
    """
    results[].filename에서 폴더/프레임을 추출해 TopK 이미지 경로를 만듭니다.
    예) ..._LVD_251216104934_300_50_0_270_1862.npy
        -> folder=251216104934_300_50_0_270, frame=01862
    """
    tokens = _numeric_tokens_any(filename)
    if len(tokens) < 2:
        raise ValueError("결과 파일명에서 폴더/프레임을 구할 수 없습니다.")

    frame = f"{int(tokens[-1]):05d}"
    folder = "_".join(tokens[:-1])
    img_base = topk_root / folder / f"{folder}_{frame}"
    found = _find_with_ext(img_base)
    return found or img_base.with_suffix(".jpg")


def _label_json_path(img_path: Path, dataset_root: Path) -> Path:
    """
    이미지 경로를 기준으로 메타(json) 파일 경로를 유추합니다.
    - reference 루트: <folder>/<folder>.json
    - 기타 루트: <folder>/<folder>_label.json (없으면 _labels.json까지 시도)
    """
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
                # BOM 이 포함된 경우 대비
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
                    # 또한 파일명/스템만으로도 조회할 수 있게 별도 키를 추가합니다.
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

    # 추가 안전장치: key에 경로가 포함된 경우 basename 비교(case-insensitive)
    target = name.lower()
    for key, item in mapping.items():
        try:
            if Path(key).name.lower() == target:
                return item
        except Exception:
            if isinstance(key, str) and key.lower() == target:
                return item
    return None


def _format_pose(meta: Optional[Dict[str, object]]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    pos = meta.get("position") if isinstance(meta.get("position"), dict) else {}
    att = meta.get("attitude") if isinstance(meta.get("attitude"), dict) else {}

    def _num(d: Dict[str, object], key: str) -> Optional[float]:
        val = d.get(key) if isinstance(d, dict) else None
        return float(val) if isinstance(val, (int, float)) else None

    lat = _num(pos, "latitude")
    lon = _num(pos, "longitude")
    alt = _num(pos, "altitude")
    roll = _num(att, "roll")
    pitch = _num(att, "pitch")
    yaw = _num(att, "yaw")

    lines: List[str] = []
    geo_parts: List[str] = []
    if lat is not None:
        geo_parts.append(f"lat={lat:.6f}")
    if lon is not None:
        geo_parts.append(f"long={lon:.6f}")
    if alt is not None:
        geo_parts.append(f"alt={alt:.1f}")
    if geo_parts:
        lines.append(", ".join(geo_parts))

    att_parts: List[str] = []
    if roll is not None:
        att_parts.append(f"roll={roll:.2f}")
    if pitch is not None:
        att_parts.append(f"pitch={pitch:.2f}")
    if yaw is not None:
        att_parts.append(f"yaw={yaw:.2f}")
    if att_parts:
        lines.append(", ".join(att_parts))

    return "\n".join(lines) if lines else None


def _pose_text(img_path: Path, dataset_root: Path) -> Optional[str]:
    return _format_pose(_lookup_metadata(img_path, dataset_root))


def _total_ms_from_payload(payload: Dict[str, object]) -> float:
    timing = payload.get("timing_ms")
    if isinstance(timing, dict):
        for key in ("total", "search", "embed"):
            val = timing.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return 0.0


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def plot_models_gallery(
    query_img: Path,
    rows: Sequence[Dict[str, object]],
    top_k: int,
    show_gps: bool,
    query_pose: Optional[str] = None,
) -> plt.Figure:
    row_count = 1 + len(rows)  # 1행은 쿼리
    cols = max(top_k, 1) + 1  # 0번 컬럼은 쿼리/라벨, 나머지는 TopK
    fig, axes = plt.subplots(row_count, cols, figsize=(3 * cols, 2.0 * row_count))
    axes = np.atleast_2d(axes)

    # Row 0: Query
    for ci in range(cols):
        ax = axes[0, ci]
        if ci == 0:
            try:
                img = load_image(query_img)
                ax.imshow(img)
            except Exception:
                ax.imshow(np.full((512, 512, 3), 220, dtype=np.uint8))
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10, color="red", transform=ax.transAxes)
            title = f"Query\n{query_img.name}"
            if show_gps and query_pose:
                title += f"\n{query_pose}"
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")

    # Rows 1..: 각 모델 TopK
    for r_idx, row in enumerate(rows, start=1):
        model = row.get("model", f"model{r_idx}")
        backend = row.get("backend")
        total_ms = float(row.get("total_ms", 0.0) or 0.0)
        hits = row.get("hits", [])

        label_ax = axes[r_idx, 0]
        label_ax.axis("off")
        label_text = f"{model} | TopK={top_k}"
        if backend:
            label_text += f" | {backend}"
        label_text += f"\nTotal(ms)={total_ms:.1f}"
        label_ax.text(0.5, 0.5, label_text, ha="center", va="center", fontsize=10)

        for ci in range(1, cols):
            ax = axes[r_idx, ci]
            hit_idx = ci - 1
            if hit_idx >= len(hits) or hit_idx >= top_k:
                ax.axis("off")
                continue
            hit = hits[hit_idx]
            rank = hit.get("rank", hit_idx + 1)
            label = f"Top{rank}"
            if SHOW_SCORE and "score" in hit:
                label += f" ({float(hit['score']):.3f})"
            filename = hit.get("filename", "") or hit.get("path", "")
            img_path = hit.get("resolved_path") or resolve_hit_image(filename, TOPK_ROOT)
            try:
                img = load_image(img_path)
                ax.imshow(img)
            except Exception:
                ax.imshow(np.full((512, 512, 3), 220, dtype=np.uint8))
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, color="red", transform=ax.transAxes)
            display_name = Path(filename).name or img_path.name
            title = f"{label}\n{display_name}"
            if show_gps:
                pose = hit.get("pose")
                if not pose:
                    pose = _pose_text(img_path, TOPK_ROOT)
                    hit["pose"] = pose
                if pose:
                    title += f"\n{pose}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="모델별 TopK 이미지 시각화 (하드코딩 경로 사용)")
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        choices=sorted(MODEL_JSONS.keys()),
        help="표시할 모델 이름(1개 이상)",
    )
    parser.add_argument("--show-gps", action="store_true", help="위치/자세(lat,long,alt,roll,pitch,yaw) 텍스트 표시")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    show_gps: bool = bool(args.show_gps)
    models: List[str] = args.model
    if not models:
        raise ValueError("모델을 한 개 이상 지정하세요.")

    selected_paths: List[Path] = []
    for m in models:
        json_path = MODEL_JSONS.get(m)
        if json_path is None:
            raise KeyError(f"등록되지 않은 모델: {m}")
        if not json_path.exists():
            raise FileNotFoundError(f"{m} JSON not found: {json_path}")
        selected_paths.append(json_path)

    # 쿼리 이미지는 첫 번째 모델 JSON 기준으로 찾습니다.
    query_img = resolve_query_image(selected_paths[0], QUERY_ROOT)

    query_pose: Optional[str] = _pose_text(query_img, QUERY_ROOT) if show_gps else None

    rows: List[Dict[str, object]] = []
    for m, json_path in zip(models, selected_paths):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        hits = payload.get("results", [])
        total_ms = _total_ms_from_payload(payload)
        backend = payload.get("backend")
        processed_hits: List[Dict[str, object]] = []
        for hit in hits:
            filename = hit.get("filename", "") or hit.get("path", "")
            img_path = resolve_hit_image(filename, TOPK_ROOT)
            pose = _pose_text(img_path, TOPK_ROOT) if show_gps else None
            processed_hit = dict(hit)
            processed_hit["resolved_path"] = img_path
            if pose:
                processed_hit["pose"] = pose
            processed_hits.append(processed_hit)
        rows.append(
            {
                "model": m,
                "backend": backend,
                "total_ms": total_ms,
                "hits": processed_hits,
            }
        )

    fig = plot_models_gallery(query_img, rows, top_k=TOP_K, show_gps=show_gps, query_pose=query_pose)
    plt.show()


if __name__ == "__main__":
    main()
