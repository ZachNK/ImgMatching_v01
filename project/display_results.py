"""
display_results

쿼리 이미지에 대한 선별된 참조이미지 출력 확인용

`verify_display_topk.py`를 따라서 만들고서, 직접 수정
"""
# 타입 힌트의 전방 선언을 문자열로 사용 가능하게 함 (Python3.7+)
from __future__ import annotations

# CLI 인자 파싱
import argparse
# 결과 JSON 읽기
import json
# 거리 계산 등 수학 함수
import math
# 정규식 처리
import re
# 경로 조작
from pathlib import Path
# 타입 힌트용
from typing import Dict, List, Optional, Sequence, Tuple

# 시각화 (Plotting) 기능
import matplotlib.pyplot as plt
# 배열 처리 기능
import numpy as np
# 이미지 로딩, 변환
from PIL import Image

# -----------------------------------------------------------------------------#
# Hardcoded paths (can be overridden via CLI)
# -----------------------------------------------------------------------------#
# 기본 경로: 비행(쿼리) 이미지 루트
QUERY_ROOT = Path(r"D:/Datasets/01_01_jamshill_data_flight")
# 기본 경로: 래퍼런스(매칭대상) 이미지 루트
TOPK_ROOT = Path(r"D:/Datasets/01_03_jamshill_data_reference")
# 기본 경로: FAISS 매칭 결과 JSON이 있는 루트
FAISS_ROOT = Path(r"D:/ImgMatching_export/dinov3_faiss_match")
# 리눅스 서버 경로 대체 값
FAISS_ROOT_FALLBACK = Path("/exports/dinov3_faiss_match")
# 기본 쿼리 ID (캡쳐_프레임)
DEFAULT_QUERY = "251124160703_00130"
# 표시할 TopK 개수 기본값 (10)
TOP_K: int = 10
# 점수 출력 여부 기본값 (True)
SHOW_SCORE = True

# 이미지 사이즈 상수 추가
IMG_PX_W = 1024
IMG_PX_H = 1024
IMG_DPI = 96
SCALE = 0.25
HEADER_INCH = 0.7
GAP_INCH = 0.3


# 쿼리 루트 자동 감지: 기본 경로가 없으면 /opt/... 대체 경로로 교체
for _candidate in (QUERY_ROOT, Path("/opt/datasets/01_01_jamshill_data_flight")):
    if _candidate.exists():
        QUERY_ROOT = _candidate
        break

# 레퍼런스 루트 자동 감지: 기본 경로가 없으면 /opt/... 대체 경로로 교체
for _candidate in (TOPK_ROOT, Path("/opt/datasets/01_03_jamshill_data_reference")):
    if _candidate.exists():
        TOPK_ROOT = _candidate
        break

# MODEL_ENCODER 딕셔너리: 모델 이름별로 encoder 토큰 문자열 매핑
MODEL_ENCODER = {
    "vitb16": "vitb16",
    "vith16+": "vith16plus",
    "vitl16": "vitl16",
    "vitl16sat": "vitl16",
    "vits16": "vits16",
    "vits16+": "vits16plus",
}

# 모델 키들을 정렬한 리스트
MODEL_KEYS = sorted(MODEL_ENCODER.keys())

# 매칭 방향 인자(data-reference 등)를 내부 키 (db2reference 등)으로 매핑.
MATCH_CHOICES = {
    "data-reference": "db2reference",
    "reference-data": "reference2db",
    "reference-reference": "reference2reference",
    "data-data": "db2db",
}

# 메타데이터 JSON 캐시 (경로별 이미지 메타 매핑 저장)
_METADATA_CACHE: Dict[Path, Dict[str, Dict[str, object]]] = {}
# 누락 메타데이터 경고를 중복 출력하지 않도록 기록하는 집합
_MISSING_META_REPORTED: set[Path] = set()

def plot_modes_gallery(
    query_img: Path,
    rows: Sequence[Dict[str, object]],
    top_k: int,
    show_gps: bool,
    query_pose: Optional[str]
) -> plt.Figure:
    CELL_W = (IMG_PX_W / IMG_DPI) * SCALE
    CELL_H = (IMG_PX_H / IMG_DPI) * SCALE
    
    row_count = 2 + 6
    cols = max(top_k, 1) + 1
    figwidth  = CELL_W * cols
    figheight = HEADER_INCH + CELL_H * (1 + 6) + GAP_INCH *(row_count-1)
    hspace = GAP_INCH / CELL_H
    height_ratios = [HEADER_INCH, CELL_H] + [CELL_H] * 6
    print(row_count, cols, height_ratios)
    

    fig, axes = plt.subplots(
        row_count, 
        cols,
        figsize = (figwidth, figheight), #캔버스 비율 조절
        gridspec_kw={"height_ratios": height_ratios, 'hspace': hspace},
        constrained_layout=False,

    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.25, hspace=0)
    axes = np.atleast_2d(axes)
    img_ratio = 0.75

    for ci in range(cols):
        ax = axes[0, ci]
        ax.axis("on")
        if ci == 0:
            ax.text(0.5, 0.99, "Query", ha="center", va="center", fontsize=10)
        else:
            ax.text(0.5, 0.99, f"Top{ci}", ha="center", va="center", fontsize=10)

    def _split_cell_axes(base_ax: plt.Axes) -> Tuple[plt.Axes, plt.Axes]:
        base_ax.axis("off")
        gap = 0.0005
        img_ax = base_ax.inset_axes([0.0, 0.0, img_ratio, 1.0])
        txt_ax = base_ax.inset_axes([img_ratio, 0.0, 1.0 - img_ratio - gap, 1.0])
        for sub_ax in (img_ax, txt_ax):
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            sub_ax.set_frame_on(False)
        return img_ax, txt_ax

    return fig
    

def main() -> None:
    query_img = Path("missing")
    top_k = TOP_K
    rows = [{"model": "dummy", "hits": [{"filename":f"hit_{i+1}.jpg"} for i in range(top_k)]}]
    fig = plot_modes_gallery(query_img, rows, top_k, show_gps=False, query_pose=None)
    manager = plt.get_current_fig_manager()
    try:
        manager.window.move(0, 120)
    except Exception:
        pass
            
    plt.show()

if __name__ == "__main__":
    main()
