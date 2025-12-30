# project/imatch/types.py
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

"""
데이터 클래스 및 열거형 정의
RansacMethod: RANSAC 방법 열거형
MatchConfig: 매칭 구성 설정 데이터 클래스   
RunContext: 실행 컨텍스트 설정 데이터 클래스
"""

class RansacMethod(Enum):
    """ 
    RANSAC 방법 열거형
    OFF: RANSAC 사용 안함
    AFFINE: 어파인 변환 사용
    HOMOGRAPHY: 호모그래피 변환 사용
    """
    OFF = "off"
    AFFINE = "affine"
    HOMOGRAPHY = "homography"

@dataclass
class MatchConfig:
    """
    매칭 구성 설정
    image_size: 목표 크기 (기본값: 224)
    mutual_k: 상호 최근접 이웃 수 (기본값: 5)
    topk: 상위 k개 매칭 (기본값: 200)   
    max_patches: 최대 패치 수 (기본값: 0, 제한 없음)
    """
    image_size: int = 224
    mutual_k: int = 5
    topk: int = 200
    max_patches: int = 0

@dataclass
class RunContext:
    """
    실행 컨텍스트 설정
    repo_dir: 저장소 디렉터리 경로
    img_root: 이미지 루트 디렉터리 경로
    export_dir: 내보내기 디렉터리 경로
    device: 실행 장치 (기본값: "cuda")
    """
    repo_dir: Path
    img_root: Path
    export_dir: Path
    device: str = "cuda"
