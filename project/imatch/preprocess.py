# imatch/imageprocessing.py
import math
import torch
from torchvision import transforms

def build_transform(
    ### 이미지 전처리 
    # patch_size: 패치 크기
    # patch_multiple: 패치 배수, 입력 이미지 크기를 패치 크기의 배수로 조정 (기본값 16)
    # interpolation: 크기변형 시 사용할 보간법 (기본값 "bicubic")
    # normalize: 정규화 적용 여부 (기본값 True)
    patch_size: int,
    patch_multiple: int = 16,
    interpolation: str = "bicubic",
    normalize: str = "vits16+",
):
    """
    build_transform() 함수는 이미지 전처리를 위한 torchvision.transforms.Compose 객체를 생성.
    입력: 패치 크기, 패치 배수, 보간법, 정규화 여부: data type: int, int, str, bool
    출력: torchvision.transforms.Compose 객체: data type: transforms.Compose
    1. 목표 크기 계산: patch_size * patch_multiple
    2. 변환 단계 구성:
       - 이미지 타입 변환 (torch.float32)
       - 크기 조정 (목표 크기, 지정된 보간법 사용, 앤티앨리어싱 적용)
       - (선택적) 정규화 (ImageNet 평균/표준편차 사용)
    3. transforms.Compose 객체 반환
    """
    target_size = patch_size * patch_multiple

    transforms_steps = [

        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize(
            (target_size, target_size),
            interpolation = getattr(transforms.InterpolationMode, interpolation.upper()),
            antialias = True,
        ),
    ]

    if normalize == "LVD":
        print("\033[34m[imageprocessing] Normalization: web dataset (LVD-1689M)\033[0m")
        transforms_steps.append(
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            )
        )
    else:
        print("\033[34m[imageprocessing] Normalization: satellite dataset (SAT-493M)\033[0m")
        transforms_steps.append(
            transforms.Normalize(
                mean = [0.430, 0.411, 0.296],
                std = [0.213, 0.156, 0.143],
            )
        )

    return transforms.Compose(transforms_steps)
