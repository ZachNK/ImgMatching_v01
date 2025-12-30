"""
run.py
- DINOv3 기반 이미지 매칭 배치 실행기
- 특징:
  * -a/-b 생략 시 전체 이미지 All-vs-All (N x (N-1))
  * --weights / --group / --all-weights 로 가중치 선택
  * Advanced setting: match/keypoint/line threshold + max features
  * 요청한 폴더 구조로 JSON 저장:
      /exports/dinov3_match/<weight>_<Aalt>_<Aframe>/
        <weight>_<Aalt.Aframe>_<Balt.Bframe>.json
"""

import argparse
import torch

from imatch.preprocess import build_transform
from imatch.loading import EMBED_ROOT, MODEL_KEY, prepare_run_context
from imatch.pretrained import execute_matching
from imatch.utils import bounded_float, bounded_int


WEIGHT_GROUPS = {
    group_name: list(models.keys())
    for group_name, models in MODEL_KEY.items()
}
ALL_WEIGHT_KEYS = list(
    dict.fromkeys(weight_name for group in WEIGHT_GROUPS.values() for weight_name in group)
)


def main():
    p = argparse.ArgumentParser(description="DINOv3 Matching Batch Runner")
    # 이미지 선택
    p.add_argument("-a", "--pair-a", help="ALT.FRAME 또는 ALT (생략시 전체)")
    p.add_argument("-b", "--pair-b", help="ALT.FRAME 또는 ALT (생략시 전체)")
    p.add_argument(
        "--regex",
        default=r".*_(?P<alt>\d{3})_(?P<frame>\d{4})\.(jpg|jpeg|png|bmp|tif|tiff|webp)$",
        help="IMG_ROOT 하위에서 ALT.FRAME을 뽑을 정규식",
    )
    p.add_argument("--exts", nargs="*", default=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    # 가중치 선택
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-w", "--weights", nargs="+", help="가중치 옵션")
    g.add_argument("-g", "--group", choices=list(WEIGHT_GROUPS.keys()))
    g.add_argument("--all-weights", action="store_true")
    # 매칭 하이퍼파라미터
    p.add_argument("-i", "--image-size", type=int, default=336)
    p.add_argument(
        "-x",
        "--max-features",
        type=bounded_int(10, 100000),
        default=2000,
        metavar="[10-100000]",
        help="패치 토큰 최대 개수",
    )
    p.add_argument(
        "-t",
        "--match-th",
        type=bounded_float(0.0, 1.0),
        default=0.05,
        metavar="[0-1]",
        help="유사도 임계값 (match threshold)",
    )
    p.add_argument(
        "-k",
        "--keypoint-th",
        type=bounded_float(0.0, 1.0),
        default=0.005,
        metavar="[0-1]",
        help="패치 토큰 보존 임계값 (keypoint threshold)",
    )
    p.add_argument(
        "-l",
        "--line-th",
        type=bounded_float(0.0, 1.0),
        default=0.2,
        metavar="[0-1]",
        help="매칭 라인 보존 임계값 (line threshold)",
    )
    p.add_argument(
        "-e",
        "--save-emb",
        action="store_true",
        help="Save global/patch embeddings to EMBED_ROOT for each pair.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    key2path, pairs, resolved_weights, type_data = prepare_run_context(args, WEIGHT_GROUPS, ALL_WEIGHT_KEYS)

    transform = build_transform(args.image_size, normalize=type_data)

    if args.save_emb:
        EMBED_ROOT.mkdir(parents=True, exist_ok=True)

    execute_matching(
        args=args,
        resolved_weights=resolved_weights,
        key2path=key2path,
        pairs=pairs,
        transform=transform,
    )


if __name__ == "__main__":
    main()
