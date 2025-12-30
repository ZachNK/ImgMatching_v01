# project/imatch/models.py
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

from imatch.extracting import (
    global_embedding,
    patch_embedding,
    cosine_similarity,
    compute_patch_matches,
)
from imatch.loading import (
    REPO_DIR,
    EMBED_ROOT,
    load_image,
    save_match_result,
)

"""
모델 로딩 유틸리티
- _block_hub_net_if_needed: torch.hub 네트워크 호출 차단 설정
- load_model: 로컬 리포지토리에서 torch.hub 모델 로딩 및 체크포인트 적용
"""
DINOV_BLOCK_NET = os.getenv("DINOV_BLOCK_NET", "1").strip() == "1"

def _block_hub_net_if_needed():
    """
    DINOV_BLOCK_NET 환경 변수가 설정된 경우 torch.hub의 네트워크 호출을 차단.
    -> main()에서 호출, load_model() 전에 호출 필요.
    """
    if DINOV_BLOCK_NET:
        import torch.hub as _th
        def _no_dl(*a, **k):
            raise RuntimeError("[pretrained:0] Blocked torch.hub.load_state_dict_from_url (offline)")
        _th.load_state_dict_from_url = _no_dl  # type: ignore

def pretrained_model(repo_dir: str, hub_entry: str, weight_dir: str, device: str) -> Tuple[torch.nn.Module, str]:
    """
    torch.hub 로 DINOv3 로컬 리포에서 모델 생성 후, state_dict 로딩.
    e.g. load_model("/path/to/repo", "vitb16", "/path/to/checkpoint.pth", "cuda") -> (model, "dinov3_vitb16")
    - 입력:
      repo_dir: str — DINOv3 로컬 리포지토리 경로
      hub_entry: str — torch.hub 모델 엔트리 이름
      weight_dir: str — 가중치 파일 경로
      device: str — 모델을 올릴 장치 (예: "cuda" 또는 "cpu")
    - 출력:
      Tuple[torch.nn.Module, str] — 로드된 모델과 모델 이름 튜플
    """
    # 블록 허브 네트워크 호출이 필요한지 확인
    _block_hub_net_if_needed()

    print(f"[pretrained:1] hub.load entry='{hub_entry}' from {repo_dir}")
    model = torch.hub.load(str(repo_dir), hub_entry, source="local", trust_repo=True, pretrained=False)
    model.eval().to(device)

    # 체크 포인트 로드 및 모델 가중치 설정
    try:
        # CUDA 장치에 맞게 체크 포인트 로드 시도
        state = torch.load(weight_dir, map_location="cuda:0", weights_only=True)
    except TypeError:
        # 실패 시 CPU에 맞게 로드
        state = torch.load(weight_dir, map_location="cpu", weights_only=True)

    # 체크 포인트에서 'state_dict' 키가 있으면 해당 값으로 설정
    if isinstance(state, dict) and "state_dict" in state:
        # 'state_dict' 키가 있는 경우 해당 값으로 설정
        state = state["state_dict"]
    # 'module.' 접두사가 있는 키를 제거하여 모델에 맞게 정리
    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # 모델에 가중치 로드, 엄격하지 않게 설정하여 누락된 키나 예기치 않은 키 경고 출력
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    
    # 경고 출력
    if missing:
        # 누락된 키 경고 출력
        print(f"[pretrained:warn1] missing keys: {len(missing)}")
    if unexpected:
        # 예기치 않은 키 경고 출력
        print(f"[pretrained:warn2] unexpected keys: {len(unexpected)}")
    
    return model, (hub_entry or "hub_model")

def execute_matching(
    args,
    resolved_weights,
    key2path,
    pairs,
    transform,
) -> None:
    """DINOv3 기반 매칭 루프를 실행한다."""
    device = args.device
    autocast_device = "cuda" if device.startswith("cuda") else None
    amp_context = torch.amp.autocast(autocast_device) if autocast_device else nullcontext()

    with torch.inference_mode():
        with amp_context:
            for weight_name, hub_entry, weight_path in resolved_weights:
                print(f"[weight] {weight_name}  hub={hub_entry}  weight={weight_path}")
                model, _ = pretrained_model(REPO_DIR, hub_entry, str(weight_path), device)

                if args.save_emb:
                    (EMBED_ROOT / weight_name).mkdir(parents=True, exist_ok=True)

                if not getattr(model, "_imatch_warmed_up", False):
                    dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
                    _ = global_embedding(model, dummy, device)
                    _ = patch_embedding(model, dummy, device)
                    model._imatch_warmed_up = True

                for a_key, b_key in pairs:
                    path_a = key2path[a_key]
                    path_b = key2path[b_key]

                    xa = transform(load_image(path_a)).unsqueeze(0)
                    xb = transform(load_image(path_b)).unsqueeze(0)

                    t0 = time.perf_counter()
                    fa = global_embedding(model, xa, device); t1 = time.perf_counter()
                    fb = global_embedding(model, xb, device); t2 = time.perf_counter()
                    tokens_a = patch_embedding(model, xa, device)
                    tokens_b = patch_embedding(model, xb, device)

                    cosine = cosine_similarity(fa, fb)
                    patch, pa_np, pb_np = compute_patch_matches(tokens_a, tokens_b, args)

                    if args.save_emb:
                        embed_dir = EMBED_ROOT / weight_name / f"{a_key}_{b_key}"
                        embed_dir.mkdir(parents=True, exist_ok=True)
                        np.save(embed_dir / "global_a.npy", fa.detach().cpu().float().numpy())
                        np.save(embed_dir / "global_b.npy", fb.detach().cpu().float().numpy())
                        if pa_np is not None and pb_np is not None:
                            np.save(embed_dir / "patch_a.npy", pa_np)
                            np.save(embed_dir / "patch_b.npy", pb_np)

                    time_ms = dict(
                        forward_a=round((t1 - t0) * 1000, 2),
                        forward_b=round((t2 - t1) * 1000, 2),
                        total=round((t2 - t0) * 1000, 2),
                    )

                    save_match_result(
                        args=args,
                        weight_name=weight_name,
                        hub_entry=hub_entry,
                        weight_path=weight_path,
                        pair_a=a_key,
                        pair_b=b_key,
                        image_a=path_a,
                        image_b=path_b,
                        cosine=cosine,
                        time_ms=time_ms,
                        patch=patch,
                    )
