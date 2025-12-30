# project/imatch/features.py
import math
import torch
import numpy as np
from typing import Optional, Tuple

"""
이미지 특징 추출 관련 유틸리티 함수들.
- 글로벌 특징 벡터 추출: global_embedding(model: torch.nn.Module, x: torch.Tensor, device: str) -> torch.Tensor
- 패치 토큰 추출: patch_embedding(model: torch.nn.Module, x: torch.Tensor, device: str) -> Optional[torch.Tensor]
- 코사인 유사도 계산: cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
- 키포인트 임계값 적용: kp_threshold(tokens: torch.Tensor, idx_map: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]
- 패치 토큰 격자 재배열: patch2grid(tokens: torch.Tensor) -> torch.Tensor
"""

from imatch.matching import matching_knn, enforced_matching, grid_side, subsample_tokens

@torch.no_grad()
def global_embedding(model: torch.nn.Module, x: torch.Tensor, device: str) -> torch.Tensor: #LEGACY: extract_global_feature
    """
    모델 출력에서 글로벌 특징을 추출한다.
    """
    # 입력 텐서를 지정된 장치로 이동
    x = x.to(device, non_blocking=True)
    # 모델의 특징 추출 메서드 호출
    out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)
    # 출력에서 특징 추출
    if isinstance(out, dict):
        # 다양한 키를 확인하여 특징을 추출: "x", "feat", "features", "pooled", "pooler_output"
        if "x" in out and isinstance(out["x"], torch.Tensor) and out["x"].ndim == 3:
            feat = out["x"].mean(dim=1)
        else:
            for k in ("feat", "features", "pooled", "pooler_output"):
                if k in out and isinstance(out[k], torch.Tensor):
                    feat = out[k]
                    break
            else:
                feat = [v for v in out.values() if torch.is_tensor(v)][-1]
    else:
        # 출력이 튜플 또는 리스트인 경우 첫 번째 요소를 특징으로 사용
        feat = out

    # 특징 텐서의 차원에 따라 글로벌 특징 벡터 생성 (평균풀링 수행)
    # ndim이 3이라는 뜻은 텐서가 보통 (batch, sequence_len, hidden_dim) 구조를 따른다는 뜻
    if feat.ndim == 3:
        # 시퀀스 차원에 대해 평균 계산 (평균풀링): 1번째 차원 -> 1=sequence_len
        feat = feat.mean(dim=1)
    
    # ndim이 4라는 뜻은 텐서가 (batch, channels, height, width) 구조를 따른다는 뜻
    if feat.ndim == 4:
        # 공간 차원에 대해 평균 계산 (평균풀링): 2번째와 3번째 차원 -> 2=height, 3=width
        feat = feat.mean(dim=(2, 3))
    
    # 배치 차원 제거 후 반환: 글로벌 특징 벡터 -> 0=batch 차원
    # feat shape: (1, hidden_dim) -> feat.squeeze(0) shape: (hidden_dim,)
    return feat.squeeze(0)


@torch.no_grad()
def patch_embedding(model: torch.nn.Module, x: torch.Tensor, device: str) -> Optional[torch.Tensor]: #LEGACY: extract_patch_tokens
    """
    패치 토큰(CLS 제외)을 추출한다.
    """
    # 입력 텐서를 지정된 장치로 이동
    x = x.to(device, non_blocking=True)
    out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)

    # 다양한 출력 형식에 대응하여 패치 토큰 추출 (딕셔너리, 튜플/리스트, 텐서)
    if isinstance(out, dict):
        
        # 'x_norm_patchtokens' 키가 있으면 해당 값 사용
        if "x_norm_patchtokens" in out and torch.is_tensor(out["x_norm_patchtokens"]):
            
            # 값이 3차원 텐서인 경우 반환
            v = out["x_norm_patchtokens"]
            
            # v.ndim: 3 인 경우 패치 토큰 반환
            if v.ndim == 3:
                # 패치 토큰 반환
                return v.squeeze(0).contiguous()
        
        # 다른 키들 중 패치 토큰을 찾아 반환: 'patch_tokens', 'tokens_patch', 'features_patch'
        for k in ["patch_tokens", "tokens_patch", "features_patch"]:
            # 키에 해당하는 값 가져오기
            v = out.get(k, None)

            # 값이 3차원 텐서인 경우 반환
            if torch.is_tensor(v) and v.ndim == 3:
                # 3차원 텐서인 경우 패치 토큰 반환
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        
        # 위에서 찾지 못한 경우, 값들 중 3차원 텐서를 찾아 반환
        for v in out.values():
            
            # 3차원 텐서인 경우 패치 토큰 반환
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] > 16:
                # 패치 토큰 반환 (CLS 제외)
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        
        # 찾지 못한 경우 None 반환
        return None

    # 출력이 튜플 또는 리스트인 경우 각 요소를 검사
    if isinstance(out, (tuple, list)):
        
        # 각 요소를 검사하여 3차원 텐서인 경우 패치 토큰 반환
        for v in out:
            # 3차원 텐서인 경우 패치 토큰 반환
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] > 16:
                # 패치 토큰 반환 (CLS 제외)
                return (v[:, 1:, :].squeeze(0) if v.shape[1] > 1 else v.squeeze(0)).contiguous()
        # 찾지 못한 경우 None 반환
        return None

    # 출력이 텐서인 경우 검사
    if torch.is_tensor(out) and out.ndim == 3 and out.shape[1] > 16:
        # 패치 토큰 반환 (CLS 제외)
        return (out[:, 1:, :].squeeze(0) if out.shape[1] > 1 else out.squeeze(0)).contiguous()

    # 찾지 못한 경우 None 반환
    return None

# 코사인 유사도 계산 함수
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    코사인 유사도를 계산한다.
    """
    # L2 정규화 후 내적 계산
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float((a * b).sum().item())


def kp_threshold( #LEGACY: apply_keypoint_threshold
    # tokens: 필터링할 토큰 텐서
    # indx_map: 각 토큰에 대한 인덱스 매핑 텐서
    # threshold: 필터링 임계값
    
    tokens: torch.Tensor,
    idx_map: torch.Tensor,
    threshold: float,

    # 반환값: 필터링된 토큰 텐서와 인덱스 매핑 텐서의 튜플
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    토큰 L2-노름 기반의 임계값 필터링을 수행한다. 모든 토큰이 걸러지는 경우
    최고 점수 토큰을 하나 남겨 매칭 단계가 비어 있지 않도록 보장한다.
    """

    # 토큰이 비어있는 경우 바로 반환
    if tokens.numel() == 0:
        return tokens, idx_map

    # 토큰의 L2-노름 계산
    scores = torch.linalg.norm(tokens, dim=1)
    
    # 정규화
    min_s = scores.min() 
    max_s = scores.max()
    
    # 정규화된 점수 계산
    if (max_s - min_s).abs() < 1e-6:
        # 모든 점수가 동일한 경우 모든 토큰 선택
        normalized = torch.ones_like(scores)
    else:
        # 정규화된 점수 계산
        normalized = (scores - min_s) / (max_s - min_s + 1e-6)

    # 임계값 기반 마스크 생성
    mask = normalized >= threshold
    
    # 모든 토큰이 걸러지는 경우 최고 점수 토큰 하나 선택
    if not torch.any(mask):
        
        # 가장 높은 정규화 점수의 인덱스 찾기
        top_idx = torch.argmax(normalized)
        # 해당 인덱스의 마스크를 True로 설정
        mask[top_idx] = True

    # 마스크에 따라 토큰과 인덱스 매핑 필터링
    keep_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    
    # 필터링된 토큰과 인덱스 매핑 반환
    filtered_tokens = tokens.index_select(0, keep_idx)
    
    # 필터링된 인덱스 매핑 반환
    filtered_idx_map = idx_map.index_select(0, keep_idx)

    # 반환    
    return filtered_tokens, filtered_idx_map


def patch2grid( #LEGACY: reshape_patch_tokens_to_grid
    tokens: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]] = None,
    keep_batch: bool = False,
) -> torch.Tensor:
    """
    패치 토큰을 (grid_h, grid_w[, embed_dim]) 형태의 격자로 재배열한다.

    tokens:
        - (N, C) 또는 (B, N, C) 형태의 패치 토큰 텐서.
    grid_hw:
        - (grid_h, grid_w)를 명시하면 해당 격자로 reshape.
        - None이면 토큰 수가 정사각형이라고 가정하고 자동으로 sqrt(n)을 사용한다.
    keep_batch:
        - 입력이 배치 차원을 포함(B, N, C)할 때 True이면 (B, grid_h, grid_w, C)를 유지.
          False이면 배치가 1이라고 가정하고 squeeze(0) 수행.
    """
    if tokens.ndim == 3:
        batch, num_tokens, dim = tokens.shape
        tok = tokens
    elif tokens.ndim == 2:
        batch, num_tokens, dim = None, tokens.shape[0], tokens.shape[1]
        tok = tokens.unsqueeze(0)
    else:
        raise ValueError(f"tokens ndim must be 2 or 3, got {tokens.ndim}")

    if grid_hw is None:
        side = int(round(math.sqrt(num_tokens)))
        if side * side != num_tokens:
            raise ValueError(
                f"Unable to infer grid size: token count {num_tokens} is not a perfect square."
            )
        grid_hw = (side, side)

    gx, gy = grid_hw
    if gx * gy != num_tokens:
        raise ValueError(
            f"grid_hw {grid_hw} does not match token count {num_tokens}."
        )

    reshaped = tok.reshape(tok.shape[0], gx, gy, dim).contiguous()

    if keep_batch:
        return reshaped if tokens.ndim == 3 else reshaped

    if reshaped.shape[0] != 1:
        raise ValueError("Cannot squeeze batch dimension when batch size != 1.")

    return reshaped.squeeze(0)

def compute_patch_matches(
    tokens_a: Optional[torch.Tensor],
    tokens_b: Optional[torch.Tensor],
    args,
):
    """패치 임베딩을 이용해 매칭 결과를 계산한다."""
    if tokens_a is None or tokens_b is None:
        return None, None, None

    orig_n_a = int(tokens_a.shape[0])
    orig_n_b = int(tokens_b.shape[0])

    ia_map = torch.arange(orig_n_a, device=tokens_a.device, dtype=torch.long)
    ib_map = torch.arange(orig_n_b, device=tokens_b.device, dtype=torch.long)

    if args.keypoint_th > 0.0:
        tokens_a, ia_map = kp_threshold(tokens_a, ia_map, args.keypoint_th)
        tokens_b, ib_map = kp_threshold(tokens_b, ib_map, args.keypoint_th)

    if args.max_features:
        tokens_a, subs_a = subsample_tokens(tokens_a, int(args.max_features))
        subs_a = subs_a.to(ia_map.device, dtype=torch.long)
        ia_map = ia_map.index_select(0, subs_a)

        tokens_b, subs_b = subsample_tokens(tokens_b, int(args.max_features))
        subs_b = subs_b.to(ib_map.device, dtype=torch.long)
        ib_map = ib_map.index_select(0, subs_b)

    pa_np = tokens_a.detach().cpu().float().numpy()
    pb_np = tokens_b.detach().cpu().float().numpy()

    topk_limit = int(args.max_features) if args.max_features else 400
    ia, ib, sim = matching_knn(pa_np, pb_np, k=1, topk=topk_limit)

    if sim.size > 0:
        keep = sim >= args.match_th
        if args.line_th > 0.0:
            rel_min = sim.max() * args.line_th
            keep = np.logical_and(keep, sim >= rel_min)
        if not np.any(keep):
            top_idx = int(np.argmax(sim))
            keep = np.zeros_like(sim, dtype=bool)
            keep[top_idx] = True
        ia = ia[keep]
        ib = ib[keep]
        sim = sim[keep]

    ia, ib, sim = enforced_matching(ia, ib, sim)

    ia_map_cpu = ia_map.detach().cpu()
    ib_map_cpu = ib_map.detach().cpu()

    if ia.size > 0:
        ia_full = ia_map_cpu[torch.from_numpy(ia)]
        ib_full = ib_map_cpu[torch.from_numpy(ib)]
    else:
        ia_full = torch.empty(0, dtype=torch.long)
        ib_full = torch.empty(0, dtype=torch.long)

    g_a = grid_side(orig_n_a)
    g_b = grid_side(orig_n_b)

    patch = dict(
        n_a=orig_n_a,
        n_b=orig_n_b,
        n_selected_a=int(ia_map.shape[0]),
        n_selected_b=int(ib_map.shape[0]),
        grid_g_a=(int(g_a) if g_a else None),
        grid_g_b=(int(g_b) if g_b else None),
        idx_a=ia_full.tolist(),
        idx_b=ib_full.tolist(),
        similarities=sim.tolist(),
    )

    return patch, pa_np, pb_np
