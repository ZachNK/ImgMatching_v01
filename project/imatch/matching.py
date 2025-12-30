# project/imatch/matching.py
import math
import numpy as np
import torch
from typing import Tuple

"""

"""

def matching_knn(pa: np.ndarray, pb: np.ndarray, k: int = 5, topk: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mutual k-NN 매칭:
    - a의 상위 k 이웃에 b가 포함되고 AND b의 상위 k 이웃에 a가 포함되면 매칭
    - 유사도 내림차순 상위 topk만 남김
    returns: (idx_a[K], idx_b[K], sim[K])
    """
    assert k >= 1
    pa = pa / (np.linalg.norm(pa, axis=1, keepdims=True) + 1e-9)
    pb = pb / (np.linalg.norm(pb, axis=1, keepdims=True) + 1e-9)
    S = pa @ pb.T  # [Na, Nb]
    Na, Nb = S.shape

    kk_a = min(k, Nb)
    kk_b = min(k, Na)

    # 각 a별 상위 k의 b 인덱스
    A2B_k = np.argpartition(-S, kk_a - 1, axis=1)[:, :kk_a]
    row_idx = np.arange(Na)[:, None]
    A2B_k = A2B_k[row_idx, np.argsort(-S[row_idx, A2B_k], axis=1)]

    # 각 b별 상위 k의 a 인덱스
    B2A_k = np.argpartition(-S, kk_b - 1, axis=0)[:kk_b, :]
    col_idx = np.arange(Nb)[None, :]
    B2A_k = B2A_k[np.argsort(-S[B2A_k, col_idx], axis=0), col_idx]

    B_sets = [set(B2A_k[:, j].tolist()) for j in range(Nb)]

    cand_ia, cand_ib, cand_sim = [], [], []
    for i in range(Na):
        for j in A2B_k[i]:
            if i in B_sets[j]:
                cand_ia.append(i); cand_ib.append(j); cand_sim.append(float(S[i, j]))
    if not cand_sim:
        return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=np.float32))

    ia = np.array(cand_ia, dtype=int)
    ib = np.array(cand_ib, dtype=int)
    sim = np.array(cand_sim, dtype=np.float32)

    keep = min(topk, sim.size)
    order = np.argsort(-sim)[:keep]
    return ia[order], ib[order], sim[order]

def enforced_matching(
    ia: np.ndarray,
    ib: np.ndarray,
    sim: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy 1:1 selection of matches based on descending similarity.
    활용: mutual k-NN 매칭 후 1:1 매칭 강제
    - 입력:
      ia: np.ndarray — 이미지 A의 매칭된 인덱스 배열
      ib: np.ndarray — 이미지 B의 매칭된 인덱스 배열
      sim: np.ndarray — 매칭 유사도 배열    
    - 출력:
      Tuple[np.ndarray, np.ndarray, np.ndarray] — 선택된 매칭의 ia, ib, sim 배열 튜플
    """
    if ia.size == 0:
        return ia, ib, sim

    order = np.argsort(-sim)
    used_a = set()
    used_b = set()
    sel_a = []
    sel_b = []
    sel_sim = []

    for idx in order:
        a = int(ia[idx])
        b = int(ib[idx])
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        sel_a.append(a)
        sel_b.append(b)
        sel_sim.append(sim[idx])

    if not sel_a:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=np.float32)

    return (
        np.array(sel_a, dtype=int),
        np.array(sel_b, dtype=int),
        np.array(sel_sim, dtype=np.float32),
    )

def grid_side(n: int):
    """
    n 이 완전제곱수이면 sqrt(n)을 반환, 아니면 None
    """
    g = int(round(math.sqrt(n)))
    return g if g * g == n else None

def subsample_tokens(tok: torch.Tensor, maxn: int):
    """
    토큰을 균등 간격으로 서브샘플링. 원본 인덱스 매핑도 반환.
    """
    if maxn <= 0 or tok.shape[0] <= maxn:
        return tok, torch.arange(tok.shape[0])
    idx = torch.linspace(0, tok.shape[0] - 1, steps=maxn).round().long()
    return tok.index_select(0, idx), idx
