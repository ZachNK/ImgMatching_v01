from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class PCABasisError(RuntimeError):
    pass


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    comps = data.get("components")
    mean = data.get("mean")
    if comps is None:
        raise PCABasisError(f"\033[91m[Error] PCA basis file missing 'components': {path}\033[0m")
    if mean is None:
        mean = np.zeros(comps.shape[0], dtype=np.float32)
    return np.asarray(comps, dtype=np.float32), np.asarray(mean, dtype=np.float32)


def _load_npy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    comps = np.load(path)
    if comps.ndim != 2:
        raise PCABasisError(f"\033[91m[Error] PCA basis .npy must be 2D (C,D); got {comps.shape}\033[0m")
    mean = np.zeros(comps.shape[0], dtype=np.float32)
    return np.asarray(comps, dtype=np.float32), mean


@lru_cache(maxsize=4)
def load_pca_basis(path: str, target_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load PCA basis and mean; supports .npz with keys 'components' and 'mean' or a 2D .npy.
    Returns (components, mean) as torch tensors on CPU.
    """
    p = Path(path)
    if not p.exists():
        raise PCABasisError(f"\033[91m[Error] PCA basis file not found: {path}\033[0m")
    if p.suffix.lower() == ".npz":
        comps_np, mean_np = _load_npz(p)
    else:
        comps_np, mean_np = _load_npy(p)

    if comps_np.ndim != 2 or comps_np.shape[1] < target_dim:
        raise PCABasisError(
            f"\033[91m[Error] PCA basis has insufficient dimensions for target_dim={target_dim}: {comps_np.shape}\033[0m"
        )

    comps_t = torch.from_numpy(comps_np[:, :target_dim].copy())
    mean_t = torch.from_numpy(mean_np.copy())
    return comps_t, mean_t


def apply_pca(tokens: torch.Tensor, comps: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """
    Apply global PCA projection: (tokens - mean) @ comps
    tokens: (N, C), comps: (C, D), mean: (C,)
    """
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2D, got {tokens.shape}")
    if comps.ndim != 2:
        raise ValueError(f"components must be 2D, got {comps.shape}")
    if mean.ndim != 1 or mean.shape[0] != comps.shape[0]:
        raise ValueError(f"mean shape mismatch; expected ({comps.shape[0]},), got {mean.shape}")

    mean_t = mean.to(tokens.device, dtype=tokens.dtype)
    comps_t = comps.to(tokens.device, dtype=tokens.dtype)
    return (tokens - mean_t) @ comps_t
