from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from imatch.faiss_runtime import faiss


@dataclass
class IndexSpec:
    index_type: str = "flat"
    metric: str = "ip"
    nlist: int = 4096
    nprobe: int = 16
    pq_m: int = 16
    pq_nbits: int = 8
    train_size: int = 100000


def _metric_const(metric: str) -> int:
    metric_l = metric.lower()
    if metric_l == "ip":
        return faiss.METRIC_INNER_PRODUCT
    if metric_l == "l2":
        return faiss.METRIC_L2
    raise ValueError(f"unsupported metric: {metric}")


def _normalized_index_type(index_type: str) -> str:
    key = index_type.lower().strip()
    aliases = {
        "flat-ip": "flat",
        "flat-l2": "flat",
        "ivf": "ivf-flat",
        "ivfflat": "ivf-flat",
        "ivf-flat": "ivf-flat",
        "ivfpq": "ivf-pq",
        "ivf-pq": "ivf-pq",
    }
    return aliases.get(key, key)


def _pick_train_vectors(db_vectors: np.ndarray, train_size: int) -> np.ndarray:
    n = int(db_vectors.shape[0])
    if train_size <= 0 or train_size >= n:
        return db_vectors
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=int(train_size), replace=False)
    return db_vectors[idx]


def _set_nprobe(index: faiss.Index, nprobe: int) -> None:
    try:
        setattr(index, "nprobe", int(nprobe))
    except Exception:
        pass


def _build_index_with_spec(db_vectors, use_gpu: bool, spec: IndexSpec) -> Tuple[faiss.Index, bool, Dict[str, Any]]:
    dim = int(db_vectors.shape[1])
    metric = spec.metric.lower()
    index_type = _normalized_index_type(spec.index_type)
    metric_const = _metric_const(metric)

    if index_type == "ivf-pq":
        if spec.pq_m <= 0:
            raise ValueError("pq_m must be > 0 for ivf-pq")
        if dim % int(spec.pq_m) != 0:
            raise ValueError(f"dim ({dim}) must be divisible by pq_m ({spec.pq_m})")

    t0 = time.perf_counter()
    train_ms = 0.0
    train_count = 0

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    elif index_type == "ivf-flat":
        quantizer = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, int(spec.nlist), metric_const)
        train_vecs = _pick_train_vectors(db_vectors, int(spec.train_size))
        train_count = int(train_vecs.shape[0])
        tt = time.perf_counter()
        index.train(train_vecs)
        train_ms = (time.perf_counter() - tt) * 1000.0
        _set_nprobe(index, min(int(spec.nprobe), int(spec.nlist)))
    elif index_type == "ivf-pq":
        quantizer = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            dim,
            int(spec.nlist),
            int(spec.pq_m),
            int(spec.pq_nbits),
            metric_const,
        )
        train_vecs = _pick_train_vectors(db_vectors, int(spec.train_size))
        train_count = int(train_vecs.shape[0])
        tt = time.perf_counter()
        index.train(train_vecs)
        train_ms = (time.perf_counter() - tt) * 1000.0
        _set_nprobe(index, min(int(spec.nprobe), int(spec.nlist)))
    else:
        raise ValueError(f"unsupported index_type: {spec.index_type}")

    cpu_ms = (time.perf_counter() - t0) * 1000.0
    gpu_ms = None
    used_gpu = False
    if use_gpu:
        try:
            tgpu = time.perf_counter()
            index = faiss.index_cpu_to_all_gpus(index)
            gpu_ms = (time.perf_counter() - tgpu) * 1000.0
            used_gpu = True
            if index_type in ("ivf-flat", "ivf-pq"):
                _set_nprobe(index, min(int(spec.nprobe), int(spec.nlist)))
        except Exception as exc:  # pragma: no cover - GPU init errors
            print(f"[WARN] GPU init failed ({exc}), falling back to CPU index.")

    index.add(db_vectors)
    add_ms = (time.perf_counter() - t0) * 1000.0
    if gpu_ms is not None:
        print(
            f"[DEBUG] build_index: type={index_type}, metric={metric}, dim={dim}, "
            f"cpu_init_ms={cpu_ms:.2f}, train_ms={train_ms:.2f}, gpu_xfer_ms={gpu_ms:.2f}, add_ms={add_ms:.2f}"
        )
    else:
        print(
            f"[DEBUG] build_index: type={index_type}, metric={metric}, dim={dim}, "
            f"cpu_init_ms={cpu_ms:.2f}, train_ms={train_ms:.2f}, add_ms={add_ms:.2f}"
        )

    meta: Dict[str, Any] = {
        "index_type": index_type,
        "metric": metric,
        "nlist": int(spec.nlist) if index_type in ("ivf-flat", "ivf-pq") else None,
        "nprobe": int(spec.nprobe) if index_type in ("ivf-flat", "ivf-pq") else None,
        "m": int(spec.pq_m) if index_type == "ivf-pq" else None,
        "nbits": int(spec.pq_nbits) if index_type == "ivf-pq" else None,
        "train_size": int(spec.train_size) if index_type in ("ivf-flat", "ivf-pq") else None,
        "train_count": train_count,
        "train_ms": train_ms,
    }
    return index, used_gpu, meta


def _build_index(db_vectors, use_gpu: bool) -> Tuple[faiss.Index, bool]:
    # Backward-compatible wrapper used by faiss_4Redis.py (FlatIP baseline).
    index, used_gpu, _ = _build_index_with_spec(
        db_vectors,
        use_gpu=use_gpu,
        spec=IndexSpec(index_type="flat", metric="ip"),
    )
    return index, used_gpu

