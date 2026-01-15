from __future__ import annotations

import time
from typing import Tuple

from imatch.faiss_runtime import faiss


def _build_index(db_vectors, use_gpu: bool) -> Tuple[faiss.Index, bool]:
    dim = db_vectors.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    gpu_ms = None
    used_gpu = False
    if use_gpu:
        try:
            tgpu = time.perf_counter()
            index = faiss.index_cpu_to_all_gpus(index)
            gpu_ms = (time.perf_counter() - tgpu) * 1000.0
            used_gpu = True
        except Exception as exc:  # pragma: no cover - GPU init errors
            print(f"[WARN] GPU init failed ({exc}), falling back to CPU index.")
    index.add(db_vectors)
    add_ms = (time.perf_counter() - t0) * 1000.0
    if gpu_ms is not None:
        print(
            f"[DEBUG] build_index: dim={dim}, cpu_init_ms={cpu_ms:.2f}, "
            f"gpu_xfer_ms={gpu_ms:.2f}, add_ms={add_ms:.2f}"
        )
    else:
        print(f"[DEBUG] build_index: dim={dim}, cpu_init_ms={cpu_ms:.2f}, add_ms={add_ms:.2f}")
    return index, used_gpu
