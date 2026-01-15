from __future__ import annotations

import sys

try:
    import faiss  # type: ignore
except ImportError as err:  # pragma: no cover
    print(
        "[FATAL] faiss is not installed. Install faiss-cpu or faiss-gpu before running.\n"
        "  pip install faiss-cpu  # or faiss-gpu\n"
        f"  import error: {err}"
    )
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

__all__ = ("faiss", "tqdm")
