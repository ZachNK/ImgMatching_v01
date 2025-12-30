FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

# Micromamba + faiss-gpu (CUDA 12.1, Python 3.11) in /opt/conda/envs/faiss
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba create -y -n faiss -c pytorch -c conda-forge \
      python=3.11 "faiss-gpu=1.10.0=py3.11_h4818125_0_cuda12.1.1" && \
    micromamba clean -ya

# Make the faiss environment the default Python on shell entry
ENV PATH="/opt/conda/envs/faiss/bin:/opt/conda/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace/project

RUN printf '#!/usr/bin/env bash\npython /workspace/project/run.py "$@"\n' \
      > /usr/local/bin/run \
 && printf '#!/usr/bin/env bash\npython /workspace/project/visualize.py "$@"\n' \
      > /usr/local/bin/vis \
 && chmod +x /usr/local/bin/run /usr/local/bin/vis

CMD ["bash", "-lc", "sleep infinity"]
