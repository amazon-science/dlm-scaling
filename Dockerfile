FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Patch known CVEs in the base image
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3.9-venv python3.9-distutils \
    git wget curl && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.4.1 + CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Flash Attention and its CUDA extensions
RUN pip install --no-cache-dir ninja packaging psutil
RUN pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

RUN git clone --branch v2.6.3 --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attn && \
    cd /tmp/flash-attn/csrc/rotary && pip install --no-build-isolation . && \
    cd /tmp/flash-attn/csrc/layer_norm && pip install --no-build-isolation . && \
    cd /tmp/flash-attn/csrc/xentropy && pip install --no-build-isolation . && \
    rm -rf /tmp/flash-attn

# Core dependencies (versions match environment.yaml)
RUN pip install --no-cache-dir \
    xformers==0.0.28.post1 \
    lightning==2.1.2 \
    pytorch-lightning==2.4.0 \
    lightning-utilities==0.11.8 \
    jsonargparse==4.33.2 \
    safetensors==0.4.5 \
    sentencepiece==0.1.99 \
    transformers==4.31.0 \
    datasets==3.0.1 \
    wandb==0.15.3 \
    numpy==1.25.0 \
    lm-eval==0.4.4 \
    einops==0.6.1 \
    bitsandbytes==0.43.1 \
    triton==3.0.0

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
