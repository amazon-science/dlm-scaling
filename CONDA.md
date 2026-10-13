# Environment setup

Step-by-step equivalent of the [Dockerfile](Dockerfile): PyTorch 2.4.1 on CUDA 12.1, Flash
Attention 2.6.3 together with its `rotary`, `layer_norm` and `xentropy` CUDA extensions,
xFormers 0.0.28 and lm-evaluation-harness 0.4.4. Every version pinned below is the one the
Dockerfile pins; [environment.yaml](environment.yaml) is the fully resolved lock of the same
environment, including transitive dependencies.

Linux x86-64 only. You need a CUDA 12.1-compatible toolchain with `nvcc` on `PATH` — the
Dockerfile gets this from `nvidia/cuda:12.1.1-devel-ubuntu22.04` — and an NVIDIA GPU.

## 1. Interpreter

```sh
conda create -n dlm-scaling python=3.9
conda activate dlm-scaling
pip install --upgrade pip
pip install setuptools==75.1.0 wheel
```

## 2. PyTorch 2.4.1 + CUDA 12.1

```sh
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

## 3. Flash Attention 2.6.3 and its CUDA extensions

Install the build dependencies first. Torch has to be installed already (step 2), and every
install in this section needs `--no-build-isolation`: each `setup.py` imports `torch` at build
time, which an isolated build environment does not have.

```sh
pip install ninja==1.11.1.1 packaging==24.1 psutil==5.9.8
```

These are compiled from source for every supported GPU architecture. If the build runs out of
memory, cap the number of parallel `nvcc` jobs with e.g. `export MAX_JOBS=4`.

```sh
git clone --branch v2.6.3 --depth 1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && pip install --no-build-isolation .
cd csrc/rotary     && pip install --no-build-isolation .
cd ../layer_norm   && pip install --no-build-isolation .
cd ../xentropy     && pip install --no-build-isolation .
cd ../../.. && rm -rf flash-attention
```

The three `csrc` extensions are required: the model code imports `rotary_emb`,
`dropout_layer_norm` and `xentropy_cuda_lib` directly. They are not published on PyPI, so this
is the only way to obtain them, and they must come from the same `v2.6.3` checkout as the main
package so that they stay version-matched.

## 4. xFormers

```sh
pip install xformers==0.0.28.post1
```

Keep this pinned and do not pass `-U`. Recent xFormers releases require `torch>=2.10`, so an
unpinned upgrade silently pulls torch out from under the extensions compiled in step 3 and
breaks them at import time. Pinning `0.0.28.post1` is also a safeguard in the other direction:
it declares `torch==2.4.1` as a hard requirement.

## 5. Remaining dependencies

Identical pins to the Dockerfile.

```sh
pip install \
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
```

## 6. Verify

```sh
pip check
python -c "
import torch, flash_attn, xformers
import rotary_emb, dropout_layer_norm, xentropy_cuda_lib
print('torch     ', torch.__version__, '/ cuda', torch.version.cuda)
print('flash-attn', flash_attn.__version__)
print('xformers  ', xformers.__version__)
print('gpu       ', torch.cuda.is_available())
"
```

`pip check` should report no broken requirements, and the versions printed should be torch
2.4.1+cu121 on CUDA 12.1, Flash Attention 2.6.3 and xFormers 0.0.28.post1. A failure to import
any of `rotary_emb`, `dropout_layer_norm` or `xentropy_cuda_lib` means step 3 did not complete;
`no kernel image is available for execution on the device` at the first `FusedRMSNorm` call
instead means they compiled, but without kernels for your GPU's compute capability.
