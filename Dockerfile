# splat 3D-human / diffusion stack for vast.ai RTX 4090 (sm_89).
# Bakes the SLOW, fragile, recompiled-every-time parts so a fresh box is ready in ~2-4 min
# instead of the 15-25 min from-source CUDA build that has cost hours (pytorch3d build-isolation/
# pkg_resources, nvdiffrast JIT, gaussian rasterizer). Built unattended on GitHub Actions:
# FORCE_CUDA + TORCH_CUDA_ARCH_LIST drive nvcc at build time with NO GPU present.
#
# Boot:  vastai create instance <offer> --image ghcr.io/briantanabe/splat-3dhuman:cu121 --disk 120 --ssh --direct
# python on the image is the system python3 (torch installed into it).
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.9+PTX" \
    FORCE_CUDA=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential ninja-build cmake ffmpeg \
        python3 python3-pip python3-dev python-is-python3 \
        libgl1-mesa-glx libglib2.0-0 libegl1 libgles2 libglvnd-dev pkg-config rclone && \
    rm -rf /var/lib/apt/lists/*

# torch FIRST (cu121) so every --no-build-isolation extension links the right ABI.
RUN python -m pip install --upgrade "pip<25" && \
    python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Build prerequisites that MUST be present in-env for --no-build-isolation extensions.
# setuptools<81 keeps pkg_resources available (its removal is what broke pytorch3d's build).
RUN python -m pip install "setuptools<81" wheel ninja fvcore iopath "numpy<2"

# pytorch3d from source — the #1 recurring failure. --no-build-isolation + FORCE_CUDA compiles
# sm_89 kernels with no GPU present.
RUN python -m pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# NOTE: nvdiffrast is intentionally NOT baked — it cannot build without a GPU/GL context and it
# JIT-compiles its CUDA/GL kernels at first import on the box anyway, so baking gives nothing and
# breaks the GPU-less Actions build. Install it at runtime (seconds): pip install nvdiffrast.

# Gaussian-splatting rasterizers + knn (graphdeco standard + the leo-frank depth fork ExAvatar uses).
RUN python -m pip install --no-build-isolation \
        "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git" \
        "git+https://gitlab.inria.fr/bkerbl/simple-knn.git" || true

# 3D-human / mesh / IO tooling (PSHuman, MagicMan, mesh->gaussian conversion, bg removal).
RUN python -m pip install \
        trimesh pymeshlab xatlas rembg onnxruntime-gpu open3d plyfile \
        opencv-python-headless pillow imageio imageio-ffmpeg einops omegaconf \
        smplx scikit-image scipy "huggingface_hub>=0.23" safetensors accelerate

# Diffusion + face stack (diffusers left UNPINNED-ish; individual models pin their own version at
# runtime in <1 min — the slow part is baked above). gfpgan/insightface for any face work.
RUN python -m pip install \
        transformers==4.46.3 diffusers==0.31.0 \
        gfpgan insightface peft kornia && \
    # patch basicsr's torchvision.functional_tensor import (removed in new torchvision) AT BUILD TIME
    python - <<'PY'
import importlib.util, os, re
spec = importlib.util.find_spec("basicsr")
if spec:
    deg = os.path.join(os.path.dirname(spec.origin), "data", "degradations.py")
    if os.path.exists(deg):
        s = open(deg).read().replace("torchvision.transforms.functional_tensor",
                                     "torchvision.transforms.functional")
        open(deg, "w").write(s)
        print("patched", deg)
PY

# Sanity: confirm the heavy imports resolve at build time (catches a broken layer before push).
RUN python -c "import torch, pytorch3d, trimesh, diffusers, gfpgan, insightface; \
print('BUILD OK torch', torch.__version__, '| pytorch3d', pytorch3d.__version__)"

WORKDIR /workspace
CMD ["/bin/bash"]
