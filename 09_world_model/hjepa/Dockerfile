# H-JEPA (hierarchical JEPA world models) on a laptop GPU: planning only, no training.
# Code: github.com/kevinghst/H-JEPA at a pinned commit; checkpoints and data go in the volume at /data.
FROM ubuntu:22.04
ARG HJEPA_COMMIT=97054ec
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    HJEPA_HOME=/data HF_HOME=/data/hf PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libegl1 libgl1 libglib2.0-0 libosmesa6 ffmpeg && \
    rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    uv python install 3.11.13 && uv venv /opt/venv --python 3.11.13
RUN git clone https://github.com/kevinghst/H-JEPA.git /opt/H-JEPA && \
    cd /opt/H-JEPA && git checkout ${HJEPA_COMMIT}
# torch 2.7.1 built for CUDA 12.6: works with NVIDIA driver 560
RUN cd /opt/H-JEPA && UV_HTTP_TIMEOUT=900 VIRTUAL_ENV=/opt/venv uv pip install \
        --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \
        "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" "torchaudio==2.7.1+cu126" \
        -e ".[train,env]" huggingface_hub "imageio[ffmpeg]"
WORKDIR /opt/H-JEPA/h_jepa
