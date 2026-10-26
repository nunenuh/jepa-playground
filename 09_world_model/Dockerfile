# Runs a JEPA world model (LeWorldModel) planning in simulators on a local GPU.
# Inference only: training happens on Colab.
#
#   docker build -t jepa-wm 09_world_model
#   docker run --rm --gpus all -v jepa-wm-data:/data -v "$PWD/09_world_model/assets:/out" jepa-wm
#
# Simulators render off-screen (pygame for Push-T, EGL for MuJoCo envs) and
# every episode is saved as an mp4 under /out.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    SDL_VIDEODRIVER=dummy \
    STABLEWM_HOME=/data/stablewm \
    HF_HOME=/data/hf \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg \
        libegl1 libgl1 libgles2 libglib2.0-0 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv venv --python 3.10 /opt/venv

# stable-worldmodel pinned to a known commit (it ships the LeWM model, the
# simulators and the CEM planner); only the Push-T / MuJoCo env deps are added
# instead of the heavy [env] extra (Atari, Craftax/JAX, ...). PyTorch is pinned
# to a CUDA 12.6 build so it runs on NVIDIA drivers >= 560; transformers < 5
# because the released LeWM checkpoints use the 4.x ViT parameter names.
ARG SWM_COMMIT=21446f1ede6d5284e981bd7b47f432b994e6d812
RUN uv pip install --python /opt/venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \
        "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" "transformers>=4.50,<5" \
        "stable-worldmodel[train] @ git+https://github.com/galilai-group/stable-worldmodel@${SWM_COMMIT}" \
        pygame pymunk shapely opencv-python-headless "mujoco<3.12" gymnasium \
        h5py hdf5plugin "imageio[ffmpeg]" huggingface_hub scikit-learn

WORKDIR /work
COPY plan.py /work/plan.py
VOLUME ["/data", "/out"]
ENTRYPOINT ["python", "/work/plan.py"]
