#!/usr/bin/env bash
# Play with the JEPA world model in a window: ./09_world_model/play.sh [--model ours_epoch3] [--episode 2]
# Lets the container's root user draw on your X display for the session, and takes that back on exit.
set -euo pipefail
cd "$(dirname "$0")"
xhost +SI:localuser:root >/dev/null
trap 'xhost -SI:localuser:root >/dev/null' EXIT
docker run --rm -it --gpus all -e HF_HUB_OFFLINE=1 \
    -e DISPLAY="$DISPLAY" -e SDL_VIDEODRIVER=x11 -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v jepa-wm-data:/data -v "$PWD:/work" --entrypoint python jepa-wm /work/live.py "${@:---episode 2}"
