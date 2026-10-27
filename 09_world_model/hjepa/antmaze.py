"""Flat vs hierarchical JEPA world models planning in Visual AntMaze (H-JEPA's released checkpoints).

A 3D ant robot has to walk to a goal 3 maze cells away, seeing only a 64x64 camera image and a goal
image. LeWM plans flat (one level); H-JEPA plans top-down through 2 or 3 levels, each predicting
farther ahead. Uses H-JEPA's own evaluation (their 50 fixed tasks, planner settings and seed 42),
on the first N tasks, and saves a top-down video per task.

    docker run --rm --gpus all -v hjepa-data:/data -v "$PWD/09_world_model/hjepa:/work" hjepa \
        python /work/antmaze.py --tasks 10        # ~5 min per task per model on an RTX A2000 laptop
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/opt/H-JEPA/h_jepa")

import imageio.v2 as imageio
import mujoco
import numpy as np
from planning_eval import load_eval_config, run_planning_eval
from stable_worldmodel.envs.ogbench.antmaze_env import AntMazeEnv

MODELS = {"lewm": "ant_flat", "hjepa_l2": "ant_l2", "hjepa_l3": "ant_l3"}   # checkpoint -> matching planner
HOME = Path(os.environ.get("HJEPA_HOME", "/data"))
TOP_PX, EVERY = 320, 2                             # top-down video size, record every 2nd step


class TopDown:
    """Records a top-down view of the maze on every EVERY-th env step (the agent itself only sees 64x64)."""
    videos, enabled, _renderer, _cam, _n = [[]], False, None, None, 0

    @classmethod
    def patch(cls):
        step, reset = AntMazeEnv.step, AntMazeEnv.reset

        def recording_reset(env, *a, **kw):           # a reset starts a new task: new video
            out = reset(env, *a, **kw)
            if cls.enabled:
                cls.videos.append([])
            return out
        AntMazeEnv.reset = recording_reset

        def recording_step(env, action):
            out = step(env, action)
            if cls.enabled:
                cls._n += 1
                if cls._n % EVERY == 0:
                    cls.videos[-1].append(cls.render(env))
            return out
        AntMazeEnv.step = recording_step

    @classmethod
    def render(cls, env):
        inner = env.unwrapped                     # the OGBench MazeEnv (MuJoCo model + data)
        model, data = inner.model, inner.data
        if cls._renderer is None:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, TOP_PX)
            model.vis.global_.offheight = max(model.vis.global_.offheight, TOP_PX)
            cls._renderer = mujoco.Renderer(model, TOP_PX, TOP_PX)
            cls._cam = mujoco.MjvCamera()
            cls._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            lo, hi = model.geom_aabb[:, :3].min(0), model.geom_aabb[:, :3].max(0)
            xs = [model.geom_pos[i][:2] for i in range(model.ngeom) if model.geom_size[i][2] > 0.5]
            centre = np.mean(xs, 0) if xs else np.zeros(2)
            cls._cam.lookat[:] = [centre[0], centre[1], 0]
            cls._cam.distance, cls._cam.elevation, cls._cam.azimuth = 34.0, -90.0, 90.0
        cls._renderer.update_scene(data, cls._cam)
        return cls._renderer.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10, help="first N of H-JEPA's 50 fixed tasks")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/work/results")
    args = ap.parse_args()
    out = Path(args.out)
    TopDown.patch()
    results = {}
    for name in args.models:
        cfg = load_eval_config(MODELS[name])
        cfg.seed = args.seed
        cfg.policy = str(HOME / f"ckpts/ant/ant_{name}/seed{args.seed}/ant_{name}_object.ckpt")
        cfg.load_eval_trajs_path = str(HOME / "assets/eval_trajs/ant/expert_grid_d3_n50.pt")
        cfg.eval.num_eval = args.tasks
        cfg.eval.chunk_size = 1                   # one task per World: one video per task
        run_dir = out / name
        TopDown.enabled, TopDown.videos = True, [[]]
        m = run_planning_eval(cfg, results_dir=run_dir)
        TopDown.enabled = False
        results[name] = {"success_rate": float(m["success_rate"]),
                         "episode_successes": [bool(x) for x in m["episode_successes"]],
                         "steps_to_success": [None if not np.isfinite(x) else int(x) for x in m["steps_to_success"]],
                         "evaluation_time_s": float(m.get("evaluation_time", 0))}
        for i, frames in enumerate(v for v in TopDown.videos if len(v) > 5):
            imageio.mimsave(run_dir / f"task{i:02d}_topdown.mp4", frames, fps=30)
        print(name, json.dumps(results[name]), flush=True)
    (out / "results_antmaze.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
