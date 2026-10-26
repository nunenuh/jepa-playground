"""Plan with a JEPA world model (LeWorldModel) in the Push-T simulator.

Follows LeWM's official evaluation settings (lucas-maes/le-wm, eval.py + config/eval/pusht.yaml):
start from a state of an expert episode, the goal is the state 25 steps later,
the planner (CEM in latent space) gets 50 steps to reach it. Compared with a
random policy. Saves results.json and one mp4 per episode.

Data: the 21 Push-T validation episodes released with H-JEPA (16 MB) instead of
LeWM's 13 GB training set; the action/state normalisers are fit on them.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from huggingface_hub import hf_hub_download
from sklearn import preprocessing
from stable_worldmodel.planning import CEMSolver, GoalMSE, ShootingCostEvaluator
from torchvision.transforms import v2 as transforms

KEYS = ["action", "proprio", "state"]
GOAL_OFFSET, BUDGET = 25, 50                   # as in LeWM's pusht.yaml
PLAN = dict(horizon=5, receding_horizon=5, action_block=5)
CEM = dict(batch_size=1, num_samples=300, var_scale=1.0, n_steps=30, topk=30)
DATA_REPO, DATA_FILE = "jepa-world-models/h-jepa", "pusht_expert_val.h5"


def img_transform():
    return transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size=224)])


def load_data():
    home = Path(os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser()
    hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset", local_dir=home / "datasets")
    return swm.data.HDF5Dataset(DATA_FILE.removesuffix(".h5"), keys_to_cache=KEYS)


def pick_starts(dataset, n, seed):
    """n (episode, start step) pairs whose goal (start + GOAL_OFFSET) is inside the episode."""
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep, step = dataset.get_col_data(col), dataset.get_col_data("step_idx")
    ep_len = {e: step[ep == e].max() + 1 for e in np.unique(ep)}
    valid = np.nonzero(step <= np.array([ep_len[e] for e in ep]) - GOAL_OFFSET - 1)[0]
    rows = np.sort(np.random.default_rng(seed).choice(valid, size=n, replace=False))
    data = dataset.get_row_data(rows)
    return data[col].tolist(), data["step_idx"].tolist()


def make_policy(name, dataset, seed):
    if name == "random":
        return swm.policy.RandomPolicy()
    process = {}
    for k in KEYS:
        x = dataset.get_col_data(k)
        process[k] = preprocessing.StandardScaler().fit(x[~np.isnan(x).any(axis=1)])
        if k != "action":
            process[f"goal_{k}"] = process[k]
    model = swm.wm.utils.load_pretrained(name).to("cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    cost = ShootingCostEvaluator(model, GoalMSE())        # distance to the goal's latent after the imagined rollout
    solver = CEMSolver(cost=cost, device="cuda", seed=seed, **CEM)
    return swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**PLAN), process=process,
                                       transform={"pixels": img_transform(), "goal": img_transform()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["quentinll/lewm-pusht", "random"],
                    help="HF repo ids or checkpoint paths (relative to $STABLEWM_HOME/checkpoints), or 'random'")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/out")
    args = ap.parse_args()

    dataset = load_data()
    episodes, starts = pick_starts(dataset, args.episodes, args.seed)
    callables = [{"method": "_set_state", "args": {"state": {"value": "state"}}},
                 {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}}]
    results = {"env": "swm/PushT-v1", "episodes": args.episodes, "goal_offset": GOAL_OFFSET,
               "eval_budget": BUDGET, "plan": PLAN, "cem": CEM, "data": f"{DATA_REPO}/{DATA_FILE}",
               "gpu": torch.cuda.get_device_name(0), "policies": {}}
    for name in args.policies:
        world = swm.World(env_name="swm/PushT-v1", num_envs=args.episodes,
                          max_episode_steps=2 * BUDGET, image_shape=(224, 224))
        world.set_policy(make_policy(name, dataset, args.seed))
        videos = Path(args.out, "videos", name.replace("/", "_"))
        videos.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        m = world.evaluate(dataset=dataset, episodes_idx=episodes, start_steps=starts, goal_offset=GOAL_OFFSET,
                           eval_budget=BUDGET, callables=callables, video=videos)
        results["policies"][name] = {"success_rate": float(m["success_rate"]),
                                     "episode_successes": np.asarray(m["episode_successes"]).astype(bool).tolist(),
                                     "seconds": round(time.time() - t0, 1)}
        print(f"{name}: success {m['success_rate']:.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    Path(args.out, "results.json").write_text(json.dumps(results, indent=1))
    owner = os.stat(args.out)                       # hand files back to the host user, not root
    for p in [Path(args.out), *Path(args.out).rglob("*")]:
        os.chown(p, owner.st_uid, owner.st_gid)


if __name__ == "__main__":
    main()
