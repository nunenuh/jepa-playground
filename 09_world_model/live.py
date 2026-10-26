"""Play with a JEPA world model: Push-T live in a window.

The JEPA (LeWorldModel) plans in latent space and pushes the T toward a goal.
You can take over the pointer, drag the T somewhere impossible, or freeze the
world; the surprise meter shows how wrong the model's prediction was.

Controls
    SPACE      who drives: JEPA (plans) / YOU (pointer follows the mouse)
    drag T     grab the T and teleport it (cheating: watch the surprise spike)
    G          put the goal T where the mouse is;  Q / E rotate it
    F          freeze the world for one block (the pointer "moves" but nothing happens)
    R          new random start          ESC  quit

Run: ./09_world_model/play.sh   (Docker image from this folder, window on your desktop)
"""
import argparse
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # no sound card in Docker

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401  (pixels are blosc-compressed)
import numpy as np
import pygame
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

FS = 5                                             # env steps per action block
PLAN_T, SAMPLES, ITERS, TOPK = 5, 300, 30, 30      # CEM, as in LeWM's Push-T eval
EXECUTE = int(os.environ.get("EXECUTE", 5))                                        # blocks executed before replanning; 5 as in the eval (replanning every 2 blocks: 43% vs 70%)
SUBGOAL, BUDGET = 25, 10                         # subgoal every 25 steps, 10 blocks (50 steps) each, as in the eval
GOAL_VIEWS, GOAL_RING = 16, 90                   # goal images per goal, pointer distance from the T
SIM, PANEL, GOAL_PX = 512, 400, 170                # window layout (pixels)
MEAN = torch.tensor(spt.data.dataset_stats.ImageNet["mean"]).view(1, 3, 1, 1)
STD = torch.tensor(spt.data.dataset_stats.ImageNet["std"]).view(1, 3, 1, 1)
DATA = Path(os.environ.get("STABLEWM_HOME", "/data/stablewm"), "datasets", "pusht_expert_val.h5")


class Jepa:
    def __init__(self, name):
        self.model = swm.wm.utils.load_pretrained(name).to("cuda").eval().requires_grad_(False)
        with h5py.File(DATA) as f:                 # action normaliser, as in plan.py
            a = f["action"][:]
        a = a[~np.isnan(a).any(1)]
        self.a_mean, self.a_std = a.mean(0), a.std(0)
        self.a_lo, self.a_hi = np.percentile(a, 0.5, 0), np.percentile(a, 99.5, 0)
        self.warm = None

    @torch.no_grad()
    def encode(self, frame):
        x = torch.from_numpy(frame).permute(2, 0, 1)[None].float() / 255
        return self.model.encode({"pixels": ((x - MEAN) / STD).cuda()[:, None]})["emb"][0, 0]

    def norm_block(self, actions):                 # (FS, 2) raw -> (10,) normalised
        return torch.from_numpy(((actions - self.a_mean) / self.a_std).reshape(-1)).float().cuda()

    @torch.no_grad()
    def surprise(self, embs, blocks, now):
        """Squared error between the predicted and the actual latent of the newest frame."""
        hist = torch.stack(list(embs))[None]
        act = self.model.action_encoder(torch.stack(list(blocks))[None])
        pred = self.model.predict(hist, act)[0, -1]
        return float(((pred - now) ** 2).mean())

    @torch.no_grad()
    def plan(self, emb, goal):
        """CEM in latent space from the current frame: returns EXECUTE raw action blocks (EXECUTE, FS, 2)."""
        mean = torch.zeros(PLAN_T, FS * 2, device="cuda")
        if self.warm is not None:                  # warm start: shift the previous plan
            mean[:PLAN_T - EXECUTE] = self.warm[EXECUTE:]
        std = torch.ones_like(mean)
        for _ in range(ITERS):
            cand = mean + std * torch.randn(SAMPLES, PLAN_T, FS * 2, device="cuda")
            cand[0] = mean
            info = {"pixels": torch.zeros(1, SAMPLES, 1, 1), "emb": emb.view(1, 1, 1, -1).expand(1, SAMPLES, 1, -1)}
            pred = self.model.rollout(info, cand[None])["predicted_emb"][0, :, -1]
            cost = ((pred - goal) ** 2).sum(-1)
            elite = cand[cost.topk(TOPK, largest=False).indices]
            mean, std = elite.mean(0), elite.std(0)
        self.warm = mean
        raw = mean[:EXECUTE].view(EXECUTE, FS, 2).cpu().numpy() * self.a_std + self.a_mean
        return raw                                 # unclipped, as in swm's WorldModelPolicy


class Game:
    """An expert episode from the validation set as a chain of subgoals, each SUBGOAL steps ahead (LeWM's eval
    setting); after BUDGET blocks on one subgoal the chain moves on anyway. G replaces it with your own goal."""

    def __init__(self, jepa, episode):
        self.jepa, self.episode = jepa, episode
        with h5py.File(DATA) as f:
            self.states, self.offsets, self.lengths = f["state"][:], f["ep_offset"][:], f["ep_len"][:]
        self.env = gym.make("swm/PushT-v1", resolution=224).unwrapped
        self.goal_env = gym.make("swm/PushT-v1", resolution=224).unwrapped
        self.env.reset(seed=0)
        self.goal_env.reset(seed=0)
        self.reset()

    def reset(self):
        e = self.episode % len(self.offsets)
        self.episode += 1
        self.first, self.last = int(self.offsets[e]), int(self.offsets[e] + self.lengths[e] - 1)
        self.env._set_state(self.states[self.first].astype(np.float64))
        self.embs, self.blocks = deque(maxlen=3), deque(maxlen=3)
        self.trace, self.events = deque(maxlen=120), deque(maxlen=120)
        self.embs.append(self.jepa.encode(self.env.render()))
        self.reached = 0
        self.set_subgoal(self.first + SUBGOAL)

    def set_subgoal(self, idx):
        self.sub_idx, self.custom, self.on_goal = min(idx, self.last), False, 0
        self.goal_state = self.states[self.sub_idx].astype(np.float64)
        self.goal_pose = self.goal_state[2:5]
        self.goal_env._set_state(self.goal_state)
        self.goal_img = self.goal_env.render()
        self.goal_emb = self.jepa.encode(self.goal_img)
        self.jepa.warm = None
        self.queue = []

    def check_goal(self, hit):
        """After each block: next subgoal when reached or out of budget. Returns True when the episode is over."""
        self.on_goal += 1
        if self.custom:
            return False
        if hit:
            self.reached += 1
        if hit or self.on_goal >= BUDGET:
            if self.sub_idx >= self.last:
                return True
            self.set_subgoal(self.sub_idx + SUBGOAL)
        return False

    def hit(self):
        s = self.state()
        if self.custom:
            d_pos = np.linalg.norm(s[2:4] - self.goal_pose[:2])
            d_ang = abs((s[4] - self.goal_pose[2] + np.pi) % (2 * np.pi) - np.pi)
            return d_pos < 20 and d_ang < np.radians(15)
        return bool(self.env.eval_state(self.goal_state, s)[0])

    def set_goal(self, pos, angle):
        """Your own goal (G). Experimental: usually beyond LeWM's ~25-step planning horizon."""
        self.custom, self.on_goal = True, 0
        self.goal_pose = np.array([*pos, angle], dtype=np.float64)
        # goal latent = average over goal images with the pointer at 16 spots around the T: the planner then
        # aims for the T's pose, not for one particular pointer position (one fixed spot: 0/20 near goals)
        embs = []
        for k in range(GOAL_VIEWS):
            a = 2 * np.pi * k / GOAL_VIEWS
            agent = np.clip(np.asarray(pos) + GOAL_RING * np.array([np.cos(a), np.sin(a)]), 20, 490)
            self.goal_env._set_state(np.r_[agent, pos, angle].astype(np.float64))
            embs.append(self.jepa.encode(self.goal_env.render()))
        self.goal_emb = torch.stack(embs).mean(0)
        self.goal_env._set_state(np.r_[np.clip(np.asarray(pos) + [0, GOAL_RING], 20, 490), pos, angle])
        self.goal_img = self.goal_env.render()
        self.jepa.warm = None
        self.queue = []

    def state(self):
        return np.asarray(self.env._get_obs(), dtype=np.float64)

    def teleport(self, pos=None, dangle=0.0):
        s = self.state()
        if pos is not None:
            s[2:4] = pos
        s[4] += dangle
        self.env._set_state(s)

    def next_block(self, manual, mouse):
        """Actions for the next FS steps: from the plan, or a placeholder filled step by step from the mouse."""
        if manual:
            return None, None
        t0 = time.time()
        if self.queue:
            return self.queue.pop(0), None
        self.queue = list(self.jepa.plan(self.embs[-1], self.goal_emb))
        return self.queue.pop(0), time.time() - t0

    def step(self, action):
        self.env.step(action)

    def end_block(self, executed, event):
        """Called after FS steps: encode the new frame, score surprise, keep the 3-frame history."""
        now = self.jepa.encode(self.env.render())
        self.blocks.append(self.jepa.norm_block(executed))
        s = self.jepa.surprise(self.embs, self.blocks, now) if len(self.embs) == 3 else 0.0
        self.trace.append(s)
        self.events.append(event)
        self.embs.append(now)
        return s


def draw(screen, font, game, manual, plan_s, last_s, typical):
    screen.fill((245, 245, 245))
    game.env.render_size = SIM                     # full-resolution frame for the window only
    frame = game.env.render()
    game.env.render_size = 224
    screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
    gx, gy, ga = game.goal_pose                    # goal ghost: a red cross where the goal T sits
    pygame.draw.line(screen, (220, 40, 40), (gx - 10, gy), (gx + 10, gy), 3)
    pygame.draw.line(screen, (220, 40, 40), (gx, gy - 10), (gx, gy + 10), 3)

    x0 = SIM + 15
    goal = pygame.transform.smoothscale(pygame.surfarray.make_surface(game.goal_img.swapaxes(0, 1)), (GOAL_PX, GOAL_PX))
    screen.blit(font.render("goal image (what the JEPA aims for)", True, (40, 40, 40)), (x0, 8))
    screen.blit(goal, (x0, 28))

    s = game.state()
    d_pos = np.linalg.norm(s[2:4] - game.goal_pose[:2])
    d_ang = np.degrees(abs((s[4] - game.goal_pose[2] + np.pi) % (2 * np.pi) - np.pi))
    n_sub = -(-(game.last - game.first) // SUBGOAL)
    k_sub = -(-(game.sub_idx - game.first) // SUBGOAL)
    goal_txt = "your goal (G), experimental" if game.custom else \
        f"expert episode {(game.episode - 1) % len(game.offsets)}: subgoal {k_sub}/{n_sub}, reached {game.reached}"
    lines = [f"driver: {'YOU (mouse)' if manual else 'JEPA (planning)'}", goal_txt,
             f"T to goal: {d_pos:5.0f} px, {d_ang:4.0f} deg  (block {game.on_goal}/{BUDGET})",
             f"last plan: {plan_s * 1000:4.0f} ms" if not manual else "move the mouse to push the T",
             f"surprise: {last_s:.3f}  (normal ~{typical:.3f})"]
    for i, t in enumerate(lines):
        screen.blit(font.render(t, True, (20, 20, 20)), (x0, 205 + 18 * i))

    gy0, gh, gw = 305, 150, PANEL - 30             # surprise graph, log scale
    pygame.draw.rect(screen, (255, 255, 255), (x0, gy0, gw, gh))
    pygame.draw.rect(screen, (180, 180, 180), (x0, gy0, gw, gh), 1)
    lo, hi = np.log10(0.005), np.log10(3.0)
    y = lambda v: gy0 + gh - (np.log10(max(v, 0.005)) - lo) / (hi - lo) * gh
    pygame.draw.line(screen, (120, 180, 120), (x0, y(typical * 3)), (x0 + gw, y(typical * 3)), 1)
    tr, ev = list(game.trace), list(game.events)
    pts = [(x0 + gw - (len(tr) - 1 - i) * gw / 120, y(v)) for i, v in enumerate(tr)]
    for (p, e) in zip(pts, ev):
        if e:
            pygame.draw.line(screen, (255, 200, 120), (p[0], gy0), (p[0], gy0 + gh), 2)
    if len(pts) > 1:
        pygame.draw.lines(screen, (220, 60, 30), False, pts, 2)
    screen.blit(font.render("surprise (log); green = 3x normal; orange = you cheated", True, (60, 60, 60)),
                (x0, gy0 + gh + 4))
    help_ = "SPACE driver | drag T | G goal, Q/E turn | F freeze | R next"
    screen.blit(font.render(help_, True, (90, 90, 90)), (x0 - 5, SIM - 22))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ours_epoch3", help="checkpoint under $STABLEWM_HOME/checkpoints or HF id")
    ap.add_argument("--episode", type=int, default=0, help="first validation episode (0-20)")
    ap.add_argument("--headless-blocks", type=int, default=0, help="test mode: run N blocks, save a screenshot")
    ap.add_argument("--eval-episodes", type=int, default=0, help="test mode: run N episodes without a window")
    args = ap.parse_args()
    if args.eval_episodes:
        return evaluate(args)
    if args.headless_blocks:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    screen = pygame.display.set_mode((SIM + PANEL, SIM))
    pygame.display.set_caption(f"JEPA world model live: {args.model}")
    font = pygame.font.SysFont("dejavusans", 13)
    game = Game(Jepa(args.model), args.episode)
    manual, dragging, goal_angle = False, False, game.goal_pose[2]
    plan_s, last_s, typical, n_blocks = 0.0, 0.0, 0.05, 0
    clock = pygame.time.Clock()

    while True:
        event_flag, freeze = False, False
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    manual = not manual
                    game.queue, game.jepa.warm = [], None
                elif e.key == pygame.K_r:
                    game.reset()
                    goal_angle = game.goal_pose[2]
                elif e.key == pygame.K_f:
                    freeze, event_flag = True, True
                elif e.key in (pygame.K_g, pygame.K_q, pygame.K_e):
                    goal_angle += {pygame.K_g: 0, pygame.K_q: -np.pi / 12, pygame.K_e: np.pi / 12}[e.key]
                    pos = pygame.mouse.get_pos() if e.key == pygame.K_g else game.goal_pose[:2]
                    if pos[0] < SIM:
                        game.set_goal(np.clip(pos, 60, 450), goal_angle)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and e.pos[0] < SIM:
                dragging = np.linalg.norm(np.array(e.pos) - game.state()[2:4]) < 70
            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False

        if args.headless_blocks == 30 and n_blocks == 20:  # test: cheat once mid-run
            game.teleport(game.state()[2:4] + [50, 0], 0.5)
            event_flag = True
        mouse = np.array(pygame.mouse.get_pos(), dtype=np.float64)
        block, plan_s_new = game.next_block(manual, mouse)
        if plan_s_new is not None:
            plan_s = plan_s_new
        executed, hit = [], False
        for k in range(FS):
            for e in pygame.event.get((pygame.MOUSEBUTTONUP,)):
                dragging = False
            if dragging and pygame.mouse.get_pressed()[0]:
                game.teleport(np.clip(pygame.mouse.get_pos(), 30, 480))
                event_flag = True
            if manual:
                target = np.array(pygame.mouse.get_pos(), dtype=np.float64)
                a = np.clip((target - game.state()[:2]) / game.env.action_scale, game.jepa.a_lo, game.jepa.a_hi)
            else:
                a = block[k]
            if not freeze:
                game.step(a)
            hit = hit or game.hit()
            executed.append(a)
            draw(screen, font, game, manual, plan_s, last_s, typical)
            pygame.display.flip()
            clock.tick(20)
        last_s = game.end_block(np.array(executed), event_flag)
        n_blocks += 1
        if not manual and game.check_goal(hit):    # episode over: show the result briefly, then the next one
            msg = font.render(f"episode done: {game.reached} subgoals reached", True, (20, 120, 20))
            screen.blit(msg, (20, 20))
            pygame.display.flip()
            if not args.headless_blocks:
                pygame.time.wait(2000)
            game.reset()
        if not event_flag and last_s > 0:          # running "normal" level, ignoring cheats
            typical = 0.95 * typical + 0.05 * last_s
        if args.headless_blocks and n_blocks >= args.headless_blocks:
            draw(screen, font, game, manual, plan_s, last_s, typical)
            out = Path(os.environ.get("OUT", "/work/assets"), "live_screenshot.png")
            pygame.image.save(screen, str(out))
            s = game.state()
            print(f"blocks {n_blocks}, plan {plan_s * 1000:.0f} ms, surprise trace {np.round(list(game.trace), 3)}")
            print(f"block {s[2:5].round(1)}, goal {game.goal_pose.round(1)}")
            return


def evaluate(args):
    """Follow N expert episodes as subgoal chains, no window; reports how far the T ends from the expert's end."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    game = Game(Jepa(args.model), args.episode)
    rows = []
    for _ in range(args.eval_episodes):
        n_sub = -(-(game.last - game.first) // SUBGOAL)
        start = game.state()
        while True:
            block, _ = game.next_block(False, None)
            hit = False
            for a in block:
                game.step(a)
                hit = hit or game.hit()
            game.end_block(block, False)
            if game.check_goal(hit):
                break
        end, final = game.state(), game.states[game.last]
        d = lambda s: (np.linalg.norm(s[2:4] - final[2:4]),
                       np.degrees(abs((s[4] - final[4] + np.pi) % (2 * np.pi) - np.pi)))
        rows.append(dict(episode=(game.episode - 1) % len(game.offsets), subgoals=n_sub, reached=game.reached,
                         start_px=d(start)[0], start_deg=d(start)[1], end_px=d(end)[0], end_deg=d(end)[1],
                         done=bool(game.env.eval_state(final, end)[0])))
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
        game.reset()
    return rows


if __name__ == "__main__":
    main()
