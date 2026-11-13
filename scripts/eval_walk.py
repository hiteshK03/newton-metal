import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nmrl import REPO
"""Deterministic eval of the PPO-trained ANYmal policy vs zero-action and random baselines.
64 envs, fixed command (0.8, 0, 0) m/s body-frame, 10 s episodes, no auto-continue after a fall.

Usage: python eval_anymal.py [ckpt]
"""
import sys
CK = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "checkpoints/anymal_walk.pt"); sys.argv = sys.argv[:1]
import numpy as np
import torch
from nmrl.anymal_walk_env import AnymalEnv, EP_LEN, FRAME_DT
from nmrl.ppo_walk import AC

CKPT = CK
N, CMD = 64, (0.8, 0.0, 0.0)
env = AnymalEnv(N, fixed_cmd=CMD)
q = env.q0.numpy().reshape(N, 19)
yaw0 = 2 * np.arctan2(q[:, 5], q[:, 6])          # spawn yaw (rotation about z only)
fwd = np.stack([np.cos(yaw0), np.sin(yaw0)], 1); lat = np.stack([-np.sin(yaw0), np.cos(yaw0)], 1)


def run(policy, label, seed=0):
    rng = np.random.default_rng(seed)
    env.reset_all()
    pos0 = env.s0.joint_q.numpy().reshape(N, 19)[:, :2].copy()
    last = pos0.copy(); alive = np.ones(N, bool); vx_sum = np.zeros(N); steps = np.zeros(N)
    for t in range(EP_LEN):
        o = env.get_obs_np()
        if policy == "trained":
            with torch.no_grad():
                a = net.actor(torch.from_numpy(o)).numpy().astype(np.float32)
        elif policy == "zero":
            a = np.zeros((N, 12), np.float32)
        else:
            a = rng.uniform(-1, 1, (N, 12)).astype(np.float32)
        _, _, done, info = env.step(a)
        qq = env.s0.joint_q.numpy().reshape(N, 19)
        steps[alive] += 1; vx_sum[alive] += info[alive, 2]
        # an env that terminates before the time limit has fallen; its position after reset is not used
        fell = (done > 0.5) & (t < EP_LEN - 1) & alive
        last[alive & ~fell] = qq[alive & ~fell, :2]
        alive &= ~fell
        # keep going but ignore dead envs
    disp = last - pos0
    f = (disp * fwd).sum(1); l = (disp * lat).sum(1)
    print(f"[{label:>10}] survived 10s: {alive.sum():2d}/{N} (fall rate {100*(1-alive.mean()):4.0f}%) | "
          f"fwd speed {f.mean()/10:+.2f} m/s (mean over all envs; cmd 0.80) | "
          f"alive-only fwd speed {(f[alive].mean()/10 if alive.any() else float('nan')):+.2f} m/s | "
          f"lateral drift {np.abs(l).mean():.2f} m | mean body-vx while alive {np.sum(vx_sum)/max(1,np.sum(steps)):+.2f}", flush=True)


net = AC(); net.load_state_dict(torch.load(CKPT, map_location="cpu")["net"]); net.eval()
run("zero", "zero action"); run("random", "random")
run("trained", "PPO trained")
print("reference: IsaacLab/PhysX-trained ONNX policy walks ~1.2 m/s at command 1.0 (earlier measurement).", flush=True)
