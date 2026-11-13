import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nmrl import REPO
"""Open-loop replay: drive PD targets from the joint reference (no policy). Does the robot even hop?"""
import os, numpy as np
os.environ.update(FORCE_SKILL='2', REF_RSI='0', REF_ET='0')
from nmrl.anymal_multi_env import AnymalEnv
env = AnymalEnv(8); N = 8
r = np.load(str(REPO / "experiments/jump/data/ref_long_jump.npz")); ref, jref = r["ref"], r["jref"]
env.reset_all()
q0 = env.q0.numpy().reshape(N, 19); p0 = q0[:, :3].copy(); yaw0 = 2*np.arctan2(q0[:,5], q0[:,6]); fh = np.stack([np.cos(yaw0), np.sin(yaw0)], 1)
qdef = env.qdef.numpy()
zs, xs, tilt = [], [], []
for k in range(120):
    a = np.tile(np.clip((jref[min(k + 1, 199)] - qdef) / 0.5, -3, 3), (N, 1)).astype(np.float32)
    _, _, done, _ = env.step(a)
    q = env.s0.joint_q.numpy().reshape(N, 19)
    zs.append(q[0, 2]); xs.append(((q[0, :2] - p0[0, :2]) * fh[0]).sum())
zs, xs = np.array(zs), np.array(xs)
print("action magnitude max |a| =", float(np.abs((jref - qdef) / 0.5).max()))
for k in [0, 20, 34, 42, 50, 59, 70, 90, 119]:
    print(f"step {k:3d}: z {zs[k]:.2f} (ref {ref[k+1,1]:.2f})  fwd {xs[k]:+.2f} (ref {ref[k+1,0]:.2f})")
print(f"open-loop: peak z {zs.max():.2f} m, final fwd {xs[-1]:+.2f} m, min z {zs.min():.2f}")
