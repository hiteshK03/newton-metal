import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nmrl import REPO
"""Calibrate a 1-parameter crouch map: joints = default + s*dir (front HFE+,KFE-; hind HFE-,KFE+ magnitudes grow with s).
Measures base height (feet on ground, level body) vs s with Newton FK, then builds per-step joint reference for the long jump."""
import os, numpy as np
os.environ['FORCE_SKILL'] = '2'
from nmrl.anymal_multi_env import AnymalEnv, NQ
env = AnymalEnv(8)
q = env.s0.joint_q.numpy().reshape(8, NQ).copy()
qdef = env.qdef.numpy()
labels = [l.split('/')[-1] for l in env.model.joint_label if l.split('/')[-1] in
          {f"{L}_{J}" for L in ("LF","RF","LH","RH") for J in ("HAA","HFE","KFE")}][:12]
print("joint order:", labels)
sgn = np.array([{"HAA": 0, "HFE": 1, "KFE": -2}[l.split('_')[1]] * (1 if l[1] == 'F' else -1) for l in labels], np.float32)
DIR = sgn * 0.5          # joint change per unit s (rad); knee moves twice the hip
fb = env.foot_b.numpy(); fp = env.foot_p.numpy()
def height(s):
    qq = q.copy(); qq[:, 7:19] = qdef + s * DIR; qq[:, 2] = 1.0
    env.s0.joint_q.assign(qq.reshape(-1)); 
    import newton; newton.eval_fk(env.model, env.s0.joint_q, env.s0.joint_qd, env.s0)
    bq = env.s0.body_q.numpy().reshape(8, 13, 7)[0]
    zs = []
    for f in range(4):
        u = bq[fb[f], 3:6]; w = bq[fb[f], 6]; v = fp[f]
        zs.append((bq[fb[f], :3] + v + 2 * np.cross(u, np.cross(u, v) + w * v))[2])
    return 1.0 - np.mean(zs) + 0.03       # base height when feet (radius 0.03) rest on the ground
S = np.linspace(-0.6, 1.4, 41); Hs = np.array([height(s) for s in S])
for s, h in zip(S[::4], Hs[::4]): print(f"s={s:+.2f}  base z on ground = {h:.3f}")
np.savez(str(REPO / "experiments/jump/data/calib_crouch.npz"), S=S, H=Hs, DIR=DIR, qdef=qdef)
