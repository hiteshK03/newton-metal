import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nmrl import REPO
"""Ablation eval: long-jump skill from GROUND starts, no RSI/early-termination. Usage: python eval_ref.py ckpt label"""
import sys, os
os.environ.update(FORCE_SKILL='2', REF_RSI='0', REF_ET='0', REF_FPR='0')
import numpy as np, torch
CK, LABEL = sys.argv[1], sys.argv[2]; sys.argv = sys.argv[:1]
from nmrl.anymal_multi_env import AnymalEnv
from nmrl.ppo_multi import AC
N = 64
env = AnymalEnv(N)
net = AC(); net.load_state_dict(torch.load(CK, map_location="cpu")["net"]); net.eval()
ref = np.load(str(REPO / "experiments/jump/data/ref_long_jump.npz"))["ref"]
env.reset_all()
q = env.s0.joint_q.numpy().reshape(N, 19); p0 = q[:, :3].copy(); yaw0 = 2 * np.arctan2(q[:, 5], q[:, 6])
fh = np.stack([np.cos(yaw0), np.sin(yaw0)], 1); lat = np.stack([-np.sin(yaw0), np.cos(yaw0)], 1)
fb = env.foot_b.numpy(); fp = env.foot_p.numpy()
alive = np.ones(N, bool); err = []; inflight = np.zeros(N, bool); st_x = np.zeros(N); flights = []; fwd120 = np.zeros(N)
for t in range(199):
    with torch.no_grad():
        a = net.actor(torch.from_numpy(env.get_obs_np())).numpy().astype(np.float32)
    _, _, done, _ = env.step(a)
    q = env.s0.joint_q.numpy().reshape(N, 19)
    alive &= ~((done > 0.5) & (t < 198))
    d = q[:, :2] - p0[:, :2]; fw = (d * fh).sum(1); lt = (d * lat).sum(1)
    k = min(t + 1, 199)
    if t < 100: err.append(np.sqrt((fw - ref[k, 0]) ** 2 + lt ** 2 + (q[:, 2] - ref[k, 1]) ** 2)[alive].mean() if alive.any() else np.nan)
    if t == 119: fwd120 = fw.copy()
    bq = env.s0.body_q.numpy().reshape(N, 13, 7); zs = []
    for f in range(4):
        u = bq[:, fb[f], 3:6]; w = bq[:, fb[f], 6:7]; v = fp[f]
        zs.append((bq[:, fb[f], :3] + v + 2 * np.cross(u, np.cross(u, v) + w * v))[:, 2])
    air4 = np.all(np.stack(zs, 1) > 0.045, 1) & (q[:, 2] > 0.45) & (done < 0.5)
    for e in range(N):
        if air4[e] and not inflight[e]: inflight[e] = True; st_x[e] = fw[e]
        elif not air4[e] and inflight[e]: inflight[e] = False; flights.append(fw[e] - st_x[e])
F = np.array(flights) if flights else np.zeros(1)
ok = alive & (fwd120 >= 0.5)
print(f"RESULT {LABEL} | survive {alive.mean()*100:3.0f}% | track err (0-2s) {np.nanmean(err):.3f} m | fwd@2.4s mean {fwd120[alive].mean() if alive.any() else float('nan'):.2f} m "
      f"| flights {len(flights)} flight-dist mean {F.mean():.2f} max {F.max():.2f} m | SUCCESS (upright & >=0.5 m) {ok.mean()*100:3.0f}%", flush=True)
