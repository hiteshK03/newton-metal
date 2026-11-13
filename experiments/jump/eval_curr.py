import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nmrl import REPO
"""Curriculum eval at a fixed distance scale. Success = upright to the end AND a real flight after first ground contact
with forward distance >= 50% of the scaled reference flight AND total forward at 2.4 s >= 70% of scaled reference.
Usage: python eval_curr.py ckpt lam"""
import sys, os
CK, LAM = sys.argv[1], float(sys.argv[2])
os.environ.update(FORCE_SKILL='2', REF_RSI='0', REF_ET='0', REF_FPR='0', LAM_MIN=str(LAM), LAM_MAX=str(LAM))
import numpy as np, torch
sys.argv = sys.argv[:1]
from nmrl.anymal_multi_env import AnymalEnv
from nmrl.ppo_multi import AC
N = 64; env = AnymalEnv(N)
net = AC(); sd = torch.load(CK, map_location="cpu")["net"]; cur = net.state_dict()
for k, v in sd.items():
    if cur[k].shape == v.shape: cur[k] = v
    else:
        w = torch.zeros_like(cur[k]); w[:, :v.shape[1]] = v; cur[k] = w
net.load_state_dict(cur); net.eval()
env.reset_all()
q = env.s0.joint_q.numpy().reshape(N, 19); p0 = q[:, :2].copy(); yaw0 = 2*np.arctan2(q[:,5], q[:,6]); fh = np.stack([np.cos(yaw0), np.sin(yaw0)], 1)
fb = env.foot_b.numpy(); fp = env.foot_p.numpy()
alive = np.ones(N, bool); touched = np.zeros(N, bool); inflight = np.zeros(N, bool); st_x = np.zeros(N); st_t = np.zeros(N)
best = np.zeros(N); peak = np.zeros(N); fwd120 = np.zeros(N); pk_cur = np.zeros(N); nfl = 0
for t in range(109):
    with torch.no_grad(): a = net.actor(torch.from_numpy(env.get_obs_np())).numpy().astype(np.float32)
    _, _, done, _ = env.step(a)
    q = env.s0.joint_q.numpy().reshape(N, 19); alive &= ~((done > 0.5) & (t < 108))
    fw = ((q[:, :2] - p0) * fh).sum(1)
    if t == 99: fwd120 = fw.copy()
    bq = env.s0.body_q.numpy().reshape(N, 13, 7); zs = []
    for f in range(4):
        u = bq[:, fb[f], 3:6]; w = bq[:, fb[f], 6:7]; v = fp[f]
        zs.append((bq[:, fb[f], :3] + v + 2*np.cross(u, np.cross(u, v) + w*v))[:, 2])
    zs = np.stack(zs, 1); touched |= (zs.min(1) < 0.045)
    air4 = (zs > 0.045).all(1) & (q[:, 2] > 0.45) & touched & (done < 0.5)
    for e in range(N):
        if air4[e] and not inflight[e]: inflight[e] = True; st_x[e] = fw[e]; st_t[e] = t; pk_cur[e] = q[e, 2]
        elif air4[e]: pk_cur[e] = max(pk_cur[e], q[e, 2])
        elif inflight[e]:
            inflight[e] = False
            if t - st_t[e] >= 5: best[e] = max(best[e], fw[e] - st_x[e]); peak[e] = max(peak[e], pk_cur[e]); nfl += 1
air_ref, tot_ref = 0.48 * LAM, 0.72 * LAM
ok = alive & (best >= 0.5 * air_ref) & (fwd120 >= 0.7 * tot_ref)
print(f"RESULT lam={LAM:.2f} (jump {tot_ref:.2f} m, air {air_ref:.2f} m) | survive {alive.mean()*100:3.0f}% | real flights {nfl} "
      f"| best flight fwd mean {best.mean():.2f} max {best.max():.2f} m | fwd@2.0s mean {fwd120.mean():.2f} m | peak z max {peak.max():.2f} "
      f"| SUCCESS {ok.mean()*100:3.0f}%", flush=True)
