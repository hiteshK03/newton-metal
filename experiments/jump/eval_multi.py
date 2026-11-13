import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nmrl import REPO
"""Eval the multi-skill policy from GROUND starts (LAUNCH_P=0). Usage: LAUNCH_P=0 python eval_multi.py [ckpt]"""
import sys, os
import numpy as np, torch
os.environ['LAUNCH_P'] = '0'
CK = sys.argv[1] if len(sys.argv) > 1 else "anymal_multi2.pt"
sys.argv = sys.argv[:1]
from nmrl.anymal_multi_env import AnymalEnv, EP_LEN
from nmrl.ppo_multi import AC
N = 64
env = AnymalEnv(N, fixed_cmd=(0.8, 0.0, 0.0))
net = AC(); net.load_state_dict(torch.load(CK, map_location="cpu")["net"]); net.eval()
sk = env.skill.numpy()
env.reset_all()
cmd = env.cmd.numpy(); cmd[sk != 0] = 0.0; env.cmd.assign(cmd)
q = env.s0.joint_q.numpy().reshape(N, 19); y0 = q[:, 1].copy(); yaw0 = 2*np.arctan2(q[:,5], q[:,6])
fwd = np.stack([np.cos(yaw0), np.sin(yaw0)], 1)
p0 = q[:, :2].copy(); foot_b = env.foot_b.numpy(); foot_p = env.foot_p.numpy()
alive = np.ones(N, bool); last = p0.copy()
inflight = np.zeros(N, bool); st_y = np.zeros(N); st_t = np.zeros(N); pk = np.zeros(N); flights = {1: [], 2: []}
for t in range(EP_LEN):
    with torch.no_grad():
        a = net.actor(torch.from_numpy(env.get_obs_np())).numpy().astype(np.float32)
    _, _, done, _ = env.step(a)
    q = env.s0.joint_q.numpy().reshape(N, 19)
    jump_over = (t >= 199)                       # jump episodes end at 200 steps
    fell = (done > 0.5) & alive & (((sk == 0) & (t < EP_LEN - 1)) | ((sk != 0) & (t < 199)))
    alive &= ~fell
    if t < 199 or True:
        walk_ok = alive & (sk == 0); last[walk_ok] = q[walk_ok, :2]
    if t < 199:
        bq = env.s0.body_q.numpy().reshape(N, 13, 7); zs = []
        for f in range(4):
            u = bq[:, foot_b[f], 3:6]; w = bq[:, foot_b[f], 6:7]; v = foot_p[f]
            zs.append((bq[:, foot_b[f], :3] + v + 2 * np.cross(u, np.cross(u, v) + w * v))[:, 2])
        air4 = np.all(np.stack(zs, 1) > 0.045, 1) & (q[:, 2] > 0.45) & (done < 0.5) & (sk != 0)
        for e in np.flatnonzero(sk != 0):
            if air4[e] and not inflight[e]:
                inflight[e] = True; st_y[e] = q[e, 1]; st_t[e] = t; pk[e] = q[e, 2]
            elif air4[e]: pk[e] = max(pk[e], q[e, 2])
            elif inflight[e]:
                inflight[e] = False
                flights[int(sk[e])].append(((t - st_t[e]) * 0.02, pk[e], q[e, 1] - st_y[e]))
w = sk == 0
disp = (last - p0)[w]; f = (disp * fwd[w]).sum(1)
print(f"WALK  : survived {alive[w].sum()}/{w.sum()} | fwd speed {f.mean()/10:+.2f} m/s (cmd 0.80)")
for s, nm in [(1, "HIGH JUMP"), (2, "LONG JUMP")]:
    m = sk == s; F = np.array(flights[s]) if flights[s] else np.zeros((0, 3))
    if len(F) == 0:
        print(f"{nm}: survived {alive[m].sum()}/{m.sum()} | NO FLIGHTS from ground starts"); continue
    print(f"{nm}: survived {alive[m].sum()}/{m.sum()} | flights {len(F)} | air time mean {F[:,0].mean():.2f}s max {F[:,0].max():.2f}s | "
          f"peak base z mean {F[:,1].mean():.2f} max {F[:,1].max():.2f} m | fwd dist/flight mean {F[:,2].mean():.2f} max {F[:,2].max():.2f} m")
