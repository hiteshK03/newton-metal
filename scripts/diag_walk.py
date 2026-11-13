import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nmrl import REPO
import sys, numpy as np, torch
CK = sys.argv[1]; sys.argv = sys.argv[:1]
from nmrl.anymal_walk_env import AnymalEnv
from nmrl.ppo_walk import AC
N=64
env = AnymalEnv(N, fixed_cmd=(0.8,0,0))
net = AC(); net.load_state_dict(torch.load(CK, map_location="cpu")["net"]); net.eval()
labels=[l.split('/')[-1] for l in env.model.joint_label[:13]]
print("joint order:", labels[1:])
qdef = env.qdef.numpy()
Q=[];W=[];V=[];G=[];ACT=[];FZ=[]
for t in range(450):
    o = env.get_obs_np()
    with torch.no_grad(): a = net.actor(torch.from_numpy(o)).numpy().astype(np.float32)
    env.step(a)
    if t>=50:
        q=env.s0.joint_q.numpy().reshape(N,19); qd=env.s0.joint_qd.numpy().reshape(N,18)
        ob=env.get_obs_np()
        bq=env.s0.body_q.numpy().reshape(N,13,7); fb=env.foot_b.numpy(); fp=env.foot_p.numpy()
        from scipy.spatial.transform import Rotation as Rr
        FZ.append(np.stack([(Rr.from_quat(bq[:,fb[f],3:7]).apply(fp[f])+bq[:,fb[f],:3])[:,2] for f in range(4)],1))
        Q.append(q.copy()); V.append(ob[:,0:3]/2.0); W.append(ob[:,3:6]/0.25); G.append(ob[:,6:9]); ACT.append(a)
Q=np.array(Q);V=np.array(V);W=np.array(W);G=np.array(G);ACT=np.array(ACT)
np.set_printoptions(precision=3, suppress=True)
print("base z mean/std: %.3f / %.3f (standing default ~0.49)"%(Q[:,:,2].mean(), Q[:,:,2].std()))
print("gravity_b mean (x,y,z):", G.mean((0,1)), " -> pitch(rad)~%.3f roll~%.3f"%(np.arcsin(G[:,:,0].mean()), np.arcsin(G[:,:,1].mean())))
print("body vel mean (vx,vy,vz):", V.mean((0,1)), "| vy std over time", V[:,:,1].mean(1).std().round(3))
print("yaw rate: mean %.3f rad/s, per-env std %.3f, |mean| per env %.3f"%(W[:,:,2].mean(), W[:,:,2].mean(0).std(), np.abs(W[:,:,2].mean(0)).mean()))
print("joint offset from default (rad), mean over time & envs:")
off=(Q[:,:,7:]-qdef).mean((0,1)); rng=(Q[:,:,7:].max((0,1))-Q[:,:,7:].min((0,1)))
for n,o_,r in zip(labels[1:],off,rng): print(f"  {n:7s} offset {o_:+.3f}  range {r:.2f}")
print("action mean |a| %.3f, saturated(|a|>1) %.1f%%"%(np.abs(ACT).mean(), 100*(np.abs(ACT)>1).mean()))

FZ=np.array(FZ); c=FZ<0.045
print("feet order (bodies):", env.foot_b.numpy())
print("foot contact fraction per foot:", c.mean((0,1)), "| feet on ground at once (mean):", c.sum(2).mean().round(2))
td=((c[1:]) & (~c[:-1])).sum(0)  # touchdowns per env per foot over 400 steps = 8 s
print("touchdowns per second per foot:", (td.mean(0)/8.0).round(2))
print("max foot height (m) per foot:", FZ.max((0,1)).round(3))
