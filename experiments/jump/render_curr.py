import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nmrl import REPO
"""Render the curriculum long-jump policy (side view, slow motion). Usage: python render_curr.py ckpt lam out.mp4"""
import subprocess, sys, os
CK, LAM, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.environ.update(FORCE_SKILL='2', REF_RSI='0', REF_ET='0', REF_FPR='0', LAM_MIN=str(LAM), LAM_MAX=str(LAM), ENV_SPACING='40')
import numpy as np, torch
sys.argv = sys.argv[:1]
from nmrl.anymal_multi_env import AnymalEnv
from nmrl.ppo_multi import AC
import warp as wp, newton
env = AnymalEnv(4)
net = AC(); sd = torch.load(CK, map_location="cpu")["net"]; cur = net.state_dict()
for k, v in sd.items():
    if cur[k].shape == v.shape: cur[k] = v
    else: w = torch.zeros_like(cur[k]); w[:, :v.shape[1]] = v; cur[k] = w
net.load_state_dict(cur); net.eval()
viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=True); viewer.set_model(env.model)
q0 = env.q0.numpy().reshape(4, 19)[0]; yaw0 = 2*np.arctan2(q0[5], q0[6]); lat = np.array([-np.sin(yaw0), np.cos(yaw0), 0.0])
ff = None; zs = []; xs = []
for k in range(109):
    with torch.no_grad(): a = net.actor(torch.from_numpy(env.get_obs_np())).numpy().astype(np.float32)
    env.step(a); q = env.s0.joint_q.numpy().reshape(4, 19)[0]; zs.append(q[2]); xs.append(q[1] - q0[1])
    base = q[:3].astype(float); g = np.array([base[0], q0[1] + 0.3, 0.0])
    viewer.set_camera(pos=wp.vec3(*(g + 5.5*lat + np.array([0, 0, 0.8])))); viewer.camera.look_at(wp.vec3(g[0], g[1], 0.45))
    viewer.begin_frame(k * 0.02); viewer.log_state(env.s0); viewer.end_frame(); fr = viewer.get_frame().numpy()
    if ff is None:
        h, w = fr.shape[:2]
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
                               "-an", "-vf", "scale=1280:720", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT], stdin=subprocess.PIPE)
    ff.stdin.write(np.ascontiguousarray(fr).tobytes())
ff.stdin.close(); ff.wait()
print(f"base z min {min(zs):.2f} max {max(zs):.2f}; forward travelled {xs[-1]:.2f} m (reference {0.72*LAM:.2f} m)")
