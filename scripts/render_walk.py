import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nmrl import REPO
"""Render the PPO-trained ANYmal policy walking (command 0.8 m/s) to an mp4 via Newton's GL viewer."""
import subprocess, sys
import numpy as np, torch
from nmrl.anymal_walk_env import AnymalEnv
from nmrl.ppo_walk import AC
import warp as wp, newton

CKPT = str(REPO / "checkpoints/anymal_walk.pt")
OUT = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "media/anymal_walk.mp4")
sys.argv = sys.argv[:1]
W, H, FPS, STEPS = 1280, 720, 25, 480
env = AnymalEnv(1, fixed_cmd=(0.8, 0.0, 0.0))
net = AC(); net.load_state_dict(torch.load(CKPT, map_location="cpu")["net"]); net.eval()
viewer = newton.viewer.ViewerGL(width=W, height=H, headless=True)
viewer.set_model(env.model)
ff = None
q0 = env.q0.numpy().reshape(19)
yaw0 = 2 * np.arctan2(q0[5], q0[6]); lat = np.array([-np.sin(yaw0), np.cos(yaw0), 0.0])
print("start", flush=True); t = 0.0; nframes = 0
for k in range(STEPS):
    with torch.no_grad():
        a = net.actor(torch.from_numpy(env.get_obs_np())).numpy().astype(np.float32)
    env.step(a)
    if k % 2 == 0:
        base = env.s0.joint_q.numpy()[:3].astype(float)
        cam = base + 3.2 * lat + np.array([0.0, 0.0, 0.9])
        viewer.set_camera(pos=wp.vec3(*cam))
        viewer.camera.look_at(wp.vec3(*(base + np.array([0, 0, 0.3]))))
        viewer.begin_frame(t); viewer.log_state(env.s0); viewer.end_frame()
        fr = viewer.get_frame().numpy()
        if ff is None:
            h, w = fr.shape[:2]
            ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                   "-r", str(FPS), "-i", "-", "-an", "-vf", "scale=1280:720", "-vcodec", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", "20", OUT], stdin=subprocess.PIPE)
        ff.stdin.write(np.ascontiguousarray(fr).tobytes()); nframes += 1
    t += 0.02
ff.stdin.close(); ff.wait()
print(f"done: {nframes} frames -> {OUT}", flush=True)
