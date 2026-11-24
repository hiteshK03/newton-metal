import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
"""Render an UNMODIFIED upstream Newton example to mp4 on the Metal backend (headless GL viewer + ffmpeg).

Usage: python scripts/render_example.py EXAMPLE_NAME OUT.mp4 [NUM_FRAMES]
  e.g. python scripts/render_example.py robot_g1 media/examples/robot_g1.mp4 300
Needs ffmpeg on PATH and a Newton checkout installed (pip install -e newton).
"""
import runpy, subprocess, shutil
import numpy as np
import newton, newton.examples, newton.viewer

NAME, OUT = sys.argv[1], sys.argv[2]
NFR = int(sys.argv[3]) if len(sys.argv) > 3 else 300
root = pathlib.Path(newton.examples.__file__).parent
hits = list(root.rglob(f"example_{NAME}.py"))
assert hits, f"no upstream example named {NAME}"
modname = ".".join(["newton", "examples"] + list(hits[0].relative_to(root).with_suffix("").parts))

ff = None; count = 0
orig = newton.viewer.ViewerGL.end_frame
def end_frame(self, *a, **k):
    global ff, count
    r = orig(self, *a, **k)
    fr = self.get_frame().numpy()
    if ff is None:
        h, w = fr.shape[:2]
        pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                               "-r", "30", "-i", "-", "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-vcodec", "libx264",
                               "-pix_fmt", "yuv420p", "-crf", "22", OUT], stdin=subprocess.PIPE)
    ff.stdin.write(np.ascontiguousarray(fr).tobytes()); count += 1
    return r
newton.viewer.ViewerGL.end_frame = end_frame
sys.argv = [NAME, "--viewer", "gl", "--headless", "--num-frames", str(NFR), "--device", "metal:0"]
try:
    runpy.run_module(modname, run_name="__main__")
finally:
    if ff is not None:
        ff.stdin.close(); ff.wait()
    print(f"{NAME}: {count} frames -> {OUT}", flush=True)
