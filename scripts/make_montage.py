import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
"""Combine the walking clip and the upstream-example clips into one 4x3 montage video (all playing at once).

Usage: python scripts/make_montage.py [OUT.mp4] [SECONDS]
Inputs: media/anymal_walk.gif and media/examples/*.mp4 (see scripts/render_example.py). Needs ffmpeg and Pillow.
Short clips are looped. Labels are drawn with Pillow because many ffmpeg builds lack drawtext.
"""
import subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont
from nmrl import REPO

OUT = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "media/montage.mp4")
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 10
TW, TH, COLS, ROWS, BANNER = 480, 270, 4, 3, 90
M, E = REPO / "media", REPO / "media/examples"
# (source, label); the first tile is our own PPO result, the rest are unmodified upstream Newton examples
TILES = [
    (M / "anymal_walk.gif", "ANYmal C, PPO-trained on this Mac (this repo)"),
    (E / "franka_cable_ik_pick_place.mp4", "franka_cable_ik_pick_place"), (E / "robot_ur10.mp4", "robot_ur10"),
    (E / "balance_bird.mp4", "balance_bird"), (E / "domino_spiral.mp4", "domino_spiral"), (E / "cloth_twist.mp4", "cloth_twist"),
    (E / "cloth_rollers.mp4", "cloth_rollers"), (E / "rigid_soft_contact.mp4", "rigid_soft_contact"), (E / "softbody_gift.mp4", "softbody_gift"),
    (E / "tendon_pulley.mp4", "tendon_pulley"), (E / "newton_cradle.mp4", "newton_cradle"), (E / "selection_cartpole.mp4", "selection_cartpole"),
]
assert len(TILES) == COLS * ROWS
for src, _ in TILES:
    assert src.exists(), f"missing {src}"


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


tmp = pathlib.Path(tempfile.mkdtemp())
labels = []
for i, (_, text) in enumerate(TILES):
    im = Image.new("RGBA", (TW, 26), (0, 0, 0, 170)); ImageDraw.Draw(im).text((8, 4), text, font=font(16), fill=(255, 255, 255, 255))
    labels.append(tmp / f"l{i}.png"); im.save(labels[-1])
ban = Image.new("RGBA", (TW * COLS, BANNER), (18, 22, 30, 255)); d = ImageDraw.Draw(ban)
d.text((16, 14), "Newton physics on Apple Metal (M5 Pro): our PPO walking + upstream Newton examples", font=font(30), fill=(255, 255, 255, 255))
d.text((16, 54), "NVIDIA Warp via warp-metal (innate-inc)  |  Newton / MuJoCo Warp  |  github.com/hiteshK03/newton-metal", font=font(20), fill=(170, 185, 205, 255))
ban_p = tmp / "banner.png"; ban.save(ban_p)

cmd = ["ffmpeg", "-y", "-loglevel", "error"]
for src, _ in TILES:
    cmd += (["-ignore_loop", "0"] if src.suffix == ".gif" else ["-stream_loop", "-1"]) + ["-i", str(src)]
for l in labels:
    cmd += ["-i", str(l)]
cmd += ["-i", str(ban_p)]
n = len(TILES); f = []
for i in range(n):
    f.append(f"[{i}:v]fps=30,scale={TW}:{TH}:force_original_aspect_ratio=decrease,pad={TW}:{TH}:(ow-iw)/2:(oh-ih)/2,setsar=1[t{i}]")
    f.append(f"[t{i}][{n+i}:v]overlay=0:{TH}-26[v{i}]")
layout = "|".join(f"{(i % COLS) * TW}_{(i // COLS) * TH}" for i in range(n))
f.append("".join(f"[v{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={layout}[grid]")
f.append(f"[grid]pad={TW*COLS}:{TH*ROWS+BANNER}:0:{BANNER}[padded]")
f.append(f"[padded][{2*n}:v]overlay=0:0,format=yuv420p[out]")
cmd += ["-filter_complex", ";".join(f), "-map", "[out]", "-t", str(SECS), "-r", "30", "-an", "-vcodec", "libx264", "-crf", "22", OUT]
subprocess.run(cmd, check=True)
print("wrote", OUT)
