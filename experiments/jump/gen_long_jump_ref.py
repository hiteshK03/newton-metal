import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nmrl import REPO
"""Generate a kinematic long-jump reference (root trajectory) for ANYmal C, 50 Hz, 200 steps.
Phases (steps): settle 0-20 | crouch 20-35 | thrust 35-43 | ballistic flight 43-60 | absorb 60-75 | recover 75-100 | hold.
Columns: forward distance x (m), base height z (m), forward speed vx (m/s), vertical speed vz (m/s).
Takeoff vz=1.7 m/s (apex +0.15 m, flight 0.35 s), takeoff vx=1.4 m/s -> ~0.5 m in the air, ~0.7 m total."""
import numpy as np
DT, G = 0.02, 9.81
T = 200
x = np.zeros(T); z = np.zeros(T)
cos = lambda a, b, u: a + (b - a) * (1 - np.cos(np.pi * u)) / 2
VX, VZ, Z_STAND, Z_CROUCH, Z_TAKE = 1.4, 1.7, 0.53, 0.40, 0.56
x_thrust = VX * (8 * DT) / 2
for k in range(T):
    if k < 20:   z[k] = cos(0.62, Z_STAND, k / 20); x[k] = 0
    elif k < 35: z[k] = cos(Z_STAND, Z_CROUCH, (k - 20) / 15); x[k] = 0
    elif k < 43:
        u = (k - 35 + 1) / 8
        z[k] = cos(Z_CROUCH, Z_TAKE, u); x[k] = x_thrust * u * u
    elif k < 60:
        tau = (k - 43 + 1) * DT
        z[k] = Z_TAKE + VZ * tau - 0.5 * G * tau * tau; x[k] = x_thrust + VX * tau
    elif k < 75:
        u = (k - 60 + 1) / 15
        z[k] = cos(Z_TAKE, 0.42, u); x[k] = x[59] + (0.13) * (1 - (1 - min(u * 1.5, 1)) ** 2)
    elif k < 100: z[k] = cos(0.42, Z_STAND, (k - 75 + 1) / 25); x[k] = x[74]
    else:        z[k] = Z_STAND; x[k] = x[74]
vx = np.gradient(x, DT); vz = np.gradient(z, DT)
ref = np.stack([x, z, vx, vz], 1).astype(np.float32)
np.savez(str(REPO / "experiments/jump/data/ref_long_jump.npz"), ref=ref)
print(f"saved ref_long_jump.npz  total forward {x[-1]:.2f} m, flight forward {x[59]-x[42]:.2f} m, apex z {z.max():.2f} m, min z {z.min():.2f} m, peak vz {vz.max():.2f} m/s, peak vx {vx.max():.2f} m/s")
