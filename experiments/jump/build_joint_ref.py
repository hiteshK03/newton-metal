"""Add a joint-level reference (jref) to ref_long_jump.npz using the crouch calibration.
Order: gen_long_jump_ref.py -> calib_crouch.py -> build_joint_ref.py.
The joint reference only encodes crouch/extension (no forward push) - see the experiment README."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from nmrl import REPO
D = REPO / "experiments/jump/data"
c = np.load(D / "calib_crouch.npz"); S, H, DIR, qdef = c["S"], c["H"], c["DIR"], c["qdef"]
ref = np.load(D / "ref_long_jump.npz")["ref"]
s_of_z = lambda z: np.interp(z, H[::-1], S[::-1])
sref = np.array([s_of_z(min(ref[k, 1], 0.619)) for k in range(200)])
sref[43:60] = s_of_z(0.56)                       # legs held at takeoff extension through flight
jref = (qdef[None, :] + sref[:, None] * DIR[None, :]).astype(np.float32)
np.savez(D / "ref_long_jump.npz", ref=ref, jref=jref, sref=sref)
print("saved ref+jref; s range", sref.min().round(2), sref.max().round(2))
