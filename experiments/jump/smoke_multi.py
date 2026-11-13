import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nmrl import REPO
import numpy as np
from nmrl.anymal_multi_env import *
env = AnymalEnv(64)
for t in range(30):
    o,r,d,i = env.step(np.zeros((64,12),np.float32))
print("obs", o.shape, o[0,48:], o[40,48:], o[63,48:], "rew walk/hj/lj", r[0], r[40], r[63])
