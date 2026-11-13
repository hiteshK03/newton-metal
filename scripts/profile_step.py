"""Where does one env step spend its time? Usage: python scripts/profile_step.py [n_envs]"""
import sys, pathlib, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
import numpy as np, warp as wp
from nmrl.anymal_walk_env import AnymalEnv, FRAME_DT, SUB, k_targets, k_reward, k_obs, NQ
env = AnymalEnv(N)
act = np.zeros((N, 12), np.float32)
def timed(fn, reps=20):
    for _ in range(3): fn()
    wp.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    wp.synchronize(); return (time.perf_counter() - t) / reps * 1e3
dt = FRAME_DT / SUB
def collide(): env.s0.clear_forces(); env.cp.collide(env.s0, env.contacts)
def solve():
    env.solver.step(env.s0, env.s1, env.ctrl, env.contacts, dt)
    env.s0, env.s1 = env.s1, env.s0
env.act.assign(act)
r = {}
r["assign actions + k_targets"] = timed(lambda: (env.act.assign(act), wp.launch(k_targets, dim=N * NQ, inputs=[env.act, env.q0, env.ctrl.joint_target_q])))
r["collide (1 substep)"] = timed(collide)
r["solver.step (1 substep)"] = timed(solve)
r["reward kernel"] = timed(lambda: wp.launch(k_reward, dim=N, inputs=[env.s0.joint_q, env.s0.joint_qd, env.qdef, env.q0, env.act, env.prev, env.cmd, env.ep_len, env.s0.body_q, env.foot_b, env.foot_p, env.air, env.fxy, env.rew, env.done, env.comp]))
r["host readback (rew, done, info, obs .numpy())"] = timed(lambda: (env.rew.numpy(), env.done.numpy(), env.comp.numpy(), env.obs.numpy()))
r["full env.step()"] = timed(lambda: env.step(act))
sub = r["collide (1 substep)"] + r["solver.step (1 substep)"]
print(f"N={N} envs, {SUB} substeps/step")
for k, v in r.items(): print(f"  {k:48s} {v:8.2f} ms")
print(f"  {'=> physics per step (collide+solve) x '+str(SUB):48s} {sub*SUB:8.2f} ms  ({sub*SUB/r['full env.step()']*100:.0f}% of full step)")
print(f"  env-steps/s from full step alone: {N / r['full env.step()'] * 1e3:,.0f}")
