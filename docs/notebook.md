# Training / RL on a Mac with Newton (Apple Metal GPU) — verified results

Hardware: Apple M5 Pro, Metal backend (warp-metal 1.17.0.3), Newton 1.9.0.dev0 fork.
All results below were produced by actual execution, headless, on `metal:0`.

## 1. Differentiable-simulation TRAINING (gradient descent through physics)
Example: `diffsim_ball` — optimize a ball's initial velocity so it bounces off the
wall + floor to hit a target. Uses `wp.Tape()` to backpropagate the loss through the
full contact simulation, then gradient-descent updates. Solver: SemiImplicit.

    python -m newton.examples diffsim_ball --device metal:0 --viewer null --headless --verbose

Verified: loss 9.11 -> 1.27 (37 iters) -> **6e-11** (converged, ~512 iters).
Gradient check: numeric [-4.961, -0.399] vs analytic [-4.973, -0.399] (agree).
This is genuine training-from-scratch on the Mac GPU through differentiable physics.

## 2. IsaacLab/PhysX-trained RL POLICY running on the Mac
Example: `robot_anymal_c_walk` — ANYmal C quadruped controlled by a walking policy
trained in IsaacLab/PhysX, exported to ONNX, run via Warp-NN (`warp_nn.runtime.OnnxRuntime`).
Solver: MuJoCo-Warp.

    python run_newton_metal.py robot_anymal_c_walk --num-frames 400

Verified (anymal_walk_measure.py): the robot walks steadily at **~1.2 m/s**, base
height stays ~0.53 m (upright, not falling), 11.7 m covered in 10 s. The RL policy
inference runs entirely on Metal.
Deps added: warp-nn, onnx, scipy; mujoco/mujoco-warp bumped 3.15 -> 3.17 to match the fork.

## 3. Massively parallel batched envs (the IsaacLab training pattern)
Script: batch_envs_bench.py — N independent cartpole envs replicated into one batched
Newton model, all stepped in parallel (SolverSemiImplicit + contacts).

    python batch_envs_bench.py metal:0 4096

Verified throughput:
    Metal   64 worlds:    234,243 env-steps/s
    Metal 1024 worlds:  3,692,957 env-steps/s   (CPU 1024: 1,594,416 -> 2.3x)
    Metal 4096 worlds: 14,435,984 env-steps/s   (near-linear GPU scaling)

## Honest scope
- (1) is gradient-based training through physics — a real "trained on the Mac" result.
- (2) is INFERENCE of a policy trained elsewhere (IsaacLab/PhysX), on the Mac.
- (3) is the batched-simulation throughput that RL training needs; a full PPO loop
  (torch + RL lib) was not run here. The RL algorithm itself is backend-agnostic.

## The claim, precisely
"Newton's GPU-accelerated batched simulation, differentiable physics, and RL-policy
runtime all run on an Apple Silicon Mac via the Metal backend — the same primitives
IsaacLab uses for GPU RL. Verified: differentiable training converges on Metal, an
IsaacLab-trained locomotion policy walks on Metal, and 4096 parallel envs step at
14.4M env-steps/s on Metal."

## df64 note
Newton's fp64 inertia kernels need the Metal df64 opt-in. run_newton_metal.py and the
bench/measure scripts patch Module.__init__ to set metal_float64=df64 before importing
newton. (This is the emulation path; see warp-metal-backend skill.)

## 4. Full PPO training loop from scratch on the Mac (added)
Script: ppo_cartpole_single_long.py (single-pole cartpole, asset cartpole_single_pendulum.usda).
Physics: Newton + MuJoCo-Warp solver, 256 batched envs on metal:0 (df64 opt-in).
Policy/PPO: PyTorch on MPS. Env: .venv-metal-main.

    python ppo_cartpole_single_long.py 256 300     # 1.84M env-steps in 148 s (~12.5k steps/s)
    python eval_cartpole_single_long.py

Result: mean episode return 14 -> 594 (max ~600). Deterministic eval, 64 envs, 10 s:
    random policy: mean survival 0.60 s, 0/64 upright at end
    PPO policy:    mean survival 10.00 s, 64/64 upright at end
Notes: the earlier USD "cartpole" asset is a double pendulum (barely learned); episode cap
of 200 steps made a 10 s eval out-of-distribution (4.2 s mean) - 600-step episodes fixed it.
~12.5k steps/s is bound by per-step host<->device numpy transfers in this simple loop, not
by the simulator (section 3 shows 14M steps/s for sim alone).


## 5. ANYmal C locomotion trained FROM SCRATCH with PPO on a Mac (added)
Files: anymal_env.py (GPU-resident batched env, obs/reward/reset as warp kernels),
train_anymal_ppo.py, eval_anymal.py, anymal_ppo.pt (final policy), anymal_train.log.
Physics config copied unchanged from newton's robot_anymal_c_walk (SolverMuJoCo, 50 Hz, 4 substeps,
PD ke=150 kd=5, action scale 0.5). Env: .venv-metal-main. PPO + MLP on MPS.

    python train_anymal_ppo.py 4096 800      # 78.6M env-steps in 26.0 min (~50k steps/s)
    python eval_anymal.py

Task: velocity tracking on flat ground, per-episode random command vx 0.3-1.0, vy +-0.3, yaw +-0.6.
Reward weights are my own choice (not IsaacLab's); no feet-air-time / foot-contact term.
Timeouts are treated as terminal (no bootstrap).

Training: ep return 65 (upd 50) -> 230 (100) -> 438 (200) -> ~530 plateau (500+); falls 67% -> ~0-3%.
Deterministic eval, 64 envs, command (0.8, 0, 0), 10 s:
    zero action : 64/64 stay up, 0.00 m/s
    random      : 64/64 stay up, 0.00 m/s (noise in PD targets barely moves it)
    PPO policy  : 64/64 stay up, fwd 0.78 m/s (body vx 0.81), lateral drift 2.34 m over 10 s
Reference: IsaacLab/PhysX ONNX policy ~1.2 m/s at command 1.0.

Caveats: lateral drift (~2.3 m over ~7.8 m forward) suggests slow heading drift; not diagnosed.
Gait quality (no air-time term) not inspected. Adaptive-LR rule pinned the LR at its 1e-5 floor for
much of the run (KL stayed 0.02-0.06 even at tiny LR; cause not investigated). Only the last
checkpoint is kept (overwritten every 25 updates). Single seed, single command, flat ground only.


## 6. ANYmal gait fixes: v1 -> v2 -> v3 (added)
Same task/physics as section 5 (command 0.8 m/s forward, 64 envs, 10 s, deterministic policy).
Each run: 4096 envs, 1000 updates, 98.3M env-steps (v2 31 min, v3 38 min) on one M5 Pro.

| Measure | v1 | v2 | v3 |
|---|---|---|---|
| Fall rate | 0% | 0% | 0% |
| Forward speed (cmd 0.8) | 0.78 | 0.77 | 0.76 m/s |
| Lateral drift over 10 s | 2.34 m | 0.58 m | 0.03 m |
| Yaw-rate bias | +0.050 | +0.018 | -0.006 rad/s |
| Base height | 0.376 | 0.530 | 0.518 m |
| Body pitch | 0.20 | -0.01 | +0.007 rad |
| Actions saturated | 48% | 0% | 4.9% |

Foot contact over 8 s (LF, RF, LH, RH):
- v2: contact fraction 0.72 0.68 1.00 1.00; touchdowns/s 3.2 3.1 0 0; peak height 8-9 cm front, 4 cm hind.
- v3: contact fraction 0.54 0.63 0.62 0.63; touchdowns/s 3.9 3.8 3.8 3.9; peak height 11-13 cm all four.

What changed:
- v1 -> v2 (anymal_env2.py, train_anymal_ppo2.py): height target + quadratic height penalty and
  orientation/joint-deviation/HAA penalties (fixes stooping), tighter yaw-tracking term (heading drift),
  action-saturation penalty, LR adapted once per update instead of per minibatch, periodic + best checkpoints.
- v2 -> v3 (anymal_env3.py, train_anymal_ppo3.py): v2 paid (air_time - 0.4) at touchdown, which is
  negative for short steps, so the hind feet never lifted. v3 pays min(air_time, 0.4) and adds a
  foot-slip penalty. All four feet now step in a symmetric trot.

Caveats: single seed; flat ground; v3 still 0.76 m/s vs 0.80 commanded; saturation rose to 4.9% in v3.
Gait judged from contact statistics plus the clips below.

v1 (stooping, drifting)      ![v1](media/anymal_v1.gif)

v2 (level body, hind feet drag)   ![v2](media/anymal_v2.gif)

v3 (all four feet step)      ![v3](media/anymal_v3.gif)

Full videos: anymal_walk.mp4 (v1), anymal_walk_v2.mp4, anymal_walk_v3.mp4. Policies: anymal_ppo.pt, anymal_v2.pt, anymal_v3.pt.


## 7. Hurdles, jumping, and one multi-skill policy (added, work in progress)
Files: anymal_env4.py / train_anymal_ppo4.py / eval_anymal4.py (hurdles), anymal_env_jump.py / train_anymal_jump.py /
eval_jump.py (jump only), anymal_env_multi.py / train_anymal_multi.py / eval_multi.py / render_multi.py (one policy, 3 skills).

Hurdles (8 heights 5-26 cm, policy sees distance+height, +5 on crossing): training from scratch learned only the 5 cm
hurdle (stood in front of taller ones). A curriculum (restart from that policy with 3-18 cm hurdles) raised 8 cm crossing
6% -> 81%; 11+ cm stayed <=12%. Base height never exceeded spawn (0.62 m): stepping over, not jumping. Run stopped there.

Jump task (no obstacles): four reward exploits found by inspecting rollouts, each visible within ~100 updates:
(1) sitting on the belly at z=0.29 with all feet curled up = "flight"; (2) tucking legs during the 0.62 m spawn drop;
(3) touch once, then tuck and fall; (4) dying early because every step carried a small penalty.
Fix: no flight-only reward; reward = height above spawn and upward velocity while all four feet are airborne
(after first ground contact), plus an alive bonus. Single-skill high jump (~200 updates): 0.3 s hops, base z up to 0.83 m.

One policy, three skills (walk / high jump / long jump): skill one-hot added to the observation, new input weights start
at zero, warm start from the walking policy (anymal_v3.pt), mixed batches (50/25/25), jump envs partly start mid-air
("launch-phase" starts). Ground-start eval after 2300 updates in total:
    walk       32/32 survive, 0.75 m/s (cmd 0.80; 0.81 before the final stage) -> walking retained
    high jump  16/16 survive, airtime mean 0.26 s (max 0.30), peak base z 0.71 m max (stand ~0.54, spawn 0.62)
    long jump  16/16 survive, airtime mean 0.16 s, 0.15 m per flight; but 3.5 m in 4 s = fast hopping/bounding, not a single long jump
Caveats: jumps are small hops, not yet a crouch-and-launch jump; single seed; no reference motions used.
Clips: media/anymal_multi_walk.gif, anymal_multi_high_jump.gif, anymal_multi_long_jump.gif.
