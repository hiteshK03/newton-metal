# How it works

## Stack
PyTorch (MPS) PPO  <-  actions / done flags  ->  Warp kernels: obs, reward, reset (one thread per env)  ->  Newton `SolverMuJoCo` (MuJoCo Warp)  ->  Warp on Metal (`metal:0`, via warp-metal)

- **warp-metal** gives Warp a Metal backend: kernels are translated to Metal Shading Language and compiled at runtime. Apple GPUs have no native float64, so float64 kernels are run with an opt-in double-single emulation (`Module.options["metal_float64"] = "df64"`), which `nmrl/anymal_walk_env.py` switches on. df64 is *not* IEEE binary64; it has float32's exponent range.
- **Newton** builds the batched model: `ModelBuilder.replicate` copies the robot into N separate worlds (robots in different envs never touch each other), then `CollisionPipeline` + `SolverMuJoCo` step all envs on the GPU.
- **Unified memory** on Apple silicon means the host can read env state without an explicit copy, but we still keep observations, rewards and resets on the GPU and move only actions in and `done` flags out each step.

## Walking environment (`nmrl/anymal_walk_env.py`)
- Physics settings copied from Newton's `robot_anymal_c_walk` example: 50 Hz control, 4 substeps, PD gains Kp=150 / Kd=5, action scale 0.5 rad around the default pose.
- Observation (48): base linear and angular velocity (body frame), gravity direction, command, joint positions relative to default, joint velocities, previous action.
- Command: forward 0.3-1.0 m/s, lateral +-0.3, yaw rate +-0.6, resampled per episode. Episode 500 steps (10 s). Terminates if the base drops below 0.25 m or tilts past about 60 degrees.
- Reward (v3), all computed in one kernel:
  linear-velocity tracking `exp(-err^2/0.25)` + yaw tracking `0.25 exp(-e^2/0.25) + 0.5 exp(-e^2/0.01)`
  minus: vertical velocity (2.0), roll/pitch rate (0.05), tilt (5.0), torque (1e-6), action change (0.02), joint deviation (0.1), hip-abduction deviation (0.5), action saturation (0.05), height `40 (z-0.48)^2`, foot slip (0.1 per squared slip);
  plus feet air time `0.5 * min(air_time, 0.4)` at touchdown.
- Weights are our own choices after iteration; see `docs/results.md` for what each term fixed.

## PPO (`nmrl/ppo_walk.py`)
4096 envs, 24-step rollouts, gamma 0.99, lambda 0.95, clip 0.2, 5 epochs x 4 minibatches, KL-adaptive learning rate (adapted once per update, not per minibatch), actor and critic MLPs 256-128-128 with ELU, initial action std 0.8. Checkpoints are written periodically and the best-scoring one is kept.

## Known rough edges
- `nmrl/ppo_*.py` parse `sys.argv` at import time; eval scripts reset `sys.argv` before importing them.
- Timeouts are treated as terminal (no value bootstrap).
