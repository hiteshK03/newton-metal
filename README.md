# newton-metal-rl

Train a quadruped to walk **from scratch with PPO, entirely on a Mac GPU**: [Newton](https://github.com/newton-physics/newton) physics (MuJoCo Warp solver) running on Apple Metal through [warp-metal](https://github.com/innate-inc/warp-metal), with the learning done by PyTorch on MPS. No CUDA, no cloud.

![ANYmal C walking, trained on an Apple-silicon Mac](media/anymal_walk.gif)

> **Status: early.** Walking is reproducible and measured. The jump work is experimental and only partly working (see below). All results are single-seed.

## What works

ANYmal C velocity tracking on flat ground, command 0.8 m/s forward, 64 envs, 10 s, deterministic policy, Apple M5 Pro:

| | v1 | v2 | **v3 (shipped)** |
|---|---|---|---|
| Fall rate | 0% | 0% | **0%** |
| Forward speed (cmd 0.8) | 0.78 | 0.77 | **0.76 m/s** |
| Lateral drift over 10 s | 2.34 m | 0.58 m | **0.03 m** |
| Yaw-rate bias | +0.050 | +0.018 | **-0.006 rad/s** |
| Base height | 0.376 m | 0.530 m | **0.518 m** |
| Hind feet lift? | n/a | no (dragged) | **yes, symmetric trot** |

Training the shipped policy: 4096 envs, 1000 PPO updates, 98.3M env-steps in about 38 minutes (roughly 40-50k env-steps/s including learning). `docs/results.md` has the full table and how each fix was found.

## Quickstart (macOS, Apple silicon)

Tested with Python 3.14.7, `warp-metal` 1.17.0.3, `mujoco-warp` 3.17.0, torch 2.15.

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # Newton: until the Metal changes are upstream, use this branch (2 commits on top of upstream main)
    git clone -b metal-steel-demos https://github.com/hiteshK03/newton && pip install -e ./newton

    python scripts/eval_walk.py        # expect: 64/64 survive, ~0.76 m/s, ~0.03 m drift
    python scripts/render_walk.py      # writes media/anymal_walk.mp4

    # train your own (writes to the path you give; the shipped checkpoint is never overwritten by default runs)
    python -m nmrl.ppo_walk 4096 1000 checkpoints/my_walk.pt

## How it works
One GPU-resident environment: observations, rewards and resets are Warp kernels (one thread per env). Only the actions and the `done` flags cross the host/device boundary each step. See [`docs/how-it-works.md`](docs/how-it-works.md).

## Experimental: jumping (`experiments/jump/`)
We tried to teach a high jump and a long jump. Honest summary:
- Reward-only jumping hit **four reward exploits** (sitting on the belly, tucking during the spawn drop, ...). Write-up: [`docs/reward-exploits.md`](docs/reward-exploits.md).
- A hand-built reference with four imitation tricks (state initialisation, early termination, failure-point resets, a joint reference) was a **null result**: no arm learned to jump in 300 updates.
- After adding a flight-timing reward and a distance-scale curriculum, one run produced real single jumps (up to 0.56 m forward vs a 0.72 m target; 100% success at half size, 0% at full size under a strict test). One seed; the walking skill is probably lost in that checkpoint (not tested). Details and caveats: [`experiments/jump/README.md`](experiments/jump/README.md).

This is not a replication of SuperMimic or BlinkMimic. Porting a mimic-style tracker to Newton-on-Metal is a roadmap item, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout
    nmrl/            environments and PPO (walk: anymal_walk_env.py, ppo_walk.py; experimental: anymal_multi_env.py, ppo_multi.py)
    scripts/         eval / diagnose / render the walking policy
    experiments/jump reference generation, curriculum, ablation, evals
    checkpoints/     shipped walking policy (0.5 MB); experimental ones under experimental/
    docs/            how-it-works, results, reward-exploits, notebook (raw lab notes)
    media/           GIFs

## Credits
Built on NVIDIA Warp, warp-metal (innate-inc), Newton, MuJoCo Warp and PyTorch. Full list with licences and links: [`ATTRIBUTION.md`](ATTRIBUTION.md). Developed with Claude Code and the Hermes agent.

## Limitations
Single seed. Flat ground only. Simulation only; no sim-to-real. The ANYmal model is downloaded at runtime from Newton's assets (not redistributed here).

## Licence
Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
