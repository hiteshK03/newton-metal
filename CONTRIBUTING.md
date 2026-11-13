# Contributing

Thanks for helping. This is an early, honest-results project: small PRs, reproducible numbers, and negative results are all welcome.

## Setup
See the README (macOS on Apple silicon, Python 3.14 tested). Smoke tests:

    python scripts/eval_walk.py            # should report 64/64 survive, ~0.76 m/s
    python experiments/jump/smoke_multi.py # env builds and steps (experimental)

## Ground rules
- **Report seeds and sample sizes.** Single-seed results must say so.
- **Check what the robot actually does,** not just the reward. We hit four reward exploits on one task (see `docs/reward-exploits.md`); a PR that adds a reward should include a rollout check.
- **No licensed data in the repo.** Motion-capture and other restricted data stay out; add a download script with checksums instead.
- **Keep upstream notices** in any file derived from upstream code, and add new dependencies to `ATTRIBUTION.md`.
- Never commit API keys, tokens or personal files.

## Good first issues
- Add a second robot (Go2 or similar) with the walking task.
- **Performance (profile first: `python scripts/profile_step.py 4096`).** On an M5 Pro the MuJoCo Warp solver is ~97% of an env step; reward, host readback and policy inference are ~2%. The GPU is not saturated (8x more envs costs ~2x solver time), which suggests per-kernel launch overhead. Ideas: measure training throughput at 8192/16384 envs; look at command-buffer batching in warp-metal; check how much of the solver time is df64 (float64 emulation); try fewer line-search iterations / substeps and re-validate the policy.
- Seeds sweep for the walking policy (3+ seeds) and report the spread.
- Replace the hard-coded `sys.argv` handling in `nmrl/ppo_*.py` with proper CLI parsing.
- Port a mimic-style tracker (SuperMimic / BlinkMimic) onto Newton-on-Metal physics.
- Reproduce the experimental jump result with other seeds, or find out why the curriculum stages collapse.
