# Results

All numbers: Apple M5 Pro, single seed, deterministic policy. Re-run `scripts/eval_walk.py` to reproduce the walking row.

## Walking (shipped)
See the table in the README. Foot contact over 8 s for v3 (LF, RF, LH, RH): contact fraction 0.54 / 0.63 / 0.62 / 0.63, touchdowns per second 3.9 / 3.8 / 3.8 / 3.9, peak foot height 11-13 cm, mean 2.4 feet on the ground at once.

## Training throughput vs number of envs
Walking trainer (`python -m nmrl.ppo_walk N 12 scratch.pt`), Apple M5 Pro, 24 GB, one run per row. Throughput is total env-steps over total time at update 11 (includes some start-up cost, so small runs read slightly low). Peak footprint is the macOS process figure from `/usr/bin/time -l`; it may not count every Metal buffer.

| Envs | Throughput (env-steps/s) | Peak footprint |
|---|---|---|
| 512 | 13k | 2.1 GB |
| 1024 | 24k | 2.0 GB |
| 2048 | 39k | 2.9 GB |
| 4096 | 55k | 3.3 GB |
| 8192 | 66k | 4.8 GB |
| 16384 | 77k | 9.0 GB |

Throughput keeps rising but each doubling past 4096 envs gives about 20% more for 1.5-2x the memory. We have not profiled why it saturates (GPU, Warp launch overhead or the PPO update on MPS). Eval (64 envs) peaked at 1.8 GB. Not measured: Macs with less memory, or a full 1000-update run.

## Jumping (experimental, see `experiments/jump/README.md`)
- Single-skill high jump (~200 updates): 0.3 s hops, base up to 0.83 m (standing ~0.54 m).
- Multi-skill policy (walk + high jump + long jump, mixed batches, warm start from walking): walking kept at 0.72-0.81 m/s while hopping appeared; long jump became vertical hopping (0.03 m forward per flight).
- Ablation of four tracking tricks on a hand-built long-jump reference, 300 updates each, same init and seed: **0% success for every arm** (`docs/results/ablation_results.txt`).
- Distance-scale curriculum + flight-window reward (`docs/results/curriculum_results.txt`): the final stage produced one real flight per robot at every scale; strict success 100% at half size, 27% at 0.75, 0% at full size (flight covers 0.19 m of a 0.48 m target).

## Caveats
One seed per run; strict success thresholds were tightened mid-project (the ablation used a looser distance-only rule); the curriculum's first two stages collapsed, so the staging itself did not work as designed.

`docs/notebook.md` has the raw lab notes (written as the work progressed; early sections were produced by the Hermes agent).
