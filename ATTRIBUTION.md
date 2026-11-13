# Attribution

This project is built on other people's work. Nothing below is endorsed by, or affiliated with, its authors.

## Code and libraries we depend on

| Project | What we use it for | Licence | Link |
|---|---|---|---|
| **warp-metal** (innate-inc) | Adds a `metal:0` device to NVIDIA Warp so kernels run on the Apple GPU. This project does not contain its code; it installs the published package. | see the project | https://github.com/innate-inc/warp-metal (backend developed in https://github.com/innate-inc/warp) |
| **NVIDIA Warp** | GPU kernel language (`warp-lang`) | Apache-2.0 | https://github.com/NVIDIA/warp |
| **Newton** (newton-physics) | Physics engine, model builder, viewer. We run it on a branch of a fork with two small Metal-compatibility commits (GJK/MPR scratch buffers, `float_flip` snippet) that are not upstream yet. | Apache-2.0 | https://github.com/newton-physics/newton , fork branch https://github.com/hiteshK03/newton/tree/metal-steel-demos |
| **MuJoCo / MuJoCo Warp** (Google DeepMind) | Rigid-body solver behind Newton's `SolverMuJoCo` | Apache-2.0 | https://github.com/google-deepmind/mujoco , https://github.com/google-deepmind/mujoco_warp |
| **PyTorch** | PPO networks and optimisation on Apple's MPS device | BSD-style | https://pytorch.org |
| **ANYmal C model** (ANYbotics) | Robot description, downloaded at runtime through Newton's asset helper. Not redistributed here; check its own licence before reuse. | see the asset | via `newton.utils.download_asset("anybotics_anymal_c")` |

## Algorithms and ideas

- PPO: Schulman et al., 2017. GAE: Schulman et al., 2016.
- The velocity-tracking task, reward terms and feet-air-time idea follow the legged-gym / Isaac Lab tradition (e.g. Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL", 2021). Our reward weights are our own choices, not copied from those projects.
- The physics settings (50 Hz control, 4 substeps, PD gains, action scale) are copied from Newton's `robot_anymal_c_walk` example.

## Related work that guided the experiments (no code from these is used)

- Motion tracking / mimic line: DeepMimic (Peng et al., 2018); SuperMimic (arXiv:2610.28480) and the community list https://github.com/miao-org/awesome-supermimic ; Apple-silicon port https://github.com/WXyuany/supermimic-m5 ; quadruped retargeting pipeline https://github.com/Houserz/supermimic-quadruped .
- Latent-action / VLA front ends considered for future work: WholeBodyVLA (arXiv:2512.11047), VFlash (arXiv:2611.08449).
- Metal patches for MuJoCo Warp / mjlab by David Dobas, referenced from the warp-metal README: https://github.com/DavidDobas/mujoco_warp , https://github.com/DavidDobas/mjlab .

## Data

No third-party motion data is included. Dog motion-capture sources discussed in the notes (AI4Animation, CC BY-NC 4.0; others) are research-only and must be downloaded by the user under their own terms.

## Development

Developed with the assistance of Claude Code (Anthropic) and the Hermes agent. All results in this repository were produced by running the code here; see `docs/results.md` for exact numbers and caveats.
