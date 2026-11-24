# Known gaps and ongoing work

This is an ongoing effort, not a finished port. What is missing or unproven as of this commit (Apple M5 Pro, warp-metal 1.17.0.4, Newton branch `metal-float32-inertia`). Corrections and contributions are welcome; see CONTRIBUTING.md.

## Metal / Warp limits
- **No float64 on Metal.** Apple GPUs have no fp64. Kernels that use `float64` fail at load: `ik_h1` (`_mesh_edge_boundary_kernel`) and `mpm_granular` (`integrate_fraction`) in Newton's examples. warp-metal has an opt-in double-single (`df64`) emulation, but it is not IEEE, has the float32 exponent range, and has had correctness bugs, so this project does **not** use it. Newton's inertia code was moved to float32 on our Newton branch; that change was checked on ANYmal walking only.
- **float64 in more Newton examples:** `cloth_h1` (mesh edge boundary kernel) joins `ik_h1` and `mpm_granular`.
- **`proxy_joint_gripper`:** SolverVBD's rigid contact output buffer needs to grow during graph capture, which fails. Not investigated.
- **VBD warning:** several passing VBD examples print a warp-metal warning that one kernel (`apply_rigid_soft_truncation...`) was refused by the record cache. We have not checked what that costs.
- **MSL compile error in `cable_pile`:** "reference type must have explicit address space qualifier". Not investigated yet.
- **CUDA-only paths.** `robot_panda_hydro`, `brick_stacking` and `nut_bolt_sdf` (SDF texture build, `wp.Texture3D`), `mujoco_sleeping` and `steel_sleeping` require CUDA by design.
- **Other Warp features** (e.g. autodiff / `Tape`, `warp.fem`, volumes) were not tested here and should be assumed unsupported until checked.

## Newton on Metal
- **We need a Newton branch.** Three small commits on top of upstream `main` (GJK/MPR scratch buffers that work on Metal, float32 inertia kernels) are not upstream. The upstream draft PR (newton-physics/newton#5264) is on hold at the maintainers' request.
- **A cylinder case still fails on Metal** in the test we adapted for that PR (a specialisation conflict between `array_t` and `fixedarray_t` at different call sites). Not fixed.
- **Examples:** 33 of 50 upstream examples pass on Metal; the rest are in the README table. `robot_anymal_c_walk` and `robot_asroballet` were not run (need `warp_nn`). `basic_pendulum` trips a 1e-4 velocity tolerance because of float noise. `ik_cube_stacking` (success rate 0.0) and `robot_omniwheel` (run too short) fail and have not been checked against CPU.

## Our own results
- **Jumping is partial.** Best run: single jumps up to 0.56 m forward against a 0.72 m target; 100% success at half size, 0% at full size under a strict test. See `experiments/jump/`.
- **Manipulation (Franka cube lift) does not work yet.** Five PPO runs reach and grasp the cube but never lift it, while a scripted IK pick lifts 16/16 through the same environment. Likely a joint-space exploration problem; the next idea is end-effector (Cartesian) actions. The lift code is not in this commit. The Allegro-hand task has not been started.
- **Single seed, single machine, flat ground, simulation only.** No sim-to-real.
- **No performance or accuracy comparison against CUDA.** The throughput numbers are for this Mac only.
- **Unofficial wheel.** The warp-metal wheel we install is our own build of the fork, not an official release. Tested on Python 3.14 and macOS arm64 only.
