"""Batched ANYmal C velocity-tracking env on Newton + warp-metal (Apple GPU).

Physics config is copied from newton's robot_anymal_c_walk example (known stable on Metal).
Obs / reward / termination / reset are warp kernels (env = tid); only `done` flags
and actions cross the host boundary each step.
"""
import time
import numpy as np
import warp as wp

wp.init(); wp.set_device("metal:0")
import warp._src.context as ctx
_o = ctx.Module.__init__
def _p(self, *a, **k):
    _o(self, *a, **k); self.options["metal_float64"] = "df64"
ctx.Module.__init__ = _p
import newton, newton.utils

NQ, NQD, NJ = 19, 18, 12
OBS_DIM = 48
ACTION_SCALE = 0.5
SUB = 4
FRAME_DT = 0.02          # 50 Hz control
KE, KD = 150.0, 5.0
EP_LEN = 500             # 10 s
Z_FALL = 0.25            # base height termination
TILT_FALL = 0.5          # terminate if gravity_b.z > -TILT_FALL  (> ~60 deg tilt)

# ---------------- kernels ----------------
@wp.kernel
def k_targets(act: wp.array2d[float], q0: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    e = i // 19
    j = i % 19
    if j < 7:
        if j == 6:
            out[i] = 1.0
        else:
            out[i] = 0.0
    else:
        a = wp.clamp(act[e, j - 7], -1.0, 1.0)
        out[i] = q0[e * 19 + j] + 0.5 * a

@wp.kernel
def k_obs(q: wp.array[float], qd: wp.array[float], qdef: wp.array[float],
          cmd: wp.array2d[float], prev: wp.array2d[float], obs: wp.array2d[float]):
    e = wp.tid()
    qo = e * 19
    do = e * 18
    r = wp.quat(q[qo + 3], q[qo + 4], q[qo + 5], q[qo + 6])
    vb = wp.quat_rotate_inv(r, wp.vec3(qd[do], qd[do + 1], qd[do + 2]))
    wb = wp.quat_rotate_inv(r, wp.vec3(qd[do + 3], qd[do + 4], qd[do + 5]))
    gb = wp.quat_rotate_inv(r, wp.vec3(0.0, 0.0, -1.0))
    for k in range(3):
        obs[e, k] = vb[k] * 2.0
        obs[e, 3 + k] = wb[k] * 0.25
        obs[e, 6 + k] = gb[k]
        obs[e, 9 + k] = cmd[e, k]
    for k in range(12):
        obs[e, 12 + k] = q[qo + 7 + k] - qdef[k]
        obs[e, 24 + k] = qd[do + 6 + k] * 0.05
        obs[e, 36 + k] = prev[e, k]

@wp.kernel
def k_reward(q: wp.array[float], qd: wp.array[float], qdef: wp.array[float], q0: wp.array[float],
             act: wp.array2d[float], prev: wp.array2d[float], cmd: wp.array2d[float],
             ep_len: wp.array[int], body_q: wp.array[wp.transformf], foot_b: wp.array[int],
             foot_p: wp.array[wp.vec3], air: wp.array2d[float], fxy: wp.array2d[float],
             rew: wp.array[float], done: wp.array[float], comp: wp.array2d[float]):
    e = wp.tid()
    qo = e * 19
    do = e * 18
    r = wp.quat(q[qo + 3], q[qo + 4], q[qo + 5], q[qo + 6])
    vb = wp.quat_rotate_inv(r, wp.vec3(qd[do], qd[do + 1], qd[do + 2]))
    wb = wp.quat_rotate_inv(r, wp.vec3(qd[do + 3], qd[do + 4], qd[do + 5]))
    gb = wp.quat_rotate_inv(r, wp.vec3(0.0, 0.0, -1.0))
    ex = vb[0] - cmd[e, 0]
    ey = vb[1] - cmd[e, 1]
    ew = wb[2] - cmd[e, 2]
    r_lin = wp.exp(-(ex * ex + ey * ey) / 0.25)
    r_yaw = 0.25 * wp.exp(-(ew * ew) / 0.25) + 0.5 * wp.exp(-(ew * ew) / 0.01)
    p_vz = vb[2] * vb[2]
    p_wxy = wb[0] * wb[0] + wb[1] * wb[1]
    p_ori = gb[0] * gb[0] + gb[1] * gb[1]
    p_tau = float(0.0)
    p_act = float(0.0)
    p_dev = float(0.0)
    p_haa = float(0.0)
    p_sat = float(0.0)
    for k in range(12):
        a = act[e, k]
        ac = wp.clamp(a, -1.0, 1.0)
        tgt = q0[qo + 7 + k] + 0.5 * ac
        tau = 150.0 * (tgt - q[qo + 7 + k]) - 5.0 * qd[do + 6 + k]
        p_tau += tau * tau
        d = a - prev[e, k]
        p_act += d * d
        dd = q[qo + 7 + k] - qdef[k]
        p_dev += dd * dd
        if k % 3 == 0:
            p_haa += dd * dd
        ov = wp.max(wp.abs(a) - 1.0, 0.0)
        p_sat += ov * ov
    # feet air time, positive-only: reward min(air, 0.4) at first touchdown; slip penalty in contact
    r_air = float(0.0)
    p_slip = float(0.0)
    for f in range(4):
        bq = body_q[e * 13 + foot_b[f]]
        fp = wp.transform_point(bq, foot_p[f])
        fz = fp[2]
        if fz < 0.045:
            if air[e, f] > 0.0:
                r_air += wp.min(air[e, f], 0.4)
            else:
                dx = fp[0] - fxy[e, 2 * f]
                dy = fp[1] - fxy[e, 2 * f + 1]
                dd2 = dx * dx + dy * dy
                if dd2 < 0.25:
                    p_slip += dd2 / (0.02 * 0.02)
            air[e, f] = 0.0
        else:
            air[e, f] = air[e, f] + 0.02
        fxy[e, 2 * f] = fp[0]
        fxy[e, 2 * f + 1] = fp[1]
    z = q[qo + 2]
    p_h = (z - 0.48) * (z - 0.48)
    total = 1.0 * r_lin + 1.0 * r_yaw - 2.0 * p_vz - 0.05 * p_wxy - 5.0 * p_ori \
        - 1.0e-6 * p_tau - 0.02 * p_act - 0.1 * p_dev - 0.5 * p_haa - 0.05 * p_sat \
        - 40.0 * p_h + 0.5 * r_air - 0.1 * p_slip
    bad = float(0.0)
    if z < 0.25:
        bad = 1.0
    if gb[2] > -0.5:
        bad = 1.0
    if not (wp.abs(z) < 100.0):
        bad = 1.0
    if not (wp.abs(total) < 1.0e6):
        bad = 1.0
        total = 0.0
    if ep_len[e] >= 500:
        bad = 1.0
    rew[e] = total
    done[e] = bad
    comp[e, 0] = r_lin
    comp[e, 1] = wb[2]
    comp[e, 2] = vb[0]

@wp.kernel
def k_reset(done: wp.array[float], seed: int, q: wp.array[float], qd: wp.array[float],
            q0: wp.array[float], qdef: wp.array[float], cmd: wp.array2d[float],
            prev: wp.array2d[float], ep_len: wp.array[int], fixed_cmd: wp.array[float], use_fixed: int,
            air: wp.array2d[float]):
    e = wp.tid()
    if done[e] > 0.5:
        st = wp.rand_init(seed, e)
        qo = e * 19
        do = e * 18
        for j in range(7):
            q[qo + j] = q0[qo + j]
        for k in range(12):
            q[qo + 7 + k] = qdef[k] + 0.1 * (wp.randf(st) * 2.0 - 1.0)
        for j in range(18):
            qd[do + j] = 0.0
        if use_fixed == 1:
            cmd[e, 0] = fixed_cmd[0]
            cmd[e, 1] = fixed_cmd[1]
            cmd[e, 2] = fixed_cmd[2]
        else:
            cmd[e, 0] = 0.3 + 0.7 * wp.randf(st)
            cmd[e, 1] = 0.6 * wp.randf(st) - 0.3
            cmd[e, 2] = 1.2 * wp.randf(st) - 0.6
        for k in range(12):
            prev[e, k] = 0.0
        for f in range(4):
            air[e, f] = 0.0
        ep_len[e] = 0


class AnymalEnv:
    def __init__(self, n_envs, fixed_cmd=None, njmax=50, nconmax=100):
        self.N = N = n_envs
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1)
        b.default_shape_cfg.ke = 5.0e4; b.default_shape_cfg.kd = 5.0e2
        b.default_shape_cfg.kf = 1.0e3; b.default_shape_cfg.mu = 0.75
        path = newton.utils.download_asset("anybotics_anymal_c")
        b.add_urdf(str(path / "urdf" / "anymal.urdf"),
                   xform=wp.transform(wp.vec3(0, 0, 0.62), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi * 0.5)),
                   floating=True, enable_self_collisions=False, collapse_fixed_joints=True,
                   ignore_inertial_definitions=False, joint_damping_mode="passive")
        init = {"RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8, "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
                "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8, "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8}
        for nme, v in init.items():
            i = next(i for i, l in enumerate(b.joint_label) if l.endswith(f"/{nme}"))
            b.joint_q[i + 6] = v
        for i in range(len(b.joint_target_ke)):
            b.joint_target_ke[i] = KE; b.joint_target_kd[i] = KD
        big = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(big)
        big.replicate(b, N, spacing=(2.0, 2.0, 0.0))
        big.add_ground_plane()
        self.model = model = big.finalize()
        assert model.joint_coord_count == N * NQ and model.joint_dof_count == N * NQD
        self.solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, solver="newton",
                                                  ls_iterations=50, njmax=njmax, nconmax=nconmax)
        self.cp = newton.CollisionPipeline(model); self.contacts = self.cp.contacts()
        self.s0, self.s1 = model.state(), model.state(); self.ctrl = model.control()
        newton.eval_fk(model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        self.q0 = wp.clone(self.s0.joint_q)                       # spawn pose, all envs
        self.qdef = wp.array(self.q0.numpy()[7:19].astype(np.float32), dtype=float)
        self.ctrl.joint_target_q = wp.zeros(N * NQ, dtype=float)
        nb = model.body_count // N
        assert nb == 13, nb
        sb = model.shape_body.numpy(); st = model.shape_type.numpy(); tf = model.shape_transform.numpy()
        fidx = [i for i in range(len(sb)) if sb[i] in range(nb) and st[i] == 3]
        assert len(fidx) == 4, fidx
        self.foot_b = wp.array(np.array([sb[i] for i in fidx], np.int32), dtype=int)
        self.foot_p = wp.array(np.array([tf[i][:3] for i in fidx], np.float32), dtype=wp.vec3)
        self.air = wp.zeros((N, 4), dtype=float)
        self.fxy = wp.zeros((N, 8), dtype=float)
        self.obs = wp.zeros((N, OBS_DIM), dtype=float)
        self.act = wp.zeros((N, NJ), dtype=float)
        self.prev = wp.zeros((N, NJ), dtype=float)
        self.cmd = wp.zeros((N, 3), dtype=float)
        self.rew = wp.zeros(N, dtype=float); self.done = wp.zeros(N, dtype=float)
        self.comp = wp.zeros((N, 3), dtype=float)
        self.ep_len = wp.zeros(N, dtype=int)
        self.use_fixed = 1 if fixed_cmd is not None else 0
        self.fixed = wp.array(np.array(fixed_cmd if fixed_cmd is not None else [0, 0, 0], np.float32), dtype=float)
        self._seed = 1
        self.reset_all()

    def reset_all(self):
        self.done.fill_(1.0)
        self._reset()
        self.done.zero_()

    def _reset(self):
        self._seed += 1
        wp.launch(k_reset, dim=self.N, inputs=[self.done, self._seed, self.s0.joint_q, self.s0.joint_qd,
                                               self.q0, self.qdef, self.cmd, self.prev, self.ep_len,
                                               self.fixed, self.use_fixed, self.air])
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        wp.launch(k_obs, dim=self.N, inputs=[self.s0.joint_q, self.s0.joint_qd, self.qdef,
                                             self.cmd, self.prev, self.obs])

    def get_obs_np(self):
        return self.obs.numpy()

    def step(self, act_np):
        """act_np: (N,12) float32 numpy. Returns obs, rew, done as numpy."""
        self.act.assign(act_np)
        wp.launch(k_targets, dim=self.N * NQ, inputs=[self.act, self.q0, self.ctrl.joint_target_q])
        dt = FRAME_DT / SUB
        for _ in range(SUB):
            self.s0.clear_forces(); self.cp.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.ctrl, self.contacts, dt)
            self.s0, self.s1 = self.s1, self.s0
        wp.launch(k_reward, dim=self.N, inputs=[self.s0.joint_q, self.s0.joint_qd, self.qdef, self.q0,
                                                self.act, self.prev, self.cmd, self.ep_len,
                                                self.s0.body_q, self.foot_b, self.foot_p, self.air, self.fxy,
                                                self.rew, self.done, self.comp])
        rew = self.rew.numpy(); done = self.done.numpy()
        info = self.comp.numpy()
        wp.copy(self.prev, self.act)
        self.ep_len.assign(self.ep_len.numpy() + 1)
        if done.any():
            self._reset()      # also refreshes obs for reset envs
        else:
            wp.launch(k_obs, dim=self.N, inputs=[self.s0.joint_q, self.s0.joint_qd, self.qdef,
                                                 self.cmd, self.prev, self.obs])
        # NOTE: obs for non-reset envs must reflect post-step state; recompute for all, then
        # overwrite reset envs' obs happens inside _reset (it runs k_obs on all envs after reset).
        return self.obs.numpy(), rew, done, info
