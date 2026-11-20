"""Baseline controllers.

* :class:`GeometricController` - the SE(3) geometric tracking controller (Lee, Leok,
  McClamroch, 2010) with differential-flatness feedforward of acceleration, body rates
  and torques: the standard choice in PX4-style and research autopilots. Rotor thrusts
  come from an attitude-priority mixer (collective thrust gives way to the torques when
  a rotor would saturate).
* :class:`LinearMPC` - MPC on the model linearised at hover (small-angle attitude
  error, LTI), tracking the same flat reference, solved by OSQP with warm start. Only the
  vectors of the QP change at run time, so each step is a cheap QP solve.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

from .model import NU, casadi_model, quat_to_rot
from .params import G, Quad


def _vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class GeometricController:
    def __init__(self, quad: Quad, traj, kp=(12.0, 12.0, 16.0), kv=(6.0, 6.0, 8.0),
                 att_freq: float = 25.0, att_damp: float = 0.9):
        self.quad, self.traj = quad, traj
        self.kp, self.kv = np.array(kp), np.array(kv)
        J = quad.Jv
        self.KR = J * att_freq ** 2
        self.Kw = J * 2 * att_damp * att_freq
        self.Minv = np.linalg.inv(quad.mixer)
        self.T_max = np.full(NU, quad.T_max)
        self.solve_times: list[float] = []

    def set_thrust_limits(self, T_max):
        self.T_max = np.minimum(np.asarray(T_max, float), self.quad.T_max)

    def step(self, t, x):
        t0 = time.perf_counter()
        q = self.quad
        D = self.traj.derivatives(t)
        p_r, v_r, a_r = D[0, 0], D[1, 0], D[2, 0]
        Xr, Ur = self.traj.reference(t, q)
        R_r = quat_to_rot(Xr[0, 3:7])
        w_r = Xr[0, 10:13]
        p, v, w = x[0:3], x[7:10], x[10:13]
        R = quat_to_rot(x[3:7])
        a_cmd = a_r + self.kp * (p_r - p) + self.kv * (v_r - v) + np.array([0, 0, G])
        # drag compensation with the known drag model
        a_cmd += R @ (np.array(q.drag) * (R.T @ v)) / q.m
        zb = a_cmd / np.linalg.norm(a_cmd)
        yb = np.cross(zb, [1.0, 0.0, 0.0])
        yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        R_d = np.c_[xb, yb, zb]
        thrust = q.m * float(a_cmd @ R[:, 2])
        e_R = 0.5 * _vee(R_d.T @ R - R.T @ R_d)
        w_d = R.T @ R_r @ w_r
        e_w = w - w_d
        # feedforward torque of the reference, mapped to the current attitude
        tau_ff = q.mixer[1:] @ Ur[0]
        tau = -self.KR * e_R - self.Kw * e_w + np.cross(w, q.Jv * w) + tau_ff \
            - np.cross(w_r, q.Jv * w_r)
        T = self.allocate(thrust, tau)
        self.solve_times.append(time.perf_counter() - t0)
        return T

    def allocate(self, thrust, tau):
        """Attitude-priority allocation (as autopilot mixers do): keep the torques and
        shift the collective thrust so that all rotors fit their limits; if the torques
        alone do not fit, scale them down. Then clip."""
        q = self.quad
        dT = self.Minv @ np.r_[0.0, tau]  # rotor thrusts that make the torques
        base = thrust / 4
        lo, hi = q.T_min - dT, self.T_max - dT  # admissible range for the common part
        if lo.max() <= hi.min():
            base = np.clip(base, lo.max(), hi.min())
        else:  # torques do not fit: scale them so that they just fit, collective in the middle
            span = dT.max() - dT.min()
            room = (self.T_max.min() - q.T_min)
            dT *= room / span
            base = q.T_min - dT.min()
        return np.clip(base + dT, q.T_min, self.T_max)


class LinearMPC:
    """LTI MPC at hover. State z = [p, 2 q_vec, v, w] (12), input u = T - hover."""

    def __init__(self, quad: Quad, traj, dt0=0.01, n_fine=10, dt_tail=0.06, n_tail=15,
                 W=(200.0, 20.0, 5.0, 0.5), W_u=1.0, terminal_scale=2.0):
        import casadi as ca
        import osqp

        self.quad, self.traj = quad, traj
        self.dts = np.r_[np.full(n_fine, dt0), np.full(n_tail, dt_tail)]
        self.N = N = len(self.dts)
        self.offsets = np.r_[0.0, np.cumsum(self.dts)]
        m = casadi_model(quad)
        xh = np.r_[0, 0, 0, 1, 0, 0, 0, np.zeros(6)]
        uh = np.full(NU, quad.hover_thrust)
        A13 = np.array(ca.Function("A", [m["x"], m["u"]], [ca.jacobian(m["f"], m["x"])])(xh, uh))
        B13 = np.array(ca.Function("B", [m["x"], m["u"]], [ca.jacobian(m["f"], m["u"])])(xh, uh))
        # reduce to 12 states: drop q_w (= 1 at hover), use e = 2 q_vec
        keep = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        S = np.eye(13)[keep]  # 12 x 13 selection
        Tm = np.diag([1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1.0])
        Ti = np.linalg.inv(Tm)
        A = Tm @ S @ A13 @ S.T @ Ti
        B = Tm @ S @ B13
        nz, nu = 12, NU
        self.nz = nz
        Ads, Bds = [], []
        for h in self.dts:
            M = np.zeros((nz + nu, nz + nu))
            M[:nz, :nz], M[:nz, nz:] = A, B
            E = expm(M * h)
            Ads.append(E[:nz, :nz]); Bds.append(E[:nz, nz:])
        wz = np.r_[np.full(3, W[0]), np.full(3, W[1]), np.full(3, W[2]), np.full(3, W[3])]
        # variables: z_0..z_N, u_0..u_{N-1}
        nv = (N + 1) * nz + N * nu
        self.iz = lambda k: k * nz  # noqa: E731
        self.iu = lambda k: (N + 1) * nz + k * nu  # noqa: E731
        P = np.zeros(nv)
        for k in range(N + 1):
            s = self.dts[k] if k < N else terminal_scale
            P[self.iz(k): self.iz(k) + nz] = 2 * wz * s
            if k < N:
                P[self.iu(k): self.iu(k) + nu] = 2 * W_u * s
        self.Pdiag = P
        self.P = sp.diags(P).tocsc()
        rows = []
        Aeq = sp.lil_matrix(((N + 1) * nz, nv))
        Aeq[0:nz, 0:nz] = np.eye(nz)
        for k in range(N):
            r = (k + 1) * nz
            Aeq[r:r + nz, self.iz(k + 1): self.iz(k + 1) + nz] = np.eye(nz)
            Aeq[r:r + nz, self.iz(k): self.iz(k) + nz] = -Ads[k]
            Aeq[r:r + nz, self.iu(k): self.iu(k) + nu] = -Bds[k]
        Ain = sp.lil_matrix((N * nu, nv))
        for k in range(N):
            Ain[k * nu:(k + 1) * nu, self.iu(k): self.iu(k) + nu] = np.eye(nu)
        self.A = sp.vstack([Aeq, Ain]).tocsc()
        self.neq = (N + 1) * nz
        self.uh = uh
        self.T_max = np.full(NU, quad.T_max)
        self.l = np.r_[np.zeros(self.neq), np.tile(quad.T_min - uh, N)]
        self.u = np.r_[np.zeros(self.neq), np.tile(self.T_max - uh, N)]
        self.solver = osqp.OSQP()
        self.solver.setup(P=self.P, q=np.zeros(nv), A=self.A, l=self.l, u=self.u, verbose=False,
                          warm_starting=True, polishing=False, eps_abs=1e-5, eps_rel=1e-5,
                          max_iter=4000)
        self.solve_times: list[float] = []
        self.wall_times: list[float] = []
        self.status: list[str] = []
        self._last = np.zeros(nu)
        del rows

    def set_thrust_limits(self, T_max):
        self.T_max = np.minimum(np.asarray(T_max, float), self.quad.T_max)
        self.u[self.neq:] = np.tile(self.T_max - self.uh, self.N)

    @staticmethod
    def _z(x):
        q = x[3:7] if x[3] >= 0 else -x[3:7]
        return np.r_[x[0:3], 2 * q[1:4], x[7:10], x[10:13]]

    def step(self, t, x):
        t0 = time.perf_counter()
        X, U = self.traj.reference(t + self.offsets, self.quad)
        Z = np.array([self._z(xr) for xr in X])
        qv = np.zeros_like(self.Pdiag)
        for k in range(self.N + 1):
            sl = slice(self.iz(k), self.iz(k) + self.nz)
            qv[sl] = -self.Pdiag[sl] * Z[k]
            if k < self.N:
                su = slice(self.iu(k), self.iu(k) + NU)
                qv[su] = -self.Pdiag[su] * (np.minimum(U[k], self.T_max) - self.uh)
        z0 = self._z(x)
        self.l[:self.nz] = self.u[:self.nz] = z0
        t1 = time.perf_counter()
        self.solver.update(q=qv, l=self.l, u=self.u)
        res = self.solver.solve()
        self.solve_times.append(time.perf_counter() - t1)
        self.status.append(res.info.status)
        if res.x is not None and res.info.status_val in (1, 2):
            self._last = res.x[self.iu(0): self.iu(0) + NU]
        T = np.clip(self._last + self.uh, self.quad.T_min, self.T_max)
        self.wall_times.append(time.perf_counter() - t0)
        return T
