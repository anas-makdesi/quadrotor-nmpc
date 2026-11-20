"""Nonlinear MPC for the quadrotor with acados.

    minimise   sum_k ||[x_k; u_k] - [x_ref; u_ref](t_k)||_W^2 + ||x_N - x_ref(t_N)||_W_N^2
    subject to x_{k+1} = F(x_k, u_k)       (RK4 of :func:`qnmpc.model.casadi_model`)
               T_min <= u_i <= T_max,i     (per rotor; can change at run time)
               |w| <= body-rate limits     (soft)

The reference is the flat full-state reference of :mod:`qnmpc.trajectory` (position,
attitude, velocity, body rates and rotor thrusts). Horizon 1 s on a non-uniform grid:
10 steps of the control period (10 ms at 100 Hz) then 15 steps of 60 ms. One real-time
iteration (Gauss-Newton SQP, HPIPM) per control step by default.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import NU, NX, casadi_model
from .params import Quad

BUILD_DIR = Path(os.environ.get("QNMPC_BUILD_DIR", Path.cwd() / "build"))


@dataclass
class NMPCConfig:
    dt0: float = 0.01
    n_fine: int = 10
    dt_tail: float = 0.06
    n_tail: int = 15
    mode: str = "rti"  # or "sqp"
    max_iter: int = 50
    W_p: float = 200.0
    W_q: float = 20.0
    W_v: float = 5.0
    W_w: float = 0.5
    W_u: float = 1.0
    terminal_scale: float = 2.0
    slack: float = 1e3
    name: str = ""

    @property
    def N(self) -> int:
        return self.n_fine + self.n_tail

    @property
    def time_steps(self) -> np.ndarray:
        return np.r_[np.full(self.n_fine, self.dt0), np.full(self.n_tail, self.dt_tail)]

    @property
    def horizon(self) -> float:
        return float(self.time_steps.sum())


def build_ocp(quad: Quad, cfg: NMPCConfig):
    from acados_template import AcadosModel, AcadosOcp

    m = casadi_model(quad)
    model = AcadosModel()
    model.name = cfg.name or f"qnmpc_{cfg.mode}_N{cfg.N}"
    model.x, model.u = m["x"], m["u"]
    model.f_expl_expr = m["f"]

    ocp = AcadosOcp()
    ocp.model = model
    so = ocp.solver_options
    so.N_horizon = cfg.N
    so.tf = cfg.horizon
    so.time_steps = cfg.time_steps

    ny = NX + NU
    W = np.diag(np.r_[np.full(3, cfg.W_p), np.full(4, cfg.W_q), np.full(3, cfg.W_v),
                      np.full(3, cfg.W_w), np.full(4, cfg.W_u)])
    c = ocp.cost
    c.cost_type = c.cost_type_e = "LINEAR_LS"
    c.W = W
    c.W_e = cfg.terminal_scale * W[:NX, :NX]
    c.Vx = np.zeros((ny, NX)); c.Vx[:NX, :NX] = np.eye(NX)
    c.Vu = np.zeros((ny, NU)); c.Vu[NX:, :] = np.eye(NU)
    c.Vx_e = np.eye(NX)
    c.yref = np.zeros(ny)
    c.yref_e = np.zeros(NX)

    k = ocp.constraints
    k.x0 = np.r_[0, 0, 0, 1, 0, 0, 0, np.zeros(6)]
    k.idxbu = np.arange(NU)
    k.lbu = np.full(NU, quad.T_min)
    k.ubu = np.full(NU, quad.T_max)
    k.idxbx = np.arange(10, 13)
    k.lbx = -np.array(quad.w_max)
    k.ubx = np.array(quad.w_max)
    k.idxsbx = np.arange(3)
    c.zl = c.zu = cfg.slack * np.ones(3)
    c.Zl = c.Zu = cfg.slack * np.ones(3)

    so.integrator_type = "ERK"
    so.sim_method_num_stages = 4
    so.sim_method_num_steps = 1
    so.hessian_approx = "GAUSS_NEWTON"
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.qp_solver_cond_N = 5
    so.qp_solver_warm_start = 1
    so.qp_solver_iter_max = 50
    if cfg.mode == "rti":
        so.nlp_solver_type = "SQP_RTI"
    else:
        so.nlp_solver_type = "SQP"
        so.nlp_solver_max_iter = cfg.max_iter
        so.tol = 1e-6
    so.print_level = 0
    return ocp


class NMPC:
    """Trajectory-tracking NMPC. ``step(t, x)`` returns the rotor thrust commands."""

    def __init__(self, quad: Quad, traj, cfg: NMPCConfig | None = None, build: bool = True):
        from acados_template import AcadosOcpSolver

        self.quad, self.traj = quad, traj
        self.cfg = cfg = cfg or NMPCConfig()
        self.ocp = build_ocp(quad, cfg)
        out = BUILD_DIR / self.ocp.model.name
        out.mkdir(parents=True, exist_ok=True)
        self.ocp.code_export_directory = str(out / "c_generated_code")
        self.ocp.json_file = "acados_ocp.json"
        self.solver = AcadosOcpSolver(self.ocp, build=build, generate=build, verbose=False)
        self.offsets = np.r_[0.0, np.cumsum(cfg.time_steps)]
        self.T_max = np.full(NU, quad.T_max)
        self.initialized = False
        self.solve_times, self.wall_times, self.status, self.iters = [], [], [], []

    def set_thrust_limits(self, T_max):
        """Per-rotor maximum thrust (N) the controller may use; no rebuild needed."""
        self.T_max = np.minimum(np.asarray(T_max, float), self.quad.T_max)
        for k in range(self.cfg.N):
            self.solver.constraints_set(k, "ubu", self.T_max)

    def _set_reference(self, t, x0):
        X, U = self.traj.reference(t + self.offsets, self.quad)
        if np.dot(X[0, 3:7], x0[3:7]) < 0:  # pick the quaternion sign closest to the state
            X[:, 3:7] *= -1
        U = np.minimum(U, self.T_max)
        for k in range(self.cfg.N):
            self.solver.set(k, "yref", np.r_[X[k], U[k]])
        self.solver.set(self.cfg.N, "yref", X[-1])
        return X, U

    def step(self, t, x0) -> np.ndarray:
        x0 = np.asarray(x0, float)
        t0 = time.perf_counter()
        X, U = self._set_reference(t, x0)
        if not self.initialized:
            for k in range(self.cfg.N + 1):
                self.solver.set(k, "x", X[k])
                if k < self.cfg.N:
                    self.solver.set(k, "u", U[k])
            if self.cfg.mode == "rti":
                for _ in range(5):
                    self.solver.solve_for_x0(x0, False, False)
            self.initialized = True
        u = self.solver.solve_for_x0(x0, fail_on_nonzero_status=False,
                                     print_stats_on_failure=False)
        self.wall_times.append(time.perf_counter() - t0)
        self.solve_times.append(float(self.solver.get_stats("time_tot")))
        self.status.append(int(self.solver.status))
        self.iters.append(int(self.solver.get_stats("sqp_iter")) if self.cfg.mode == "sqp" else 1)
        return np.asarray(u, float)
