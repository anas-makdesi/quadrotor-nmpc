"""Quadrotor rigid-body model.

State  x = [p (3), q (4, w-x-y-z unit quaternion, body to world), v (3, world), w (3, body)]
Input  u = rotor thrusts T (4) in newtons

    p' = v
    q' = 1/2 q * [0, w]
    v' = R(q) [0, 0, sum T] / m - g e_z - R D R^T v / m        (linear rotor drag D)
    w' = J^-1 (tau - w x J w),  [sum T, tau] = mixer T

* :func:`casadi_model` - the model the NMPC plans with (rotor thrusts act instantly).
* :func:`plant_step` - the simulated quadrotor: the same rigid body plus first-order
  motor dynamics (time constant ``tau_motor``) and optional rotor faults (lower
  efficiency, lower maximum thrust) that the nominal controller does not know about.
"""
from __future__ import annotations

import numpy as np

from .params import Faults, G, Quad

NX, NU = 13, 4


# --------------------------------------------------------------------- quaternions
def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    """Rotation matrix to unit quaternion (w >= 0)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def tilt_angle(q) -> float:
    """Angle between the body z axis and the world z axis (rad)."""
    w, x, y, z = q
    return float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


# -------------------------------------------------------------------------- CasADi
def casadi_model(quad: Quad):
    import casadi as ca

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    q, v, w = x[3:7], x[7:10], x[10:13]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = ca.vertcat(
        ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
        ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)),
    )
    M = ca.DM(quad.mixer)
    ft = ca.mtimes(M, u)
    D = ca.diag(ca.DM(quad.drag))
    J = ca.DM(quad.Jv)
    acc = ca.mtimes(R, ca.vertcat(0, 0, ft[0])) / quad.m - ca.vertcat(0, 0, G) \
        - ca.mtimes(R, ca.mtimes(D, ca.mtimes(R.T, v))) / quad.m
    qdot = 0.5 * ca.vertcat(
        -qx * w[0] - qy * w[1] - qz * w[2],
        qw * w[0] + qy * w[2] - qz * w[1],
        qw * w[1] - qx * w[2] + qz * w[0],
        qw * w[2] + qx * w[1] - qy * w[0],
    )
    Jw = J * w
    wdot = (ft[1:4] - ca.cross(w, Jw)) / J
    f = ca.vertcat(v, qdot, acc, wdot)
    return dict(x=x, u=u, f=f, R=R)


# --------------------------------------------------------------------------- plant
def rhs(x, T, quad: Quad):
    """Rigid-body dynamics for actual rotor thrusts T (NumPy)."""
    q = x[3:7]
    v, w = x[7:10], x[10:13]
    R = quat_to_rot(q)
    ft = quad.mixer @ T
    D = np.array(quad.drag)
    acc = R @ np.array([0, 0, ft[0]]) / quad.m - np.array([0, 0, G]) - R @ (D * (R.T @ v)) / quad.m
    qdot = 0.5 * quat_mul(q, np.r_[0.0, w])
    J = quad.Jv
    wdot = (ft[1:4] - np.cross(w, J * w)) / J
    return np.r_[v, qdot, acc, wdot]


class Plant:
    """Simulated quadrotor with motor lag and optional rotor faults."""

    def __init__(self, quad: Quad, x0, faults: Faults | None = None, dt_int: float = 1e-3):
        self.quad = quad
        self.faults = faults or Faults()
        self.x = np.asarray(x0, float).copy()
        # actual rotor thrusts (motor states), at steady state for the hover command
        self.T = self.actual_target(np.full(4, quad.hover_thrust))
        self.dt_int = dt_int

    def actual_target(self, cmd):
        """Thrust each motor settles to for a command (efficiency and caps applied)."""
        cmd = np.clip(np.asarray(cmd, float), self.quad.T_min, self.quad.T_max)
        t = np.array(self.faults.efficiency) * cmd
        return np.minimum(t, np.array(self.faults.max_thrust))

    def step(self, cmd, dt):
        target = self.actual_target(cmd)
        n = max(1, int(round(dt / self.dt_int)))
        h = dt / n
        a = np.exp(-h / self.quad.tau_motor)
        for _ in range(n):
            T = self.T
            k1 = rhs(self.x, T, self.quad)
            k2 = rhs(self.x + 0.5 * h * k1, T, self.quad)
            k3 = rhs(self.x + 0.5 * h * k2, T, self.quad)
            k4 = rhs(self.x + h * k3, T, self.quad)
            self.x = self.x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.x[3:7] /= np.linalg.norm(self.x[3:7])
            self.T = target + (self.T - target) * a  # exact first-order motor response
        return self.x.copy()
