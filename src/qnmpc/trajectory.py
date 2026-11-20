"""Figure-8 reference with full-state feedforward from differential flatness.

Position  p(t) = [A sin(om t), B sin(2 om t), z0 + C sin(om t)], yaw = 0.
The peak speed (at the crossing point) is om * sqrt(A^2 + 4 B^2 + C^2).

From the position derivatives (up to snap) we get, for zero yaw and ignoring drag, the
attitude, body rates, collective thrust and rotor thrusts that fly the curve exactly:
    a = p'' + g e_z,  z_b = a/|a|,  T = m |a|,
    h = (m/T)(p''' - (z_b . p''') z_b),  w_x = -h . y_b,  w_y = h . x_b,  w_z = 0,
and the torques from J w' + w x J w (w' by finite differences).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import rot_to_quat
from .params import G, Quad


@dataclass
class Figure8:
    v_peak: float = 8.0  # m/s at the crossing point
    A: float = 5.0
    B: float = 2.5
    C: float = 0.5
    z0: float = 2.0

    @property
    def om(self) -> float:
        return self.v_peak / np.sqrt(self.A ** 2 + 4 * self.B ** 2 + self.C ** 2)

    @property
    def period(self) -> float:
        return 2 * np.pi / self.om

    def derivatives(self, t):
        """Position and its first four derivatives at times t: array (5, len(t), 3)."""
        t = np.atleast_1d(np.asarray(t, float))
        om = self.om
        out = np.zeros((5, t.size, 3))
        for k in range(5):
            s1 = om ** k * np.sin(om * t + k * np.pi / 2)
            s2 = (2 * om) ** k * np.sin(2 * om * t + k * np.pi / 2)
            out[k, :, 0] = self.A * s1
            out[k, :, 1] = self.B * s2
            out[k, :, 2] = self.C * s1 + (self.z0 if k == 0 else 0.0)
        return out

    def reference(self, t, quad: Quad):
        """Flat reference at times t: states (K, 13) and rotor thrusts (K, 4)."""
        t = np.atleast_1d(np.asarray(t, float))
        P, V, Acc, Jerk, _ = self.derivatives(t)
        K = t.size
        X = np.zeros((K, 13))
        W = np.zeros((K, 3))
        Tc = np.zeros(K)
        for i in range(K):
            a = Acc[i] + np.array([0, 0, G])
            Tc[i] = quad.m * np.linalg.norm(a)
            zb = a / np.linalg.norm(a)
            yb = np.cross(zb, [1.0, 0.0, 0.0])
            yb /= np.linalg.norm(yb)
            xb = np.cross(yb, zb)
            R = np.c_[xb, yb, zb]
            q = rot_to_quat(R)
            if i and np.dot(q, X[i - 1, 3:7]) < 0:  # keep the quaternion sign continuous
                q = -q
            h = quad.m / Tc[i] * (Jerk[i] - np.dot(zb, Jerk[i]) * zb)
            W[i] = [-np.dot(h, yb), np.dot(h, xb), 0.0]
            X[i, 0:3], X[i, 3:7], X[i, 7:10], X[i, 10:13] = P[i], q, V[i], W[i]
        # rotor thrusts from collective thrust and torques J w' + w x J w
        eps = 1e-3
        Wp = self._rates(t + eps, quad)
        Wm = self._rates(t - eps, quad)
        wdot = (Wp - Wm) / (2 * eps)
        J = quad.Jv
        tau = wdot * J + np.cross(W, W * J)
        Minv = np.linalg.inv(quad.mixer)
        U = (Minv @ np.c_[Tc, tau].T).T
        return X, U

    def _rates(self, t, quad):
        _, _, Acc, Jerk, _ = self.derivatives(t)
        W = np.zeros((t.size, 3))
        for i in range(t.size):
            a = Acc[i] + np.array([0, 0, G])
            Tc = quad.m * np.linalg.norm(a)
            zb = a / np.linalg.norm(a)
            yb = np.cross(zb, [1.0, 0.0, 0.0])
            yb /= np.linalg.norm(yb)
            xb = np.cross(yb, zb)
            h = quad.m / Tc * (Jerk[i] - np.dot(zb, Jerk[i]) * zb)
            W[i] = [-np.dot(h, yb), np.dot(h, xb), 0.0]
        return W
