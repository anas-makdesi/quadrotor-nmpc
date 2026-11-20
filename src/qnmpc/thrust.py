"""Certified rotor thrust from flight data, with ddma.

A damaged propeller, a worn motor or a sagging battery lowers the thrust a rotor can
give. Planning with the rated thrust then asks the rotor for more than it has: the
controller loses attitude authority exactly when it needs it most. This module turns
measured thrust data into a per-rotor maximum thrust that is *guaranteed* to be there.

Data: for each rotor, pairs (filtered command c, measured thrust y), e.g. from a short
"thrust check" hop. Rotor thrusts are measured in flight from the IMU through the
inverse mixer (collective thrust from the accelerometer, torques from the angular
acceleration). Assumptions:

* each rotor's steady-state thrust is a function f_i(c) of its command that is
  non-decreasing, no steeper than ``g_hi`` (never more efficient than ``g_hi`` times its
  rating) and concave (it can only saturate, not get stronger at high command);
* the motor responds like a first-order lag with known time constant, and the
  thrust measurements are within +-``noise``.

The command is passed through the same first-order lag. For a concave f, the lagged
thrust never exceeds f(lagged command) (Jensen's inequality), so each sample gives
f_i(c) >= y - noise: a one-sided bound that holds during transients too, so no data has
to be discarded. ddma's :class:`ddma.GrowthMap` turns these samples and the slope
bounds into a lower bound on f_i over the whole command range that holds for every
function consistent with the data and the assumptions. The certified maximum thrust is
its largest value (the rotor provably reaches it). The NMPC uses it as the rotor's
input bound, which can be changed at run time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import Quad


@dataclass
class ThrustCertificate:
    quad: Quad
    cells: int = 140
    g_hi: float = 1.1
    noise: float = 0.15  # measurement error bound (N)
    prior: float = None  # thrust assumed before data (default: hover thrust x 1.3)
    models: list = field(init=False, repr=False)
    n_used: np.ndarray = field(init=False)

    def __post_init__(self):
        from ddma import GrowthMap, RectangularPartition

        q = self.quad
        part = RectangularPartition([np.linspace(q.T_min, q.T_max, self.cells + 1)])
        # y = f(c) + d with d in [-T_max - noise, noise]: during transients the measured
        # thrust may fall short of f(c) by any amount, never exceed it by more than noise
        self.models = [GrowthMap(part, g_lo=[0.0], g_hi=[self.g_hi],
                                 d_lo=-(q.T_max + self.noise), d_hi=self.noise)
                       for _ in range(4)]
        self.n_used = np.zeros(4, int)
        if self.prior is None:
            self.prior = 1.3 * q.hover_thrust

    def add(self, cmd_filtered, thrust):
        """Add samples: lagged commands (K, 4) (see :func:`lag_filter`) and measured
        thrusts (K, 4)."""
        cmd, thrust = np.atleast_2d(cmd_filtered), np.atleast_2d(thrust)
        for i in range(4):
            self.models[i].update(cmd[:, i:i + 1], thrust[:, i])
            self.n_used[i] += len(cmd)

    def bounds(self, i):
        lo, hi = self.models[i].f_bounds()
        e = np.linspace(self.quad.T_min, self.quad.T_max, self.cells + 1)
        return 0.5 * (e[1:] + e[:-1]), np.asarray(lo).ravel(), np.asarray(hi).ravel()

    @property
    def certified_max(self) -> np.ndarray:
        """Per-rotor thrust (N) the data guarantee (never below the prior)."""
        out = np.empty(4)
        for i in range(4):
            _, lo, _ = self.bounds(i)
            lo = lo[np.isfinite(lo)]
            lo = lo[lo > -1e6]
            out[i] = max(self.prior, float(lo.max()) if lo.size else -np.inf)
        return np.minimum(out, self.quad.T_max)


def lag_filter(cmd, dt: float, tau: float, c0):
    """Commands passed through the motor's first-order lag (exact for zero-order hold)."""
    a = np.exp(-dt / tau)
    out = np.empty_like(np.asarray(cmd, float))
    c = np.asarray(c0, float).copy()
    for k, u in enumerate(cmd):
        out[k] = c  # thrust measured at step k responds to commands before k
        c = u + (c - u) * a
    return out


def thrust_check(quad: Quad, faults, controller_factory, height: float = 1.5,
                 duration: float = 1.0, dt: float = 0.01, noise: float = 0.15, seed: int = 0):
    """Fly a vertical "hop" (up ``height`` metres and back in ``duration`` seconds, high
    vertical acceleration) and record (lagged command, measured thrust) per rotor.

    Returns (lagged commands (K, 4), measured thrusts (K, 4)). ``controller_factory(traj)``
    builds the controller that flies it (any controller with ``step(t, x)``)."""
    from .model import Plant

    rng = np.random.default_rng(seed)

    class Hop:  # z(t) = z0 + h sin^2(pi t / T): smooth up-and-down
        def __init__(self):
            self.z0, self.h, self.T = 2.0, height, duration

        def derivatives(self, t):
            t = np.atleast_1d(np.asarray(t, float))
            w = 2 * np.pi / self.T
            out = np.zeros((5, t.size, 3))
            inside = (t >= 0) & (t <= self.T)
            # z = z0 + h/2 (1 - cos(w t))
            out[0, :, 2] = self.z0 + np.where(inside, self.h / 2 * (1 - np.cos(w * t)), 0.0)
            for k in range(1, 5):
                out[k, :, 2] = np.where(inside, self.h / 2 * w ** k
                                        * -np.cos(w * t + k * np.pi / 2), 0.0)
            return out

        def reference(self, t, quad):
            from .trajectory import Figure8

            return Figure8.reference(self, t, quad)

        def _rates(self, t, quad):
            from .trajectory import Figure8

            return Figure8._rates(self, t, quad)

    hop = Hop()
    ctrl = controller_factory(hop)
    X0, _ = hop.reference(0.0, quad)
    plant = Plant(quad, X0[0], faults)
    C, Y = [], []
    for k in range(int(duration / dt) + 50):
        t = k * dt
        Y.append(plant.T + rng.uniform(-noise, noise, 4))  # measured at the start of the step
        u = ctrl.step(t, plant.x.copy())
        C.append(np.clip(u, quad.T_min, quad.T_max))
        plant.step(u, dt)
    C, Y = np.array(C), np.array(Y)
    Cf = lag_filter(C, dt, quad.tau_motor, np.full(4, quad.hover_thrust))
    return Cf, Y
