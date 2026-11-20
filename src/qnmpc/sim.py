"""Closed-loop flights and tracking metrics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Plant, tilt_angle
from .params import Faults, Quad


@dataclass
class Flight:
    name: str
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray  # commanded thrusts
    T: np.ndarray  # actual thrusts
    x_ref: np.ndarray
    crashed: bool
    solve_ms: np.ndarray
    status: list = field(default_factory=list)

    @property
    def err(self) -> np.ndarray:
        return np.linalg.norm(self.x[:, :3] - self.x_ref[:, :3], axis=1)

    def summary(self) -> dict:
        e = self.err
        st = self.solve_ms
        return dict(
            name=self.name, crashed=bool(self.crashed), duration=round(float(self.t[-1]), 2),
            rms_error=round(float(np.sqrt(np.mean(e ** 2))), 3),
            max_error=round(float(e.max()), 3),
            max_tilt_deg=round(float(np.rad2deg(max(tilt_angle(x[3:7]) for x in self.x))), 1),
            max_thrust=round(float(self.u.max()), 2),
            solve_ms_median=round(float(np.median(st)), 3) if st.size else None,
            solve_ms_p99=round(float(np.percentile(st, 99)), 3) if st.size else None,
            solve_ms_max=round(float(st.max()), 3) if st.size else None,
        )


def fly(controller, quad: Quad, traj, duration: float, name: str = "", dt: float = 0.01,
        faults: Faults | None = None, x0=None, crash_error: float = 2.0, on_step=None) -> Flight:
    """Fly ``duration`` seconds from the reference start (or ``x0``).

    The flight counts as a crash if the position error exceeds ``crash_error`` metres or
    the quadrotor goes below the ground (z < 0)."""
    X0, _ = traj.reference(0.0, quad)
    plant = Plant(quad, X0[0] if x0 is None else x0, faults)
    ts, xs, us, Ts, xr = [], [], [], [], []
    crashed = False
    n = int(round(duration / dt))
    for k in range(n):
        t = k * dt
        x = plant.x.copy()
        u = controller.step(t, x)
        Xr, _ = traj.reference(t, quad)
        ts.append(t); xs.append(x); us.append(u); Ts.append(plant.T.copy()); xr.append(Xr[0])
        if on_step is not None:
            on_step(k, t, x, u, plant)
        plant.step(u, dt)
        if np.linalg.norm(plant.x[:3] - Xr[0, :3]) > crash_error or plant.x[2] < 0 \
                or not np.all(np.isfinite(plant.x)):
            crashed = True
            break
    st = np.asarray(getattr(controller, "solve_times", []), float) * 1e3
    return Flight(name, np.array(ts), np.array(xs), np.array(us), np.array(Ts), np.array(xr),
                  crashed, st, list(getattr(controller, "status", [])))
