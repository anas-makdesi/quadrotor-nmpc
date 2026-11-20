"""Quadrotor parameters (a ~0.75 kg racing-class quadrotor, X configuration)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass(frozen=True)
class Quad:
    m: float = 0.75  # mass (kg)
    J: tuple = (2.5e-3, 2.1e-3, 4.3e-3)  # principal inertia (kg m^2)
    arm: float = 0.14  # rotor distance from the centre (m)
    c_tau: float = 0.022  # yaw torque per newton of thrust (m)
    T_max: float = 7.0  # rated thrust per rotor (N); thrust-to-weight 3.8
    T_min: float = 0.0
    drag: tuple = (0.26, 0.28, 0.42)  # linear rotor drag in the body frame (kg/s)
    tau_motor: float = 0.03  # motor time constant (s), plant only
    w_max: tuple = (12.0, 12.0, 4.0)  # body-rate limits (rad/s)

    @property
    def Jv(self) -> np.ndarray:
        return np.array(self.J)

    @property
    def hover_thrust(self) -> float:
        return self.m * G / 4

    @property
    def rotor_xy(self) -> np.ndarray:
        """Rotor positions in the body frame (x forward, y left), X configuration."""
        a = self.arm / np.sqrt(2)
        return np.array([[a, -a], [-a, a], [a, a], [-a, -a]])  # front-right, rear-left, ...

    @property
    def spin(self) -> np.ndarray:
        """Rotor spin directions (+1 counter-clockwise from above), for the yaw torque."""
        return np.array([1.0, 1.0, -1.0, -1.0])

    @property
    def mixer(self) -> np.ndarray:
        """Maps the four rotor thrusts to [collective thrust, tau_x, tau_y, tau_z]."""
        xy = self.rotor_xy
        return np.vstack([np.ones(4), xy[:, 1], -xy[:, 0], self.c_tau * self.spin])


@dataclass(frozen=True)
class Faults:
    """Actuator faults of the simulated quadrotor (unknown to the nominal controller)."""

    efficiency: tuple = (1.0, 1.0, 1.0, 1.0)  # thrust per command, per rotor
    max_thrust: tuple = field(default=(np.inf,) * 4)  # hard cap per rotor (N)
