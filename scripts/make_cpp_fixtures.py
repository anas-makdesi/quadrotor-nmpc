"""Reference data for the C++ tests (ros2/quadrotor_nmpc_ros/test/fixtures).

    python scripts/make_cpp_fixtures.py

plant_traj.csv   plant trajectory (with motor lag and a capped rotor 3) under fixed commands
reference.csv    figure-8 flat reference (state and rotor thrusts) at sample times
solver_seq.csv   closed-loop states/times and the commands the Python NMPC (RTI) returned
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from qnmpc.model import Plant  # noqa: E402
from qnmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from qnmpc.params import Faults, Quad  # noqa: E402
from qnmpc.trajectory import Figure8  # noqa: E402

FIX = ROOT / "ros2" / "quadrotor_nmpc_ros" / "test" / "fixtures"


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    quad, tr = Quad(), Figure8(v_peak=10.0)
    # plant
    X0, _ = tr.reference(0.0, quad)
    plant = Plant(quad, X0[0], Faults(max_thrust=(np.inf, np.inf, 3.5, np.inf)))
    rows = []
    for k in range(200):
        t = 0.01 * k
        cmd = quad.hover_thrust + np.array(
            [
                1.5 * np.sin(3 * t),
                0.8 * np.cos(2 * t),
                2.5 * np.sin(1.5 * t + 0.3),
                -0.5 * np.sin(t),
            ]
        )
        x = plant.x.copy()
        plant.step(cmd, 0.01)
        rows.append(np.r_[x, cmd, plant.x])
    np.savetxt(
        FIX / "plant_traj.csv",
        np.array(rows),
        delimiter=",",
        fmt="%.17g",
        header="x(13),cmd(4),x_next(13); dt=0.01, rotor 3 capped at 3.5 N, start at hover thrust",
    )
    # reference
    t = np.linspace(0.0, 3.0, 61)
    X, U = tr.reference(t, quad)
    np.savetxt(
        FIX / "reference.csv",
        np.c_[t, X, U],
        delimiter=",",
        fmt="%.17g",
        header="t,x_ref(13),u_ref(4); v_peak=10 (quaternion sign continuous along the samples)",
    )
    # solver closed loop
    c = NMPC(quad, tr, NMPCConfig())
    plant = Plant(quad, X0[0])
    rows = []
    for k in range(150):
        tk = 0.01 * k
        x = plant.x.copy()
        u = c.step(tk, x)
        rows.append(np.r_[tk, x, u])
        plant.step(u, 0.01)
    np.savetxt(
        FIX / "solver_seq.csv",
        np.array(rows),
        delimiter=",",
        fmt="%.17g",
        header="t,x(13),u(4); RTI, v_peak=10, rated limits",
    )
    print("fixtures written to", FIX)


if __name__ == "__main__":
    main()
