"""Timing benchmark, in closed loop on two laps of the figure-8 at 10 m/s peak.

    python examples/benchmark.py            # writes results/timing.json and a figure

Per controller: solver time per control step (median, p99, worst), share of steps over
the 10 ms (100 Hz) and 5 ms (200 Hz) budgets, and the tracking error. Variants: horizon
discretisation (N = 15, 25, 40 for the same 1 s horizon), 200 Hz control, SQP iterated to
convergence, the linear MPC (OSQP) and the geometric controller.
Timings depend on the machine (and vary by about +-30 % between runs on a shared one).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from qnmpc.baselines import GeometricController, LinearMPC  # noqa: E402
from qnmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from qnmpc.params import Quad  # noqa: E402
from qnmpc.sim import fly  # noqa: E402
from qnmpc.trajectory import Figure8  # noqa: E402

OUT = ROOT / "results"
FIG = ROOT / "docs" / "figures"


def stats(ms, extra=None):
    ms = np.asarray(ms, float)
    d = dict(
        steps=int(ms.size),
        median_ms=round(float(np.median(ms)), 3),
        p99_ms=round(float(np.percentile(ms, 99)), 3),
        max_ms=round(float(ms.max()), 3),
        over_10ms_pct=round(100 * float(np.mean(ms > 10)), 2),
        over_5ms_pct=round(100 * float(np.mean(ms > 5)), 2),
    )
    d.update(extra or {})
    return d


def main(fig: bool = True, quick: bool = False):
    OUT.mkdir(exist_ok=True)
    quad, tr = Quad(), Figure8(v_peak=10.0)
    dur = 1.0 if quick else 2 * tr.period
    T = {}
    variants = {
        "rti_N15": NMPCConfig(n_fine=5, dt_tail=0.9 / 10, n_tail=10),
        "rti_N25": NMPCConfig(),
        "rti_N40": NMPCConfig(n_fine=10, dt_tail=0.9 / 30, n_tail=30),
        "rti_N30_200Hz": NMPCConfig(dt0=0.005, n_fine=20, dt_tail=0.09, n_tail=10),
        "sqp_N25": NMPCConfig(mode="sqp", max_iter=50),
    }
    for key, cfg in variants.items():
        c = NMPC(quad, tr, cfg)
        f = fly(c, quad, tr, dur, key, dt=cfg.dt0)
        extra = dict(
            N=cfg.N,
            rate_hz=round(1 / cfg.dt0),
            rms_error_cm=round(100 * f.summary()["rms_error"], 2),
            crashed=f.crashed,
        )
        if cfg.mode == "sqp":
            extra.update(
                iterations_median=int(np.median(c.iters)), iterations_max=int(max(c.iters))
            )
        if key == "rti_N25":
            extra["python_step_median_ms"] = round(1e3 * float(np.median(c.wall_times)), 3)
        T[key] = stats(f.solve_ms, extra)
        print(key, T[key])
    lm = LinearMPC(quad, Figure8(v_peak=8.0))  # at 10 m/s it crashes: time it at 8 m/s
    f = fly(lm, quad, lm.traj, dur, "lmpc")
    T["linear_mpc_8ms"] = stats(
        f.solve_ms,
        dict(
            rms_error_cm=round(100 * f.summary()["rms_error"], 2),
            crashed=f.crashed,
            python_step_median_ms=round(1e3 * float(np.median(lm.wall_times)), 3),
            note="figure-8 at 8 m/s (crashes at 10 m/s); QP solve time only",
        ),
    )
    g = GeometricController(quad, tr)
    f = fly(g, quad, tr, dur, "geo")
    T["geometric"] = stats(
        f.solve_ms,
        dict(
            rms_error_cm=round(100 * f.summary()["rms_error"], 2),
            crashed=f.crashed,
            note="Python implementation",
        ),
    )
    print("lmpc", T["linear_mpc_8ms"])
    (OUT / "timing.json").write_text(json.dumps(T, indent=2))
    if fig:
        figure(T)
    return T


def figure(T):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from qnmpc.plotting import INK2, MUTED, SERIES, style

    style()
    FIG.mkdir(parents=True, exist_ok=True)
    rows = [
        ("acados RTI, N = 15", "rti_N15", 0),
        ("acados RTI, N = 25 (default)", "rti_N25", 0),
        ("acados RTI, N = 40", "rti_N40", 0),
        ("acados RTI, N = 30, 200 Hz", "rti_N30_200Hz", 0),
        ("Linear MPC, OSQP (QP only, 8 m/s)", "linear_mpc_8ms", 1),
        ("acados SQP iterated to convergence", "sqp_N25", 2),
    ]
    ros = OUT / "ros2_flight.json"
    if ros.exists():
        R = json.loads(ros.read_text())
        T = dict(
            T,
            ros2=dict(
                median_ms=R["latency_ms_median"],
                p99_ms=R["latency_ms_p99"],
                max_ms=R["latency_ms_max"],
            ),
        )
        rows.insert(2, ("ROS 2 C++ node, state -> command", "ros2", 0))
    fig, ax = plt.subplots(figsize=(11, 3.9))
    for i, (label, key, c) in enumerate(rows[::-1]):
        d = T[key]
        ax.plot(
            [d["median_ms"], d["p99_ms"]], [i, i], color=SERIES[c], lw=3, solid_capstyle="round"
        )
        ax.plot([d["p99_ms"], d["max_ms"]], [i, i], color=SERIES[c], lw=1, alpha=0.6)
        ax.plot(d["median_ms"], i, "o", ms=8, color=SERIES[c], mec="white", mew=1.5)
        ax.text(
            d["max_ms"] * 1.15, i, f"{d['median_ms']:.2g} ms", va="center", color=INK2, fontsize=9
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xscale("log")
    for b, txt in ((5, "200 Hz\nbudget"), (10, "100 Hz\nbudget")):
        ax.axvline(b, color=MUTED, lw=1, ls="--")
        ax.text(b * 1.05, -0.9, txt, color=INK2, fontsize=8.5, va="bottom")
    ax.set_ylim(-1.0, len(rows) - 0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.1, 200)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        "time per control step (ms), log scale: dot = median, thick = up to p99, thin = worst case"
    )
    ax.set_title("Solve time per control step, figure-8 at 10 m/s")
    fig.tight_layout()
    fig.savefig(FIG / "timing.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--fig-only" in sys.argv:
        figure(json.loads((OUT / "timing.json").read_text()))
    else:
        main(fig="--no-fig" not in sys.argv, quick="--quick" in sys.argv)
