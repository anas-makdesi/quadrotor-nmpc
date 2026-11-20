"""Figure-8 experiments: NMPC (acados) vs the geometric controller and a linear MPC, and
flying with a damaged rotor using a thrust limit certified from data (ddma).

    python examples/figure8_comparison.py      # writes results/figure8.json and figures

Experiment 1 - speed sweep: two laps of the figure-8 at peak speeds 4 to 12 m/s.
Experiment 2 - damaged rotor: rotor 3 gives at most 3.5 N (half its rating); figure-8 at
10 m/s with the NMPC planning with the rated thrust, with the thrust certified by ddma
from a 1 s "thrust check" hop, and with the true limits (oracle); and the geometric
controller.
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
from qnmpc.params import Faults, Quad  # noqa: E402
from qnmpc.sim import fly  # noqa: E402
from qnmpc.thrust import ThrustCertificate, thrust_check  # noqa: E402
from qnmpc.trajectory import Figure8  # noqa: E402

OUT = ROOT / "results"
FIG = ROOT / "docs" / "figures"
SPEEDS = (4, 6, 8, 10, 11, 12)
CAP = 3.5


def main(fig: bool = True):
    OUT.mkdir(exist_ok=True)
    quad = Quad()
    nmpc = NMPC(quad, Figure8(8.0), NMPCConfig())
    sweep = {}
    flights = {}
    for v in SPEEDS:
        tr = Figure8(v_peak=v)
        dur = 2 * tr.period
        nmpc.traj, nmpc.initialized, nmpc.solve_times = tr, False, []
        nmpc.set_thrust_limits(np.full(4, quad.T_max))
        runs = {
            "nmpc": fly(nmpc, quad, tr, dur, "NMPC (acados)"),
            "geometric": fly(GeometricController(quad, tr), quad, tr, dur, "Geometric + flatness"),
            "linear_mpc": fly(LinearMPC(quad, tr), quad, tr, dur, "Linear MPC (OSQP)"),
        }
        sweep[v] = {k: r.summary() for k, r in runs.items()}
        flights[v] = runs
        print(v, {k: (s["crashed"], s["rms_error"]) for k, s in sweep[v].items()})

    # ------------------------------------------------------------ damaged rotor
    faults = Faults(max_thrust=(np.inf, np.inf, CAP, np.inf))
    Cf, Y = thrust_check(quad, faults, lambda tr: GeometricController(quad, tr))
    cert = ThrustCertificate(quad)
    cert.add(Cf, Y)
    T_cert = cert.certified_max
    tr = Figure8(v_peak=10.0)
    dur = 2 * tr.period
    fault = {}
    for key, label, lim in [
        ("nominal", "NMPC, rated thrust 7 N", np.full(4, quad.T_max)),
        ("certified", "NMPC, ddma-certified thrust", T_cert),
        ("oracle", "NMPC, true limits (oracle)", np.array([7.0, 7.0, CAP, 7.0])),
    ]:
        nmpc.traj, nmpc.initialized, nmpc.solve_times = tr, False, []
        nmpc.set_thrust_limits(lim)
        fault[key] = fly(nmpc, quad, tr, dur, label, faults=faults)
    fault["geometric"] = fly(GeometricController(quad, tr), quad, tr, dur,
                             "Geometric + flatness", faults=faults)
    summary = dict(
        sweep={str(v): s for v, s in sweep.items()},
        damaged_rotor={k: f.summary() for k, f in fault.items()},
        meta=dict(rated_thrust=quad.T_max, damaged_rotor=3, damaged_max_thrust=CAP,
                  certified_max_thrust=[round(float(t), 2) for t in T_cert],
                  thrust_check_samples=int(len(Cf)), thrust_check_duration_s=1.0,
                  measurement_noise_N=cert.noise),
    )
    (OUT / "figure8.json").write_text(json.dumps(summary, indent=2))
    for k, f in fault.items():
        print(k, f.summary())
    print("certified", T_cert.round(2))
    if fig:
        figures(quad, flights, fault, cert, Cf, Y)
    return summary


def figures(quad, flights, fault, cert, Cf, Y):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from qnmpc.plotting import CRITICAL, INK2, SERIES, speed_cmap, style

    style()
    FIG.mkdir(parents=True, exist_ok=True)
    names = [("nmpc", "NMPC (acados)"), ("geometric", "Geometric + flatness"),
             ("linear_mpc", "Linear MPC (OSQP)")]

    # --- hero: NMPC at 12 m/s (top view, coloured by speed) + error vs speed
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    f = flights[12]["nmpc"]
    P = f.x[:, :2]
    seg = np.stack([P[:-1], P[1:]], axis=1)
    lc = LineCollection(seg, cmap=speed_cmap(), linewidths=2.4)
    lc.set_array(np.linalg.norm(f.x[:-1, 7:10], axis=1))
    ax.add_collection(lc)
    ax.plot(f.x_ref[:, 0], f.x_ref[:, 1], color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
    g = flights[12]["geometric"]
    ax.plot(g.x[:, 0], g.x[:, 1], color=SERIES[1], lw=1.4)
    ax.plot(*g.x[-1, :2], marker="X", ms=11, color=CRITICAL, mec="white", mew=1.5, zorder=5)
    ax.annotate("geometric controller\nloses control", g.x[-1, :2], xytext=(10, -28),
                textcoords="offset points", color=INK2, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.03)
    cb.set_label("speed (m/s)")
    cb.outline.set_visible(False)
    s = f.summary()
    ax.set_title(f"NMPC at 12 m/s peak: {100 * s['rms_error']:.1f} cm RMS error, "
                 f"{s['max_tilt_deg']:.0f}° tilt")
    ax = axs[1]
    for i, (key, label) in enumerate(names):
        vs, es, crash = [], [], None
        for v in SPEEDS:
            r = flights[v][key]
            if r.crashed:
                crash = v
                break
            vs.append(v)
            es.append(100 * r.summary()["rms_error"])
        ax.plot(vs, es, "-o", color=SERIES[i], ms=6, mec="white", mew=1.2, label=label)
        if crash is not None:
            ax.plot(crash, 60, marker="X", ms=11, color=CRITICAL, mec="white", mew=1.5)
            ax.plot([vs[-1], crash], [es[-1], 60], color=SERIES[i], lw=1, ls=":")
    ax.text(SPEEDS[-1], 75, "✕ = crash", ha="right", color=INK2, fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 100)
    ax.set_xlabel("peak speed of the figure-8 (m/s)")
    ax.set_ylabel("RMS position error (cm)")
    ax.set_title("Tracking error vs speed (2 laps each)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "figure8_speed.png", dpi=150)
    plt.close(fig)

    # --- damaged rotor
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    ref = fault["certified"].x_ref
    ax.plot(ref[:, 0], ref[:, 1], color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0,
            label="reference")
    for i, key in enumerate(["certified", "nominal", "geometric"]):
        r = fault[key]
        ax.plot(r.x[:, 0], r.x[:, 1], color=SERIES[i], lw=2.2 if i == 0 else 1.3,
                label=r.name, zorder=3 - i)
        if r.crashed:
            ax.plot(*r.x[-1, :2], marker="X", ms=11, color=CRITICAL, mec="white", mew=1.5,
                    zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-3.6, 4.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.set_title(f"Rotor 3 limited to {CAP} N, figure-8 at 10 m/s")
    ax = axs[1]
    c, lo, _ = cert.bounds(2)
    ok = lo > -1e6
    ax.scatter(Cf[:, 2], Y[:, 2], s=10, color=SERIES[0], alpha=0.6, lw=0,
               label="rotor 3 samples (lagged command, measured thrust)")
    ax.plot(c[ok], lo[ok], color=SERIES[0], lw=2, label="ddma guaranteed lower bound")
    cc = np.linspace(0, quad.T_max, 100)
    ax.plot(cc, np.minimum(cc, CAP), color=INK2, lw=1.3, ls="--", label="true rotor map (unknown)")
    ax.axhline(cert.certified_max[2], color=SERIES[1], lw=1.6)
    ax.text(0.2, cert.certified_max[2] + 0.15,
            f"certified max thrust {cert.certified_max[2]:.2f} N", color=INK2, fontsize=10)
    ax.set_xlim(0, quad.T_max)
    ax.set_ylim(0, quad.T_max)
    ax.set_xlabel("command (N)")
    ax.set_ylabel("thrust (N)")
    ax.set_title(f"Thrust certified from a 1 s hop ({len(Cf)} samples)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "damaged_rotor.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(fig="--no-fig" not in sys.argv)
