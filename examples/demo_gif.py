"""Short demo for the README and the website: a figure-8 at 10 m/s with a damaged rotor.

Rotor 3 gives at most 3.5 N (half its rating). The same NMPC flies twice: once planning with
the rated thrust, once with the thrust limits certified by ddma from a 1 s hop. Runs the
damaged-rotor part of examples/figure8_comparison.py, then draws it from above.

    python examples/demo_gif.py      (writes docs/figures/demo.gif; needs acados and ddma)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from qnmpc.baselines import GeometricController  # noqa: E402
from qnmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from qnmpc.params import Faults, Quad  # noqa: E402
from qnmpc.sim import fly  # noqa: E402
from qnmpc.thrust import ThrustCertificate, thrust_check  # noqa: E402
from qnmpc.trajectory import Figure8  # noqa: E402

INK, TEAL, ORANGE, MUTED, RED = "#0B1220", "#2a9d8f", "#FB923C", "#6B7280", "#C2410C"
PAPER, GRID = "#F7F7F5", "#E4E4DF"
CAP = 3.5
FRAME_S, FRAME_MS = 0.1, 100     # real time


def run():
    quad = Quad()
    faults = Faults(max_thrust=(np.inf, np.inf, CAP, np.inf))
    Cf, Y = thrust_check(quad, faults, lambda tr: GeometricController(quad, tr))
    cert = ThrustCertificate(quad)
    cert.add(Cf, Y)
    tr = Figure8(v_peak=10.0)
    nmpc = NMPC(quad, tr, NMPCConfig())
    flights = {}
    for key, lim in (("certified", cert.certified_max), ("nominal", np.full(4, quad.T_max))):
        nmpc.traj, nmpc.initialized, nmpc.solve_times = tr, False, []
        nmpc.set_thrust_limits(lim)
        flights[key] = fly(nmpc, quad, tr, 2 * tr.period, key, faults=faults)
        print(key, flights[key].summary())
    return flights, cert.certified_max


def draw(flights, t_cert):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    runs = [(flights["certified"], TEAL,
             f"Thrust certified by ddma ({t_cert[2]:.2f} N on rotor 3)"),
            (flights["nominal"], ORANGE, "Plans with the rated thrust (7 N)")]
    t_end = max(f.t[-1] for f, _, _ in runs)
    times = np.r_[np.arange(0, t_end, FRAME_S), t_end + 1e-6]

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=100)
    fig.patch.set_facecolor(PAPER)
    frames = []
    for T in times:
        for ax, (f, color, label) in zip(axs, runs):
            ax.clear()
            ax.set_facecolor(PAPER)
            ref = f.x_ref
            ax.plot(ref[:, 0], ref[:, 1], color=MUTED, lw=0.8, ls=(0, (3, 3)))
            k = int(np.searchsorted(f.t, T, side="right")) - 1
            k = max(0, min(k, len(f.t) - 1))
            ax.plot(f.x[: k + 1, 0], f.x[: k + 1, 1], color=color, lw=2)
            if T >= f.t[-1] and f.crashed:
                ax.plot(*f.x[-1, :2], "X", color=RED, ms=14, mec="white", mew=1.5, zorder=6)
                status, scol = f"crashed after {f.t[-1]:.1f} s", RED
            else:
                ax.plot(*f.x[k, :2], "o", color=color, ms=9, mec="white", mew=1.2, zorder=6)
                status, scol = f"tracking error {f.err[k]:.2f} m", INK
                if T >= f.t[-1]:
                    status = (f"flew the whole run, RMS error "
                              f"{np.sqrt(np.mean(f.err ** 2)):.2f} m")
            ax.set(xlim=(-6.2, 6.2), ylim=(-3.2, 3.2), aspect="equal")
            ax.axis("off")
            ax.set_title(label, loc="left", fontsize=9.5, color=INK, fontweight="bold")
            ax.text(6.2, -3.2, status, ha="right", va="top", color=scol, fontsize=9.5)
        fig.suptitle(f"Figure-8 at 10 m/s, rotor 3 limited to {CAP} N   ·   t = {T:3.1f} s",
                     x=0.02, ha="left", fontsize=9.5, color=MUTED)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.08, wspace=0.08)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    durations = [FRAME_MS] * len(frames)
    durations[-1] = 3000
    images = [Image.fromarray(fr).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
              for fr in frames]
    out = ROOT / "docs" / "figures" / "demo.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"{len(frames)} frames; saved {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    draw(*run())
