"""Tests for the quadrotor NMPC package (Python side)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from qnmpc.baselines import GeometricController, LinearMPC
from qnmpc.model import Plant, casadi_model, quat_to_rot, rhs, rot_to_quat
from qnmpc.params import Faults, Quad
from qnmpc.sim import fly
from qnmpc.trajectory import Figure8

warnings.filterwarnings("ignore")

try:
    import acados_template  # noqa: F401

    HAVE_ACADOS = True
except Exception:  # pragma: no cover
    HAVE_ACADOS = False
try:
    import ddma  # noqa: F401

    HAVE_DDMA = True
except Exception:  # pragma: no cover
    HAVE_DDMA = False
needs_acados = pytest.mark.skipif(not HAVE_ACADOS, reason="acados not installed")
needs_ddma = pytest.mark.skipif(not HAVE_DDMA, reason="ddma not installed")


@pytest.fixture(scope="module")
def quad():
    return Quad()


def test_mixer_and_hover(quad):
    T = np.full(4, quad.hover_thrust)
    f = quad.mixer @ T
    assert f[0] == pytest.approx(quad.m * 9.81)
    np.testing.assert_allclose(f[1:], 0.0, atol=1e-12)
    x = np.r_[0, 0, 1, 1, 0, 0, 0, np.zeros(6)]
    np.testing.assert_allclose(rhs(x, T, quad), 0.0, atol=1e-12)


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        np.testing.assert_allclose(rot_to_quat(quat_to_rot(q)), q, atol=1e-12)


def test_casadi_model_matches_numpy(quad):
    import casadi as ca

    m = casadi_model(quad)
    f = ca.Function("f", [m["x"], m["u"]], [m["f"]])
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        x = np.r_[rng.normal(size=3), q, rng.normal(size=3) * 3, rng.normal(size=3)]
        u = rng.uniform(0, quad.T_max, 4)
        np.testing.assert_allclose(np.array(f(x, u)).ravel(), rhs(x, u, quad), atol=1e-10)


def test_flat_reference_is_dynamically_consistent(quad):
    """Integrating the drag-free model with the reference thrusts stays on the reference."""
    from dataclasses import replace

    q0 = replace(quad, drag=(0.0, 0.0, 0.0))
    tr = Figure8(v_peak=8.0)
    t = np.arange(0, 0.3, 1e-3)
    X, U = tr.reference(t, q0)
    x = X[0].copy()
    for k in range(len(t) - 1):
        h, u = 1e-3, U[k]
        k1 = rhs(x, u, q0); k2 = rhs(x + h / 2 * k1, u, q0)
        k3 = rhs(x + h / 2 * k2, u, q0); k4 = rhs(x + h * k3, u, q0)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[3:7] /= np.linalg.norm(x[3:7])
    assert np.linalg.norm(x[:3] - X[-1, :3]) < 0.01


def test_plant_fault_caps_thrust(quad):
    p = Plant(quad, np.r_[0, 0, 5, 1, 0, 0, 0, np.zeros(6)],
              Faults(max_thrust=(np.inf, np.inf, 3.5, np.inf)))
    for _ in range(50):
        p.step(np.full(4, 6.0), 0.01)
    np.testing.assert_allclose(p.T, [6.0, 6.0, 3.5, 6.0], atol=1e-6)


def test_geometric_tracks_moderate_speed(quad):
    tr = Figure8(v_peak=6.0)
    f = fly(GeometricController(quad, tr), quad, tr, tr.period)
    assert not f.crashed and f.summary()["rms_error"] < 0.03


def test_linear_mpc_tracks_slow_speed(quad):
    tr = Figure8(v_peak=4.0)
    f = fly(LinearMPC(quad, tr), quad, tr, 4.0)
    assert not f.crashed and f.summary()["rms_error"] < 0.03


@needs_acados
def test_nmpc_tracks_aggressive_figure8(quad):
    from qnmpc.nmpc import NMPC, NMPCConfig

    tr = Figure8(v_peak=12.0)
    c = NMPC(quad, tr, NMPCConfig(name="test_q"))
    f = fly(c, quad, tr, tr.period)
    s = f.summary()
    assert not f.crashed and s["rms_error"] < 0.05
    assert s["max_tilt_deg"] > 70  # it really is aggressive
    assert all(st == 0 for st in c.status)


@needs_acados
def test_nmpc_respects_thrust_limits(quad):
    from qnmpc.nmpc import NMPC, NMPCConfig

    tr = Figure8(v_peak=10.0)
    c = NMPC(quad, tr, NMPCConfig(name="test_q"))
    c.set_thrust_limits([7.0, 7.0, 3.0, 7.0])
    f = fly(c, quad, tr, 2.0)
    assert f.u[:, 2].max() <= 3.0 + 1e-6


@needs_ddma
def test_thrust_certificate_is_sound_and_tight(quad):
    from qnmpc.thrust import ThrustCertificate, thrust_check

    for cap, eff in [(3.5, 1.0), (4.5, 1.0), (np.inf, 0.7)]:
        faults = Faults(max_thrust=(np.inf, np.inf, cap, np.inf), efficiency=(1, 1, eff, 1))
        Cf, Y = thrust_check(quad, faults, lambda tr: GeometricController(quad, tr), seed=2)
        cert = ThrustCertificate(quad)
        cert.add(Cf, Y)
        true = Plant(quad, np.r_[0, 0, 0, 1, np.zeros(9)], faults).actual_target(
            np.full(4, quad.T_max))
        T = cert.certified_max
        assert np.all(T <= true + 1e-9)  # never above what the rotor can give
        assert T[2] >= 0.95 * true[2]  # tight on the damaged rotor
        for i in range(4):  # the lower bound holds over the whole command range
            c, lo, _ = cert.bounds(i)
            ok = lo > -1e6
            f_true = np.minimum(faults.efficiency[i] * c, faults.max_thrust[i])
            assert np.all(lo[ok] <= f_true[ok] + 1e-9)


@needs_acados
@needs_ddma
def test_certified_thrust_prevents_the_crash(quad):
    from qnmpc.nmpc import NMPC, NMPCConfig
    from qnmpc.thrust import ThrustCertificate, thrust_check

    faults = Faults(max_thrust=(np.inf, np.inf, 3.5, np.inf))
    Cf, Y = thrust_check(quad, faults, lambda tr: GeometricController(quad, tr))
    cert = ThrustCertificate(quad)
    cert.add(Cf, Y)
    tr = Figure8(v_peak=10.0)
    c = NMPC(quad, tr, NMPCConfig(name="test_q"))
    c.set_thrust_limits(cert.certified_max)
    assert not fly(c, quad, tr, 2 * tr.period, faults=faults).crashed
    c.initialized = False
    c.set_thrust_limits(np.full(4, quad.T_max))
    assert fly(c, quad, tr, 2 * tr.period, faults=faults).crashed  # within the second lap
