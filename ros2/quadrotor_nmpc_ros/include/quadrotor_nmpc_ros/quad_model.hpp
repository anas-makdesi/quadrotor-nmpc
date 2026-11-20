// Quadrotor plant and figure-8 flat reference: C++ mirrors of qnmpc.model.Plant and
// qnmpc.trajectory.Figure8 (checked against Python in test/test_model.cpp).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "quad_params.hpp"

namespace quadrotor_nmpc
{

constexpr int NX = 13;  // p(3), q(4: w x y z), v(3), w(3)
constexpr int NU = 4;   // rotor thrusts
using State = std::array<double, NX>;
using Thrusts = std::array<double, NU>;
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

inline Vec3 cross(const Vec3 & a, const Vec3 & b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double dot(const Vec3 & a, const Vec3 & b) {return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];}
inline double norm(const Vec3 & a) {return std::sqrt(dot(a, a));}

inline Mat3 quat_to_rot(double w, double x, double y, double z)
{
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
    {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
    {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

// rotation matrix -> unit quaternion with w >= 0 (same branches as the Python code)
inline std::array<double, 4> rot_to_quat(const Mat3 & R)
{
  std::array<double, 4> q{};
  const double tr = R[0][0] + R[1][1] + R[2][2];
  if (tr > 0) {
    const double s = 2 * std::sqrt(tr + 1);
    q = {0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s};
  } else {
    int i = 0;
    if (R[1][1] > R[i][i]) {i = 1;}
    if (R[2][2] > R[i][i]) {i = 2;}
    const int j = (i + 1) % 3, k = (i + 2) % 3;
    const double s = 2 * std::sqrt(1 + R[i][i] - R[j][j] - R[k][k]);
    q[0] = (R[k][j] - R[j][k]) / s;
    q[1 + i] = 0.25 * s;
    q[1 + j] = (R[j][i] + R[i][j]) / s;
    q[1 + k] = (R[k][i] + R[i][k]) / s;
  }
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (auto & c : q) {c /= n;}
  if (q[0] < 0) {for (auto & c : q) {c = -c;}}
  return q;
}

inline std::array<double, 4> mix(const Thrusts & T)
{
  std::array<double, 4> f{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {f[r] += params::mixer[4 * r + c] * T[c];}
  }
  return f;
}

// rigid-body dynamics for actual rotor thrusts
inline State rhs(const State & x, const Thrusts & T)
{
  const double qw = x[3], qx = x[4], qy = x[5], qz = x[6];
  const Vec3 v{x[7], x[8], x[9]}, w{x[10], x[11], x[12]};
  const Mat3 R = quat_to_rot(qw, qx, qy, qz);
  const auto f = mix(T);
  Vec3 vb{}, acc{};
  for (int i = 0; i < 3; ++i) {vb[i] = R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2];}
  for (int i = 0; i < 3; ++i) {vb[i] *= params::drag[i];}
  for (int i = 0; i < 3; ++i) {
    acc[i] = R[i][2] * f[0] / params::m - (R[i][0] * vb[0] + R[i][1] * vb[1] + R[i][2] * vb[2]) /
      params::m;
  }
  acc[2] -= params::g;
  const Vec3 Jw{params::J[0] * w[0], params::J[1] * w[1], params::J[2] * w[2]};
  const Vec3 c = cross(w, Jw);
  State d{};
  d[0] = v[0]; d[1] = v[1]; d[2] = v[2];
  // q' = 1/2 q * [0, w]
  d[3] = 0.5 * (-qx * w[0] - qy * w[1] - qz * w[2]);
  d[4] = 0.5 * (qw * w[0] + qy * w[2] - qz * w[1]);
  d[5] = 0.5 * (qw * w[1] - qx * w[2] + qz * w[0]);
  d[6] = 0.5 * (qw * w[2] + qx * w[1] - qy * w[0]);
  d[7] = acc[0]; d[8] = acc[1]; d[9] = acc[2];
  for (int i = 0; i < 3; ++i) {d[10 + i] = (f[1 + i] - c[i]) / params::J[i];}
  return d;
}

struct Faults
{
  Thrusts efficiency{1.0, 1.0, 1.0, 1.0};
  Thrusts max_thrust{
    std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
};

// Simulated quadrotor: rigid body (RK4 at dt_int) + first-order motors + faults.
class Plant
{
public:
  Plant(const State & x0, Faults faults = {}, double dt_int = 1e-3)
  : x_(x0), faults_(faults), dt_int_(dt_int)
  {
    Thrusts hover;
    hover.fill(params::m * params::g / 4);
    T_ = target(hover);  // motors at steady state for the hover command
  }

  const State & x() const {return x_;}
  const Thrusts & thrust() const {return T_;}

  Thrusts target(const Thrusts & cmd) const
  {
    Thrusts t{};
    for (int i = 0; i < 4; ++i) {
      const double c = std::clamp(cmd[i], params::T_min, params::T_max);
      t[i] = std::min(faults_.efficiency[i] * c, faults_.max_thrust[i]);
    }
    return t;
  }

  void step(const Thrusts & cmd, double dt)
  {
    const Thrusts tg = target(cmd);
    const int n = std::max(1, static_cast<int>(std::lround(dt / dt_int_)));
    const double h = dt / n, a = std::exp(-h / params::tau_motor);
    auto axpy = [](const State & s, double c, const State & k) {
        State o;
        for (int i = 0; i < NX; ++i) {o[i] = s[i] + c * k[i];}
        return o;
      };
    for (int it = 0; it < n; ++it) {
      const State k1 = rhs(x_, T_);
      const State k2 = rhs(axpy(x_, 0.5 * h, k1), T_);
      const State k3 = rhs(axpy(x_, 0.5 * h, k2), T_);
      const State k4 = rhs(axpy(x_, h, k3), T_);
      for (int i = 0; i < NX; ++i) {x_[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);}
      const double qn = std::sqrt(x_[3] * x_[3] + x_[4] * x_[4] + x_[5] * x_[5] + x_[6] * x_[6]);
      for (int i = 3; i < 7; ++i) {x_[i] /= qn;}
      for (int i = 0; i < 4; ++i) {T_[i] = tg[i] + (T_[i] - tg[i]) * a;}
    }
  }

private:
  State x_;
  Thrusts T_{};
  Faults faults_;
  double dt_int_;
};

// Figure-8 with full-state flat reference (see qnmpc/trajectory.py)
struct Figure8
{
  double v_peak = 10.0, A = 5.0, B = 2.5, C = 0.5, z0 = 2.0;

  double om() const {return v_peak / std::sqrt(A * A + 4 * B * B + C * C);}

  // k-th derivative of the position at t
  Vec3 deriv(int k, double t) const
  {
    const double w = om();
    const double s1 = std::pow(w, k) * std::sin(w * t + k * M_PI / 2);
    const double s2 = std::pow(2 * w, k) * std::sin(2 * w * t + k * M_PI / 2);
    return {A * s1, B * s2, C * s1 + (k == 0 ? z0 : 0.0)};
  }

  // body frame of the flat attitude and its rates
  void frame(double t, Mat3 & R, Vec3 & w, double & thrust) const
  {
    Vec3 a = deriv(2, t);
    a[2] += params::g;
    const Vec3 j = deriv(3, t);
    const double na = norm(a);
    thrust = params::m * na;
    const Vec3 zb{a[0] / na, a[1] / na, a[2] / na};
    Vec3 yb = cross(zb, {1.0, 0.0, 0.0});
    const double ny = norm(yb);
    for (auto & c : yb) {c /= ny;}
    const Vec3 xb = cross(yb, zb);
    for (int i = 0; i < 3; ++i) {R[i][0] = xb[i]; R[i][1] = yb[i]; R[i][2] = zb[i];}
    const double zj = dot(zb, j);
    Vec3 h{};
    for (int i = 0; i < 3; ++i) {h[i] = params::m / thrust * (j[i] - zj * zb[i]);}
    w = {-dot(h, yb), dot(h, xb), 0.0};
  }

  // reference state and rotor thrusts at t
  void reference(double t, State & x, Thrusts & u) const
  {
    Mat3 R;
    Vec3 w;
    double thrust;
    frame(t, R, w, thrust);
    const auto q = rot_to_quat(R);
    const Vec3 p = deriv(0, t), v = deriv(1, t);
    x = {p[0], p[1], p[2], q[0], q[1], q[2], q[3], v[0], v[1], v[2], w[0], w[1], w[2]};
    const double eps = 1e-3;
    Mat3 Rp, Rm;
    Vec3 wp, wm;
    double tp, tm;
    frame(t + eps, Rp, wp, tp);
    frame(t - eps, Rm, wm, tm);
    double tau[3];
    const Vec3 Jw{params::J[0] * w[0], params::J[1] * w[1], params::J[2] * w[2]};
    const Vec3 c = cross(w, Jw);
    for (int i = 0; i < 3; ++i) {tau[i] = (wp[i] - wm[i]) / (2 * eps) * params::J[i] + c[i];}
    const double f[4] = {thrust, tau[0], tau[1], tau[2]};
    for (int r = 0; r < 4; ++r) {
      u[r] = 0.0;
      for (int k = 0; k < 4; ++k) {u[r] += params::mixer_inv[4 * r + k] * f[k];}
    }
  }
};

}  // namespace quadrotor_nmpc
