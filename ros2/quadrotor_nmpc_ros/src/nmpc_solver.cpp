#include "quadrotor_nmpc_ros/nmpc_solver.hpp"

#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_quadrotor_nmpc.h"
}

namespace quadrotor_nmpc
{

namespace
{
inline quadrotor_nmpc_solver_capsule * cap(void * p)
{
  return static_cast<quadrotor_nmpc_solver_capsule *>(p);
}
constexpr int N = QUADROTOR_NMPC_N;
}  // namespace

NmpcSolver::NmpcSolver(Figure8 traj)
: traj_(traj)
{
  auto * c = quadrotor_nmpc_acados_create_capsule();
  const int st = quadrotor_nmpc_acados_create(c);
  if (st != 0) {
    quadrotor_nmpc_acados_free_capsule(c);
    throw std::runtime_error("acados create failed with status " + std::to_string(st));
  }
  capsule_ = c;
  T_max_.fill(params::T_max);
}

NmpcSolver::~NmpcSolver()
{
  if (capsule_) {
    quadrotor_nmpc_acados_free(cap(capsule_));
    quadrotor_nmpc_acados_free_capsule(cap(capsule_));
  }
}

void NmpcSolver::set_thrust_limits(const Thrusts & T_max)
{
  auto * c = cap(capsule_);
  double ub[NU];
  for (int i = 0; i < NU; ++i) {
    T_max_[i] = std::min(T_max[i], params::T_max);
    ub[i] = T_max_[i];
  }
  for (int k = 0; k < N; ++k) {
    ocp_nlp_constraints_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, c->nlp_out, k, "ubu", ub);
  }
}

void NmpcSolver::set_x0(const State & x0)
{
  auto * c = cap(capsule_);
  double x[NX];
  for (int i = 0; i < NX; ++i) {x[i] = x0[i];}
  ocp_nlp_constraints_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, c->nlp_out, 0, "lbx", x);
  ocp_nlp_constraints_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, c->nlp_out, 0, "ubx", x);
}

void NmpcSolver::set_reference(double t, const State & x0, State * X, Thrusts * U)
{
  auto * c = cap(capsule_);
  double tk = t;
  for (int k = 0; k <= N; ++k) {
    traj_.reference(tk, X[k], U[k]);
    if (k > 0) {  // keep the quaternion sign continuous along the horizon
      double d = 0;
      for (int i = 3; i < 7; ++i) {d += X[k][i] * X[k - 1][i];}
      if (d < 0) {for (int i = 3; i < 7; ++i) {X[k][i] = -X[k][i];}}
    }
    if (k < N) {tk += params::time_steps[k];}
  }
  double d0 = 0;
  for (int i = 3; i < 7; ++i) {d0 += X[0][i] * x0[i];}
  const double sgn = d0 < 0 ? -1.0 : 1.0;
  double y[NX + NU];
  for (int k = 0; k <= N; ++k) {
    for (int i = 3; i < 7; ++i) {X[k][i] *= sgn;}
    for (int i = 0; i < NX; ++i) {y[i] = X[k][i];}
    if (k < N) {
      for (int i = 0; i < NU; ++i) {
        U[k][i] = std::min(U[k][i], T_max_[i]);
        y[NX + i] = U[k][i];
      }
    }
    ocp_nlp_cost_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, k, "yref", y);
  }
}

SolveResult NmpcSolver::step(double t, const State & x0)
{
  auto * c = cap(capsule_);
  std::vector<State> X(N + 1);
  std::vector<Thrusts> U(N + 1);
  set_reference(t, x0, X.data(), U.data());
  if (!initialized_) {
    for (int k = 0; k <= N; ++k) {
      double xk[NX];
      for (int i = 0; i < NX; ++i) {xk[i] = X[k][i];}
      ocp_nlp_out_set(c->nlp_config, c->nlp_dims, c->nlp_out, c->nlp_in, k, "x", xk);
      if (k < N) {
        double uk[NU];
        for (int i = 0; i < NU; ++i) {uk[i] = U[k][i];}
        ocp_nlp_out_set(c->nlp_config, c->nlp_dims, c->nlp_out, c->nlp_in, k, "u", uk);
      }
    }
    for (int i = 0; i < 5; ++i) {
      set_x0(x0);
      quadrotor_nmpc_acados_solve(c);
    }
    initialized_ = true;
  }
  set_x0(x0);
  SolveResult r;
  r.status = quadrotor_nmpc_acados_solve(c);
  double tt = 0;
  ocp_nlp_get(c->nlp_solver, "time_tot", &tt);
  r.solve_time_ms = 1e3 * tt;
  double u[NU];
  ocp_nlp_out_get(c->nlp_config, c->nlp_dims, c->nlp_out, 0, "u", u);
  for (int i = 0; i < NU; ++i) {r.u[i] = u[i];}
  return r;
}

}  // namespace quadrotor_nmpc
