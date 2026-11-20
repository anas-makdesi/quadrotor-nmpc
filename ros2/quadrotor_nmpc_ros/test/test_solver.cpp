// The C++ wrapper must return the commands the Python NMPC returned for the same states.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "quadrotor_nmpc_ros/nmpc_solver.hpp"

using namespace quadrotor_nmpc;

static std::vector<std::vector<double>> read_csv(const std::string & path)
{
  std::ifstream f(path);
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {continue;}
    std::vector<double> r;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {r.push_back(std::stod(tok));}
    rows.push_back(r);
  }
  return rows;
}

static Figure8 traj10()
{
  Figure8 f;
  f.v_peak = 10.0;
  return f;
}

TEST(Solver, ReproducesPythonClosedLoop)
{
  NmpcSolver s(traj10());
  auto rows = read_csv(std::string(FIXTURES) + "/solver_seq.csv");
  ASSERT_EQ(rows.size(), 150u);
  double worst = 0;
  for (const auto & r : rows) {
    State x;
    for (int i = 0; i < NX; ++i) {x[i] = r[1 + i];}
    auto res = s.step(r[0], x);
    EXPECT_EQ(res.status, 0);
    for (int i = 0; i < NU; ++i) {
      worst = std::max(worst, std::abs(res.u[i] - r[14 + i]) / params::T_max);
    }
  }
  EXPECT_LT(worst, 1e-6);
}

TEST(Solver, FitsTheRealTimeBudget)
{
  NmpcSolver s(traj10());
  auto rows = read_csv(std::string(FIXTURES) + "/solver_seq.csv");
  std::vector<double> ms;
  for (const auto & r : rows) {
    State x;
    for (int i = 0; i < NX; ++i) {x[i] = r[1 + i];}
    auto t0 = std::chrono::steady_clock::now();
    s.step(r[0], x);
    ms.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ms.begin() + 1, ms.end());
  EXPECT_LT(ms[1 + (ms.size() - 1) / 2], 10.0);  // generous: 100 Hz budget on any CI machine
}

TEST(Solver, ThrustLimitsAreRespected)
{
  NmpcSolver s(traj10());
  s.set_thrust_limits({7.0, 7.0, 3.0, 7.0});
  State x;
  Thrusts u;
  traj10().reference(0.0, x, u);
  for (int k = 0; k < 50; ++k) {
    auto res = s.step(0.01 * k, x);
    EXPECT_LE(res.u[2], 3.0 + 1e-6);
  }
}
