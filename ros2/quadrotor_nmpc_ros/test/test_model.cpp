// The C++ plant and figure-8 reference must reproduce the Python ones.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "quadrotor_nmpc_ros/quad_model.hpp"

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

TEST(Plant, MatchesPythonWithMotorLagAndFault)
{
  auto rows = read_csv(std::string(FIXTURES) + "/plant_traj.csv");
  ASSERT_EQ(rows.size(), 200u);
  State x0;
  for (int i = 0; i < NX; ++i) {x0[i] = rows[0][i];}
  Faults f;
  f.max_thrust[2] = 3.5;
  Plant p(x0, f);
  double worst = 0;
  for (const auto & r : rows) {
    Thrusts cmd{r[13], r[14], r[15], r[16]};
    p.step(cmd, 0.01);
    for (int i = 0; i < NX; ++i) {
      worst = std::max(worst, std::abs(p.x()[i] - r[17 + i]) / (1.0 + std::abs(r[17 + i])));
    }
  }
  EXPECT_LT(worst, 1e-9);
}

TEST(Reference, MatchesPythonFlatness)
{
  auto rows = read_csv(std::string(FIXTURES) + "/reference.csv");
  ASSERT_EQ(rows.size(), 61u);
  Figure8 traj;
  traj.v_peak = 10.0;
  double worst_x = 0, worst_u = 0;
  for (const auto & r : rows) {
    State x;
    Thrusts u;
    traj.reference(r[0], x, u);
    double d = 0;
    for (int i = 3; i < 7; ++i) {d += x[i] * r[1 + i];}
    const double s = d < 0 ? -1.0 : 1.0;  // quaternions equal up to sign
    for (int i = 0; i < NX; ++i) {
      const double xi = (i >= 3 && i < 7) ? s * x[i] : x[i];
      worst_x = std::max(worst_x, std::abs(xi - r[1 + i]));
    }
    for (int i = 0; i < NU; ++i) {worst_u = std::max(worst_u, std::abs(u[i] - r[14 + i]));}
  }
  EXPECT_LT(worst_x, 1e-9);
  EXPECT_LT(worst_u, 1e-6);  // finite-difference torques
}
