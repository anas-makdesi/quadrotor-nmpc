// NMPC controller node: tracks the figure-8 with one real-time iteration of the generated
// acados solver per state message. No Python in the loop.
//
// Subscribes  /quad/state      quadrotor_nmpc_msgs/QuadState
// Publishes   /quad/thrust_cmd quadrotor_nmpc_msgs/RotorThrusts
//             /nmpc/stats      quadrotor_nmpc_msgs/ControllerStats
// Parameters  v_peak (figure-8 peak speed, m/s)
//             thrust_limits (4 values, N; e.g. the ddma-certified thrusts; run time)
// Time on the trajectory is counted from the first state message received.

#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include "quadrotor_nmpc_msgs/msg/controller_stats.hpp"
#include "quadrotor_nmpc_msgs/msg/quad_state.hpp"
#include "quadrotor_nmpc_msgs/msg/rotor_thrusts.hpp"
#include "quadrotor_nmpc_ros/nmpc_solver.hpp"
#include "rclcpp/rclcpp.hpp"

using quadrotor_nmpc_msgs::msg::ControllerStats;
using quadrotor_nmpc_msgs::msg::QuadState;
using quadrotor_nmpc_msgs::msg::RotorThrusts;
namespace qn = quadrotor_nmpc;

class NmpcNode : public rclcpp::Node
{
public:
  NmpcNode()
  : Node("nmpc"), solver_(make_traj(declare_parameter("v_peak", 10.0)))
  {
    traj_ = make_traj(get_parameter("v_peak").as_double());
    auto lim = declare_parameter("thrust_limits", std::vector<double>(4, qn::params::T_max));
    apply_limits(lim);
    param_cb_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & ps) {
        rcl_interfaces::msg::SetParametersResult res;
        res.successful = true;
        for (const auto & p : ps) {
          if (p.get_name() == "thrust_limits") {
            auto v = p.as_double_array();
            res.successful = v.size() == 4;
            if (res.successful) {apply_limits(v);}
          }
        }
        return res;
      });
    auto qos = rclcpp::QoS(1).best_effort();
    cmd_pub_ = create_publisher<RotorThrusts>("/quad/thrust_cmd", qos);
    stats_pub_ = create_publisher<ControllerStats>("/nmpc/stats", 10);
    sub_ = create_subscription<QuadState>(
      "/quad/state", qos, [this](QuadState::ConstSharedPtr m) {on_state(*m);});
    RCLCPP_INFO(get_logger(), "NMPC ready: figure-8 at %.1f m/s peak", traj_.v_peak);
  }

private:
  static qn::Figure8 make_traj(double v)
  {
    qn::Figure8 f;
    f.v_peak = v;
    return f;
  }

  void apply_limits(const std::vector<double> & v)
  {
    qn::Thrusts T{v[0], v[1], v[2], v[3]};
    solver_.set_thrust_limits(T);
    RCLCPP_INFO(get_logger(), "thrust limits %.2f %.2f %.2f %.2f N", T[0], T[1], T[2], T[3]);
  }

  void on_state(const QuadState & m)
  {
    const auto c0 = std::chrono::steady_clock::now();
    const rclcpp::Time stamp(m.header.stamp);
    if (!t0_) {t0_ = stamp;}
    const double t = (stamp - *t0_).seconds();
    qn::State x;
    for (int i = 0; i < 3; ++i) {
      x[i] = m.position[i];
      x[7 + i] = m.velocity[i];
      x[10 + i] = m.body_rates[i];
    }
    for (int i = 0; i < 4; ++i) {x[3 + i] = m.orientation[i];}
    const auto res = solver_.step(t, x);

    RotorThrusts cmd;
    cmd.header.stamp = now();
    for (int i = 0; i < 4; ++i) {cmd.thrust[i] = res.u[i];}
    cmd_pub_->publish(cmd);

    qn::State xr;
    qn::Thrusts ur;
    traj_.reference(t, xr, ur);
    ControllerStats st;
    st.header = m.header;
    st.solve_time_ms = res.solve_time_ms;
    st.step_time_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
    st.latency_ms = (rclcpp::Time(cmd.header.stamp) - stamp).seconds() * 1e3;
    st.status = res.status;
    st.position_error = std::sqrt(std::pow(x[0] - xr[0], 2) + std::pow(x[1] - xr[1], 2) +
      std::pow(x[2] - xr[2], 2));
    stats_pub_->publish(st);
  }

  qn::NmpcSolver solver_;
  qn::Figure8 traj_;
  std::optional<rclcpp::Time> t0_;
  rclcpp::Publisher<RotorThrusts>::SharedPtr cmd_pub_;
  rclcpp::Publisher<ControllerStats>::SharedPtr stats_pub_;
  rclcpp::Subscription<QuadState>::SharedPtr sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NmpcNode>());
  rclcpp::shutdown();
  return 0;
}
