// Quadrotor simulator node: rigid body + motor lag + optional rotor faults, integrated in
// real time at 1 kHz with the latest thrust command; state published at the control rate.
//
// Publishes   /quad/state      quadrotor_nmpc_msgs/QuadState  (state_rate_hz)
// Subscribes  /quad/thrust_cmd quadrotor_nmpc_msgs/RotorThrusts
// The simulation starts with the first command, so a slow-starting controller does not
// find the quadrotor already off its reference.
// Parameters  v_peak (start on the figure-8 reference at t = 0), state_rate_hz,
//             rotor_max_thrust (4 values, N; e.g. [7, 7, 3.5, 7] for a damaged rotor 3)

#include <chrono>
#include <memory>
#include <vector>

#include "quadrotor_nmpc_msgs/msg/quad_state.hpp"
#include "quadrotor_nmpc_msgs/msg/rotor_thrusts.hpp"
#include "quadrotor_nmpc_ros/quad_model.hpp"
#include "rclcpp/rclcpp.hpp"

using quadrotor_nmpc_msgs::msg::QuadState;
using quadrotor_nmpc_msgs::msg::RotorThrusts;
using namespace std::chrono_literals;
namespace qn = quadrotor_nmpc;

class SimNode : public rclcpp::Node
{
public:
  SimNode()
  : Node("quad_sim")
  {
    qn::Figure8 f;
    f.v_peak = declare_parameter("v_peak", 10.0);
    const double rate = declare_parameter("state_rate_hz", 100.0);
    auto caps = declare_parameter("rotor_max_thrust", std::vector<double>(4, 1e9));
    qn::Faults faults;
    for (int i = 0; i < 4; ++i) {faults.max_thrust[i] = caps.at(i);}
    qn::State x0;
    qn::Thrusts u0;
    f.reference(0.0, x0, u0);
    plant_ = std::make_unique<qn::Plant>(x0, faults);
    cmd_.fill(qn::params::m * qn::params::g / 4);
    every_ = std::max(1, static_cast<int>(std::lround(1000.0 / rate)));
    auto qos = rclcpp::QoS(1).best_effort();
    pub_ = create_publisher<QuadState>("/quad/state", qos);
    sub_ = create_subscription<RotorThrusts>(
      "/quad/thrust_cmd", qos, [this](RotorThrusts::ConstSharedPtr m) {
        for (int i = 0; i < 4; ++i) {cmd_[i] = m->thrust[i];}
        started_ = true;
      });
    timer_ = create_wall_timer(1ms, [this]() {tick();});
  }

private:
  void tick()
  {
    if (k_ % every_ == 0) {
      const auto & x = plant_->x();
      QuadState m;
      m.header.stamp = now();
      m.header.frame_id = "world";
      for (int i = 0; i < 3; ++i) {
        m.position[i] = x[i];
        m.velocity[i] = x[7 + i];
        m.body_rates[i] = x[10 + i];
      }
      for (int i = 0; i < 4; ++i) {m.orientation[i] = x[3 + i];}
      pub_->publish(m);
    }
    if (started_) {plant_->step(cmd_, 1e-3);}  // hold the start state until the controller runs
    ++k_;
  }

  std::unique_ptr<qn::Plant> plant_;
  qn::Thrusts cmd_{};
  long k_ = 0;
  bool started_ = false;
  int every_ = 10;
  rclcpp::Publisher<QuadState>::SharedPtr pub_;
  rclcpp::Subscription<RotorThrusts>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimNode>());
  rclcpp::shutdown();
  return 0;
}
