# Quadrotor state (world frame z up; quaternion w-x-y-z, body to world).
std_msgs/Header header
float64[3] position        # m
float64[4] orientation     # w, x, y, z
float64[3] velocity        # m/s, world frame
float64[3] body_rates      # rad/s, body frame
