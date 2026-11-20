# Per-step diagnostics of the NMPC node.
std_msgs/Header header       # stamp of the state this step answered
float64 solve_time_ms        # acados internal time (time_tot)
float64 step_time_ms         # whole callback: reference, solve, publish
float64 latency_ms           # state stamp -> command published
int32 status                 # acados status (0 = success)
float64 position_error       # distance to the reference at this state (m)
