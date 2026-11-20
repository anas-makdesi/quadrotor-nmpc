# Changelog

## Unreleased
- ddma pinned to `v0.12.0` (the `thrust` extra). CI installs it, so the thrust
  certificate tests now run on every push instead of being skipped.

## 0.1.0 (2026-11-16)
First release.
- 13-state quaternion quadrotor model with rotor thrust inputs and rotor drag; plant with
  motor lag and rotor faults; flat full-state figure-8 reference.
- acados NMPC (Gauss-Newton SQP, real-time iteration or converged), HPIPM, non-uniform
  1 s horizon; per-rotor thrust limits at run time.
- Baselines: SE(3) geometric controller with flatness feedforward and attitude-priority
  mixer; linear MPC at hover with OSQP.
- Rotor thrust certificate from a 1 s hop with ddma's `GrowthMap` (concave rotor maps,
  motor lag handled exactly).
- Speed sweep, damaged-rotor experiment and timing benchmark with figures and JSON results.
- ROS 2 Jazzy packages: messages, C++ NMPC node and 1 kHz C++ simulator node, launch file,
  parity gtests, launch test, flight probe.
- CI (ruff, pytest, colcon build/test in a ROS container), Dockerfile.
