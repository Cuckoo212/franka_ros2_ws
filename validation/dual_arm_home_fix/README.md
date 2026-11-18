# HOME/FIX offline geometric validation — 2026-10-28

Result: PASS. No ROS node, robot driver, or trajectory execution was started.

Inputs are captured in dual.urdf, dual.srdf and targets.txt. They include the
recorded HOME/FIX angles, A mount at B +X 0.47 m with 180-degree Z rotation,
and the fitted rectangular board with 0.01 m thickness.

- 16 combined endpoint tests: four HOME/FIX combinations, each with both
  grippers independently fully open or fully closed. All within bounds and
  collision-free against the modeled table, self, and other arm.
- 8 OMPL RRTConnect planning tests: each arm HOME to FIX and FIX to HOME,
  moving gripper open/closed, other arm at HOME with open gripper. All passed.
- Every returned path rechecked by MoveIt at maximum joint increment 0.01 rad.

See report.txt for detailed results and TCP positions in B coordinates.
The reported solutions are geometric paths, not executable timed trajectories.
This does not validate current robot pose to HOME, grasp pose to FIX, cable
forces/deformation, camera occlusion, or unmodeled objects. Real execution must
replan from the current measured state. execution_enabled remains false.
