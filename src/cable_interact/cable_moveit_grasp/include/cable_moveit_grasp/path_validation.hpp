#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

namespace cable_grasp {
// Resampling resolution is configurable; validity decisions belong to MoveIt.
inline robot_trajectory::RobotTrajectory validateTrajectory(
    const robot_trajectory::RobotTrajectory& trajectory,
    const planning_scene::PlanningScene& scene, double joint_step,
    const moveit_msgs::msg::Constraints& path_constraints = moveit_msgs::msg::Constraints(),
    const std::vector<moveit_msgs::msg::Constraints>& goal_constraints = {}) {
  if (!std::isfinite(joint_step) || joint_step <= 0.)
    throw std::runtime_error("Invalid trajectory validation resolution");
  if (trajectory.getWayPointCount() < 2 || !trajectory.getGroup())
    throw std::runtime_error("Empty trajectory or missing planning group");
  const auto* group = trajectory.getGroup();
  robot_trajectory::RobotTrajectory sampled(trajectory.getRobotModel(), trajectory.getGroupName());
  auto append = [&](moveit::core::RobotState state, double dt) {
    state.update();
    for (const auto& name : group->getVariableNames())
      if (!std::isfinite(state.getVariablePosition(name)))
        throw std::runtime_error("Non-finite trajectory joint position: " + name);
    if (!state.satisfiesBounds(group))
      throw std::runtime_error("MoveIt joint bounds check failed at sample " + std::to_string(sampled.getWayPointCount()));
    sampled.addSuffixWayPoint(state, dt);
  };
  append(trajectory.getFirstWayPoint(), 0.);
  for (size_t i = 1; i < trajectory.getWayPointCount(); ++i) {
    const auto& a = trajectory.getWayPoint(i-1);
    const auto& b = trajectory.getWayPoint(i);
    double delta = 0.;
    for (const auto& name : group->getVariableNames()) {
      const double value = b.getVariablePosition(name);
      if (!std::isfinite(value)) throw std::runtime_error("Non-finite trajectory joint position: " + name);
      delta = std::max(delta, std::abs(value-a.getVariablePosition(name)));
    }
    const int steps = std::max(1, static_cast<int>(std::ceil(delta/joint_step)));
    for (int k = 1; k <= steps; ++k) {
      moveit::core::RobotState state(a);
      a.interpolate(b, static_cast<double>(k)/steps, state);
      append(state, trajectory.getWayPointDurationFromPrevious(i)/steps);
    }
  }
  std::vector<size_t> invalid;
  if (!scene.isPathValid(sampled, path_constraints, goal_constraints,
                         trajectory.getGroupName(), false, &invalid)) {
    throw std::runtime_error("MoveIt isPathValid rejected collision/feasibility/path or goal constraints; first invalid sample=" +
                             (invalid.empty() ? std::string("unknown") : std::to_string(invalid.front())));
  }
  return sampled;
}
} // namespace cable_grasp
