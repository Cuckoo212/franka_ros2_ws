#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <moveit_msgs/srv/get_motion_plan.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

namespace cable_grasp {
// Own the service response until the nested trajectory has been moved out.
// std::future<SharedPtr>::get() returns a temporary owner: a reference to one of
// its members would dangle as soon as that statement ended.
inline moveit_msgs::msg::MotionPlanResponse takeSelectedGraspPlan(
    moveit_msgs::srv::GetMotionPlan::Response::SharedPtr response,const std::string& group) {
  if(!response || response->motion_plan_response.error_code.val!=moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
      response->motion_plan_response.group_name!=group)
    throw std::runtime_error("No valid selected candidate for this arm; select A/B in RViz, click cloud and wait for preview");
  return std::move(response->motion_plan_response);
}

inline void requireCandidateJointNames(const moveit_msgs::msg::RobotTrajectory& trajectory,
                                      std::vector<std::string> expected) {
  auto actual=trajectory.joint_trajectory.joint_names;
  std::sort(expected.begin(),expected.end()); std::sort(actual.begin(),actual.end());
  if(actual!=expected || trajectory.joint_trajectory.points.size()<2 ||
      !trajectory.multi_dof_joint_trajectory.points.empty())
    throw std::runtime_error("Malformed or wrong-arm candidate trajectory");
}
inline void requireUnchangedArms(const moveit::core::RobotState& planned,
                                const moveit::core::RobotState& current,
                                const std::vector<std::string>& groups,double tolerance) {
  if(!std::isfinite(tolerance) || tolerance<=0.) throw std::runtime_error("Invalid start tolerance");
  for(const auto& name:groups) {
    const auto* group=current.getJointModelGroup(name);
    if(!group) throw std::runtime_error("Unknown candidate arm group");
    for(const auto& joint:group->getVariableNames()) {
      const double before=planned.getVariablePosition(joint),now=current.getVariablePosition(joint);
      if(!std::isfinite(before) || !std::isfinite(now) || std::abs(before-now)>tolerance)
        throw std::runtime_error("Either arm moved since candidate preview; click again to replan");
    }
  }
}
} // namespace cable_grasp
