#pragma once
#include <cable_moveit_grasp/path_validation.hpp>
#include <cable_moveit_grasp/grasp_retry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit/robot_state/conversions.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <franka_msgs/action/grasp.hpp>
#include <functional>
namespace cable_grasp {
using MotionGroup=moveit::planning_interface::MoveGroupInterface;
planning_scene::PlanningScenePtr requestMotionScene(
  const rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr& client,
  const moveit::core::RobotModelConstPtr& model,bool require_camera);
MotionGroup::Plan timeMotionPath(const moveit::core::RobotState& start,
  robot_trajectory::RobotTrajectory path,double velocity,double acceleration);
template<class Action> void sendGripperGoal(const rclcpp::Node::SharedPtr& node,
 const std::string& action,const typename Action::Goal& goal,double timeout,
 const std::function<void(const std::string&)>& report) {
 auto client=rclcpp_action::create_client<Action>(node,action);
 if(!client->wait_for_action_server(std::chrono::seconds(3))) throw std::runtime_error("Gripper unavailable: "+action);
 auto future=client->async_send_goal(goal);
 if(future.wait_for(std::chrono::seconds(3))!=std::future_status::ready) throw std::runtime_error("Unknown gripper goal state; recovery required");
 auto handle=future.get();if(!handle) throw std::runtime_error("Gripper goal rejected");
 auto result=client->async_get_result(handle);
 if(result.wait_for(std::chrono::duration<double>(timeout))!=std::future_status::ready) {
   client->async_cancel_goal(handle);throw std::runtime_error("Gripper timeout; cancel requested; recovery required");
 }
 const auto wrapped=result.get();
 if(!wrapped.result) throw std::runtime_error("Empty gripper result; recovery required");
 report("gripper result: action_code="+std::to_string(static_cast<int>(wrapped.code))+" success="+
   (wrapped.result->success ? "true" : "false")+" error='"+wrapped.result->error+"'");
 if constexpr(std::is_same<Action,franka_msgs::action::Grasp>::value) {
   if(!wrapped.result->success && wrapped.result->error.empty() &&
      (wrapped.code==rclcpp_action::ResultCode::ABORTED || wrapped.code==rclcpp_action::ResultCode::SUCCEEDED))
     throw GraspMiss("Gripper completed but did not confirm a grasp");
 }
 if(wrapped.code!=rclcpp_action::ResultCode::SUCCEEDED || !wrapped.result->success)
   throw std::runtime_error("Gripper failed; action_code="+std::to_string(static_cast<int>(wrapped.code))+"; error='"+wrapped.result->error+"'; recovery required");
}
}
