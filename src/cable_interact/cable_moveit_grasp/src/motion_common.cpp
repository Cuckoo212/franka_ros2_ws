#include <cable_moveit_grasp/motion_common.hpp>
namespace cable_grasp {
planning_scene::PlanningScenePtr requestMotionScene(
 const rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr& client,
 const moveit::core::RobotModelConstPtr& model,bool require_camera) {
 auto request=std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();request->components.components=1023;
 auto future=client->async_send_request(request);
 if(future.wait_for(std::chrono::seconds(3))!=std::future_status::ready) throw std::runtime_error("Planning scene unavailable");
 auto scene=std::make_shared<planning_scene::PlanningScene>(model);
 const auto response=future.get();
 if(!response || !scene->setPlanningSceneMsg(response->scene)) throw std::runtime_error("Invalid planning scene response");
 if(!scene->getWorld()->hasObject("cable_platform")) throw std::runtime_error("Table missing from scene");
 if(require_camera && !scene->getWorld()->hasObject("realsense_camera")) throw std::runtime_error("Camera obstacle missing from scene");
 return scene;
}
MotionGroup::Plan timeMotionPath(const moveit::core::RobotState& start,robot_trajectory::RobotTrajectory path,double velocity,double acceleration) {
 if(path.getWayPointCount()<2 || !std::isfinite(velocity) || velocity<=0. || velocity>1. || !std::isfinite(acceleration) || acceleration<=0. || acceleration>1.)
   throw std::runtime_error("Invalid motion path/scaling");
 trajectory_processing::IterativeParabolicTimeParameterization timing;
 if(!timing.computeTimeStamps(path,velocity,acceleration)) throw std::runtime_error("Time parameterization failed");
 MotionGroup::Plan plan;moveit::core::robotStateToRobotStateMsg(start,plan.start_state_);path.getRobotTrajectoryMsg(plan.trajectory_);return plan;
}
}
