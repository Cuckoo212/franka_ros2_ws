#include <cable_moveit_grasp/single_arm_motion.hpp>
using namespace std::chrono_literals;
SingleArmMotion::MGI::Plan SingleArmMotion::planLift(const moveit::core::RobotState& start) {
  const double distance=get_parameter("lift_distance").as_double();
  if(!std::isfinite(distance) || distance<=0.) throw std::runtime_error("Invalid lift_distance");
  const auto frame=get_parameter("base_frame").as_string(),tcp=get_parameter("tcp_link").as_string();
  Eigen::Isometry3d target=start.getGlobalLinkTransform(frame).inverse()*start.getGlobalLinkTransform(tcp);
  target.translation().y()+=distance; // Historical behavior: fixed link0 +Y, not TCP-local Y or world Z.
  geometry_msgs::msg::Pose pose;const Eigen::Quaterniond q(target.linear());
  pose.position.x=target.translation().x();pose.position.y=target.translation().y();pose.position.z=target.translation().z();
  pose.orientation.x=q.x();pose.orientation.y=q.y();pose.orientation.z=q.z();pose.orientation.w=q.w();
  group_->clearPoseTargets();group_->setStartState(start);
  moveit_msgs::msg::RobotTrajectory msg;
  const double fraction=group_->computeCartesianPath({pose},.003,2.,msg,true);
  if(!std::isfinite(fraction) || fraction<.999999) throw std::runtime_error("Incomplete collision-free +Y lift");
  robot_trajectory::RobotTrajectory path(group_->getRobotModel(),group_->getName());path.setRobotTrajectoryMsg(start,msg);
  auto world=scene();const auto checked=cable_grasp::validateTrajectory(path,*world,.01);
  const auto& end=checked.getLastWayPoint();const auto reached=end.getGlobalLinkTransform(frame).inverse()*end.getGlobalLinkTransform(tcp);
  if((reached.translation()-target.translation()).norm()>.001 || Eigen::Quaterniond(reached.linear()).angularDistance(q)>.005)
    throw std::runtime_error("Lift goal pose mismatch");
  return cable_grasp::timeMotionPath(start,path,get_parameter("velocity_scaling").as_double(),get_parameter("acceleration_scaling").as_double());
}
void SingleArmMotion::executeLiftSkill() {
  report("lift: replanning from measured post-grasp state");
  const auto current=measured();const auto frame=get_parameter("base_frame").as_string(),tcp=get_parameter("tcp_link").as_string();
  Eigen::Vector3d target=(current.getGlobalLinkTransform(frame).inverse()*current.getGlobalLinkTransform(tcp)).translation();
  target.y()+=get_parameter("lift_distance").as_double();
  executePlan(planLift(current));
  const double tolerance=get_parameter("lift_position_tolerance").as_double();
  if(!std::isfinite(tolerance) || tolerance<=0.) throw std::runtime_error("Invalid lift_position_tolerance");
  const auto deadline=std::chrono::steady_clock::now()+3s;
  while(rclcpp::ok() && std::chrono::steady_clock::now()<deadline) {
    const auto state=measured();const Eigen::Vector3d reached=(state.getGlobalLinkTransform(frame).inverse()*state.getGlobalLinkTransform(tcp)).translation();
    if((reached-target).norm()<=tolerance) {report("lift completed");return;}
    std::this_thread::sleep_for(20ms);
  }
  throw std::runtime_error("Lift position arrival failed; keeping grasp");
}
