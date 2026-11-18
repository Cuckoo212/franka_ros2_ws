#include <cable_moveit_grasp/single_arm_motion.hpp>
SingleArmMotion::MGI::Plan SingleArmMotion::planPlace(const moveit::core::RobotState& start) {
  const auto target=get_parameter("place_joint_positions").as_double_array();
  if(target.size()!=7 || !std::all_of(target.begin(),target.end(),[](double q){return std::isfinite(q);})) throw std::runtime_error("Place requires seven finite joints");
  const double seconds=get_parameter("place_planning_time").as_double(),tolerance=get_parameter("place_goal_tolerance").as_double();
  if(!std::isfinite(seconds) || seconds<=0. || !std::isfinite(tolerance) || tolerance<=0.) throw std::runtime_error("Invalid place planning configuration");
  group_->setStartState(start);group_->clearPoseTargets();
  if(!group_->setJointValueTarget(target)) throw std::runtime_error("Place goal outside joint bounds");
  group_->setPlanningTime(seconds);group_->setGoalJointTolerance(tolerance);
  MGI::Plan plan;if(group_->plan(plan)!=moveit::core::MoveItErrorCode::SUCCESS) throw std::runtime_error("Place planning failed");
  robot_trajectory::RobotTrajectory path(group_->getRobotModel(),group_->getName());path.setRobotTrajectoryMsg(start,plan.trajectory_);
  cable_grasp::validateTrajectory(path,*scene(),.01);
  const auto& names=start.getJointModelGroup(group_->getName())->getVariableNames();
  for(size_t i=0;i<names.size();++i) if(std::abs(path.getLastWayPoint().getVariablePosition(names[i])-target[i])>tolerance+1e-6) throw std::runtime_error("Place planned goal mismatch");
  return plan;
}
void SingleArmMotion::executePlaceSkill() {
  report("place: replanning from measured lifted state");const auto plan=planPlace(measured());executePlan(plan);
  const auto configured=get_parameter("place_joint_positions").as_double_array();
  const auto& names=group_->getRobotModel()->getJointModelGroup(group_->getName())->getVariableNames();
  const double timeout=get_parameter("arrival_timeout").as_double(),tolerance=get_parameter("place_arrival_tolerance").as_double();
  if(!std::isfinite(timeout) || timeout<=0. || !std::isfinite(tolerance) || tolerance<=0.) throw std::runtime_error("Invalid place arrival configuration");
  const auto finished=now().nanoseconds();const auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(timeout);
  cable_grasp::ArrivalSettling settled(.1);
  while(rclcpp::ok() && std::chrono::steady_clock::now()<deadline) {
    std::map<std::string,double> positions;std::map<std::string,std::int64_t> stamps;
    {std::lock_guard<std::mutex> lock(mutex_);positions=positions_;stamps=stamps_;}
    if(settled.update(cable_grasp::checkArrival(names,configured,positions,stamps,finished,now().nanoseconds(),tolerance))) {report("place arrival confirmed");return;}
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  throw std::runtime_error("Place arrival failed; keeping grasp, no release");
}
