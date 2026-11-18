#include <cable_moveit_grasp/dual_motion.hpp>

void DualMotion::executeGraspSkill(const std::string& arm,MGI& group) {
  if(!home_[arm]) throw std::runtime_error("Grasp requires this arm to start at home");
  const bool rviz=get_parameter("use_rviz_grasp_plan").as_bool();
  const int reselections=get_parameter("grasp_max_reselections").as_int();
  if(reselections<0 || reselections>3) throw std::runtime_error("grasp_max_reselections must be in [0,3]");
  auto open=[&]() {
    franka_msgs::action::Move::Goal goal;goal.width=.08;goal.speed=.02;
    gripper<franka_msgs::action::Move>(arm,goal);
  };
  auto return_home=[&]() {
    report("Three grasp positions failed; opening and returning failed arm to HOME");
    executeReleaseSkill(arm,"open_for_home");
    executeHomeSkill(arm,group); // Only a successful HOME permits re-selection.
  };
  for(int round=0;round<=reselections;++round) {
    if(rviz) {
      MGI::Plan selected;
      try {selected=selectedGraspPlan(arm,group,round>0);checkPlan(group,selected,true);}
      catch(const std::exception& error) {
        if(round>0) throw cable_grasp::GraspRecovered(std::string("Recovery point planning failed after HOME: ")+error.what());
        throw;
      }
      home_[arm]=false;open();
      execute(group,&selected,false);
    } else {
      if(round>0) throw cable_grasp::GraspRecovered("Fresh point required after HOME; publish a new grasp target");
      geometry_msgs::msg::PoseStamped target;
      {std::lock_guard<std::mutex> lock(mutex_);
        if(!goals_.count(arm) || (now()-goal_times_.at(arm)).seconds()>5.) throw std::runtime_error("Publish a fresh grasp_target");
        target=goals_.at(arm);goals_.erase(arm);
      }
      if(target.header.frame_id!=(arm=="a" ? "a_fr3_link0" : "fr3_link0")) throw std::runtime_error("Target frame mismatch");
      const auto& q=target.pose.orientation;const auto& p=target.pose.position;
      if(!std::isfinite(p.x+p.y+p.z+q.x+q.y+q.z+q.w) || std::abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.)>1e-3)
        throw std::runtime_error("Invalid target pose");
      home_[arm]=false;open();group.setStartStateToCurrentState();group.clearPoseTargets();group.setPoseTarget(target);execute(group,nullptr,false);
    }
    const std::string frame=arm=="a" ? "a_fr3_link0" : "fr3_link0";
    const std::string tcp=arm=="a" ? "a_fr3_hand_tcp" : "fr3_hand_tcp";
    const auto original_state=measuredState(group);
    const Eigen::Isometry3d original=original_state.getGlobalLinkTransform(frame).inverse()*original_state.getGlobalLinkTransform(tcp);
    for(int attempt=0;attempt<3;++attempt) {
      if(attempt>0) {
        open();
        const auto current=measuredState(group);
        const auto target=cable_grasp::graspRetryTarget(original,attempt);
        geometry_msgs::msg::Pose pose;const Eigen::Quaterniond q(target.linear());
        pose.position.x=target.translation().x();pose.position.y=target.translation().y();pose.position.z=target.translation().z();
        pose.orientation.x=q.x();pose.orientation.y=q.y();pose.orientation.z=q.z();pose.orientation.w=q.w();
        const auto previous_frame=group.getPoseReferenceFrame();
        group.setPoseReferenceFrame(frame);group.setStartState(current);group.clearPoseTargets();
        moveit_msgs::msg::RobotTrajectory message;double fraction=0.;
        try {fraction=group.computeCartesianPath({pose},.001,2.,message,group.getPathConstraints(),true);}
        catch(...) {group.setPoseReferenceFrame(previous_frame);throw;}
        group.setPoseReferenceFrame(previous_frame);
        if(!std::isfinite(fraction) || fraction<.999999) throw std::runtime_error("Grasp Z retry Cartesian path incomplete; recovery required");
        robot_trajectory::RobotTrajectory trajectory(group.getRobotModel(),group.getName());trajectory.setRobotTrajectoryMsg(current,message);
        if(trajectory.getWayPointCount()<2) throw std::runtime_error("Empty grasp retry trajectory");
        trajectory_processing::IterativeParabolicTimeParameterization timing;
        if(!timing.computeTimeStamps(trajectory,get_parameter("velocity_scaling").as_double(),get_parameter("acceleration_scaling").as_double()))
          throw std::runtime_error("Grasp retry time parameterization failed");
        MGI::Plan plan;moveit::core::robotStateToRobotStateMsg(current,plan.start_state_);trajectory.getRobotTrajectoryMsg(plan.trajectory_);
        report("Executing grasp retry offset from original TCP-Z="+std::to_string(attempt==1 ? -.01 : .01)+" m");
        execute(group,&plan,false); // Full live-state, limits and collision checks before execution.
      }
      franka_msgs::action::Grasp::Goal close;close.width=.004;close.speed=.02;close.force=10.;close.epsilon.inner=close.epsilon.outer=.003;
      report("Grasp point round="+std::to_string(round)+" close attempt="+std::to_string(attempt+1));
      try {
        gripper<franka_msgs::action::Grasp>(arm,close);
        holding_[arm]=true;recordBGrasp(arm,group);return;
      } catch(const cable_grasp::GraspMiss& error) {report("Confirmed grasp miss: "+std::string(error.what()));}
    }
    return_home();
    if(round==reselections) throw cable_grasp::GraspRecovered("All bounded grasp retries exhausted; failed arm is open at HOME; select a fresh cable point");
    report("Re-selecting recovery point from fresh cloud after HOME");
  }
}
