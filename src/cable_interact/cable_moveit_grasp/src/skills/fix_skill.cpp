#include <cable_moveit_grasp/dual_motion.hpp>
#include <cable_moveit_grasp/fix_projection.hpp>

moveit::core::RobotState DualMotion::measuredState(MGI& group) {
  freshState();
  auto current=group.getCurrentState(2.);
  if(!current) throw std::runtime_error("FIX current robot state unavailable");
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Both arms are frozen into every collision check; use raw measured joints.
    for(const std::string prefix:{std::string(""),std::string("a_")}) for(int i=1;i<=7;++i) {
      const auto joint=prefix+"fr3_joint"+std::to_string(i);
      if(!joint_positions_.count(joint) || !joint_times_.count(joint) || (now()-joint_times_.at(joint)).seconds()>1.)
        throw std::runtime_error("FIX raw joint feedback stale: "+joint);
      current->setVariablePosition(joint,joint_positions_.at(joint));
    }
  }
  current->update();return *current;
}

void DualMotion::executeFixSkill(const std::string& arm,MGI& group) {
  fix_preview_pub_.at(arm)->publish(moveit_msgs::msg::DisplayTrajectory{});
  const std::string frame=arm=="a" ? "a_fr3_link0" : "fr3_link0";
  const std::string tcp=arm=="a" ? "a_fr3_hand_tcp" : "fr3_hand_tcp";
  const auto current=measuredState(group);
  const auto* jmg=current.getJointModelGroup(group.getName());
  const Eigen::Isometry3d base=current.getGlobalLinkTransform(frame);
  const Eigen::Isometry3d start=base.inverse()*current.getGlobalLinkTransform(tcp);
  const auto recorded=get_parameter(arm=="a" ? "fixA" : "fixB").as_double_array();
  if(recorded.size()!=7 || !std::all_of(recorded.begin(),recorded.end(),[](double q){return std::isfinite(q);}))
    throw std::runtime_error("FIX requires seven finite recorded joint values");
  auto recorded_state=current;recorded_state.setJointGroupPositions(jmg,recorded);recorded_state.update();
  if(!recorded_state.satisfiesBounds(jmg)) throw std::runtime_error("Recorded FIX joints outside bounds");
  const Eigen::Vector3d position=(base.inverse()*recorded_state.getGlobalLinkTransform(tcp)).translation();
  const auto candidates=cable_grasp::fixPositionCandidates(position,start.linear());
  const double limit=get_parameter("fix_tcp_max_total_rotation").as_double();
  if(!std::isfinite(limit) || limit<=0. || limit>M_PI)
    throw std::runtime_error("Invalid FIX terminal projection limit");
  report("FIX position-only target in "+frame+" ["+std::to_string(position.x())+", "+std::to_string(position.y())+", "+std::to_string(position.z())+"]; recorded orientation ignored; prefer current orientation");
  const auto previous=group.getPathConstraints();
  const auto previous_frame=group.getPoseReferenceFrame();
  const double previous_time=group.getPlanningTime(),previous_position=group.getGoalPositionTolerance(),previous_orientation=group.getGoalOrientationTolerance();
  auto restore=[&]() {
    group.setPathConstraints(previous);group.setPoseReferenceFrame(previous_frame);
    group.setPlanningTime(previous_time);group.setGoalPositionTolerance(previous_position);group.setGoalOrientationTolerance(previous_orientation);group.clearPoseTargets();
  };
  auto constraints=previous;constraints.name="fix_projection_terminal_guard"; // No intermediate orientation/peak constraint.
  group.setPathConstraints(constraints);group.setPoseReferenceFrame(frame);
  group.setGoalPositionTolerance(.0005);group.setGoalOrientationTolerance(.001);
  struct Result {MGI::Plan plan;bool valid=false;double final_angle=std::numeric_limits<double>::infinity();double joint_travel=std::numeric_limits<double>::infinity();std::string name;};
  Result best;
  auto timedPlan=[&](robot_trajectory::RobotTrajectory path) {
    trajectory_processing::IterativeParabolicTimeParameterization timing;
    if(!timing.computeTimeStamps(path,get_parameter("velocity_scaling").as_double(),get_parameter("acceleration_scaling").as_double()))
      throw std::runtime_error("FIX time parameterization failed");
    MGI::Plan plan;moveit::core::robotStateToRobotStateMsg(current,plan.start_state_);path.getRobotTrajectoryMsg(plan.trajectory_);return plan;
  };
  try {
    // Already at the requested position: no orientation adjustment is needed.
    if((position-start.translation()).norm()<.001) {
      report("FIX position already reached; keeping current orientation");
      restore();home_[arm]=false;return;
    }
    for(size_t index=0;index<candidates.size();++index) {
      const auto& target=candidates[index];
      const std::string name=index==0 ? "keep-current" : "adaptive-"+std::to_string(index);
      report("FIX evaluating position-only candidate="+name);
      const int routes=index==0 ? 1 : 3; // With fixed orientation all three routes reduce to one line.
      for(int route=0;route<routes;++route) {
        group.setStartState(current);group.clearPoseTargets();
        try {
          std::vector<geometry_msgs::msg::Pose> waypoints;
          for(const auto& transform:cable_grasp::fixCartesianWaypoints(start,target,route)) {
            geometry_msgs::msg::Pose pose;const Eigen::Quaterniond q(transform.linear());
            pose.position.x=transform.translation().x();pose.position.y=transform.translation().y();pose.position.z=transform.translation().z();
            pose.orientation.x=q.x();pose.orientation.y=q.y();pose.orientation.z=q.z();pose.orientation.w=q.w();waypoints.push_back(pose);
          }
          moveit_msgs::msg::RobotTrajectory message;
          const double fraction=group.computeCartesianPath(waypoints,.003,2.,message,constraints,true);
          report("FIX candidate="+name+" Cartesian route="+std::to_string(route)+" completion="+std::to_string(fraction));
          if(!std::isfinite(fraction) || fraction<.999999) throw std::runtime_error("Cartesian FIX path incomplete");
          robot_trajectory::RobotTrajectory path(group.getRobotModel(),group.getName());path.setRobotTrajectoryMsg(current,message);
          if(path.getWayPointCount()<2) throw std::runtime_error("Empty FIX Cartesian path");
          auto plan=timedPlan(path);
          const auto checked=checkPlan(group,plan);
          const Eigen::Isometry3d reached=checked.getLastWayPoint().getGlobalLinkTransform(frame).inverse()*checked.getLastWayPoint().getGlobalLinkTransform(tcp);
          // Only the recorded POSITION is a task goal; candidate orientation is a planning choice.
          if((reached.translation()-position).norm()>.001) throw std::runtime_error("FIX target position mismatch");
          const double angle=cable_grasp::requireFixTCPRotationTotal(checked,frame,tcp,limit);
          double travel=0.;
          for(size_t i=1;i<checked.getWayPointCount();++i)
            travel+=checked.getWayPoint(i-1).distance(checked.getWayPoint(i),jmg);
          report("FIX qualified candidate="+name+"; terminal projected absolute angle="+std::to_string(angle*180./M_PI)+" deg; joint travel="+std::to_string(travel));
          if(!best.valid || angle<best.final_angle-1e-6 || (std::abs(angle-best.final_angle)<=1e-6 && travel<best.joint_travel)) {
            best.plan=std::move(plan);best.valid=true;best.final_angle=angle;best.joint_travel=travel;best.name=name;
          }
        } catch(const std::exception& error) {report("FIX candidate="+name+" path rejected: "+error.what());}
      }
      // A valid constant-orientation line has priority over all adaptive orientations.
      if(index==0 && best.valid) break;
    }
    if(!best.valid) throw std::runtime_error("No qualified position-only Cartesian FIX path; keeping grasp; no sampling fallback");
    report("FIX selected candidate="+best.name+"; terminal projected absolute angle="+std::to_string(best.final_angle*180./M_PI)+" deg");
    fix_preview_pub_.at(arm)->publish(moveit_msgs::msg::DisplayTrajectory{});
    execute(group,&best.plan); // Exact saved plan; revalidated before execution, arrival checked afterward.
    restore();home_[arm]=false;
  } catch(...) {restore();throw;}
}
