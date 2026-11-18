#include <iostream>
#include <functional>
#include <future>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <cable_moveit_grasp/path_validation.hpp>
#include <cable_moveit_grasp/candidate_handoff.hpp>
#include <cable_moveit_grasp/orientation_checks.hpp>
#include <cable_moveit_grasp/fix_projection.hpp>

void require(bool value, const std::string& text) { if (!value) throw std::runtime_error(text); }
void rejects(const std::function<void()>& function, const std::string& text) {
  bool failed=false;
  try { function(); } catch(const std::runtime_error&) { failed=true; }
  require(failed,text);
}
int main() {
  const std::string urdf_text=R"(<robot name="validation_fixture">
    <link name="base"/><link name="tool"><collision><geometry><sphere radius="0.03"/></geometry></collision></link>
    <joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
      <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>)";
  auto urdf=urdf::parseURDF(urdf_text);
  auto srdf=std::make_shared<srdf::Model>();
  require(srdf->initString(*urdf,"<robot name='validation_fixture'><group name='arm'><joint name='slide'/></group></robot>"),"SRDF parse");
  auto model=std::make_shared<moveit::core::RobotModel>(urdf,srdf);
  planning_scene::PlanningScene scene(model);
  auto trajectory=[&](double from,double to) {
    robot_trajectory::RobotTrajectory result(model,"arm");
    moveit::core::RobotState state(model); state.setToDefaultValues();
    state.setVariablePosition("slide",from);state.update();result.addSuffixWayPoint(state,0.);
    state.setVariablePosition("slide",to);state.update();result.addSuffixWayPoint(state,1.);
    return result;
  };
  // Reproduce the ownership boundary of async_send_request without ROS/network.
  std::promise<moveit_msgs::srv::GetMotionPlan::Response::SharedPtr> promise;
  auto future=promise.get_future();
  auto response=std::make_shared<moveit_msgs::srv::GetMotionPlan::Response>();
  response->motion_plan_response.group_name="arm";
  response->motion_plan_response.error_code.val=moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  trajectory(-.2,.2).getRobotTrajectoryMsg(response->motion_plan_response.trajectory);
  std::weak_ptr<moveit_msgs::srv::GetMotionPlan::Response> weak=response;
  promise.set_value(std::move(response));
  const auto selected=cable_grasp::takeSelectedGraspPlan(future.get(),"arm");
  require(weak.expired(),"Temporary service response owner should have been released");
  require(selected.trajectory.joint_trajectory.joint_names==std::vector<std::string>{"slide"},"Candidate must retain owned data after response destruction");
  require(selected.trajectory.joint_trajectory.points.size()==2,"Candidate points lost after response destruction");
  rejects([&]{cable_grasp::takeSelectedGraspPlan(nullptr,"arm");},"Null response must be rejected");
  auto wrong=std::make_shared<moveit_msgs::srv::GetMotionPlan::Response>();
  wrong->motion_plan_response.error_code.val=moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  wrong->motion_plan_response.group_name="other_arm";
  rejects([&]{cable_grasp::takeSelectedGraspPlan(wrong,"arm");},"Response for wrong arm must be rejected");
  auto saved=trajectory(-.2,.2).getFirstWayPoint();
  auto moved=saved; moved.setVariablePosition("slide",-.1); moved.update();
  cable_grasp::requireUnchangedArms(saved,saved,{"arm"},.02);
  rejects([&]{cable_grasp::requireUnchangedArms(saved,moved,{"arm"},.02);},"Must reject movement after candidate preview");
  moveit_msgs::msg::RobotTrajectory candidate;
  trajectory(-.2,.2).getRobotTrajectoryMsg(candidate);
  cable_grasp::requireCandidateJointNames(candidate,{"slide"});
  rejects([&]{cable_grasp::requireCandidateJointNames(candidate,{"other_arm_joint"});},"Must reject wrong-arm candidate");
  candidate.joint_trajectory.joint_names.push_back("slide");
  rejects([&]{cable_grasp::requireCandidateJointNames(candidate,{"slide"});},"Must reject duplicate candidate joint");
  const std::string wrist_urdf=R"(<robot name="wrist_fixture"><link name="base"/><link name="middle"/><link name="wrist_link"/><link name="tcp"/>
    <joint name="shoulder" type="revolute"><parent link="base"/><child link="middle"/><axis xyz="0 1 0"/><limit lower="-3.2" upper="3.2" effort="10" velocity="1"/></joint>
    <joint name="wrist" type="revolute"><parent link="middle"/><child link="wrist_link"/><axis xyz="0 1 0"/><limit lower="-3.2" upper="3.2" effort="10" velocity="1"/></joint><joint name="tool" type="fixed"><parent link="wrist_link"/><child link="tcp"/><origin rpy="1.5707963267948966 0 0" xyz="0 0 0"/></joint></robot>)";
  auto wu=urdf::parseURDF(wrist_urdf);auto ws=std::make_shared<srdf::Model>();
  require(ws->initString(*wu,"<robot name='wrist_fixture'><group name='arm'><chain base_link='base' tip_link='tcp'/></group></robot>"),"Wrist SRDF");
  auto wm=std::make_shared<moveit::core::RobotModel>(wu,ws);
  robot_trajectory::RobotTrajectory compensated(wm,"arm");moveit::core::RobotState wstate(wm);wstate.setToDefaultValues();
  for(double angle:{0.,M_PI/2,M_PI}) {
    wstate.setVariablePosition("shoulder",angle);wstate.setVariablePosition("wrist",-angle);wstate.update();compensated.addSuffixWayPoint(wstate,1.);
  }
  const Eigen::Matrix3d wrist_reference=Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d::UnitX()).toRotationMatrix();
  const auto orientation=cable_grasp::fixTCPConstraint("base","tcp",wrist_reference,M_PI/9);
  require(orientation.parameterization==orientation.ROTATION_VECTOR,"Use rotation-vector constraints");
  planning_scene::PlanningScene wrist_scene(wm);moveit_msgs::msg::Constraints oc;oc.orientation_constraints.push_back(orientation);
  auto sampled=cable_grasp::validateTrajectory(compensated,wrist_scene,.01,oc);
  require(cable_grasp::requireFixTCPOrientation(sampled,orientation)<1e-10,"Compensated joint7 half turn must preserve TCP");
  robot_trajectory::RobotTrajectory excursion(wm,"arm");
  for(double angle:{0.,.4,0.}) {
    wstate.setVariablePosition("shoulder",angle);wstate.setVariablePosition("wrist",0.);wstate.update();excursion.addSuffixWayPoint(wstate,1.);
  }
  rejects([&]{cable_grasp::requireFixTCPOrientation(excursion,orientation);},"Must reject intermediate TCP twist despite matching endpoints");
  rejects([&]{cable_grasp::validateTrajectory(excursion,wrist_scene,.01,oc);},"MoveIt must enforce TCP guard on full path");
  const auto wide_guard=cable_grasp::fixTCPConstraint("base","tcp",wrist_reference,M_PI/2);
  cable_grasp::requireFixTCPOrientation(excursion,wide_guard); // 0.4 rad now allowed.
  wstate.setVariablePosition("shoulder",1.7);wstate.update();excursion.addSuffixWayPoint(wstate,1.);
  rejects([&]{cable_grasp::requireFixTCPOrientation(excursion,wide_guard);},"90-degree bound must still reject excessive rotation");
  const auto half_turn_guard=cable_grasp::fixTCPConstraint("base","tcp",wrist_reference,M_PI);
  cable_grasp::requireFixTCPOrientation(excursion,half_turn_guard); // Intermediate 1.7 rad is allowed.
  wstate.setVariablePosition("shoulder",M_PI);wstate.update();excursion.addSuffixWayPoint(wstate,1.);
  require(std::abs(cable_grasp::requireFixTCPOrientation(excursion,half_turn_guard)-M_PI)<1e-10,"180-degree path boundary must be accepted");
  // Projection rollback: terminal-only signed accumulation, never modulo 360.
  auto projected_path=[&](const std::vector<double>& degrees) {
    robot_trajectory::RobotTrajectory result(wm,"arm");
    auto state=wstate;
    for(double value:degrees) {
      state.setVariablePosition("shoulder",value*M_PI/360.);
      state.setVariablePosition("wrist",value*M_PI/360.);state.update();result.addSuffixWayPoint(state,1.);
    }
    return cable_grasp::validateTrajectory(result,wrist_scene,.01);
  };
  const auto long_way=projected_path({0.,100.,200.,296.78});
  rejects([&]{cable_grasp::requireFixTCPRotationTotal(long_way,"base","tcp",M_PI/2);},"296.78 must not be wrapped to -63.22");
  require(std::abs(cable_grasp::requireFixTCPRotationTotal(projected_path({0.,-63.22}),"base","tcp",M_PI/2)*180./M_PI-63.22)<1e-7,"Negative short path is qualified");
  require(std::abs(cable_grasp::requireFixTCPRotationTotal(projected_path({0.,200.,60.}),"base","tcp",M_PI/2)*180./M_PI-60.)<1e-7,"Intermediate peak is diagnostic only; terminal 60 is qualified");
  rejects([&]{cable_grasp::requireFixTCPRotationTotal(projected_path({0.,116.78}),"base","tcp",M_PI/2);},"Terminal recorded path above 90 must be rejected");
  require(std::abs(cable_grasp::requireFixTCPRotationTotal(projected_path({0.,90.}),"base","tcp",M_PI/2)-M_PI/2)<1e-8,"Inclusive 90-degree boundary");
  // Inside model bounds but far below the previous 2% margin: must remain valid.
  auto near_limit=cable_grasp::validateTrajectory(trajectory(.90,.9999),scene,.03);
  require(near_limit.getWayPointCount()>2,"Expected interpolated samples");
  rejects([&]{cable_grasp::validateTrajectory(trajectory(.9,1.01),scene,.03);},"Must reject out-of-bounds endpoint");
  // Verify standard path/goal constraints are actually enforced.
  moveit_msgs::msg::Constraints constraints;
  moveit_msgs::msg::JointConstraint joint;
  joint.joint_name="slide";joint.position=.5;joint.tolerance_above=.01;joint.tolerance_below=.01;joint.weight=1.;
  constraints.joint_constraints.push_back(joint);
  rejects([&]{cable_grasp::validateTrajectory(trajectory(-.2,.2),scene,.03,constraints);},"Must enforce path constraints");
  rejects([&]{cable_grasp::validateTrajectory(trajectory(-.2,.2),scene,.03,moveit_msgs::msg::Constraints(), {constraints});},"Must enforce goal constraints");
  auto satisfied_goal = constraints;
  satisfied_goal.joint_constraints.front().position=.2;
  cable_grasp::validateTrajectory(trajectory(-.2,.2),scene,.03,moveit_msgs::msg::Constraints(),{satisfied_goal});
  scene.setStateFeasibilityPredicate([](const moveit::core::RobotState& state,bool){return state.getVariablePosition("slide")<.15;});
  rejects([&]{cable_grasp::validateTrajectory(trajectory(-.2,.2),scene,.03);},"Must enforce scene feasibility callback");
  planning_scene::PlanningScene obstacle_scene(model);
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id="base";object.id="middle_obstacle";object.operation=object.ADD;
  shape_msgs::msg::SolidPrimitive sphere;sphere.type=sphere.SPHERE;sphere.dimensions={.02};
  geometry_msgs::msg::Pose pose;pose.orientation.w=1.;object.primitives.push_back(sphere);object.primitive_poses.push_back(pose);
  require(obstacle_scene.processCollisionObjectMsg(object),"Obstacle insertion");
  // Endpoints are collision-free; only interpolation reveals this obstacle.
  require(!obstacle_scene.isStateColliding(trajectory(-.2,.2).getFirstWayPoint()),"Start should be collision-free");
  require(!obstacle_scene.isStateColliding(trajectory(-.2,.2).getLastWayPoint()),"Goal should be collision-free");
  rejects([&]{cable_grasp::validateTrajectory(trajectory(-.2,.2),obstacle_scene,.03);},"Must detect obstacle between endpoints");
  std::cout<<"MoveIt validation checks passed: low margin accepted, bounds/collision/constraints/feasibility rejected\n";
}
