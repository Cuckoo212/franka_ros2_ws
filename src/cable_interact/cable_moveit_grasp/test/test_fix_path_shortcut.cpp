#include <iostream>
#include <functional>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <cable_moveit_grasp/fix_path_shortcut.hpp>
void require(bool value,const char* reason) {if(!value) throw std::runtime_error(reason);}
int main() {
  const Eigen::Vector3d fix_position(.2,-.3,.6);
  const Eigen::Matrix3d initial_rotation=Eigen::AngleAxisd(.4,Eigen::Vector3d::UnitX()).toRotationMatrix();
  const auto position_candidates=cable_grasp::fixPositionCandidates(fix_position,initial_rotation);
  require(position_candidates.size()==34,"Bounded adaptive orientation set");
  require(position_candidates.front().linear().isApprox(initial_rotation),"First FIX candidate must preserve current orientation");
  for(const auto& candidate:position_candidates) {
    require(candidate.translation().isApprox(fix_position),"Every adaptive orientation must retain recorded position");
    require((candidate.linear().transpose()*candidate.linear()).isApprox(Eigen::Matrix3d::Identity()),"Candidate rotation must be orthonormal");
  }

  auto urdf=urdf::parseURDF(R"(<robot name="shortcut_fixture"><link name="world"/><link name="base"/><joint name="mount" type="fixed"><parent link="world"/><child link="base"/><origin rpy="1.5707963267948966 0 0" xyz="0.2 0.1 0.3"/></joint><link name="x_link"/><link name="y_link"/>
    <link name="wrist"/><link name="tcp"><collision><geometry><sphere radius="0.03"/></geometry></collision></link>
    <joint name="x" type="prismatic"><parent link="base"/><child link="x_link"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="1" effort="10"/></joint>
    <joint name="y" type="prismatic"><parent link="x_link"/><child link="y_link"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="1" effort="10"/></joint>
    <joint name="turn" type="revolute"><parent link="y_link"/><child link="wrist"/><axis xyz="0 1 0"/><limit lower="-3.2" upper="3.2" velocity="1" effort="10"/></joint><joint name="tool" type="fixed"><parent link="wrist"/><child link="tcp"/><origin rpy="1.5707963267948966 0 0" xyz="0 0 0"/></joint></robot>)");
  auto srdf=std::make_shared<srdf::Model>();
  require(srdf->initString(*urdf,"<robot name='shortcut_fixture'><group name='arm'><chain base_link='base' tip_link='tcp'/></group></robot>"),"SRDF");
  auto model=std::make_shared<moveit::core::RobotModel>(urdf,srdf);
  planning_scene::PlanningScene scene(model);
  robot_trajectory::RobotTrajectory path(model,"arm");
  moveit::core::RobotState state(model);state.setToDefaultValues();
  for(const auto& q:std::vector<std::vector<double>>{{-.2,0.,0.},{-.2,.2,3.2},{.2,.2,0.},{.2,0.,0.}}) {
    state.setJointGroupPositions("arm",q);state.update();path.addSuffixWayPoint(state,1.);
  }
  moveit_msgs::msg::Constraints constraints;
  // Empty constraints: no intermediate orientation restriction; cumulative checks remain.
  cable_grasp::FixShortcutReport report;
  auto direct=cable_grasp::shortcutFixRotation(path,scene,constraints,"base","tcp",M_PI/2,32,report);
  require(!direct && report.before<1e-8,"Final zero must pass unchanged despite intermediate 3.2 radians");
  moveit_msgs::msg::CollisionObject object;object.header.frame_id="base";object.id="blocked_direct";object.operation=object.ADD;
  shape_msgs::msg::SolidPrimitive sphere;sphere.type=sphere.SPHERE;sphere.dimensions={.02};
  geometry_msgs::msg::Pose pose;pose.orientation.w=1.;object.primitives.push_back(sphere);object.primitive_poses.push_back(pose);
  require(scene.processCollisionObjectMsg(object),"Obstacle insertion");
  const auto checked=cable_grasp::validateTrajectory(path,scene,.01,constraints);
  cable_grasp::requireFixTCPRotationTotal(checked,"base","tcp",M_PI/2);
  robot_trajectory::RobotTrajectory blocked(model,"arm");
  blocked.addSuffixWayPoint(path.getFirstWayPoint(),0.);blocked.addSuffixWayPoint(path.getLastWayPoint(),1.);
  bool collision_rejected=false;
  try {cable_grasp::validateTrajectory(blocked,scene,.01,constraints);}
  catch(const std::runtime_error&) {collision_rejected=true;}
  require(collision_rejected,"Final-angle rule must never bypass collision checks");
  auto nonzero_goal=path;auto endpoint=path.getLastWayPoint();endpoint.setVariablePosition("turn",.4);endpoint.update();
  nonzero_goal.addSuffixWayPoint(endpoint,1.);
  require(!cable_grasp::shortcutFixRotation(nonzero_goal,scene,constraints,"base","tcp",.1,32,report),"Shortcut must still reject excessive terminal angle");
  auto rotation_path=[&](const std::vector<double>& angles) {
    robot_trajectory::RobotTrajectory result(model,"arm");
    auto sample=state;
    for(double degrees:angles) {
      sample.setJointGroupPositions("arm",std::vector<double>{-.5,.5,degrees*M_PI/180.});sample.update();
      result.addSuffixWayPoint(sample,1.);
    }
    return cable_grasp::validateTrajectory(result,scene,.01,constraints);
  };
  const auto example=rotation_path({0.,120.,20.,127.055166});
  const auto vector=cable_grasp::fixTCPVectorRotation(example,"base","tcp",M_PI);
  require(std::abs(vector.accumulated.y()*180./M_PI-127.055166)<1e-7,"Signed increments must cancel: 120-100+107.055166");
  require(std::abs(vector.accumulated.x())<1e-9 && std::abs(vector.accumulated.z())<1e-9,"Vector must be expressed in rotated link0, not world or moving TCP");
  require(std::abs(cable_grasp::requireFixTCPRotationTotal(example,"base","tcp",M_PI)*180./M_PI-127.055166)<1e-7,"327-degree scalar travel must no longer reject the example");
  const auto returned=rotation_path({0.,120.,0.});
  const auto back=cable_grasp::fixTCPVectorRotation(returned,"base","tcp",M_PI);
  require(back.accumulated.norm()<1e-9 && std::abs(back.peak-120.*M_PI/180.)<1e-8,"Reverse rotation must cancel while preserving peak");
  require(cable_grasp::requireFixTCPRotationTotal(rotation_path({0.,180.5,0.}),"base","tcp",M_PI/2)<1e-8,"Intermediate >180 degrees followed by return to zero must pass");
  const auto short_final=rotation_path({0.,120.,20.,63.22});
  require(std::abs(cable_grasp::requireFixTCPRotationTotal(short_final,"base","tcp",M_PI/2)*180./M_PI-63.22)<1e-7,"Final 63.22 degrees must pass even when peak is 120 degrees");
  bool terminal_rejected=false;
  try {cable_grasp::requireFixTCPRotationTotal(rotation_path({0.,116.78}),"base","tcp",M_PI/2);}
  catch(const std::runtime_error&) {terminal_rejected=true;}
  require(terminal_rejected,"Final 116.78 degrees must fail 90-degree terminal limit");
  require(std::abs(cable_grasp::requireFixTCPRotationTotal(rotation_path({0.,-63.22}),"base","tcp",M_PI/2)*180./M_PI-63.22)<1e-7,"Negative angle must use absolute terminal value");
  const Eigen::Matrix3d projected_start=Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d spin_about_tcp_y=projected_start*Eigen::AngleAxisd(2.5,Eigen::Vector3d::UnitY()).toRotationMatrix();
  require(std::abs(cable_grasp::tcpYAxisXZSignedIncrement(projected_start,spin_about_tcp_y))<1e-8,"Full TCP rotation about its own Y must not add projected rotation");
  bool undefined=false;
  try {cable_grasp::tcpYAxisXZSignedIncrement(Eigen::Matrix3d::Identity(),projected_start);}
  catch(const std::runtime_error&) {undefined=true;}
  require(undefined,"Undefined projection must not silently count as zero");
  std::cout<<"FIX shortcuts: preserved endpoints, reduced total rotation, obstacle avoidance, strict limit and full-sum diagnostics passed\n";
}
