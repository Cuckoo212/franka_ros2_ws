#include <iostream>
#include <functional>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <cable_moveit_grasp/path_validation.hpp>

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
