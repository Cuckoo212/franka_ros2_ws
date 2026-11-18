#include <cable_moveit_grasp/gripper_collision.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <iostream>
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
int main() {
  const auto urdf=urdf::parseURDF(R"(<robot name="hand_fixture"><link name="base"/>
    <link name="finger1"><collision><geometry><sphere radius="0.001"/></geometry></collision></link>
    <joint name="finger1" type="prismatic"><parent link="base"/><child link="finger1"/><axis xyz="1 0 0"/><limit lower="0" upper="0.04" effort="10" velocity="1"/></joint>
    <link name="finger2"><collision><geometry><sphere radius="0.001"/></geometry></collision></link>
    <joint name="finger2" type="prismatic"><parent link="base"/><child link="finger2"/><origin xyz="0 0.1 0"/><axis xyz="-1 0 0"/><limit lower="0" upper="0.04" effort="10" velocity="1"/></joint>
    <link name="other_arm"><collision><geometry><sphere radius="0.001"/></geometry></collision></link>
    <joint name="other_arm" type="prismatic"><parent link="base"/><child link="other_arm"/><axis xyz="1 0 0"/><limit lower="0" upper="0.3" effort="10" velocity="1"/></joint></robot>)");
  auto srdf=std::make_shared<srdf::Model>();require(srdf->initString(*urdf,"<robot name='hand_fixture'/>"),"SRDF");
  auto model=std::make_shared<moveit::core::RobotModel>(urdf,srdf);
  planning_scene::PlanningScene scene(model);
  moveit::core::RobotState state(model);state.setToDefaultValues();
  state.setVariablePosition("finger1",.005);state.setVariablePosition("finger2",.005);state.setVariablePosition("other_arm",.2);state.update();
  const std::array<std::string,2> fingers{"finger1","finger2"};
  require(cable_grasp::validateGripperSweep(scene,state,fingers,.08,false)>2,"Free opening must pass");
  moveit_msgs::msg::CollisionObject object;object.id="realsense_camera";object.header.frame_id="base";object.operation=object.ADD;
  shape_msgs::msg::SolidPrimitive sphere;sphere.type=sphere.SPHERE;sphere.dimensions={.002};
  geometry_msgs::msg::Pose pose;pose.orientation.w=1.;pose.position.x=.02;
  object.primitives.push_back(sphere);object.primitive_poses.push_back(pose);require(scene.processCollisionObjectMsg(object),"World object");
  bool rejected=false;
  try {cable_grasp::validateGripperSweep(scene,state,fingers,.08,false);} catch(const std::runtime_error& e) {rejected=std::string(e.what()).find("realsense_camera")!=std::string::npos;}
  require(rejected,"Opening must catch an obstacle between safe start/end finger positions");
  object.operation=object.REMOVE;require(scene.processCollisionObjectMsg(object),"Remove world obstacle");
  state.setVariablePosition("other_arm",.02);state.update();rejected=false;
  try {cable_grasp::validateGripperSweep(scene,state,fingers,.08,false);} catch(const std::runtime_error& e) {rejected=std::string(e.what()).find("other_arm")!=std::string::npos;}
  require(rejected,"Hand sweep must include the opposite arm outside a hand planning group");
  state.setVariablePosition("finger1",.04);state.setVariablePosition("finger2",.04);state.setVariablePosition("other_arm",0.);state.update();
  require(cable_grasp::validateGripperSweep(scene,state,fingers,.01,false)>2,"Move only to requested width must pass");
  rejected=false;try {cable_grasp::validateGripperSweep(scene,state,fingers,.01,true);} catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"Force grasp must check beyond requested width through complete closure");
  require(cable_grasp::boundedFingerPosition(state,"finger1",-.000003)==0.,"Tiny hard-stop measurement noise is bounded");
  rejected=false;try {cable_grasp::boundedFingerPosition(state,"finger1",-.001);} catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"Large invalid measurements must not be clamped");
  std::cout<<"Gripper sweep: world collision, inter-arm collision, free opening, full closure and bounds passed\n";
}
