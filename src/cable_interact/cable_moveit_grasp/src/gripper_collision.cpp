#include <cable_moveit_grasp/dual_motion.hpp>
#include <cable_moveit_grasp/gripper_collision.hpp>

void DualMotion::checkGripperMotion(const std::string& arm,double width,bool grasp) {
  freshState();
  auto scene_ptr=cable_grasp::requestMotionScene(scene_client_,a_->getRobotModel(),get_parameter("camera_obstacle_enabled").as_bool());
  const auto& scene=*scene_ptr;
  auto state=scene.getCurrentState();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Require raw measurements for all 14 arm joints and all four finger joints.
    // Never rely on a re-stamped mux message or URDF default finger positions.
    for(const std::string prefix:{std::string(""),std::string("a_")}) {
      std::vector<std::string> names;
      for(int i=1;i<=7;++i) names.push_back(prefix+"fr3_joint"+std::to_string(i));
      for(int i=1;i<=2;++i) names.push_back(prefix+"fr3_finger_joint"+std::to_string(i));
      for(const auto& name:names) {
        if(!joint_positions_.count(name) || !joint_times_.count(name) ||
           (now()-joint_times_.at(name)).seconds()>1. || (now()-joint_times_.at(name)).seconds()<-.5)
          throw std::runtime_error("Gripper collision check needs fresh measurement: "+name);
        const double value=joint_positions_.at(name);
        state.setVariablePosition(name,name.find("finger")!=std::string::npos ? cable_grasp::boundedFingerPosition(state,name,value) : value);
      }
    }
  }
  state.update();
  const std::string prefix=arm=="a" ? "a_" : "";
  const auto samples=cable_grasp::validateGripperSweep(scene,state,{prefix+"fr3_finger_joint1",prefix+"fr3_finger_joint2"},width,grasp);
  report("gripper "+arm+" collision sweep passed; samples="+std::to_string(samples)+"; table/camera/full dual robot checked");
}
