#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <moveit/planning_scene/planning_scene.h>

namespace cable_grasp {
// Normalize only sub-0.1mm sensor noise at a hard stop, never a missing measurement.
inline double boundedFingerPosition(const moveit::core::RobotState& state,const std::string& name,double value) {
  if(!state.getRobotModel()->hasJointModel(name) || !std::isfinite(value))
    throw std::runtime_error("Missing/non-finite finger measurement: "+name);
  const auto& bounds=state.getRobotModel()->getVariableBounds(name);
  if(!bounds.position_bounded_ || value<bounds.min_position_-1e-4 || value>bounds.max_position_+1e-4)
    throw std::runtime_error("Finger measurement outside model bounds: "+name);
  return std::clamp(value,bounds.min_position_,bounds.max_position_);
}

// Both fingers follow the symmetric Franka Hand width command; check every
// interpolation sample against the full robot and the world, honoring the SRDF ACM.
// A force grasp may close past the requested width after missing the cable, so
// conservatively validate all the way to fully closed, not only goal.width.
inline size_t validateGripperSweep(const planning_scene::PlanningScene& scene,
    moveit::core::RobotState state,const std::array<std::string,2>& fingers,
    double width,bool force_grasp,double finger_step=.0005) {
  if(!std::isfinite(width) || width<0. || !std::isfinite(finger_step) || finger_step<=0.)
    throw std::runtime_error("Invalid gripper sweep width/resolution");
  std::array<double,2> initial{},target{};
  double delta=0.;
  for(size_t i=0;i<2;++i) {
    initial[i]=boundedFingerPosition(state,fingers[i],state.getVariablePosition(fingers[i]));
    const auto& bounds=state.getRobotModel()->getVariableBounds(fingers[i]);
    const double commanded=width/2.;
    if(commanded<bounds.min_position_ || commanded>bounds.max_position_+1e-9)
      throw std::runtime_error("Gripper target width outside model bounds");
    target[i]=force_grasp ? bounds.min_position_ : std::clamp(commanded,bounds.min_position_,bounds.max_position_);
    if(force_grasp && initial[i]+1e-4<commanded)
      throw std::runtime_error("Force grasp target is wider than measured hand; open first");
    delta=std::max(delta,std::abs(target[i]-initial[i]));
  }
  const int steps=std::max(1,static_cast<int>(std::ceil(delta/finger_step)));
  collision_detection::CollisionRequest request;
  request.contacts=true;request.max_contacts=10;request.max_contacts_per_pair=1;
  request.group_name=""; // Whole robot: include opposite hand and both arms.
  for(int sample=0;sample<=steps;++sample) {
    for(size_t i=0;i<2;++i) state.setVariablePosition(fingers[i],initial[i]+(target[i]-initial[i])*sample/steps);
    state.update();
    if(!state.satisfiesBounds()) throw std::runtime_error("Gripper sweep: robot joint bounds invalid");
    collision_detection::CollisionResult result;scene.checkCollision(request,result,state);
    if(result.collision) {
      std::string pair=result.contacts.empty() ? "unknown pair" : result.contacts.begin()->first.first+" <-> "+result.contacts.begin()->first.second;
      throw std::runtime_error("Gripper sweep collision at sample "+std::to_string(sample)+"/"+std::to_string(steps)+": "+pair+"; no gripper goal sent");
    }
  }
  return steps+1;
}
}
