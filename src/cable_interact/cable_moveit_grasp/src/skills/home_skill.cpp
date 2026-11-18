#include <cable_moveit_grasp/dual_motion.hpp>
void DualMotion::executeHomeSkill(const std::string& arm,MGI& group) {
  const auto recorded=get_parameter(arm=="a" ? "homeA" : "homeB").as_double_array();
  if(recorded.size()!=7) throw std::runtime_error("Configure seven HOME joints");
  for(double q:recorded) if(!std::isfinite(q)) throw std::runtime_error("Non-finite HOME joint");
  auto current=group.getCurrentState(3.);
  if(!current) throw std::runtime_error("Missing HOME current state");
  group.setStartState(*current);group.clearPoseTargets();
  if(!group.setJointValueTarget(recorded)) throw std::runtime_error("HOME joint target out of bounds");
  const auto& names=current->getJointModelGroup(group.getName())->getVariableNames();
  bool already=true;
  for(size_t i=0;i<names.size();++i) already=already && std::abs(current->getVariablePosition(names[i])-recorded[i])<=.001;
  if(!already) execute(group);
  else report("already at HOME; no trajectory required");
  home_[arm]=true;
}
