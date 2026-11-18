#include <cable_moveit_grasp/dual_motion.hpp>
void DualMotion::executeVerifySkill(const std::string& arm) {
  if(!holding_[arm]) throw std::runtime_error("No successful grasp recorded");
  // Action-result evidence only; slip/vision verification is not implied.
}
