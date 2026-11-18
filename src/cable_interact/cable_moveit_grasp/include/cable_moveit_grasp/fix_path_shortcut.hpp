#pragma once
#include <optional>
#include <limits>
#include <cable_moveit_grasp/path_validation.hpp>
#include <cable_moveit_grasp/fix_projection.hpp>
namespace cable_grasp {
struct FixShortcutReport {double before=0.,after=0.;size_t checks=0;};
// Remove unnecessary joint-space detours, never relaxing collision or TCP bounds.
// All accepted edges and the resulting whole path are densely checked in the same scene.
inline std::optional<robot_trajectory::RobotTrajectory> shortcutFixRotation(
    const robot_trajectory::RobotTrajectory& original,const planning_scene::PlanningScene& scene,
    const moveit_msgs::msg::Constraints& constraints,const std::string& frame,const std::string& tcp,double limit,size_t budget,FixShortcutReport& report) {
  if(frame.empty() || tcp.empty() || !std::isfinite(limit) || limit<=0.)
    throw std::runtime_error("Invalid FIX shortcut frame/link or cumulative rotation limit");
  auto cost=[&](const robot_trajectory::RobotTrajectory& path) {
    const auto sampled=validateTrajectory(path,scene,.01,constraints);
    return requireFixTCPRotationTotal(sampled,frame,tcp,std::numeric_limits<double>::max());
  };
  report={};report.before=report.after=cost(original);
  if(report.before<=limit+1e-6 || budget==0) return std::nullopt;
  robot_trajectory::RobotTrajectory result(original.getRobotModel(),original.getGroupName());
  result.addSuffixWayPoint(original.getFirstWayPoint(),0.);
  size_t index=0;const size_t last=original.getWayPointCount()-1;
  while(index<last) {
    bool found=false;
    const size_t stride=std::max<size_t>(1,(last-index)/8);
    for(size_t target=last;target>index && report.checks<budget;) {
      auto prefix=result;
      prefix.addSuffixWayPoint(original.getWayPoint(target),1.);
      ++report.checks;
      try {
        (void)cost(prefix); // Validate the whole prefix, but NEVER limit its intermediate accumulated angle.
        result.addSuffixWayPoint(original.getWayPoint(target),1.);index=target;found=true;break;
      } catch(const std::runtime_error&) {} // Invalid edge: never use it as a shortcut.
      if(target<=index+1) break;
      target=target>index+stride ? target-stride : index+1;
    }
    if(!found) return std::nullopt;
  }
  const double final_cost=cost(result);
  if(final_cost>limit+1e-6 || final_cost>=report.before-1e-8) return std::nullopt;
  report.after=final_cost;
  return result;
}
}
