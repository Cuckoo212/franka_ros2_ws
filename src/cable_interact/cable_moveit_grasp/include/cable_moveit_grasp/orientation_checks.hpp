#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit_msgs/msg/constraints.hpp>

namespace cable_grasp {
inline double tcpOrientationAngle(const Eigen::Matrix3d& reference,const Eigen::Matrix3d& rotation) {
  Eigen::Quaterniond a(reference),b(rotation);
  if(!a.coeffs().allFinite() || !b.coeffs().allFinite() || a.norm()<1e-9 || b.norm()<1e-9)
    throw std::runtime_error("Invalid FIX TCP rotation");
  return a.normalized().angularDistance(b.normalized());
}
inline moveit_msgs::msg::OrientationConstraint fixTCPConstraint(const std::string& frame,const std::string& tcp,
    const Eigen::Matrix3d& reference,double maximum) {
  if(frame.empty() || tcp.empty() || !std::isfinite(maximum) || maximum<=0. || maximum>M_PI)
    throw std::runtime_error("Invalid FIX orientation frame/link or maximum angle");
  moveit_msgs::msg::OrientationConstraint result;
  result.header.frame_id=frame;result.link_name=tcp;
  const Eigen::Quaterniond q(reference);
  result.orientation.x=q.x();result.orientation.y=q.y();result.orientation.z=q.z();result.orientation.w=q.w();
  result.absolute_x_axis_tolerance=result.absolute_y_axis_tolerance=result.absolute_z_axis_tolerance=maximum;
  result.parameterization=result.ROTATION_VECTOR;result.weight=1.;return result;
}
inline double requireFixTCPOrientation(const robot_trajectory::RobotTrajectory& trajectory,
                                       const moveit_msgs::msg::OrientationConstraint& constraint) {
  const Eigen::Quaterniond q(constraint.orientation.w,constraint.orientation.x,constraint.orientation.y,constraint.orientation.z);
  const Eigen::Matrix3d reference=q.normalized().toRotationMatrix();
  double worst=0.;
  for(size_t i=0;i<trajectory.getWayPointCount();++i) {
    const auto& state=trajectory.getWayPoint(i);
    const Eigen::Isometry3d tcp=state.getGlobalLinkTransform(constraint.header.frame_id).inverse()*state.getGlobalLinkTransform(constraint.link_name);
    const double angle=tcpOrientationAngle(reference,tcp.linear());
    if(angle>constraint.absolute_x_axis_tolerance+1e-6)
      throw std::runtime_error("FIX TCP orientation exceeds start-reference bound at path sample "+std::to_string(i)+
                               ": "+std::to_string(angle*180./M_PI)+" deg");
    worst=std::max(worst,angle);
  }
  return worst;
}
} // namespace cable_grasp
