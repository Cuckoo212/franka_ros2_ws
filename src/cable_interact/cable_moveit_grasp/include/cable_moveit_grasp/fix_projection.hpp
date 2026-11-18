#pragma once
#include <cable_moveit_grasp/orientation_checks.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit_msgs/msg/constraints.hpp>

namespace cable_grasp {
// Rotation matrices map TCP axes into the SAME fixed arm link0 frame.
// Column 1 is TCP +Y; XZ and ZX give the same unsigned planar angle.
inline double tcpYAxisXZProjectionAngle(const Eigen::Matrix3d& reference,const Eigen::Matrix3d& target) {
  if(!reference.allFinite() || !target.allFinite()) throw std::runtime_error("Non-finite FIX TCP orientation");
  const Eigen::Vector2d a(reference(0,1),reference(2,1));
  const Eigen::Vector2d b(target(0,1),target(2,1));
  if(a.norm()<1e-6 || b.norm()<1e-6)
    throw std::runtime_error("FIX TCP Y-axis XZ projection is near zero in link0; planar angle undefined");
  return std::acos(std::clamp(a.normalized().dot(b.normalized()),-1.,1.));
}
// Positive signed turn follows the right-hand rule about fixed link0 +Y.
inline double tcpYAxisXZSignedIncrement(const Eigen::Matrix3d& from,const Eigen::Matrix3d& to) {
  const Eigen::Vector2d a(from(0,1),from(2,1)),b(to(0,1),to(2,1));
  if(!a.allFinite() || !b.allFinite() || a.norm()<1e-6 || b.norm()<1e-6)
    throw std::runtime_error("FIX TCP Y-axis XZ projection is near zero; signed angle undefined");
  const Eigen::Vector2d u=a.normalized(),v=b.normalized();
  return std::atan2(u.y()*v.x()-u.x()*v.y(),u.dot(v));
}
struct FixOrientationCandidate {
  Eigen::Isometry3d target;
  bool flipped;
  double angle; // Unsigned TCP Y-axis XZ projection angle in arm link0.
};
inline std::vector<FixOrientationCandidate> fixOrientationCandidates(
    const Eigen::Isometry3d& nominal,const Eigen::Matrix3d& reference) {
  if(!nominal.matrix().allFinite()) throw std::runtime_error("Invalid FIX TCP target");
  std::vector<FixOrientationCandidate> result;
  for(bool flipped:{false,true}) {
    Eigen::Isometry3d target=nominal;
    if(flipped) target.linear()=nominal.linear()*Eigen::AngleAxisd(M_PI,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    result.push_back({target,flipped,tcpYAxisXZProjectionAngle(reference,target.linear())});
  }
  std::stable_sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.angle<b.angle;});
  return result;
}
// Position-only FIX: recorded orientation is deliberately not an input.
// First candidate preserves the measured orientation; later candidates rotate
// about the starting TCP local axes, in increasing angular-change order.
inline std::vector<Eigen::Isometry3d> fixPositionCandidates(
    const Eigen::Vector3d& position,const Eigen::Matrix3d& reference) {
  if(!position.allFinite() || !reference.allFinite()) throw std::runtime_error("Invalid FIX position/start orientation");
  Eigen::Isometry3d target=Eigen::Isometry3d::Identity();
  target.translation()=position;target.linear()=reference;
  std::vector<Eigen::Isometry3d> result{target};
  for(double degrees:{30.,60.,90.,120.,150.,180.}) {
    for(int axis=0;axis<3;++axis) for(int sign:{1,-1}) {
      if(degrees==180. && sign==-1) continue; // Same orientation.
      target.linear()=reference*Eigen::AngleAxisd(sign*degrees*M_PI/180.,Eigen::Vector3d::Unit(axis)).toRotationMatrix();
      result.push_back(target);
    }
  }
  return result;
}
// These three task-space routes have the same endpoint and minimal intended angular travel.
inline std::vector<Eigen::Isometry3d> fixCartesianWaypoints(const Eigen::Isometry3d& start,
    const Eigen::Isometry3d& target,int mode) {
  if(mode==0) return {target};
  Eigen::Isometry3d intermediate=mode==1 ? target : start;
  if(mode==1) intermediate.linear()=start.linear();
  else if(mode==2) intermediate.linear()=target.linear();
  else throw std::runtime_error("Unknown FIX Cartesian route");
  return {intermediate,target};
}
// Only TCP +Y projected into fixed link0 XZ is accumulated.
// Keep the old vector container for callers: its Y component is the signed planar angle.
struct FIXVectorRotation {
  Eigen::Vector3d accumulated=Eigen::Vector3d::Zero();
  double peak=0.;
  size_t first_exceeded=0;
};
inline FIXVectorRotation fixTCPVectorRotation(const robot_trajectory::RobotTrajectory& sampled,
    const std::string& frame,const std::string& tcp,double maximum) {
  if(!std::isfinite(maximum) || maximum<=0. || sampled.getWayPointCount()<2)
    throw std::runtime_error("Invalid FIX vector rotation limit or empty path");
  FIXVectorRotation result;result.first_exceeded=sampled.getWayPointCount();
  Eigen::Matrix3d previous=Eigen::Matrix3d::Identity();
  for(size_t i=0;i<sampled.getWayPointCount();++i) {
    const auto& state=sampled.getWayPoint(i);
    const Eigen::Matrix3d rotation=(state.getGlobalLinkTransform(frame).inverse()*state.getGlobalLinkTransform(tcp)).linear();
    // Check the initial projection too, including a two-point zero-motion path.
    tcpYAxisXZSignedIncrement(rotation,rotation);
    if(i) result.accumulated.y()+=tcpYAxisXZSignedIncrement(previous,rotation);
    result.peak=std::max(result.peak,result.accumulated.norm());
    if(result.accumulated.norm()>maximum+1e-6 && result.first_exceeded==sampled.getWayPointCount()) result.first_exceeded=i;
    previous=rotation;
  }
  return result;
}
// Compatibility name: ONLY the terminal unwrapped signed planar sum is limited.
inline double requireFixTCPRotationTotal(const robot_trajectory::RobotTrajectory& sampled,
    const std::string& frame,const std::string& tcp,double maximum) {
  const auto result=fixTCPVectorRotation(sampled,frame,tcp,maximum);
  const double final=std::abs(result.accumulated.y());
  if(final>maximum+1e-6)
    throw std::runtime_error("FIX final accumulated TCP-Y XZ projected angle exceeds limit: final signed="+
        std::to_string(result.accumulated.y()*180./M_PI)+" deg; final absolute="+std::to_string(final*180./M_PI)+
        " deg > "+std::to_string(maximum*180./M_PI)+" deg; diagnostic peak="+std::to_string(result.peak*180./M_PI)+" deg (not limited)");
  return final;
}

} // namespace cable_grasp
