#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>
namespace cable_grasp {
inline Eigen::Vector3d selectFixedYPoint(const std::vector<Eigen::Vector3d>& points,double y,double tolerance,
                                       const Eigen::Vector3d& home) {
  if(!std::isfinite(y) || !std::isfinite(tolerance) || tolerance<=0. || !home.allFinite())
    throw std::runtime_error("Invalid fixed-Y selection parameters");
  double distance=std::numeric_limits<double>::infinity(),error=distance;Eigen::Vector3d best;
  for(const auto& point:points) {
    if(!point.allFinite() || std::abs(point.y()-y)>tolerance+1e-12) continue;
    const double d=(point-home).squaredNorm(),e=std::abs(point.y()-y);
    if(d<distance || (d==distance && e<error)) {distance=d;error=e;best=point;}
  }
  if(!std::isfinite(distance)) throw std::runtime_error("No cable point within fixed Y tolerance; no grasp plan generated");
  return best;
}
inline Eigen::Vector3d selectPositiveXPoint(const std::vector<Eigen::Vector3d>& points,const Eigen::Vector3d& anchor,
                                          double offset,double tolerance) {
  if(!anchor.allFinite() || !std::isfinite(offset) || offset<=0. || !std::isfinite(tolerance) || tolerance<=0. || tolerance>=offset)
    throw std::runtime_error("Invalid A positive-X point selection parameters");
  const Eigen::Vector3d target=anchor+Eigen::Vector3d(offset,0.,0.);
  double best=std::numeric_limits<double>::infinity();Eigen::Vector3d selected;
  for(const auto& point:points) {
    if(!point.allFinite() || std::abs(point.x()-target.x())>tolerance) continue;
    const double distance=(point-target).squaredNorm();
    if(distance<best) {best=distance;selected=point;}
  }
  if(!std::isfinite(best)) throw std::runtime_error("No cable point near B grasp X + requested offset");
  return selected;
}

}
