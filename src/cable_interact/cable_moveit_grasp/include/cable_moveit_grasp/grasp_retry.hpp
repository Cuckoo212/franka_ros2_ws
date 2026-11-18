#pragma once
#include <array>
#include <stdexcept>
#include <Eigen/Geometry>
namespace cable_grasp {
class GraspMiss : public std::runtime_error {public: using std::runtime_error::runtime_error;};
// Only raised after successful release and HOME; coordinator may request a fresh manual plan.
class GraspRecovered : public std::runtime_error {public: using std::runtime_error::runtime_error;};
inline Eigen::Isometry3d graspRetryTarget(const Eigen::Isometry3d& original,int attempt) {
  if(attempt<0 || attempt>2) throw std::runtime_error("Invalid grasp retry index");
  constexpr std::array<double,3> offsets{0.,-.01,.01};
  auto target=original;target.translation()+=original.linear().col(2)*offsets[attempt];return target;
}
}
