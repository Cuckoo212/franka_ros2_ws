#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace cable_grasp {
inline Eigen::Vector3d estimateTangent(const std::vector<Eigen::Vector3d>& points,
                                      double min_ratio, double max_offset,
                                      const Eigen::Vector3d& selected) {
  if (points.size() < 6) throw std::runtime_error("Need at least six local points");
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto& p : points) center += p;
  center /= points.size();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& p : points) covariance += (p-center)*(p-center).transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) throw std::runtime_error("PCA failed");
  const auto values = solver.eigenvalues();
  if (values[2] < 1e-12 || values[2] / std::max(values[1], 1e-12) < min_ratio)
    throw std::runtime_error("Ambiguous local cable tangent; select away from crossings");
  Eigen::Vector3d tangent = solver.eigenvectors().col(2).normalized();
  if (((selected-center) - tangent*tangent.dot(selected-center)).norm() > max_offset)
    throw std::runtime_error("Clicked point is too far from fitted local cable line");
  return tangent;
}

inline Eigen::Matrix3d candidateRotation(const Eigen::Vector3d& tangent, int sign,
                                        double angle) {
  const Eigen::Vector3d x = sign*tangent.normalized();
  Eigen::Vector3d reference = Eigen::Vector3d::UnitZ();
  if (std::abs(reference.dot(x)) > 0.9) reference = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d z = (reference-x*reference.dot(x)).normalized();
  Eigen::Matrix3d basis;
  basis.col(0) = x;
  basis.col(1) = z.cross(x).normalized();
  basis.col(2) = z;
  return basis * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
}

// Exact base -Y approach; tangent's Y component must be removed to keep a
// right-handed orthonormal TCP frame. Reject a cable pointing along approach.
inline Eigen::Matrix3d boardNormalRotation(const Eigen::Vector3d& tangent, int sign) {
  if (!tangent.allFinite() || (sign!=1 && sign!=-1))
    throw std::runtime_error("Invalid board-normal tangent/sign");
  const Eigen::Vector3d z(0.,-1.,0.);
  Eigen::Vector3d x=tangent-z*tangent.dot(z);
  if (x.norm()<1e-6) throw std::runtime_error("Cable tangent is perpendicular to board; no board-normal grasp frame");
  x=sign*x.normalized();
  Eigen::Matrix3d rotation;
  rotation.col(0)=x; rotation.col(1)=z.cross(x); rotation.col(2)=z;
  return rotation;
}
// Height difference measured along base Y to the robot-facing (+Y) box face.
inline double boardHeightGap(const Eigen::Vector3d& point, const Eigen::Vector3d& center,
                             const Eigen::Vector3d& normal, double thickness) {
  if (!point.allFinite() || !center.allFinite() || !normal.allFinite() ||
      !std::isfinite(thickness) || thickness<=0. || std::abs(normal.y())<1e-6)
    throw std::runtime_error("Invalid board height geometry");
  Eigen::Vector3d n=normal.normalized(); if (n.y()<0.) n=-n;
  return std::abs((n.dot(point-center)-thickness/2.)/n.y());
}
inline bool needsBoardNormalGrasp(double gap, double threshold) {
  if (!std::isfinite(gap) || gap<0. || !std::isfinite(threshold) || threshold<0.)
    throw std::runtime_error("Invalid board proximity threshold");
  return gap<=threshold+1e-12;
}

// Keep the measurement/PCA point intact; offset only the planned TCP target.
inline Eigen::Vector3d raisedBoardTarget(const Eigen::Vector3d& selected, double gap,
                                        double threshold, double raise) {
  if (!selected.allFinite() || !std::isfinite(gap) || gap<0. ||
      !std::isfinite(threshold) || threshold<0. || !std::isfinite(raise) || raise<0.)
    throw std::runtime_error("Invalid near-board target offset");
  Eigen::Vector3d target=selected;
  if (gap<threshold) target.y()+=raise;
  return target;
}

inline double normalizedMargin(double q, double lower, double upper) {
  if (!(upper > lower)) throw std::runtime_error("Invalid joint bounds");
  return std::min(q-lower, upper-q)/(upper-lower);
}
}  // namespace cable_grasp
