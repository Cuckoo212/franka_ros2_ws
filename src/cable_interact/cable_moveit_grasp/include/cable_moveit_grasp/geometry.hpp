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

inline double normalizedMargin(double q, double lower, double upper) {
  if (!(upper > lower)) throw std::runtime_error("Invalid joint bounds");
  return std::min(q-lower, upper-q)/(upper-lower);
}
}  // namespace cable_grasp
