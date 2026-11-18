#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <shape_msgs/msg/mesh.hpp>

namespace cable_grasp {
struct Platform {
  shape_msgs::msg::Mesh mesh;
  Eigen::Vector3d center, normal;
  double max_residual;
};
// Ordered convex polygon, projected onto its least-squares plane and extruded
// symmetrically. Reject invalid input rather than silently changing its boundary.
inline Platform makePlatform(const std::vector<double>& xyz, double thickness, bool rectangle = false) {
  if (xyz.size()<9 || xyz.size()%3 || !std::isfinite(thickness) || thickness<=0.)
    throw std::runtime_error("Invalid platform points or thickness");
  const size_t count=xyz.size()/3;
  Eigen::MatrixXd points(count,3);
  for (size_t i=0;i<count;++i) for (size_t j=0;j<3;++j) {
    if (!std::isfinite(xyz[3*i+j])) throw std::runtime_error("Non-finite platform point");
    points(i,j)=xyz[3*i+j];
  }
  Platform result; result.center=points.colwise().mean();
  Eigen::MatrixXd centered=points.rowwise()-result.center.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered,Eigen::ComputeFullV);
  if (svd.singularValues()[1]<1e-6) throw std::runtime_error("Collinear platform points");
  result.normal=svd.matrixV().col(2);
  if (result.normal.y()<0.) result.normal=-result.normal;
  result.max_residual=(centered*result.normal).cwiseAbs().maxCoeff();
  if (result.max_residual>0.01) throw std::runtime_error("Platform points deviate from plane by more than 10 mm");
  std::vector<Eigen::Vector3d> projected;
  for (size_t i=0;i<count;++i) {
    Eigen::Vector3d p=points.row(i).transpose();
    projected.push_back(p-result.normal.dot(p-result.center)*result.normal);
  }
  if (rectangle) {
    // Stable in-plane axes aligned to base X, rather than an arbitrary PCA yaw.
    Eigen::Vector3d u=Eigen::Vector3d::UnitX()-result.normal.x()*result.normal;
    if (u.norm()<1e-6) u=Eigen::Vector3d::UnitZ()-result.normal.z()*result.normal;
    u.normalize();
    const Eigen::Vector3d v=result.normal.cross(u).normalized();
    double u0=INFINITY,u1=-INFINITY,v0=INFINITY,v1=-INFINITY;
    for (const auto& point:projected) {
      const Eigen::Vector3d d=point-result.center;
      u0=std::min(u0,d.dot(u)); u1=std::max(u1,d.dot(u));
      v0=std::min(v0,d.dot(v)); v1=std::max(v1,d.dot(v));
    }
    projected={result.center+u0*u+v0*v,result.center+u1*u+v0*v,
               result.center+u1*u+v1*v,result.center+u0*u+v1*v};
  }
  const size_t boundary_count=projected.size();
  double area=0.;
  for (size_t i=0;i<boundary_count;++i)
    area+=result.normal.dot((projected[i]-result.center).cross(projected[(i+1)%boundary_count]-result.center));
  if (std::abs(area)<1e-8) throw std::runtime_error("Degenerate platform polygon");
  if (area<0.) std::reverse(projected.begin(),projected.end());
  // Every vertex must be on the left of each oriented edge; this also rejects
  // crossed boundaries and duplicate vertices for this convex mesh builder.
  for (size_t i=0;i<boundary_count;++i) {
    const auto edge=projected[(i+1)%boundary_count]-projected[i];
    if (edge.norm()<1e-6) throw std::runtime_error("Duplicate platform point");
    for (size_t j=0;j<boundary_count;++j)
      if (result.normal.dot(edge.cross(projected[j]-projected[i])) < -1e-8)
        throw std::runtime_error("Platform boundary must be ordered and convex");
  }
  for (int layer=0;layer<2;++layer) for (const auto& p:projected) {
    const Eigen::Vector3d v=p+(layer ? 0.5 : -0.5)*thickness*result.normal;
    geometry_msgs::msg::Point point; point.x=v.x(); point.y=v.y(); point.z=v.z();
    result.mesh.vertices.push_back(point);
  }
  auto triangle=[&](size_t a,size_t b,size_t c) {
    shape_msgs::msg::MeshTriangle t;
    t.vertex_indices={static_cast<uint32_t>(a),static_cast<uint32_t>(b),static_cast<uint32_t>(c)};
    result.mesh.triangles.push_back(t);
  };
  for (size_t i=1;i+1<boundary_count;++i) {
    triangle(0,i+1,i); triangle(boundary_count,boundary_count+i,boundary_count+i+1);
  }
  for (size_t i=0;i<boundary_count;++i) {
    const size_t next=(i+1)%boundary_count;
    triangle(i,next,boundary_count+next); triangle(i,boundary_count+next,boundary_count+i);
  }
  return result;
}
}
