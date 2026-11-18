#pragma once
#include <cable_moveit_grasp/geometry.hpp>
#include <array>
#include <functional>
#include <algorithm>
#include <map>
#include <numeric>
#include <limits>

namespace cable_grasp {
struct ContinuousCableSelection {
  Eigen::Vector3d point;
  double span;
  size_t support_points;
};
// Discrete continuity: voxel representatives are connected only by 3D edges
// <= max_gap. A component must contain two measured points >= min_length apart
// (a conservative lower bound on connected cable length, not sum of branch edges).
// Select the first original point in increasing X whose local tangent is reliable.
inline ContinuousCableSelection selectContinuousCablePoint(
    const std::vector<Eigen::Vector3d>& points,double x_min,double x_max,
    double min_length,double max_gap,double voxel,double local_radius,
    size_t min_local_points,double pca_ratio,double max_line_offset,
    const std::function<bool(const Eigen::Vector3d&)>& acceptable_position = {}) {
  if(!std::isfinite(x_min) || !std::isfinite(x_max) || x_max<x_min ||
     !std::isfinite(min_length) || min_length<=0. || !std::isfinite(max_gap) || max_gap<=0. || max_gap>=min_length ||
     !std::isfinite(voxel) || voxel<=0. || voxel>max_gap || !std::isfinite(local_radius) || local_radius<=0. ||
     min_local_points<6 || !std::isfinite(pca_ratio) || pca_ratio<=0. || !std::isfinite(max_line_offset) || max_line_offset<=0.)
    throw std::runtime_error("Invalid continuous-cable X interval/quality parameters");
  using Key=std::array<long long,3>;
  auto key=[](const Eigen::Vector3d& p,double cell) -> Key {
    return {static_cast<long long>(std::floor(p.x()/cell)),static_cast<long long>(std::floor(p.y()/cell)),static_cast<long long>(std::floor(p.z()/cell))};
  };
  std::vector<size_t> ordered;std::vector<Eigen::Vector3d> nodes;
  std::map<Key,size_t> voxels;
  std::map<Key,std::vector<size_t>> local_grid,graph_grid;
  // Local tangent uses all finite cloud points, matching the downstream planner.
  for(size_t i=0;i<points.size();++i) {
    const auto& p=points[i];if(!p.allFinite() || p.cwiseAbs().maxCoeff()>1e4) continue;
    local_grid[key(p,local_radius)].push_back(i);
    if(p.x()<x_min-1e-12 || p.x()>x_max+1e-12) continue;
    ordered.push_back(i);
    if(voxels.emplace(key(p,voxel),nodes.size()).second) nodes.push_back(p);
  }
  if(ordered.empty()) throw std::runtime_error("No finite cable points in requested X search interval");
  for(size_t i=0;i<nodes.size();++i) graph_grid[key(nodes[i],max_gap)].push_back(i);
  const size_t unknown=std::numeric_limits<size_t>::max();
  std::vector<size_t> component(nodes.size(),unknown);
  std::vector<double> spans;
  std::vector<size_t> counts;
  for(size_t root=0;root<nodes.size();++root) {
    if(component[root]!=unknown) continue;
    const size_t id=spans.size();std::vector<size_t> queue{root};component[root]=id;
    for(size_t head=0;head<queue.size();++head) {
      const size_t i=queue[head];const auto center=key(nodes[i],max_gap);
      for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) for(int dz=-1;dz<=1;++dz) {
        auto cell=graph_grid.find({center[0]+dx,center[1]+dy,center[2]+dz});
        if(cell==graph_grid.end()) continue;
        for(size_t j:cell->second) if(component[j]==unknown && (nodes[i]-nodes[j]).norm()<=max_gap+1e-12) {
          component[j]=id;queue.push_back(j);
        }
      }
    }
    auto farthest=[&](size_t anchor) {
      size_t far=anchor;double distance=0.;
      for(size_t i:queue) if((nodes[i]-nodes[anchor]).squaredNorm()>distance) {far=i;distance=(nodes[i]-nodes[anchor]).squaredNorm();}
      return far;
    };
    const size_t a=farthest(root),b=farthest(a);
    spans.push_back((nodes[a]-nodes[b]).norm());counts.push_back(queue.size());
  }
  std::stable_sort(ordered.begin(),ordered.end(),[&](size_t a,size_t b){return points[a].x()<points[b].x();});
  for(size_t index:ordered) {
    const auto& selected=points[index];const size_t id=component[voxels.at(key(selected,voxel))];
    if(spans[id]+1e-12<min_length) continue;
    if(acceptable_position && !acceptable_position(selected)) continue;
    std::vector<Eigen::Vector3d> local;const auto center=key(selected,local_radius);
    for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) for(int dz=-1;dz<=1;++dz) {
      auto cell=local_grid.find({center[0]+dx,center[1]+dy,center[2]+dz});
      if(cell==local_grid.end()) continue;
      for(size_t i:cell->second) if((points[i]-selected).norm()<=local_radius) local.push_back(points[i]);
    }
    if(local.size()<min_local_points) continue;
    try {estimateTangent(local,pca_ratio,max_line_offset,selected);}
    catch(const std::runtime_error&) {continue;}
    return {selected,spans[id],counts[id]};
  }
  throw std::runtime_error("No reliable cable point: need a connected >=5cm segment a valid local tangent and any required position clearance in X search interval");
}
}
