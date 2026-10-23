#include <cable_moveit_grasp/platform.hpp>
#include <map>
#include <utility>
#include <iostream>
int main() {
  const std::vector<double> points={-.252,-.480,.422,-.234,-.491,.936,.108,-.500,.953,.380,-.506,.907,.557,-.507,.740,.644,-.504,.573,.653,-.501,.454};
  auto p=cable_grasp::makePlatform(points,.01);
  if (p.mesh.vertices.size()!=14 || p.mesh.triangles.size()!=24 || p.max_residual>.001 || p.normal.y()<.99) return 1;
  std::map<std::pair<unsigned,unsigned>,std::pair<int,int>> edges;
  for (const auto& t:p.mesh.triangles) for(int j=0;j<3;++j) {
    unsigned a=t.vertex_indices[j],b=t.vertex_indices[(j+1)%3];
    auto& e=edges[std::minmax(a,b)]; ++e.first; e.second+=a<b ? 1 : -1;
  }
  for (const auto& edge:edges) if(edge.second.first!=2 || edge.second.second!=0) return 2;
  for (size_t i=0;i<7;++i) {
    const auto& a=p.mesh.vertices[i]; const auto& b=p.mesh.vertices[i+7];
    const Eigen::Vector3d d(b.x-a.x,b.y-a.y,b.z-a.z);
    if(std::abs(d.dot(p.normal)-.01)>1e-10) return 3;
    if(std::abs(p.normal.dot(Eigen::Vector3d(a.x,a.y,a.z)-p.center)+.005)>1e-10) return 4;
  }
  auto rejected=[](const std::vector<double>& q,double thickness) {
    try { cable_grasp::makePlatform(q,thickness); return false; } catch(const std::runtime_error&) {return true;}
  };
  if(!rejected({0,0,0,1,0,0,2,0,0},.01) || !rejected(points,0)) return 5;
  if(!rejected({0,0,0,1,0,1,0,0,1,1,0,0},.01)) return 6;
  std::cout << "Platform fit, closed mesh winding, thickness and invalid input checks passed\n";
}
