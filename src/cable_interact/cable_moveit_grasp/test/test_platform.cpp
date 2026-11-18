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
  auto rectangle=cable_grasp::makePlatform(points,.01,true);
  if(rectangle.mesh.vertices.size()!=8 || rectangle.mesh.triangles.size()!=12) return 7;
  // Both in-plane rectangle axes are orthogonal, with consistent thickness.
  auto vertex=[&](size_t i) {const auto& q=rectangle.mesh.vertices[i]; return Eigen::Vector3d(q.x,q.y,q.z);};
  if(std::abs((vertex(1)-vertex(0)).dot(vertex(3)-vertex(0)))>1e-10) return 8;
  for(size_t i=0;i<4;++i)
    if(std::abs((vertex(i+4)-vertex(i)).norm()-.01)>1e-10) return 9;
  std::map<std::pair<unsigned,unsigned>,std::pair<int,int>> rect_edges;
  for(const auto& t:rectangle.mesh.triangles) for(int j=0;j<3;++j) {
    unsigned a=t.vertex_indices[j],b=t.vertex_indices[(j+1)%3];
    auto& edge=rect_edges[std::minmax(a,b)]; ++edge.first; edge.second+=a<b ? 1 : -1;
  }
  for(const auto& edge:rect_edges) if(edge.second.first!=2 || edge.second.second!=0) return 10;
  const Eigen::Vector3d u=(vertex(1)-vertex(0)).normalized(), v=(vertex(3)-vertex(0)).normalized();
  for(size_t i=0;i<points.size()/3;++i) {
    Eigen::Vector3d q(points[3*i],points[3*i+1],points[3*i+2]);
    auto delta=q-vertex(0);
    if(delta.dot(u)<-1e-10 || delta.dot(u)>(vertex(1)-vertex(0)).norm()+1e-10 ||
       delta.dot(v)<-1e-10 || delta.dot(v)>(vertex(3)-vertex(0)).norm()+1e-10) return 11;
  }
  auto rejected=[](const std::vector<double>& q,double thickness) {
    try { cable_grasp::makePlatform(q,thickness); return false; } catch(const std::runtime_error&) {return true;}
  };
  if(!rejected({0,0,0,1,0,0,2,0,0},.01) || !rejected(points,0)) return 5;
  if(!rejected({0,0,0,1,0,1,0,0,1,1,0,0},.01)) return 6;
  std::cout << "Platform fit, closed mesh winding, thickness and invalid input checks passed\n";
}
