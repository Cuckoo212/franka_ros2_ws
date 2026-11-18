#include <cable_moveit_grasp/fixed_y_selection.hpp>
#include <iostream>
int main() {
  using Eigen::Vector3d;using cable_grasp::selectFixedYPoint;
  const double y=-.374;const Vector3d home(1.,0.,1.);
  const std::vector<Vector3d> points={{0.,y,0.},{.9,y+.001,1.},{1.,y+.003,1.},{NAN,y,1.}};
  auto selected=selectFixedYPoint(points,y,.002,home);
  if((selected-points[1]).norm()>1e-12) return 1; // HOME TCP distance, not origin distance or exact-Y priority.
  auto same_y=selectFixedYPoint({{0.,y,0.},{.9,y,1.}},y,.002,home);
  if(std::abs(same_y.x()-.9)>1e-12) return 2;
  bool rejected=false;try{selectFixedYPoint({{0.,y+.003,0.}},y,.002,home);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected) return 3;
  rejected=false;try{selectFixedYPoint(points,y,0.,home);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected) return 4;
  // New asymmetric expansion relative to the old center: keep -0.474,
  // extend only the upper edge to -0.174; rank by full HOME distance.
  const Vector3d new_home(0.,-.17,0.);
  auto expanded=selectFixedYPoint({{0.,-.40,0.},{0.,-.20,0.},{0.,-.173,0.},{0.,-.475,0.}},-.324,.15,new_home);
  if((expanded-Vector3d(0.,-.20,0.)).norm()>1e-12) return 7;
  for(double edge:{-.474,-.174}) {
    auto boundary=selectFixedYPoint({{0.,edge,0.}},-.324,.15,new_home);
    if(std::abs(boundary.y()-edge)>1e-12) return 8;
  }
  const Vector3d anchor(.1,-.3,.7);
  auto positive=cable_grasp::selectPositiveXPoint({{-.1,-.3,.7},{.3,-.3,.7},{.3,.2,.7},{.32,-.3,.7}},anchor,.2,.01);
  if((positive-Vector3d(.3,-.3,.7)).norm()>1e-12) return 5;
  rejected=false;try{cable_grasp::selectPositiveXPoint({{-.1,-.3,.7},{.32,-.3,.7}},anchor,.2,.01);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected) return 6;
  std::cout<<"Fixed Y: finite points, tolerance band, HOME TCP proximity and no-match rejection passed\n";
}
