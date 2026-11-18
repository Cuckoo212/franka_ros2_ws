#include <cable_moveit_grasp/grasp_retry.hpp>
#include <cable_moveit_grasp/fixed_y_selection.hpp>
#include <iostream>
int main() {
  Eigen::Isometry3d start=Eigen::Isometry3d::Identity();
  start.translation()=Eigen::Vector3d(.2,.3,.4);
  start.linear()=Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d::UnitY()).toRotationMatrix();
  auto down=cable_grasp::graspRetryTarget(start,1),up=cable_grasp::graspRetryTarget(start,2);
  if(!cable_grasp::graspRetryTarget(start,0).matrix().isApprox(start.matrix())) return 1;
  if(!(down.translation()-start.translation()).isApprox(Eigen::Vector3d(-.01,0.,0.))) return 2;
  if(!(up.translation()-down.translation()).isApprox(Eigen::Vector3d(.02,0.,0.))) return 3;
  if(!(up.translation()-start.translation()).isApprox(Eigen::Vector3d(.01,0.,0.))) return 4;
  if(!down.linear().isApprox(start.linear()) || !up.linear().isApprox(start.linear())) return 5;
  const Eigen::Vector3d anchor(.1,-.3,.7);
  auto selected=cable_grasp::selectPositiveXPoint({{.15,-.3,.7},{.15,.2,.7},{.1,-.3,.7}},anchor,.05,.01);
  if(!selected.isApprox(Eigen::Vector3d(.15,-.3,.7))) return 6;
  bool rejected=false;try {cable_grasp::graspRetryTarget(start,3);} catch(const std::runtime_error&) {rejected=true;}
  if(!rejected) return 7;
  std::cout<<"Local-TCP retry offsets, fixed orientation and B recovery selection passed\n";
}
