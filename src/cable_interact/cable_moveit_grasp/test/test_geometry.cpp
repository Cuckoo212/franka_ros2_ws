#include <cable_moveit_grasp/geometry.hpp>
#include <iostream>
void require(bool value) { if(!value) throw std::runtime_error("geometry assertion failed"); }
int main() {
  std::vector<Eigen::Vector3d> line;
  for(int i=-10;i<=10;++i) line.emplace_back(i*.001,0,0);
  auto t=cable_grasp::estimateTangent(line,4.,.005,Eigen::Vector3d::Zero());
  require(std::abs(t.dot(Eigen::Vector3d::UnitX()))>0.999);
  for(int sign:{1,-1}) for(int degrees=0;degrees<360;degrees+=10) {
    auto r=cable_grasp::candidateRotation(t,sign,degrees*3.141592653589793/180.);
    require((r.col(0)-sign*t).norm()<1e-9);
    require((r.transpose()*r-Eigen::Matrix3d::Identity()).norm()<1e-9);
    require(std::abs(r.determinant()-1.)<1e-9);
    Eigen::Vector3d p(.4,.1,.2);
    auto entry=(p-.04*r.col(2)).eval();
    require(std::abs((p-entry).norm()-.04)<1e-9);
  }
  auto r=cable_grasp::candidateRotation(Eigen::Vector3d::UnitZ(),1,0.);
  require(std::abs(r.determinant()-1.)<1e-9);
  auto crossing=line;
  for(int i=-10;i<=10;++i) crossing.emplace_back(0,i*.001,0);
  bool rejected=false;
  try { cable_grasp::estimateTangent(crossing,4.,.005,Eigen::Vector3d::Zero()); }
  catch(const std::runtime_error&) {rejected=true;}
  require(rejected);
  rejected=false;
  try { cable_grasp::estimateTangent(line,4.,.005,Eigen::Vector3d(0,.02,0)); }
  catch(const std::runtime_error&) {rejected=true;}
  require(rejected);
  require(std::abs(cable_grasp::normalizedMargin(0.,-2.,2.)-.5)<1e-9);
  require(cable_grasp::normalizedMargin(1.99,-2.,2.)<.01);
  // TCP-to-link offset rotates with each candidate, preserving the intended TCP.
  Eigen::Isometry3d tcp=Eigen::Isometry3d::Identity(),tool=Eigen::Isometry3d::Identity();
  tcp.linear()=cable_grasp::candidateRotation(Eigen::Vector3d::UnitX(),1,1.);
  tool.translation()=Eigen::Vector3d(0,0,.1);
  require(((tcp*tool)*tool.inverse()).matrix().isApprox(tcp.matrix()));
  for (int sign : {1,-1}) {
    const auto upright=cable_grasp::boardNormalRotation(Eigen::Vector3d(1.,.2,.5),sign);
    require(upright.col(2).isApprox(Eigen::Vector3d(0.,-1.,0.)));
    require(std::abs(upright.col(0).y())<1e-12);
    require((upright.transpose()*upright).isApprox(Eigen::Matrix3d::Identity()));
    require(std::abs(upright.determinant()-1.)<1e-12);
    require(upright.col(0).isApprox(sign*Eigen::Vector3d(1.,0.,.5).normalized()));
    // Entry lies above target in base +Y for a -Y approach.
    require((Eigen::Vector3d::Zero()-.04*upright.col(2)).isApprox(Eigen::Vector3d(0.,.04,0.)));
  }
  const Eigen::Vector3d center(0.,-.5,0.),normal(0.,1.,0.);
  require(cable_grasp::needsBoardNormalGrasp(cable_grasp::boardHeightGap(
      Eigen::Vector3d(.2,-.485,.3),center,normal,.01),.01));
  require(!cable_grasp::needsBoardNormalGrasp(cable_grasp::boardHeightGap(
      Eigen::Vector3d(.2,-.4849,.3),center,normal,.01),.01));
  const Eigen::Vector3d tilted=Eigen::Vector3d(.02,1.,.03).normalized();
  const Eigen::Vector3d face=center+.005*tilted;
  require(std::abs(cable_grasp::boardHeightGap(face,center,tilted,.01))<1e-12);
  rejected=false;
  try { cable_grasp::boardNormalRotation(Eigen::Vector3d::UnitY(),1); }
  catch (const std::runtime_error&) {rejected=true;}
  require(rejected);
  const Eigen::Vector3d picked(.2,-.51,.6);
  for (double gap : {0.,.007999}) {
    const auto raised=cable_grasp::raisedBoardTarget(picked,gap,.008,.005);
    require(raised.isApprox(picked+Eigen::Vector3d(0.,.005,0.)));
    require(raised.x()==picked.x() && raised.z()==picked.z());
    require(cable_grasp::needsBoardNormalGrasp(gap,.01));
  }
  for (double gap : {.008,.008001,.02})
    require(cable_grasp::raisedBoardTarget(picked,gap,.008,.005).isApprox(picked));
  // Original measured point is unchanged and repeated generation never accumulates offsets.
  require(picked.isApprox(Eigen::Vector3d(.2,-.51,.6)));
  require(cable_grasp::raisedBoardTarget(picked,.003,.008,.005).isApprox(
      cable_grasp::raisedBoardTarget(picked,.003,.008,.005)));
  std::cout<<"Geometry checks passed\n";
}
