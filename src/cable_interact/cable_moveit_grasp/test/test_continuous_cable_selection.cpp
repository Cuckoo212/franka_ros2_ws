#include <cable_moveit_grasp/continuous_cable_selection.hpp>
#include <iostream>
void require(bool yes,const char* message) {if(!yes) throw std::runtime_error(message);}
int main() {
  using Eigen::Vector3d;
  auto pick=[](const std::vector<Vector3d>& p,double lo,double hi) {
    return cable_grasp::selectContinuousCablePoint(p,lo,hi,.05,.006,.002,.025,12,4.,.01);
  };
  std::vector<Vector3d> cloud{{-.29,.2,.7},{-.285,.2,.7}}; // Earlier isolated fragment must not win.
  for(int i=0;i<=40;++i) cloud.emplace_back(-.2+i*.002,-.3,.7);
  for(int i=0;i<=40;++i) cloud.emplace_back(.1+i*.002,-.3,.7);
  auto selected=pick(cloud,-.3,.2);
  require(std::abs(selected.point.x()+.2)<1e-10,"Select first reliable X segment, not earlier noise or nearest HOME");
  require(selected.span>=.05,"At least 5cm measured connected extent");
  auto a=pick(cloud,.1,.4);
  require(std::abs(a.point.x()-.1)<1e-10,"A scan starts at B TCP X+10cm and uses increasing X");
  std::vector<Vector3d> fragments;
  for(int i=0;i<=10;++i) fragments.emplace_back(i*.002,0.,0.);
  for(int i=0;i<=10;++i) fragments.emplace_back(.04+i*.002,0.,0.);
  bool rejected=false;try {pick(fragments,0.,.1);} catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"Separated short fragments must not count as one 6cm cable");
  std::vector<Vector3d> blob;
  for(int i=0;i<20;++i) for(int j=0;j<20;++j) blob.emplace_back(i*.0002,j*.0002,0.);
  rejected=false;try {pick(blob,0.,.1);} catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"Dense short noise must not accumulate edge lengths to 5cm");
  rejected=false;try {pick(cloud,.1,-.3);} catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"B upper bound below -0.3 must fail closed");
  std::vector<Vector3d> sideways;
  for(int i=0;i<=40;++i) sideways.emplace_back(.15,i*.002,.7);
  require(pick(sideways,.1,.4).span>=.05,"Continuity is 3D length, not X span");
  std::vector<Vector3d> exact;
  for(int i=0;i<=25;++i) exact.emplace_back(i*.002,0.,0.);
  require(pick(exact,0.,.05).span>=.05-1e-12,"Inclusive 5cm length and X interval boundaries");
  // A valid low-X segment exactly at the near-board boundary must be skipped,
  // without removing its support points from the continuity/tangent calculation.
  std::vector<Vector3d> board_points;
  for(int i=0;i<=40;++i) board_points.emplace_back(-.2+i*.002,.015,0.);
  for(int i=0;i<=40;++i) board_points.emplace_back(i*.002,.025,0.);
  auto clear=[](const Vector3d& point) {
    return !cable_grasp::needsBoardNormalGrasp(cable_grasp::boardHeightGap(point,Vector3d::Zero(),Vector3d::UnitY(),.01),.01);
  };
  auto elevated=cable_grasp::selectContinuousCablePoint(board_points,-.3,.2,.05,.006,.002,.025,12,4.,.01,clear);
  require(std::abs(elevated.point.x())<1e-10,"B must continue increasing X after rejecting a near-board segment");
  require(clear(elevated.point),"Selected B point must not trigger near-board two-candidate branch");
  require(!clear(Vector3d(0.,.015,0.)),"Exactly 1cm surface clearance must be rejected");
  require(clear(Vector3d(0.,.015001,0.)),"Strictly above 1cm surface clearance can pass");
  rejected=false;try {
    cable_grasp::selectContinuousCablePoint(board_points,-.3,-.1,.05,.006,.002,.025,12,4.,.01,clear);
  } catch(const std::runtime_error&) {rejected=true;}
  require(rejected,"All reliable points near board must fail rather than fall back to a blocked point");
  auto clear3=[](const Vector3d& point) {
    return !cable_grasp::needsBoardNormalGrasp(cable_grasp::boardHeightGap(point,Vector3d::Zero(),Vector3d::UnitY(),.01),.03);
  };
  require(!clear3(Vector3d(0.,.025,0.)),"2cm board gap must now be skipped");
  require(!clear3(Vector3d(0.,.035,0.)),"Exactly 3cm board gap must be skipped");
  require(clear3(Vector3d(0.,.035001,0.)),"Above 3cm board gap can pass");
  std::cout<<"Continuous cable: ordered search, isolated noise, gaps, dense blob, dynamic bounds and 3D support passed\n";
}
