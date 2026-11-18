#include <cable_moveit_grasp/arrival_check.hpp>
#include <iostream>
int main() {
  using namespace cable_grasp;
  const std::vector<std::string> names={"a_fr3_joint2","a_fr3_joint1"};
  const std::vector<double> target={-1.,.5};
  std::map<std::string,double> p={{"a_fr3_joint1",.5},{"a_fr3_joint2",-1.}};
  std::map<std::string,std::int64_t> stamps={{"a_fr3_joint1",1000000000},{"a_fr3_joint2",1000000000}};
  ArrivalSettling settling(.1);
  auto sample=checkArrival(names,target,p,stamps,1000000000,1000000000,.02);
  if(sample.fresh || settling.update(sample)) return 1; // Cached state from before completion cannot pass.
  for(auto& s:stamps) s.second=1010000000;
  sample=checkArrival(names,target,p,stamps,1000000000,1010000000,.02);
  if(!sample.fresh || !sample.within || settling.update(sample)) return 2;
  if(settling.update(sample)) return 3; // Re-reading the same sample cannot simulate settling.
  for(auto& s:stamps) s.second=1110000000;
  sample=checkArrival(names,target,p,stamps,1000000000,1110000000,.02);
  if(!settling.update(sample)) return 4;
  p["a_fr3_joint2"]=-.96;
  sample=checkArrival(names,target,p,stamps,1000000000,1110000000,.02);
  if(sample.within || settling.update(sample) || sample.worst_error<.039) return 5;
  if(sample.details.find("a_fr3_joint2 target=-1 measured=-0.96")==std::string::npos) return 6;
  p["a_fr3_joint2"]=-1.;
  sample=checkArrival(names,target,p,stamps,1000000000,2000000000,.02);
  if(sample.fresh || settling.update(sample)) return 7;
  stamps.erase("a_fr3_joint2");
  sample=checkArrival(names,target,p,stamps,1000000000,1110000000,.02);
  if(sample.fresh || sample.within) return 8;
  const std::vector<std::string> a_names={"a_fr3_joint1","a_fr3_joint2","a_fr3_joint3","a_fr3_joint4","a_fr3_joint5","a_fr3_joint6","a_fr3_joint7"};
  const std::vector<double> a_goal={-.29641062,-1.4727197,.50501399,-2.0452574,1.9312703,1.6803881,.075845883};
  const std::vector<double> a_measured={-.29701,-1.4730461,.50572693,-2.0474389,1.9246041,1.6769445,.054482248};
  std::map<std::string,double> a_positions;
  std::map<std::string,std::int64_t> a_stamps;
  for(size_t i=0;i<a_names.size();++i) {a_positions[a_names[i]]=a_measured[i];a_stamps[a_names[i]]=1110000000;}
  auto strict=checkArrival(a_names,a_goal,a_positions,a_stamps,1000000000,1110000000,.02);
  if(strict.within) return 9;
  auto targeted=checkArrival(a_names,a_goal,a_positions,a_stamps,1000000000,1110000000,.02,{{"a_fr3_joint7",.025}});
  if(!targeted.within) return 10;
  a_positions["a_fr3_joint5"]=a_goal[4]-.03;
  if(checkArrival(a_names,a_goal,a_positions,a_stamps,1000000000,1110000000,.02,{{"a_fr3_joint7",.025}}).within) return 11;
  std::cout<<"Arrival requires fresh post-execution samples, settling and explicit per-joint tolerances; HOME/FIX diagnostics verified\n";
}
