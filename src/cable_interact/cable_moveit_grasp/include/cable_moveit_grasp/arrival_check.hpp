#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cable_grasp {
struct ArrivalReport {
  bool fresh{true}, within{true};
  std::int64_t oldest_stamp{INT64_MAX};
  double worst_error{-1.};
  std::string details;
};
inline ArrivalReport checkArrival(const std::vector<std::string>& names,const std::vector<double>& target,
    const std::map<std::string,double>& positions,const std::map<std::string,std::int64_t>& stamps,
    std::int64_t finished,std::int64_t now,double tolerance,
    const std::map<std::string,double>& joint_tolerances={}) {
  if(names.empty() || names.size()!=target.size() || !std::isfinite(tolerance) || tolerance<=0.)
    throw std::runtime_error("Invalid arrival-check configuration");
  ArrivalReport result;std::ostringstream details;details.precision(8);
  for(size_t i=0;i<names.size();++i) {
    const auto p=positions.find(names[i]);const auto stamp=stamps.find(names[i]);
    if(p==positions.end() || stamp==stamps.end() || !std::isfinite(p->second) || !std::isfinite(target[i])) {
      result.fresh=false;result.within=false;details<<names[i]<<" missing/nonfinite; ";continue;
    }
    const auto override_tolerance=joint_tolerances.find(names[i]);
    const double joint_tolerance=override_tolerance==joint_tolerances.end() ? tolerance : override_tolerance->second;
    if(!std::isfinite(joint_tolerance) || joint_tolerance<=0.) throw std::runtime_error("Invalid per-joint arrival tolerance");
    const double age=(now-stamp->second)*1e-9;
    const double error=std::abs(p->second-target[i]);
    result.fresh=result.fresh && stamp->second>finished && age>=-.5 && age<=.5;
    result.within=result.within && error<=joint_tolerance;
    result.oldest_stamp=std::min(result.oldest_stamp,stamp->second);
    result.worst_error=std::max(result.worst_error,error);
    details<<names[i]<<" target="<<target[i]<<" measured="<<p->second<<" abs_error="<<error
           <<" tolerance="<<joint_tolerance<<" age="<<age<<"s; ";
  }
  result.details=details.str();return result;
}
class ArrivalSettling {
 public:
  explicit ArrivalSettling(double seconds): duration_(0) {
    if(!std::isfinite(seconds) || seconds<0. || seconds>600.) throw std::runtime_error("Invalid settling duration");
    duration_=static_cast<std::int64_t>(seconds*1e9);
  }
  bool update(const ArrivalReport& sample) {
    if(!sample.fresh || !sample.within) {since_=-1;return false;}
    if(since_<0) since_=sample.oldest_stamp;
    return sample.oldest_stamp-since_>=duration_;
  }
 private:
  std::int64_t duration_,since_{-1};
};
}
