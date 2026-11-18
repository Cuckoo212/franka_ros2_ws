#pragma once
#include <stdexcept>
#include <string>
#include <vector>

namespace cable_grasp {
// Manual HOME always opens first, regardless of the internal holding record.
inline std::vector<std::string> manualOperations(const std::string& command,
                                                 bool /*holding*/, bool /*other_holding*/) {
  if(command=="grasp") return {"grasp","verify"};
  if(command=="fix") return {"fix"};
  if(command=="home") return {"open_for_home","home"};
  throw std::runtime_error("Unknown manual arm command");
}
}
