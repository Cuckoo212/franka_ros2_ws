#include <cable_moveit_grasp/manual_commands.hpp>
#include <iostream>
int main() {
  using cable_grasp::manualOperations;
  if(manualOperations("grasp",false,true)!=std::vector<std::string>{"grasp","verify"}) return 1;
  for(bool holding:{false,true}) for(bool other:{false,true})
    if(manualOperations("home",holding,other)!=std::vector<std::string>{"open_for_home","home"}) return 2;
  if(manualOperations("fix",true,false)!=std::vector<std::string>{"fix"}) return 3;
  bool rejected=false;
  try {manualOperations("unknown",false,false);} catch(const std::runtime_error&) {rejected=true;}
  if(!rejected) return 4;
  std::cout<<"Manual HOME always opens before arm motion, including unknown/sole-holder state\n";
}
