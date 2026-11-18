#include <cable_moveit_grasp/dual_motion.hpp>
void DualMotion::executeReleaseSkill(const std::string& arm,const std::string& op) {
        report("opening gripper "+arm+" before "+(op=="open_for_home" ? "HOME" : "release"));
        franka_msgs::action::Move::Goal open; open.width=.08; open.speed=.02; gripper<franka_msgs::action::Move>(arm,open); holding_[arm]=false;
        if(arm=="b") b_grasp_pub_->publish(geometry_msgs::msg::PoseStamped{});
}
