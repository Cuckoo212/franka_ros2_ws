#pragma once
#include <cable_moveit_grasp/motion_common.hpp>
#include <cable_moveit_grasp/candidate_handoff.hpp>
#include <cable_moveit_grasp/arrival_check.hpp>
#include <cable_moveit_grasp/gripper_collision.hpp>
#include <cable_moveit_grasp/platform.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/srv/get_motion_plan.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <franka_msgs/action/move.hpp>
#include <mutex>
#include <optional>
#include <thread>

class SingleArmMotion : public rclcpp::Node {
 public:
  SingleArmMotion();
 private:
  using MGI=cable_grasp::MotionGroup;
  using Trigger=std_srvs::srv::Trigger;
  void initialize();
  void report(const std::string& message);
  void receive(const sensor_msgs::msg::JointState& message,const std::vector<std::string>& allowed);
  moveit::core::RobotState measured();
  planning_scene::PlanningScenePtr scene();
  MGI::Plan selected(bool preview);
  robot_trajectory::RobotTrajectory validate(const MGI::Plan& plan,bool assume_open=false);
  void executePlan(const MGI::Plan& plan);
  MGI::Plan planLift(const moveit::core::RobotState& start);
  MGI::Plan planPlace(const moveit::core::RobotState& start);
  void executeLiftSkill();
  void executePlaceSkill();
  void preview();
  void executeSelected();
  void publishPreview(const MGI::Plan& grasp,const MGI::Plan& lift,const MGI::Plan& place);
  template<class Action> void gripper(const typename Action::Goal& goal) {
    auto world=scene();auto state=measured();
    const std::string prefix=get_parameter("arm_prefix").as_string();
    const auto samples=cable_grasp::validateGripperSweep(*world,state,{prefix+"fr3_finger_joint1",prefix+"fr3_finger_joint2"},goal.width,
        std::is_same<Action,franka_msgs::action::Grasp>::value);
    report("gripper collision sweep passed, samples="+std::to_string(samples));
    physical_sent_=true;
    cable_grasp::sendGripperGoal<Action>(shared_from_this(),get_parameter("gripper_namespace").as_string()+
       (std::is_same<Action,franka_msgs::action::Move>::value ? "/move" : "/grasp"),goal,
       get_parameter("gripper_timeout").as_double(),[this](const std::string& message){report(message);});
  }
  void open();
  std::optional<MGI::Plan> previewed_;
  std::unique_ptr<MGI> group_;
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> world_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr scene_client_;
  rclcpp::CallbackGroup::SharedPtr operations_;
  std::vector<rclcpp::Service<Trigger>::SharedPtr> services_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr> subscriptions_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr preview_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::vector<std::string> required_,arm_groups_;
  std::mutex mutex_;
  std::map<std::string,double> positions_;
  std::map<std::string,std::int64_t> stamps_;
  bool ready_=false,holding_=false,uncertain_=false,fault_=false,physical_sent_=false;
  std::string phase_="idle";
};
