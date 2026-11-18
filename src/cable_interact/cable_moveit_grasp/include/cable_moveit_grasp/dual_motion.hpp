#pragma once
#include <optional>
#include <cable_moveit_grasp/grasp_retry.hpp>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <map>
#include <sstream>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/srv/get_motion_plan.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit/robot_state/conversions.h>
#include <franka_msgs/action/move.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <cable_moveit_grasp/platform.hpp>
#include <cable_moveit_grasp/path_validation.hpp>
#include <cable_moveit_grasp/candidate_handoff.hpp>
#include <cable_moveit_grasp/manual_commands.hpp>
#include <cable_moveit_grasp/arrival_check.hpp>
#include <cable_moveit_grasp/orientation_checks.hpp>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
using namespace std::chrono_literals;
using MGI=moveit::planning_interface::MoveGroupInterface;
using Trigger=std_srvs::srv::Trigger;

class DualMotion : public rclcpp::Node {
 public:
  DualMotion();
 private:
  void executeHomeSkill(const std::string& arm,MGI& group);
  void executeGraspSkill(const std::string& arm,MGI& group);
  void executeFixSkill(const std::string& arm,MGI& group);
  void executeReleaseSkill(const std::string& arm,const std::string& op);
  void executeVerifySkill(const std::string& arm);
  moveit::core::RobotState measuredState(MGI& group);
  void performRequest(const std::string& arm,const std::string& command,bool manual,
                      Trigger::Response::SharedPtr response,bool coordinated=false);
  std::string executionState() const;
  void publishExecutionState();
  void report(const std::string& phase);
  void initialize();
  void freshState();
  void checkGripperMotion(const std::string& arm,double width,bool grasp);
  template<class Action> void gripper(const std::string& arm,const typename Action::Goal& goal) {
    report((std::is_same<Action,franka_msgs::action::Move>::value ? "opening/releasing gripper " : "closing gripper ")+arm);
    auto client=rclcpp_action::create_client<Action>(shared_from_this(),(arm=="a" ? "/NS_2" : "/NS_1")+std::string("/franka_gripper/")+(std::is_same<Action,franka_msgs::action::Move>::value ? "move" : "grasp"));
    if (!client->wait_for_action_server(3s)) throw std::runtime_error("Gripper unavailable");
    checkGripperMotion(arm,goal.width,std::is_same<Action,franka_msgs::action::Grasp>::value);
    physical_command_sent_=true;
    auto sent=client->async_send_goal(goal);
    if(sent.wait_for(3s)!=std::future_status::ready) throw std::runtime_error("Unknown gripper goal state; recovery required");
    auto handle=sent.get(); if(!handle) throw std::runtime_error("Gripper goal rejected");
    auto result=client->async_get_result(handle);
    if(result.wait_for(15s)!=std::future_status::ready) {client->async_cancel_goal(handle); throw std::runtime_error("Gripper timeout; recovery required");}
    auto wrapped=result.get();
    if(!wrapped.result) throw std::runtime_error("Gripper returned an empty result; recovery required");
    report("gripper "+arm+" result: action_code="+std::to_string(static_cast<int>(wrapped.code))+
        " success="+std::string(wrapped.result->success ? "true" : "false")+" error='"+wrapped.result->error+"'");
    if constexpr(std::is_same<Action,franka_msgs::action::Grasp>::value) {
      // A completed negative grasp with no device error may be retried. Unknown,
      // canceled, timed-out or explicit device-error results remain hard failures.
      if(!wrapped.result->success && wrapped.result->error.empty() &&
         (wrapped.code==rclcpp_action::ResultCode::ABORTED || wrapped.code==rclcpp_action::ResultCode::SUCCEEDED))
        throw cable_grasp::GraspMiss("Gripper completed but did not confirm a grasp");
    }
    if(wrapped.code!=rclcpp_action::ResultCode::SUCCEEDED || !wrapped.result->success)
      throw std::runtime_error("Gripper failed; action_code="+std::to_string(static_cast<int>(wrapped.code))+
          "; error='"+wrapped.result->error+"'; recovery required");
  }
  MGI::Plan selectedGraspPlan(const std::string& arm, MGI& group,bool reselect=false);
  robot_trajectory::RobotTrajectory checkPlan(MGI& group,const MGI::Plan& plan,bool assume_open=false);
  void execute(MGI& group,const MGI::Plan* cached=nullptr,bool verify_arrival=true);
  void waitForArrival(MGI& group,const robot_trajectory::RobotTrajectory& trajectory,std::int64_t finished);




  void recordBGrasp(const std::string& arm,MGI& group);
  void run(const std::string& arm,const std::string& op);
  std::map<std::string,rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr> fix_preview_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr b_grasp_pub_;
  bool recovering_home_{false};
  std::string fault_arm_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr override_pub_,execution_state_pub_;
  std::string active_command_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  std::unique_ptr<MGI> a_,b_;
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> scene_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr scene_client_;
  rclcpp::CallbackGroup::SharedPtr operations_;
  std::vector<rclcpp::Service<Trigger>::SharedPtr> services_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr> subscriptions_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> targets_;
  std::map<std::string,rclcpp::Time> joint_times_,goal_times_;
  std::map<std::string,double> joint_positions_;
  std::map<std::string,geometry_msgs::msg::PoseStamped> goals_;
  std::map<std::string,bool> holding_{{"a",false},{"b",false}};
  std::map<std::string,bool> home_{{"a",false},{"b",false}};
  bool camera_ready_=false,platform_ready_=false,fault_=false,physical_command_sent_=false;
  std::mutex mutex_;
  rclcpp::TimerBase::SharedPtr timer_;
};
