// Copyright (c) 2026
//
// Cartesian controller for validating AprilTag hand-eye calibration. It accepts
// an AprilTag pose in the robot base frame, moves to a hover pose, then moves the
// gripper TCP to the tag center while aligning TCP +Z with the tag +Z axis.

#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_msgs/action/homing.hpp>
#include <franka_msgs/action/move.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/empty.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

class CartesianApriltagPositionController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    for (int i = 1; i <= 7; ++i) {
      config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/effort");
    }
    return config;
  }

  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    config.names = franka_cartesian_pose_->get_state_interface_names();
    for (int i = 1; i <= 7; ++i) {
      config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/position");
    }
    for (int i = 1; i <= 7; ++i) {
      config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/velocity");
    }
    for (int i = 1; i <= 7; ++i) {
      config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/effort");
    }
    for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
      config.names.push_back(franka_robot_model_name);
    }
    return config;
  }

  controller_interface::return_type update(const rclcpp::Time&, const rclcpp::Duration&) override {
    if (initialization_flag_) {
      std::tie(initial_orientation_, initial_position_) =
          franka_cartesian_pose_->getCurrentOrientationAndTranslation();
      commanded_position_ = initial_position_;
      commanded_orientation_ = initial_orientation_;
      update_joint_states();
      previous_tau_commanded_.setZero();
      phase_start_time_sec_ = get_node()->now().seconds();
      motion_phase_ = MotionPhase::kIdle;
      initialization_flag_ = false;
      RCLCPP_INFO(get_node()->get_logger(),
                  "AprilTag position controller activated. Waiting for %s.",
                  target_pose_topic_.c_str());
    }

    update_joint_states();
    apply_pending_target_update();
    const auto [current_orientation, current_position] =
        franka_cartesian_pose_->getCurrentOrientationAndTranslation();
    update_motion_state(current_position, current_orientation);

    const Vector7d tau_d_calculated = compute_torque_command(
        current_position, current_orientation, vector_from_std(joint_velocities_current_));
    for (int i = 0; i < 7; ++i) {
      command_interfaces_[i].set_value(tau_d_calculated(i));
    }

    return controller_interface::return_type::OK;
  }

  CallbackReturn on_init() override {
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("target_pose_topic", "/apriltag_target_pose");
    auto_declare<std::string>("start_approach_topic", "/cartesian_apriltag_position_controller/start_approach");
    auto_declare<std::string>("expected_frame_id", "fr3_link0");
    auto_declare<std::vector<double>>("hover_offset", {0.0, 0.0, 0.15});
    auto_declare<std::vector<double>>("cartesian_stiffness",
                                      {800.0, 900.0, 1050.0, 35.0, 35.0, 30.0});
    auto_declare<std::vector<double>>("cartesian_damping",
                                      {56.7, 60.0, 64.8, 11.8, 11.8, 11.0});
    auto_declare<std::vector<double>>("max_torque_deltas",
                                      {0.25, 0.25, 0.22, 0.20, 0.15, 0.12, 0.10});
    auto_declare<double>("max_translation_error", 0.10);
    auto_declare<double>("max_rotation_error", 0.35);
    auto_declare<double>("move_to_hover_duration", 8.0);
    auto_declare<double>("move_to_target_duration", 8.0);
    auto_declare<double>("target_position_tolerance", 0.01);
    auto_declare<double>("target_orientation_tolerance", 0.10);
    auto_declare<bool>("strict_frame_check", true);
    auto_declare<bool>("track_target_updates", false);
    auto_declare<bool>("close_gripper_on_start", true);
    auto_declare<bool>("home_gripper_on_start", true);
    auto_declare<double>("gripper_width", 0.004);
    auto_declare<double>("gripper_speed", 0.02);
    auto_declare<double>("gripper_force", 10.0);
    auto_declare<double>("gripper_epsilon_inner", 0.004);
    auto_declare<double>("gripper_epsilon_outer", 0.020);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State&) override {
    is_gazebo_ = get_node()->get_parameter("gazebo").as_bool();
    arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
    arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
    target_pose_topic_ = get_node()->get_parameter("target_pose_topic").as_string();
    start_approach_topic_ = get_node()->get_parameter("start_approach_topic").as_string();
    expected_frame_id_ = get_node()->get_parameter("expected_frame_id").as_string();
    strict_frame_check_ = get_node()->get_parameter("strict_frame_check").as_bool();
    track_target_updates_ = get_node()->get_parameter("track_target_updates").as_bool();
    close_gripper_on_start_ = get_node()->get_parameter("close_gripper_on_start").as_bool();
    home_gripper_on_start_ = get_node()->get_parameter("home_gripper_on_start").as_bool();
    gripper_width_ = get_node()->get_parameter("gripper_width").as_double();
    gripper_speed_ = get_node()->get_parameter("gripper_speed").as_double();
    gripper_force_ = get_node()->get_parameter("gripper_force").as_double();
    gripper_epsilon_inner_ = get_node()->get_parameter("gripper_epsilon_inner").as_double();
    gripper_epsilon_outer_ = get_node()->get_parameter("gripper_epsilon_outer").as_double();
    move_to_hover_duration_sec_ = get_node()->get_parameter("move_to_hover_duration").as_double();
    move_to_target_duration_sec_ = get_node()->get_parameter("move_to_target_duration").as_double();
    target_position_tolerance_m_ = get_node()->get_parameter("target_position_tolerance").as_double();
    target_orientation_tolerance_rad_ =
        get_node()->get_parameter("target_orientation_tolerance").as_double();

    const auto hover_offset = get_node()->get_parameter("hover_offset").as_double_array();
    const auto cartesian_stiffness =
        get_node()->get_parameter("cartesian_stiffness").as_double_array();
    const auto cartesian_damping = get_node()->get_parameter("cartesian_damping").as_double_array();
    const auto max_torque_deltas = get_node()->get_parameter("max_torque_deltas").as_double_array();
    if (hover_offset.size() != 3 || cartesian_stiffness.size() != 6 ||
        cartesian_damping.size() != 6 || max_torque_deltas.size() != 7) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "hover_offset must have 3 values; cartesian_stiffness/cartesian_damping must "
                   "have 6; max_torque_deltas must have 7.");
      return CallbackReturn::ERROR;
    }
    if (move_to_hover_duration_sec_ <= 0.0 || move_to_target_duration_sec_ <= 0.0 ||
        target_position_tolerance_m_ <= 0.0 || target_orientation_tolerance_rad_ <= 0.0 ||
        gripper_width_ < 0.0 || gripper_speed_ <= 0.0 || gripper_force_ <= 0.0 ||
        gripper_epsilon_inner_ < 0.0 || gripper_epsilon_outer_ < 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Motion durations/tolerances and gripper speed/force must be positive.");
      return CallbackReturn::ERROR;
    }

    hover_offset_ = Eigen::Vector3d(hover_offset[0], hover_offset[1], hover_offset[2]);
    for (int i = 0; i < 6; ++i) {
      cartesian_stiffness_(i) = cartesian_stiffness.at(i);
      cartesian_damping_(i) = cartesian_damping.at(i);
    }
    for (int i = 0; i < 7; ++i) {
      max_torque_deltas_(i) = max_torque_deltas.at(i);
    }
    max_translation_error_ = get_node()->get_parameter("max_translation_error").as_double();
    max_rotation_error_ = get_node()->get_parameter("max_rotation_error").as_double();

    franka_cartesian_pose_ =
        std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
            franka_semantic_components::FrankaCartesianPoseInterface(arm_prefix_, false));
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        franka_semantic_components::FrankaRobotModel(
            arm_prefix_ + robot_type_ + "/" + k_robot_model_interface_name,
            arm_prefix_ + robot_type_ + "/" + k_robot_state_interface_name));

    if (!is_gazebo_) {
      auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
          "service_server/set_full_collision_behavior");
      auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();
      auto future_result = client->async_send_request(request);
      future_result.wait_for(robot_utils::time_out);
      auto success = future_result.get();
      if (!success) {
        RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
        return CallbackReturn::ERROR;
      }
      RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");
    } else {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Gazebo/fake mode enabled: skip set_full_collision_behavior service call.");
    }

    auto parameters_client =
        std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
    parameters_client->wait_for_service();
    auto future = parameters_client->get_parameters({"robot_description"});
    auto result = future.get();
    if (!result.empty()) {
      robot_description_ = result[0].value_to_string();
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
      return CallbackReturn::ERROR;
    }
    robot_type_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());

    target_pose_subscription_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
        target_pose_topic_, rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
          handle_target_pose(message);
        });
    start_approach_subscription_ = get_node()->create_subscription<std_msgs::msg::Empty>(
        start_approach_topic_, rclcpp::SystemDefaultsQoS(),
        [this](const std_msgs::msg::Empty::SharedPtr) {
          handle_start_approach_command();
        });

    std::string action_namespace = get_node()->get_namespace();
    if (action_namespace == "/") {
      action_namespace.clear();
    }
    gripper_homing_action_client_ = rclcpp_action::create_client<franka_msgs::action::Homing>(
        get_node(), action_namespace + "/franka_gripper/homing");
    gripper_move_action_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
        get_node(), action_namespace + "/franka_gripper/move");
    assign_homing_goal_options_callbacks();
    assign_move_goal_options_callbacks();

    RCLCPP_INFO(get_node()->get_logger(),
                "Configured AprilTag position controller. target_topic=%s start_topic=%s "
                "expected_frame=%s hover_offset=[%.3f, %.3f, %.3f] close_gripper_on_start=%s",
                target_pose_topic_.c_str(), start_approach_topic_.c_str(), expected_frame_id_.c_str(),
                hover_offset_.x(), hover_offset_.y(), hover_offset_.z(),
                close_gripper_on_start_ ? "true" : "false");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State&) override {
    initialization_flag_ = true;
    franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);
    franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
    pose_state_interface_count_ = franka_cartesian_pose_->get_state_interface_names().size();
    joint_position_state_start_index_ = pose_state_interface_count_;
    joint_velocity_state_start_index_ = joint_position_state_start_index_ + 7;
    joint_effort_state_start_index_ = joint_velocity_state_start_index_ + 7;
    if (close_gripper_on_start_ && home_gripper_on_start_ && gripper_homing_action_client_ &&
        !gripper_homing_action_client_->wait_for_action_server(std::chrono::milliseconds(500))) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Homing Action server is not visible during activation. "
                  "The controller will wait for it when an AprilTag target arrives.");
    }
    if (close_gripper_on_start_ && gripper_move_action_client_ &&
        !gripper_move_action_client_->wait_for_action_server(std::chrono::milliseconds(500))) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Move Action server is not visible during activation. "
                  "The controller will wait for it when an AprilTag target arrives.");
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override {
    franka_cartesian_pose_->release_interfaces();
    franka_robot_model_->release_interfaces();
    return CallbackReturn::SUCCESS;
  }

 private:
  enum class MotionPhase { kIdle, kWaitForGripperHoming, kWaitForGripperClose, kMoveToHover, kMoveToTarget, kHold };

  struct TargetPose {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d hover_position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  };

  static double compute_progress(double elapsed_time, double duration_sec) {
    const double normalized_time = std::clamp(elapsed_time / duration_sec, 0.0, 1.0);
    const double t2 = normalized_time * normalized_time;
    const double t3 = t2 * normalized_time;
    const double t4 = t3 * normalized_time;
    const double t5 = t4 * normalized_time;
    return 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
  }

  static Vector7d vector_from_std(const std::vector<double>& values) {
    Vector7d vector = Vector7d::Zero();
    for (size_t i = 0; i < 7 && i < values.size(); ++i) {
      vector(static_cast<Eigen::Index>(i)) = values[i];
    }
    return vector;
  }

  static Eigen::Vector3d compute_orientation_error(const Eigen::Quaterniond& current_orientation,
                                                   const Eigen::Quaterniond& desired_orientation) {
    Eigen::Quaterniond desired = desired_orientation;
    if (desired.coeffs().dot(current_orientation.coeffs()) < 0.0) {
      desired.coeffs() *= -1.0;
    }
    Eigen::Quaterniond delta = desired * current_orientation.conjugate();
    Eigen::AngleAxisd angle_axis(delta);
    if (!std::isfinite(angle_axis.angle())) {
      return Eigen::Vector3d::Zero();
    }
    double angle = angle_axis.angle();
    if (angle > M_PI) {
      angle -= 2.0 * M_PI;
    }
    if (std::abs(angle) < 1e-6 || !std::isfinite(angle_axis.axis().norm())) {
      return Eigen::Vector3d::Zero();
    }
    return angle_axis.axis() * angle;
  }

  static bool quaternion_is_valid(const Eigen::Quaterniond& quaternion) {
    return std::isfinite(quaternion.x()) && std::isfinite(quaternion.y()) &&
           std::isfinite(quaternion.z()) && std::isfinite(quaternion.w()) &&
           quaternion.norm() > 1e-6;
  }

  static Eigen::Quaterniond tcp_orientation_from_tag_orientation(
      const Eigen::Quaterniond& tag_orientation) {
    Eigen::Quaterniond target_orientation = tag_orientation.normalized();
    target_orientation.normalize();
    return target_orientation;
  }

  void handle_target_pose(const geometry_msgs::msg::PoseStamped::SharedPtr message) {
    if (!track_target_updates_ && motion_phase_ != MotionPhase::kIdle &&
        motion_phase_ != MotionPhase::kHold) {
      RCLCPP_INFO_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "Ignoring live AprilTag target updates while executing the current approach.");
      return;
    }

    if (strict_frame_check_ && !expected_frame_id_.empty() &&
        message->header.frame_id != expected_frame_id_) {
      RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "Ignoring AprilTag target in frame '%s'. Expected '%s'. Publish the pose after "
          "transforming it to the robot base frame.",
          message->header.frame_id.c_str(), expected_frame_id_.c_str());
      return;
    }

    Eigen::Quaterniond tag_orientation(message->pose.orientation.w, message->pose.orientation.x,
                                       message->pose.orientation.y, message->pose.orientation.z);
    if (!quaternion_is_valid(tag_orientation)) {
      RCLCPP_WARN(get_node()->get_logger(), "Ignoring AprilTag target with invalid orientation.");
      return;
    }

    TargetPose target;
    target.position = Eigen::Vector3d(message->pose.position.x, message->pose.position.y,
                                      message->pose.position.z);
    target.orientation = tcp_orientation_from_tag_orientation(tag_orientation);
    target.hover_position = target.position + tag_orientation.normalized().toRotationMatrix() * hover_offset_;
    if (!std::isfinite(target.position.x()) || !std::isfinite(target.position.y()) ||
        !std::isfinite(target.position.z())) {
      RCLCPP_WARN(get_node()->get_logger(), "Ignoring AprilTag target with non-finite position.");
      return;
    }

    {
      std::lock_guard<std::mutex> lock(target_update_mutex_);
      requested_target_pose_ = target;
      pending_target_available_ = true;
    }
    RCLCPP_INFO_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 2000,
        "AprilTag target detected and cached: target=[%.6f, %.6f, %.6f] "
        "hover=[%.6f, %.6f, %.6f], frame=%s. Publish std_msgs/msg/Empty on %s to start approach.",
        target.position.x(), target.position.y(), target.position.z(),
        target.hover_position.x(), target.hover_position.y(), target.hover_position.z(),
        message->header.frame_id.c_str(), start_approach_topic_.c_str());
  }

  void handle_start_approach_command() {
    if (motion_phase_ != MotionPhase::kIdle && motion_phase_ != MotionPhase::kHold) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Ignoring start approach command because the controller is already executing.");
      return;
    }
    {
      std::lock_guard<std::mutex> lock(target_update_mutex_);
      if (!pending_target_available_) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "Ignoring start approach command because no AprilTag target has been cached yet.");
        return;
      }
    }
    target_update_requested_ = true;
    RCLCPP_INFO(get_node()->get_logger(), "Start approach command accepted. Entering motion state machine.");
  }

  void update_joint_states() {
    for (int i = 0; i < 7; ++i) {
      joint_positions_current_[i] =
          state_interfaces_.at(joint_position_state_start_index_ + i).get_value();
      joint_velocities_current_[i] =
          state_interfaces_.at(joint_velocity_state_start_index_ + i).get_value();
      joint_efforts_current_[i] = state_interfaces_.at(joint_effort_state_start_index_ + i).get_value();
    }
  }

  void apply_pending_target_update() {
    if (!target_update_requested_.exchange(false)) {
      return;
    }

    TargetPose requested_target;
    {
      std::lock_guard<std::mutex> lock(target_update_mutex_);
      requested_target = requested_target_pose_;
      pending_target_available_ = false;
    }
    if (initialization_flag_) {
      return;
    }

    initial_position_ = commanded_position_;
    initial_orientation_ = commanded_orientation_;
    target_position_ = requested_target.position;
    target_orientation_ = requested_target.orientation;
    hover_position_ = requested_target.hover_position;
    hover_orientation_ = target_orientation_;
    phase_start_time_sec_ = get_node()->now().seconds();
    homing_goal_sent_ = false;
    homing_result_received_ = false;
    homing_succeeded_ = false;
    close_goal_sent_ = false;
    close_result_received_ = false;
    close_succeeded_ = false;
    if (close_gripper_on_start_ && home_gripper_on_start_) {
      send_homing_goal();
      motion_phase_ = MotionPhase::kWaitForGripperHoming;
    } else if (close_gripper_on_start_) {
      send_close_goal();
      motion_phase_ = MotionPhase::kWaitForGripperClose;
    } else {
      motion_phase_ = MotionPhase::kMoveToHover;
    }

    RCLCPP_INFO(get_node()->get_logger(),
                "Starting AprilTag approach: hover=[%.6f, %.6f, %.6f], target=[%.6f, %.6f, %.6f], first_phase=%s",
                hover_position_.x(), hover_position_.y(), hover_position_.z(),
                target_position_.x(), target_position_.y(), target_position_.z(),
                close_gripper_on_start_ ? (home_gripper_on_start_ ? "home_then_close_gripper" : "close_gripper") : "move_to_hover");
  }

  void update_motion_state(const Eigen::Vector3d& current_position,
                           const Eigen::Quaterniond& current_orientation) {
    switch (motion_phase_) {
      case MotionPhase::kIdle:
        commanded_position_ = initial_position_;
        commanded_orientation_ = initial_orientation_;
        break;
      case MotionPhase::kWaitForGripperHoming:
        update_wait_for_gripper_homing_phase();
        break;
      case MotionPhase::kWaitForGripperClose:
        update_wait_for_gripper_close_phase();
        break;
      case MotionPhase::kMoveToHover:
        update_move_to_hover_phase();
        break;
      case MotionPhase::kMoveToTarget:
        update_move_to_target_phase(current_position, current_orientation);
        break;
      case MotionPhase::kHold:
        commanded_position_ = target_position_;
        commanded_orientation_ = target_orientation_;
        break;
    }
  }

  void update_wait_for_gripper_homing_phase() {
    commanded_position_ = initial_position_;
    commanded_orientation_ = initial_orientation_;

    if (!homing_goal_sent_) {
      send_homing_goal();
    }

    if (!homing_result_received_) {
      RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                           "Waiting for gripper homing before closing.");
      return;
    }

    if (!homing_succeeded_) {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                            "Gripper homing did not succeed. Holding the initial pose; "
                            "check /franka_gripper/homing and the physical gripper state.");
      return;
    }

    send_close_goal();
    motion_phase_ = MotionPhase::kWaitForGripperClose;
    RCLCPP_INFO(get_node()->get_logger(), "Gripper homing succeeded. Closing gripper.");
  }

  void update_wait_for_gripper_close_phase() {
    commanded_position_ = initial_position_;
    commanded_orientation_ = initial_orientation_;

    if (!close_goal_sent_) {
      send_close_goal();
    }

    if (!close_result_received_) {
      RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                           "Waiting for gripper close before moving to AprilTag hover pose.");
      return;
    }

    if (!close_succeeded_) {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                            "Gripper close did not succeed. Holding the initial pose; "
                            "check /franka_gripper/move and load_gripper:=true.");
      return;
    }

    phase_start_time_sec_ = get_node()->now().seconds();
    motion_phase_ = MotionPhase::kMoveToHover;
    RCLCPP_INFO(get_node()->get_logger(),
                "Gripper close move succeeded. Moving to AprilTag hover pose with the gripper closed.");
  }

  void update_move_to_hover_phase() {
    const double elapsed_time = std::max(0.0, get_node()->now().seconds() - phase_start_time_sec_);
    const double alpha = compute_progress(elapsed_time, move_to_hover_duration_sec_);
    commanded_position_ = initial_position_ + alpha * (hover_position_ - initial_position_);
    commanded_orientation_ = initial_orientation_.slerp(alpha, hover_orientation_);

    RCLCPP_INFO_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "Moving to AprilTag hover pose: progress=%.1f%% commanded=[%.4f, %.4f, %.4f] hover=[%.4f, %.4f, %.4f]",
        100.0 * std::clamp(elapsed_time / move_to_hover_duration_sec_, 0.0, 1.0),
        commanded_position_.x(), commanded_position_.y(), commanded_position_.z(),
        hover_position_.x(), hover_position_.y(), hover_position_.z());

    if (elapsed_time >= move_to_hover_duration_sec_) {
      commanded_position_ = hover_position_;
      commanded_orientation_ = hover_orientation_;
      phase_start_time_sec_ = get_node()->now().seconds();
      motion_phase_ = MotionPhase::kMoveToTarget;
      RCLCPP_INFO(get_node()->get_logger(), "Reached AprilTag hover pose. Moving to tag center.");
    }
  }

  void update_move_to_target_phase(const Eigen::Vector3d& current_position,
                                   const Eigen::Quaterniond& current_orientation) {
    const double elapsed_time = std::max(0.0, get_node()->now().seconds() - phase_start_time_sec_);
    const double alpha = compute_progress(elapsed_time, move_to_target_duration_sec_);
    commanded_position_ = hover_position_ + alpha * (target_position_ - hover_position_);
    commanded_orientation_ = hover_orientation_.slerp(alpha, target_orientation_);

    if (elapsed_time >= move_to_target_duration_sec_) {
      commanded_position_ = target_position_;
      commanded_orientation_ = target_orientation_;

      const Eigen::Vector3d translation_error = target_position_ - current_position;
      const Eigen::Vector3d orientation_error_vector =
          compute_orientation_error(current_orientation, target_orientation_);
      if (translation_error.norm() <= target_position_tolerance_m_ &&
          orientation_error_vector.norm() <= target_orientation_tolerance_rad_) {
        motion_phase_ = MotionPhase::kHold;
        RCLCPP_INFO(
            get_node()->get_logger(),
            "Reached AprilTag target. Holding pose. position_error=%.4f m orientation_error=%.4f rad",
            translation_error.norm(), orientation_error_vector.norm());
      } else {
        RCLCPP_INFO_THROTTLE(
            get_node()->get_logger(), *get_node()->get_clock(), 2000,
            "Waiting for AprilTag target convergence. position_error=%.4f/%.4f m "
            "orientation_error=%.4f/%.4f rad",
            translation_error.norm(), target_position_tolerance_m_, orientation_error_vector.norm(),
            target_orientation_tolerance_rad_);
      }
    }
  }

  void send_homing_goal() {
    if (!gripper_homing_action_client_) {
      RCLCPP_ERROR(get_node()->get_logger(), "Cannot home gripper: Homing Action client is not configured.");
      homing_goal_sent_ = false;
      homing_result_received_ = true;
      homing_succeeded_ = false;
      return;
    }
    if (!gripper_homing_action_client_->action_server_is_ready() &&
        !gripper_homing_action_client_->wait_for_action_server(std::chrono::milliseconds(100))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Cannot home gripper: /franka_gripper/homing Action server is not available.");
      homing_goal_sent_ = false;
      homing_result_received_ = true;
      homing_succeeded_ = false;
      return;
    }

    franka_msgs::action::Homing::Goal homing_goal;
    homing_goal_sent_ = true;
    homing_result_received_ = false;
    homing_succeeded_ = false;

    auto goal_handle_future =
        gripper_homing_action_client_->async_send_goal(homing_goal, homing_goal_options_);
    if (!goal_handle_future.valid()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to submit gripper homing goal.");
      homing_result_received_ = true;
      homing_succeeded_ = false;
      return;
    }

    RCLCPP_INFO(get_node()->get_logger(), "Submitted gripper homing goal.");
  }

  void send_close_goal() {
    if (!gripper_move_action_client_) {
      RCLCPP_ERROR(get_node()->get_logger(), "Cannot close gripper: Move Action client is not configured.");
      close_goal_sent_ = false;
      close_result_received_ = true;
      close_succeeded_ = false;
      return;
    }
    if (!gripper_move_action_client_->action_server_is_ready() &&
        !gripper_move_action_client_->wait_for_action_server(std::chrono::milliseconds(100))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Cannot close gripper: /franka_gripper/move Action server is not available.");
      close_goal_sent_ = false;
      close_result_received_ = true;
      close_succeeded_ = false;
      return;
    }

    franka_msgs::action::Move::Goal move_goal;
    move_goal.width = gripper_width_;
    move_goal.speed = gripper_speed_;
    close_goal_sent_ = true;
    close_result_received_ = false;
    close_succeeded_ = false;

    auto goal_handle_future =
        gripper_move_action_client_->async_send_goal(move_goal, move_goal_options_);
    if (!goal_handle_future.valid()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to submit gripper close move goal.");
      close_result_received_ = true;
      close_succeeded_ = false;
      return;
    }

    RCLCPP_INFO(get_node()->get_logger(),
                "Submitted gripper close move goal: width=%.4f speed=%.3f",
                gripper_width_, gripper_speed_);
  }

  void assign_homing_goal_options_callbacks() {
    homing_goal_options_.goal_response_callback =
        [this](const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::Homing>>&
                   goal_handle) {
          if (!goal_handle) {
            homing_result_received_ = true;
            homing_succeeded_ = false;
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper homing goal was rejected.");
          } else {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper homing goal accepted.");
          }
        };

    homing_goal_options_.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Homing>::WrappedResult&
                   result) {
          homing_result_received_ = true;
          homing_succeeded_ = result.code == rclcpp_action::ResultCode::SUCCEEDED &&
                             result.result && result.result->success;

          if (homing_succeeded_) {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper homing succeeded.");
          } else {
            const std::string error_message =
                result.result ? result.result->error : "unknown gripper homing error";
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper homing failed: %s",
                         error_message.c_str());
          }
        };
  }

  void assign_move_goal_options_callbacks() {
    move_goal_options_.goal_response_callback =
        [this](const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>>&
                   goal_handle) {
          if (!goal_handle) {
            close_result_received_ = true;
            close_succeeded_ = false;
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper close move goal was rejected.");
          } else {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper close move goal accepted.");
          }
        };

    move_goal_options_.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>::WrappedResult&
                   result) {
          close_result_received_ = true;
          close_succeeded_ = result.code == rclcpp_action::ResultCode::SUCCEEDED &&
                             result.result && result.result->success;

          if (close_succeeded_) {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper close move succeeded.");
          } else {
            const std::string error_message =
                result.result ? result.result->error : "unknown gripper close error";
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper close move failed: %s",
                         error_message.c_str());
          }
        };
  }

  Vector7d compute_torque_command(const Eigen::Vector3d& current_position,
                                  const Eigen::Quaterniond& current_orientation,
                                  const Vector7d& joint_velocities_current) {
    const std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
    const std::array<double, 42> jacobian_array =
        franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
    Vector7d coriolis(coriolis_array.data());
    const Vector6d cartesian_velocity = jacobian * joint_velocities_current;

    Vector6d pose_error = Vector6d::Zero();
    pose_error.head<3>() = commanded_position_ - current_position;
    pose_error.tail<3>() = compute_orientation_error(current_orientation, commanded_orientation_);
    for (int i = 0; i < 3; ++i) {
      pose_error(i) = std::clamp(pose_error(i), -max_translation_error_, max_translation_error_);
      pose_error(i + 3) = std::clamp(pose_error(i + 3), -max_rotation_error_, max_rotation_error_);
    }

    const Vector6d desired_wrench =
        cartesian_stiffness_.cwiseProduct(pose_error) -
        cartesian_damping_.cwiseProduct(cartesian_velocity);
    const Vector7d tau_d_calculated = jacobian.transpose() * desired_wrench + coriolis;
    Vector7d tau_d_saturated = tau_d_calculated;
    for (int i = 0; i < 7; ++i) {
      const double lower = previous_tau_commanded_(i) - max_torque_deltas_(i);
      const double upper = previous_tau_commanded_(i) + max_torque_deltas_(i);
      tau_d_saturated(i) = std::clamp(tau_d_saturated(i), lower, upper);
    }
    previous_tau_commanded_ = tau_d_saturated;
    return tau_d_saturated;
  }

  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_subscription_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr start_approach_subscription_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Homing>> gripper_homing_action_client_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Move>> gripper_move_action_client_;
  rclcpp_action::Client<franka_msgs::action::Homing>::SendGoalOptions homing_goal_options_;
  rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions move_goal_options_;

  bool initialization_flag_{true};
  bool is_gazebo_{false};
  bool strict_frame_check_{true};
  bool track_target_updates_{false};
  bool close_gripper_on_start_{true};
  bool home_gripper_on_start_{true};
  MotionPhase motion_phase_{MotionPhase::kIdle};
  std::atomic_bool homing_goal_sent_{false};
  std::atomic_bool homing_result_received_{false};
  std::atomic_bool homing_succeeded_{false};
  std::atomic_bool close_goal_sent_{false};
  std::atomic_bool close_result_received_{false};
  std::atomic_bool close_succeeded_{false};
  std::string robot_description_;
  std::string robot_type_{"fr3"};
  std::string arm_prefix_;
  std::string target_pose_topic_{"/apriltag_target_pose"};
  std::string start_approach_topic_{"/cartesian_apriltag_position_controller/start_approach"};
  std::string expected_frame_id_{"fr3_link0"};
  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};

  double phase_start_time_sec_{0.0};
  double move_to_hover_duration_sec_{8.0};
  double move_to_target_duration_sec_{8.0};
  double target_position_tolerance_m_{0.01};
  double target_orientation_tolerance_rad_{0.10};
  double max_translation_error_{0.10};
  double max_rotation_error_{0.35};
  double gripper_width_{0.004};
  double gripper_speed_{0.02};
  double gripper_force_{10.0};
  double gripper_epsilon_inner_{0.004};
  double gripper_epsilon_outer_{0.020};

  Eigen::Quaterniond initial_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Quaterniond hover_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Quaterniond target_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Quaterniond commanded_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d initial_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d hover_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d target_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d commanded_position_{Eigen::Vector3d::Zero()};
  // Expressed in the AprilTag frame, then rotated into the robot base frame.
  Eigen::Vector3d hover_offset_{0.0, 0.0, 0.15};

  TargetPose requested_target_pose_;
  bool pending_target_available_{false};
  std::atomic_bool target_update_requested_{false};
  std::mutex target_update_mutex_;
  size_t pose_state_interface_count_{0};
  size_t joint_position_state_start_index_{0};
  size_t joint_velocity_state_start_index_{0};
  size_t joint_effort_state_start_index_{0};
  Vector7d previous_tau_commanded_{Vector7d::Zero()};
  Vector6d cartesian_stiffness_{Vector6d::Zero()};
  Vector6d cartesian_damping_{Vector6d::Zero()};
  Vector7d max_torque_deltas_{Vector7d::Zero()};
  std::vector<double> joint_positions_current_{0, 0, 0, 0, 0, 0, 0};
  std::vector<double> joint_velocities_current_{0, 0, 0, 0, 0, 0, 0};
  std::vector<double> joint_efforts_current_{0, 0, 0, 0, 0, 0, 0};
};

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianApriltagPositionController,
                       controller_interface::ControllerInterface)
