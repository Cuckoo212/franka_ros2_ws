// Copyright (c) 2023 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <atomic>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/homing.hpp>
#include <franka_msgs/action/move.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <limits>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

namespace {

constexpr const char* kGraspPointRoot = "/home/flexcycle/cv_models/cmcor/info_for_3Dpoint";
const Eigen::Vector3d kHoverApproachOffset(0.0, -0.20, 0.0);
const Eigen::Vector3d kHoverTargetZAxis = Eigen::Vector3d::UnitY();

struct GraspPlan {
  Eigen::Vector3d grasp_point;
  Eigen::Vector3d gripper_direction_a;
  Eigen::Vector3d gripper_direction_b;
};

struct JointLimits {
  double lower;
  double upper;
};

std::string normalize_cable_id(const std::string& cable_id) {
  if (cable_id.empty()) {
    return cable_id;
  }

  bool all_digits = std::all_of(cable_id.begin(), cable_id.end(), [](unsigned char c) {
    return std::isdigit(c) != 0;
  });
  if (!all_digits) {
    return cable_id;
  }

  std::ostringstream stream;
  stream << std::setw(3) << std::setfill('0') << std::stoi(cable_id);
  return stream.str();
}

std::filesystem::path build_grasp_plan_path(const std::string& cable_id) {
  const std::string normalized_cable_id = normalize_cable_id(cable_id);
  const std::filesystem::path cable_dir =
      std::filesystem::path(kGraspPointRoot) / ("cable_" + normalized_cable_id);
  return cable_dir / ("grasp_point_cable_" + normalized_cable_id);
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open file: " + path.string());
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::vector<double> extract_number_list(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error("Failed to find list field '" + key + "' in grasp plan file.");
  }

  std::vector<double> values;
  std::stringstream stream(match[1].str());
  std::string token;
  while (std::getline(stream, token, ',')) {
    values.push_back(std::stod(token));
  }
  return values;
}

GraspPlan load_grasp_plan_from_file(const std::string& cable_id) {
  const std::filesystem::path grasp_plan_path = build_grasp_plan_path(cable_id);
  const std::string file_text = read_text_file(grasp_plan_path);
  const std::vector<double> grasp_point = extract_number_list(file_text, "grasp_point");
  if (grasp_point.size() != 3) {
    throw std::runtime_error("Field 'grasp_point' must contain exactly 3 values.");
  }

  return GraspPlan{
      Eigen::Vector3d(grasp_point[0], grasp_point[1], grasp_point[2]),
      Eigen::Vector3d(
          extract_number_list(file_text, "gripper_direction_a")[0],
          extract_number_list(file_text, "gripper_direction_a")[1],
          extract_number_list(file_text, "gripper_direction_a")[2]),
      Eigen::Vector3d(
          extract_number_list(file_text, "gripper_direction_b")[0],
          extract_number_list(file_text, "gripper_direction_b")[1],
          extract_number_list(file_text, "gripper_direction_b")[2]),
  };
}

JointLimits extract_joint_limits_from_description(const std::string& robot_description,
                                                  const std::string& joint_name) {
  const std::regex joint_regex("<joint\\s+name=\"" + joint_name + "\"[^>]*>([\\s\\S]*?)</joint>",
                               std::regex::ECMAScript | std::regex::icase);
  std::smatch joint_match;
  if (!std::regex_search(robot_description, joint_match, joint_regex)) {
    throw std::runtime_error("Failed to find joint '" + joint_name + "' in robot_description.");
  }

  const std::string joint_block = joint_match[1].str();
  const std::regex limit_regex(
      "<limit[^>]*lower=\"([-+0-9.eE]+)\"[^>]*upper=\"([-+0-9.eE]+)\"[^>]*/?>",
      std::regex::ECMAScript | std::regex::icase);
  std::smatch limit_match;
  if (!std::regex_search(joint_block, limit_match, limit_regex)) {
    throw std::runtime_error("Failed to find joint limits for '" + joint_name + "'.");
  }

  return JointLimits{std::stod(limit_match[1].str()), std::stod(limit_match[2].str())};
}

}  // namespace

class CartesianTargetPointController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    config.names = franka_cartesian_pose_->get_command_interface_names();
    return config;
  }

  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    config.names = franka_cartesian_pose_->get_state_interface_names();
    config.names.push_back(arm_prefix_ + robot_type_ + "/robot_time");
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint7/position");
    return config;
  }

  controller_interface::return_type update(const rclcpp::Time&,
                                           const rclcpp::Duration&) override {
    if (initialization_flag_) {
      std::tie(initial_orientation_, initial_position_) =
          franka_cartesian_pose_->getCurrentOrientationAndTranslation();
      target_orientation_ = compute_target_orientation();
      commanded_position_ = initial_position_;
      commanded_orientation_ = initial_orientation_;
      initial_robot_time_ = state_interfaces_.back().get_value();
      phase_start_time_ = initial_robot_time_;
      motion_phase_ = MotionPhase::kIdle;
      open_goal_sent_ = false;
      open_result_received_ = false;
      open_succeeded_ = false;
      grasp_goal_sent_ = false;
      grasp_result_received_ = false;
      grasp_succeeded_ = false;
      initialization_flag_ = false;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Controller activated. Holding current pose until a new target is received.");
    }

    robot_time_ = state_interfaces_.at(robot_time_interface_index_).get_value();
    current_joint7_position_ = state_interfaces_.at(joint7_position_interface_index_).get_value();
    apply_pending_target_update();
    update_motion_state();

    if (franka_cartesian_pose_->setCommand(commanded_orientation_, commanded_position_)) {
      return controller_interface::return_type::OK;
    }

    RCLCPP_FATAL(get_node()->get_logger(),
                 "Set command failed. Did you activate the elbow command interface?");
    return controller_interface::return_type::ERROR;
  }

  CallbackReturn on_init() override {
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("target_cable_id", "");
    auto_declare<double>("motion_duration", 8.0);
    auto_declare<double>("lift_distance", 0.30);
    auto_declare<double>("gripper_width", 0.004);
    auto_declare<double>("gripper_speed", 0.02);
    auto_declare<double>("gripper_force", 10.0);
    auto_declare<double>("gripper_epsilon_inner", 0.003);
    auto_declare<double>("gripper_epsilon_outer", 0.003);
    auto_declare<double>("open_width", 0.08);
    auto_declare<double>("open_speed", 0.05);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State&) override {
    is_gazebo_ = get_node()->get_parameter("gazebo").as_bool();
    arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
    arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
    target_position_ = Eigen::Vector3d::Zero();
    requested_target_position_ = target_position_;
    requested_gripper_direction_ = Eigen::Vector3d::UnitX();
    selected_gripper_direction_ = Eigen::Vector3d::UnitX();
    motion_duration_sec_ = get_node()->get_parameter("motion_duration").as_double();
    if (motion_duration_sec_ <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "Parameter motion_duration must be greater than 0.");
      return CallbackReturn::ERROR;
    }
    lift_distance_ = get_node()->get_parameter("lift_distance").as_double();
    gripper_width_ = get_node()->get_parameter("gripper_width").as_double();
    gripper_speed_ = get_node()->get_parameter("gripper_speed").as_double();
    gripper_force_ = get_node()->get_parameter("gripper_force").as_double();
    gripper_epsilon_inner_ = get_node()->get_parameter("gripper_epsilon_inner").as_double();
    gripper_epsilon_outer_ = get_node()->get_parameter("gripper_epsilon_outer").as_double();
    open_width_ = get_node()->get_parameter("open_width").as_double();
    open_speed_ = get_node()->get_parameter("open_speed").as_double();

    franka_cartesian_pose_ =
        std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
            franka_semantic_components::FrankaCartesianPoseInterface(arm_prefix_,
                                                                     k_elbow_activated_));

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
    }

    robot_type_ =
        robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
    try {
      joint7_limits_ =
          extract_joint_limits_from_description(robot_description_, robot_type_ + "_joint7");
      RCLCPP_INFO(get_node()->get_logger(),
                  "Loaded %s limits: [%.4f, %.4f] rad",
                  (robot_type_ + "_joint7").c_str(), joint7_limits_.lower, joint7_limits_.upper);
    } catch (const std::exception& exception) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s", exception.what());
      return CallbackReturn::ERROR;
    }

    std::string action_namespace = get_node()->get_namespace();
    if (action_namespace == "/") {
      action_namespace.clear();
    }

    gripper_homing_action_client_ = rclcpp_action::create_client<franka_msgs::action::Homing>(
        get_node(), action_namespace + "/franka_gripper/homing");
    gripper_grasp_action_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(
        get_node(), action_namespace + "/franka_gripper/grasp");
    gripper_move_action_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
        get_node(), action_namespace + "/franka_gripper/move");
    assign_homing_goal_options_callbacks();
    assign_move_goal_options_callbacks();
    assign_grasp_goal_options_callbacks();
    parameter_callback_handle_ = get_node()->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) {
          return handle_parameter_update(parameters);
        });

    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State&) override {
    initialization_flag_ = true;
    initial_robot_time_ = 0.0;
    robot_time_ = 0.0;
    franka_cartesian_pose_->assign_loaned_command_interfaces(command_interfaces_);
    franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);
    robot_time_interface_index_ = franka_cartesian_pose_->get_state_interface_names().size();
    joint7_position_interface_index_ = robot_time_interface_index_ + 1;
    if (!gripper_homing_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_node()->get_logger(), "Homing Action server not available after waiting.");
      return CallbackReturn::ERROR;
    }
    if (!gripper_move_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_node()->get_logger(), "Move Action server not available after waiting.");
      return CallbackReturn::ERROR;
    }
    if (!gripper_grasp_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_node()->get_logger(), "Grasp Action server not available after waiting.");
      return CallbackReturn::ERROR;
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override {
    franka_cartesian_pose_->release_interfaces();
    return CallbackReturn::SUCCESS;
  }

 private:
  enum class MotionPhase {
    kIdle,
    kWaitForHoming,
    kWaitForOpen,
    kPrepareMoveToHover,
    kMoveToHover,
    kPrepareMoveToTarget,
    kMoveToTarget,
    kWaitForGrasp,
    kLift,
    kHold
  };

  double compute_progress(double elapsed_time) const {
    const double normalized_time = std::clamp(elapsed_time / motion_duration_sec_, 0.0, 1.0);
    const double t2 = normalized_time * normalized_time;
    const double t3 = t2 * normalized_time;
    const double t4 = t3 * normalized_time;
    const double t5 = t4 * normalized_time;
    return 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
  }

  static Eigen::Vector3d project_onto_plane(const Eigen::Vector3d& vector,
                                            const Eigen::Vector3d& plane_normal) {
    return vector - vector.dot(plane_normal) * plane_normal;
  }

  static Eigen::Quaterniond build_orientation_from_yz_axes(const Eigen::Vector3d& desired_y_axis,
                                                           const Eigen::Vector3d& desired_z_axis) {
    Eigen::Vector3d z_axis = desired_z_axis.normalized();
    Eigen::Vector3d y_axis = project_onto_plane(desired_y_axis, z_axis);
    if (y_axis.norm() < 1e-6) {
      y_axis = Eigen::Vector3d::UnitX();
      y_axis = project_onto_plane(y_axis, z_axis);
    }
    y_axis.normalize();
    Eigen::Vector3d x_axis = y_axis.cross(z_axis).normalized();
    y_axis = z_axis.cross(x_axis).normalized();

    Eigen::Matrix3d rotation_matrix;
    rotation_matrix.col(0) = x_axis;
    rotation_matrix.col(1) = y_axis;
    rotation_matrix.col(2) = z_axis;
    return Eigen::Quaterniond(rotation_matrix);
  }

  static double signed_angle_about_axis(const Eigen::Vector3d& from_vector,
                                        const Eigen::Vector3d& to_vector,
                                        const Eigen::Vector3d& axis) {
    Eigen::Vector3d normalized_axis = axis.normalized();
    Eigen::Vector3d from_projected = project_onto_plane(from_vector, normalized_axis);
    Eigen::Vector3d to_projected = project_onto_plane(to_vector, normalized_axis);

    if (from_projected.norm() < 1e-6 || to_projected.norm() < 1e-6) {
      return 0.0;
    }

    from_projected.normalize();
    to_projected.normalize();
    const double sine = normalized_axis.dot(from_projected.cross(to_projected));
    const double cosine = std::clamp(from_projected.dot(to_projected), -1.0, 1.0);
    return std::atan2(sine, cosine);
  }

  Eigen::Quaterniond compute_hover_orientation() const {
    const Eigen::Vector3d current_y_axis =
        (initial_orientation_ * Eigen::Vector3d::UnitY()).normalized();
    return build_orientation_from_yz_axes(current_y_axis, kHoverTargetZAxis);
  }

  Eigen::Vector3d choose_gripper_direction_for_target() const {
    const auto [current_hover_orientation, current_hover_position] =
        franka_cartesian_pose_->getCurrentOrientationAndTranslation();
    (void)current_hover_position;
    const Eigen::Vector3d hover_y_axis =
        (current_hover_orientation * Eigen::Vector3d::UnitY()).normalized();

    struct CandidateEvaluation {
      Eigen::Vector3d direction;
      double alignment_score;
      double estimated_joint7;
      double limit_margin;
      bool within_limits;
    };

    auto evaluate_candidate = [&](const Eigen::Vector3d& direction) -> CandidateEvaluation {
      const double alignment_score = hover_y_axis.dot(direction);
      const double delta_joint7 =
          signed_angle_about_axis(hover_y_axis, direction, kHoverTargetZAxis);
      const double estimated_joint7 = current_joint7_position_ + delta_joint7;
      const double limit_margin =
          std::min(estimated_joint7 - joint7_limits_.lower, joint7_limits_.upper - estimated_joint7);
      return CandidateEvaluation{
          direction, alignment_score, estimated_joint7, limit_margin,
          estimated_joint7 >= joint7_limits_.lower && estimated_joint7 <= joint7_limits_.upper};
    };

    const CandidateEvaluation candidate_a = evaluate_candidate(gripper_direction_a_);
    const CandidateEvaluation candidate_b = evaluate_candidate(gripper_direction_b_);

    const CandidateEvaluation* selected_candidate = nullptr;
    if (candidate_a.within_limits != candidate_b.within_limits) {
      selected_candidate = candidate_a.within_limits ? &candidate_a : &candidate_b;
    } else if (candidate_a.limit_margin != candidate_b.limit_margin) {
      selected_candidate = candidate_a.limit_margin > candidate_b.limit_margin ? &candidate_a : &candidate_b;
    } else {
      selected_candidate =
          candidate_a.alignment_score >= candidate_b.alignment_score ? &candidate_a : &candidate_b;
    }

    if (!selected_candidate->within_limits) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Neither gripper direction keeps estimated joint7 within limits [%.4f, %.4f]. "
          "Choosing the one with larger remaining margin. A=%.4f rad, B=%.4f rad.",
          joint7_limits_.lower, joint7_limits_.upper, candidate_a.estimated_joint7,
          candidate_b.estimated_joint7);
    } else {
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Selected gripper direction with estimated joint7 %.4f rad inside limits [%.4f, %.4f].",
          selected_candidate->estimated_joint7, joint7_limits_.lower, joint7_limits_.upper);
    }

    return selected_candidate->direction;
  }

  Eigen::Quaterniond compute_target_orientation() const {
    return build_orientation_from_yz_axes(selected_gripper_direction_, kHoverTargetZAxis);
  }

  void update_motion_state() {
    switch (motion_phase_) {
      case MotionPhase::kIdle:
        commanded_position_ = initial_position_;
        commanded_orientation_ = initial_orientation_;
        break;
      case MotionPhase::kWaitForHoming:
        update_wait_for_homing_phase();
        break;
      case MotionPhase::kWaitForOpen:
        update_wait_for_open_phase();
        break;
      case MotionPhase::kPrepareMoveToHover:
        update_prepare_move_to_hover_phase();
        break;
      case MotionPhase::kMoveToHover:
        update_move_to_hover_phase();
        break;
      case MotionPhase::kMoveToTarget:
        update_move_to_target_phase();
        break;
      case MotionPhase::kPrepareMoveToTarget:
        update_prepare_move_to_target_phase();
        break;
      case MotionPhase::kWaitForGrasp:
        update_wait_for_grasp_phase();
        break;
      case MotionPhase::kLift:
        update_lift_phase();
        break;
      case MotionPhase::kHold:
        commanded_position_ = lift_target_position_;
        commanded_orientation_ = target_orientation_;
        break;
    }
  }

  rcl_interfaces::msg::SetParametersResult handle_parameter_update(
      const std::vector<rclcpp::Parameter>& parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    Eigen::Vector3d requested_target = requested_target_position_;
    Eigen::Vector3d requested_gripper_direction = requested_gripper_direction_;
    Eigen::Vector3d requested_gripper_direction_a = gripper_direction_a_;
    Eigen::Vector3d requested_gripper_direction_b = gripper_direction_b_;
    bool target_changed = false;

    for (const auto& parameter : parameters) {
      if (parameter.get_name() == "target_cable_id") {
        try {
          const auto grasp_plan = load_grasp_plan_from_file(parameter.as_string());
          requested_target = grasp_plan.grasp_point;
          requested_gripper_direction = grasp_plan.gripper_direction_a;
          requested_gripper_direction_a = grasp_plan.gripper_direction_a;
          requested_gripper_direction_b = grasp_plan.gripper_direction_b;
          target_changed = true;
        } catch (const std::exception& exception) {
          result.successful = false;
          result.reason = exception.what();
          return result;
        }
      }
    }

    if (target_changed) {
      {
        std::lock_guard<std::mutex> lock(target_update_mutex_);
        requested_target_position_ = requested_target;
        requested_gripper_direction_ = requested_gripper_direction;
        gripper_direction_a_ = requested_gripper_direction_a;
        gripper_direction_b_ = requested_gripper_direction_b;
      }
      target_update_requested_ = true;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Received new target pose: [%.6f, %.6f, %.6f]",
                  requested_target.x(), requested_target.y(), requested_target.z());
    }

    return result;
  }

  void apply_pending_target_update() {
    if (!target_update_requested_.exchange(false)) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(target_update_mutex_);
      target_position_ = requested_target_position_;
      selected_gripper_direction_ = requested_gripper_direction_;
    }
    if (initialization_flag_) {
      return;
    }

    initial_position_ = commanded_position_;
    initial_orientation_ = commanded_orientation_;
    hover_position_ = target_position_ + kHoverApproachOffset;
    hover_orientation_ = compute_hover_orientation();
    phase_start_time_ = robot_time_;
    motion_phase_ = MotionPhase::kWaitForHoming;
    homing_goal_sent_ = false;
    homing_result_received_ = false;
    homing_succeeded_ = false;
    open_goal_sent_ = false;
    open_result_received_ = false;
    open_succeeded_ = false;
    grasp_goal_sent_ = false;
    grasp_result_received_ = false;
    grasp_succeeded_ = false;
    if (!homing_goal_sent_) {
      send_homing_goal();
    }

    RCLCPP_INFO(get_node()->get_logger(),
                "Applying updated target position and starting state machine.");
  }

  void update_wait_for_open_phase() {
    commanded_position_ = initial_position_;
    commanded_orientation_ = initial_orientation_;

    if (!open_result_received_) {
      return;
    }

    if (!open_succeeded_) {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                            "Gripper open failed. Holding initial pose.");
      return;
    }

    phase_start_time_ = robot_time_;
    motion_phase_ = MotionPhase::kPrepareMoveToHover;
    RCLCPP_INFO(get_node()->get_logger(), "Gripper opened. Starting motion to hover pose.");
  }

  void update_wait_for_homing_phase() {
    commanded_position_ = initial_position_;
    commanded_orientation_ = initial_orientation_;

    if (!homing_result_received_) {
      return;
    }

    if (!homing_succeeded_) {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                            "Gripper homing failed. Holding initial pose.");
      return;
    }

    motion_phase_ = MotionPhase::kWaitForOpen;
    if (!open_goal_sent_) {
      send_open_goal();
    }
    RCLCPP_INFO(get_node()->get_logger(), "Gripper homing succeeded. Starting open action.");
  }

  void update_prepare_move_to_hover_phase() {
    commanded_position_ = initial_position_;
    commanded_orientation_ = initial_orientation_;

    initial_robot_time_ = robot_time_;
    motion_phase_ = MotionPhase::kMoveToHover;
  }

  void update_move_to_hover_phase() {
    const double elapsed_time = std::max(0.0, robot_time_ - initial_robot_time_);
    const double alpha = compute_progress(elapsed_time);

    commanded_position_ = initial_position_ + alpha * (hover_position_ - initial_position_);
    commanded_orientation_ = initial_orientation_.slerp(alpha, hover_orientation_);

    if (elapsed_time >= motion_duration_sec_) {
      commanded_position_ = hover_position_;
      commanded_orientation_ = hover_orientation_;
      selected_gripper_direction_ = choose_gripper_direction_for_target();
      target_orientation_ = compute_target_orientation();
      phase_start_time_ = robot_time_;
      motion_phase_ = MotionPhase::kPrepareMoveToTarget;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Reached hover pose. Selected gripper direction [%.4f, %.4f, %.4f].",
                  selected_gripper_direction_.x(), selected_gripper_direction_.y(),
                  selected_gripper_direction_.z());
    }
  }

  void update_prepare_move_to_target_phase() {
    commanded_position_ = hover_position_;
    commanded_orientation_ = hover_orientation_;

    initial_robot_time_ = robot_time_;
    motion_phase_ = MotionPhase::kMoveToTarget;
  }

  void update_move_to_target_phase() {
    const double elapsed_time = std::max(0.0, robot_time_ - initial_robot_time_);
    const double alpha = compute_progress(elapsed_time);

    commanded_position_ = hover_position_ + alpha * (target_position_ - hover_position_);
    commanded_orientation_ = hover_orientation_.slerp(alpha, target_orientation_);

    if (elapsed_time >= motion_duration_sec_) {
      commanded_position_ = target_position_;
      commanded_orientation_ = target_orientation_;
      if (!grasp_goal_sent_) {
        send_grasp_goal();
      }
      motion_phase_ = MotionPhase::kWaitForGrasp;
    }
  }

  void update_wait_for_grasp_phase() {
    commanded_position_ = target_position_;
    commanded_orientation_ = target_orientation_;

    if (!grasp_result_received_) {
      return;
    }

    if (!grasp_succeeded_) {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                            "Gripper grasp failed. Holding target pose without lifting.");
      return;
    }

    lift_start_position_ = target_position_;
    lift_target_position_ = target_position_ + Eigen::Vector3d(0.0, -lift_distance_, 0.0);
    phase_start_time_ = robot_time_;
    motion_phase_ = MotionPhase::kLift;
    RCLCPP_INFO(get_node()->get_logger(), "Grasp succeeded. Starting lift motion.");
  }

  void update_lift_phase() {
    const double elapsed_time = std::max(0.0, robot_time_ - phase_start_time_);
    const double alpha = compute_progress(elapsed_time);

    commanded_position_ = lift_start_position_ + alpha * (lift_target_position_ - lift_start_position_);
    commanded_orientation_ = target_orientation_;

    if (elapsed_time >= motion_duration_sec_) {
      commanded_position_ = lift_target_position_;
      motion_phase_ = MotionPhase::kHold;
      RCLCPP_INFO(get_node()->get_logger(), "Lift finished. Holding lifted pose.");
    }
  }

  void send_grasp_goal() {
    franka_msgs::action::Grasp::Goal grasp_goal;
    grasp_goal.width = gripper_width_;
    grasp_goal.speed = gripper_speed_;
    grasp_goal.force = gripper_force_;
    grasp_goal.epsilon.inner = gripper_epsilon_inner_;
    grasp_goal.epsilon.outer = gripper_epsilon_outer_;

    grasp_goal_sent_ = true;
    grasp_result_received_ = false;
    grasp_succeeded_ = false;

    auto goal_handle_future =
        gripper_grasp_action_client_->async_send_goal(grasp_goal, grasp_goal_options_);
    if (!goal_handle_future.valid()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to submit grasp goal.");
      grasp_result_received_ = true;
      grasp_succeeded_ = false;
      return;
    }

      RCLCPP_INFO(get_node()->get_logger(),
                "Target pose reached. Submitted grasp goal: width=%.4f speed=%.3f force=%.1f",
                gripper_width_, gripper_speed_, gripper_force_);
  }

  void send_homing_goal() {
    franka_msgs::action::Homing::Goal homing_goal;

    homing_goal_sent_ = true;
    homing_result_received_ = false;
    homing_succeeded_ = false;

    auto goal_handle_future =
        gripper_homing_action_client_->async_send_goal(homing_goal, homing_goal_options_);
    if (!goal_handle_future.valid()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to submit homing goal.");
      homing_result_received_ = true;
      homing_succeeded_ = false;
      return;
    }

    RCLCPP_INFO(get_node()->get_logger(), "Submitted gripper homing goal.");
  }

  void send_open_goal() {
    franka_msgs::action::Move::Goal move_goal;
    move_goal.width = open_width_;
    move_goal.speed = open_speed_;

    open_goal_sent_ = true;
    open_result_received_ = false;
    open_succeeded_ = false;

    auto goal_handle_future =
        gripper_move_action_client_->async_send_goal(move_goal, move_goal_options_);
    if (!goal_handle_future.valid()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to submit open goal.");
      open_result_received_ = true;
      open_succeeded_ = false;
      return;
    }

    RCLCPP_INFO(get_node()->get_logger(),
                "Submitted open goal: width=%.4f speed=%.3f", open_width_, open_speed_);
  }

  void assign_move_goal_options_callbacks() {
    move_goal_options_.goal_response_callback =
        [this](const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>>&
                   goal_handle) {
          if (!goal_handle) {
            open_result_received_ = true;
            open_succeeded_ = false;
            RCLCPP_ERROR(get_node()->get_logger(), "Open goal was rejected.");
          } else {
            RCLCPP_INFO(get_node()->get_logger(), "Open goal accepted.");
          }
        };

    move_goal_options_.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>::WrappedResult&
                   result) {
          open_result_received_ = true;
          open_succeeded_ =
              result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success;

          if (open_succeeded_) {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper open succeeded.");
          } else {
            const std::string error_message =
                result.result ? result.result->error : "unknown open error";
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper open failed: %s",
                         error_message.c_str());
          }
        };
  }

  void assign_homing_goal_options_callbacks() {
    homing_goal_options_.goal_response_callback =
        [this](const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::Homing>>&
                   goal_handle) {
          if (!goal_handle) {
            homing_result_received_ = true;
            homing_succeeded_ = false;
            RCLCPP_ERROR(get_node()->get_logger(), "Homing goal was rejected.");
          } else {
            RCLCPP_INFO(get_node()->get_logger(), "Homing goal accepted.");
          }
        };

    homing_goal_options_.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Homing>::WrappedResult&
                   result) {
          homing_result_received_ = true;
          homing_succeeded_ =
              result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success;

          if (homing_succeeded_) {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper homing succeeded.");
          } else {
            const std::string error_message =
                result.result ? result.result->error : "unknown homing error";
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper homing failed: %s",
                         error_message.c_str());
          }
        };
  }

  void assign_grasp_goal_options_callbacks() {
    grasp_goal_options_.goal_response_callback =
        [this](const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>>&
                   goal_handle) {
          if (!goal_handle) {
            grasp_result_received_ = true;
            grasp_succeeded_ = false;
            RCLCPP_ERROR(get_node()->get_logger(), "Grasp goal was rejected.");
          } else {
            RCLCPP_INFO(get_node()->get_logger(), "Grasp goal accepted.");
          }
        };

    grasp_goal_options_.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::WrappedResult&
                   result) {
          grasp_result_received_ = true;
          grasp_succeeded_ =
              result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success;

          if (grasp_succeeded_) {
            RCLCPP_INFO(get_node()->get_logger(), "Grasp succeeded.");
          } else {
            const std::string error_message =
                result.result ? result.result->error : "unknown grasp error";
            RCLCPP_ERROR(get_node()->get_logger(), "Grasp failed: %s", error_message.c_str());
          }
        };
  }

  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Homing>> gripper_homing_action_client_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Grasp>> gripper_grasp_action_client_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Move>> gripper_move_action_client_;
  rclcpp_action::Client<franka_msgs::action::Homing>::SendGoalOptions homing_goal_options_;
  rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions move_goal_options_;
  rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions grasp_goal_options_;

  const bool k_elbow_activated_{false};
  bool initialization_flag_{true};
  bool is_gazebo_{false};
  MotionPhase motion_phase_{MotionPhase::kIdle};
  std::atomic_bool homing_goal_sent_{false};
  std::atomic_bool homing_result_received_{false};
  std::atomic_bool homing_succeeded_{false};
  std::atomic_bool open_goal_sent_{false};
  std::atomic_bool open_result_received_{false};
  std::atomic_bool open_succeeded_{false};
  std::atomic_bool grasp_goal_sent_{false};
  std::atomic_bool grasp_result_received_{false};
  std::atomic_bool grasp_succeeded_{false};

  double initial_robot_time_{0.0};
  double robot_time_{0.0};
  double phase_start_time_{0.0};
  double motion_duration_sec_{8.0};
  double lift_distance_{0.30};
  double gripper_width_{0.004};
  double gripper_speed_{0.02};
  double gripper_force_{10.0};
  double gripper_epsilon_inner_{0.001};
  double gripper_epsilon_outer_{0.001};
  double open_width_{0.08};
  double open_speed_{0.05};
  double current_joint7_position_{0.0};

  Eigen::Quaterniond initial_orientation_;
  Eigen::Quaterniond hover_orientation_;
  Eigen::Quaterniond target_orientation_;
  Eigen::Quaterniond commanded_orientation_;
  Eigen::Vector3d initial_position_;
  Eigen::Vector3d hover_position_;
  Eigen::Vector3d commanded_position_;
  //Eigen::Vector3d target_position_{0.2, 0.2, 0.6};0.497664, 0.386541, 0.487742
  Eigen::Vector3d target_position_{0.397664, 0.386541, 0.587742};
  Eigen::Vector3d requested_target_position_{0.397664, 0.386541, 0.587742};
  Eigen::Vector3d gripper_direction_a_{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d gripper_direction_b_{-Eigen::Vector3d::UnitX()};
  Eigen::Vector3d selected_gripper_direction_{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d requested_gripper_direction_{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d lift_start_position_;
  Eigen::Vector3d lift_target_position_;

  std::string robot_description_;
  std::string robot_type_{"fr3"};
  std::string arm_prefix_;
  JointLimits joint7_limits_{-3.0159, 3.0159};
  std::atomic_bool target_update_requested_{false};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
  std::mutex target_update_mutex_;
  size_t robot_time_interface_index_{0};
  size_t joint7_position_interface_index_{0};
};

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianTargetPointController,
                       controller_interface::ControllerInterface)
