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
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/homing.hpp>
#include <franka_msgs/action/move.hpp>
#include <franka_msgs/msg/franka_robot_state.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <franka_semantic_components/franka_robot_state.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <limits>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

namespace {

constexpr const char* kGraspPointRoot = "/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint";
const Eigen::Vector3d kHoverApproachOffset(0.0, -0.15, 0.0);
const Eigen::Vector3d kHoverTargetZAxis = Eigen::Vector3d::UnitY();
constexpr double kMaxTcpZAxisAngleToLink0YAxisDeg = 60.0;

struct GraspPlan {
  Eigen::Vector3d grasp_point;
  Eigen::Vector3d plane_normal;
  std::vector<Eigen::Vector3d> gripper_directions;
};

struct JointLimits {
  double lower;
  double upper;
};

struct SelectedDirectionDecision {
  Eigen::Vector3d direction;
  Eigen::Vector3d plane_normal;
  std::size_t index;
  double y_axis_rotation_about_positive_y;
  double x_axis_roll_about_selected_y;
  double z_axis_angle_to_link0_y_deg;
  double estimated_joint7;
  double limit_margin;
  bool within_tcp_z_constraint;
  bool within_limits;
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

bool is_relative_grasp_plan_id(const std::string& target_id) {
  return target_id.find('/') != std::string::npos || target_id.find('\\') != std::string::npos;
}

std::filesystem::path checked_relative_grasp_plan_path(const std::string& target_id) {
  if (target_id.empty()) {
    throw std::runtime_error("target_cable_id must not be empty.");
  }

  std::filesystem::path relative_path(target_id);
  if (relative_path.is_absolute()) {
    throw std::runtime_error("target_cable_id relative grasp plan paths must not be absolute.");
  }
  for (const auto& part : relative_path) {
    if (part == "..") {
      throw std::runtime_error("target_cable_id relative grasp plan paths must not contain '..'.");
    }
  }
  return std::filesystem::path(kGraspPointRoot) / relative_path;
}

std::filesystem::path build_grasp_plan_path(const std::string& target_id) {
  if (is_relative_grasp_plan_id(target_id)) {
    return checked_relative_grasp_plan_path(target_id);
  }

  const std::string normalized_cable_id = normalize_cable_id(target_id);
  const std::filesystem::path cable_dir =
      std::filesystem::path(kGraspPointRoot) / ("cable_" + normalized_cable_id);
  return cable_dir / ("grasp_point_cable_" + normalized_cable_id);
}

std::filesystem::path build_viewer_selection_asset_path(const std::string& target_id) {
  if (is_relative_grasp_plan_id(target_id)) {
    const std::filesystem::path grasp_plan_path = checked_relative_grasp_plan_path(target_id);
    return grasp_plan_path.parent_path() / (grasp_plan_path.filename().string() + "_selection.js");
  }

  const std::string normalized_cable_id = normalize_cable_id(target_id);
  const std::filesystem::path cable_dir =
      std::filesystem::path(kGraspPointRoot) / ("cable_" + normalized_cable_id);
  return cable_dir / ("grasp_curve_selection_cable_" + normalized_cable_id + ".js");
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

void write_text_file(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to write file: " + path.string());
  }

  output << contents;
}

void persist_viewer_selection_asset(const std::string& cable_id,
                                    const SelectedDirectionDecision& decision) {
  if (cable_id.empty()) {
    return;
  }

  const Eigen::Vector3d selected_x_axis = decision.plane_normal.normalized();
  const Eigen::Vector3d selected_y_axis = decision.direction.normalized();
  const Eigen::Vector3d selected_z_axis = selected_x_axis.cross(selected_y_axis).normalized();

  std::ostringstream script;
  script << "window.viewerSelection = {"
         << "\"selected_gripper_direction_index\":" << decision.index << ","
         << "\"selected_tcp_x_axis\":["
         << selected_x_axis.x() << "," << selected_x_axis.y() << "," << selected_x_axis.z() << "],"
         << "\"selected_gripper_direction\":["
         << selected_y_axis.x() << "," << selected_y_axis.y() << "," << selected_y_axis.z() << "],"
         << "\"selected_tcp_z_axis\":["
         << selected_z_axis.x() << "," << selected_z_axis.y() << "," << selected_z_axis.z() << "],"
         << "\"rotation_about_positive_y\":"
         << (decision.y_axis_rotation_about_positive_y + decision.x_axis_roll_about_selected_y) << ","
         << "\"estimated_joint7\":" << decision.estimated_joint7 << ","
         << "\"limit_margin\":" << decision.limit_margin << ","
         << "\"within_limits\":" << (decision.within_limits ? "true" : "false") << "};\n";
  write_text_file(build_viewer_selection_asset_path(cable_id), script.str());
}

std::vector<double> extract_number_list(const std::string& text, const std::string& key) {
  const std::string key_token = "\"" + key + "\"";
  const std::size_t key_position = text.find(key_token);
  if (key_position == std::string::npos) {
    throw std::runtime_error("Failed to find list field '" + key + "' in grasp plan file.");
  }

  const std::size_t array_start = text.find('[', key_position);
  if (array_start == std::string::npos) {
    throw std::runtime_error("Failed to find array start for field '" + key + "'.");
  }

  int bracket_depth = 0;
  std::size_t array_end = std::string::npos;
  for (std::size_t idx = array_start; idx < text.size(); ++idx) {
    if (text[idx] == '[') {
      ++bracket_depth;
    } else if (text[idx] == ']') {
      --bracket_depth;
      if (bracket_depth == 0) {
        array_end = idx;
        break;
      }
    }
  }

  if (array_end == std::string::npos) {
    throw std::runtime_error("Failed to find array end for field '" + key + "'.");
  }

  const std::string array_text = text.substr(array_start, array_end - array_start + 1);
  std::vector<double> values;
  const std::regex number_pattern("[-+0-9.eE]+");
  for (std::sregex_iterator it(array_text.begin(), array_text.end(), number_pattern), end; it != end;
       ++it) {
    values.push_back(std::stod(it->str()));
  }
  return values;
}

std::vector<Eigen::Vector3d> extract_vector3_list(const std::string& text, const std::string& key) {
  const std::string key_token = "\"" + key + "\"";
  const std::size_t key_position = text.find(key_token);
  if (key_position == std::string::npos) {
    throw std::runtime_error("Failed to find vector list field '" + key + "' in grasp plan file.");
  }

  const std::size_t array_start = text.find('[', key_position);
  if (array_start == std::string::npos) {
    throw std::runtime_error("Failed to find array start for field '" + key + "'.");
  }

  int bracket_depth = 0;
  std::size_t array_end = std::string::npos;
  for (std::size_t idx = array_start; idx < text.size(); ++idx) {
    if (text[idx] == '[') {
      ++bracket_depth;
    } else if (text[idx] == ']') {
      --bracket_depth;
      if (bracket_depth == 0) {
        array_end = idx;
        break;
      }
    }
  }

  if (array_end == std::string::npos) {
    throw std::runtime_error("Failed to find array end for field '" + key + "'.");
  }

  const std::string array_text = text.substr(array_start, array_end - array_start + 1);
  const std::regex vector_pattern(
      "\\[\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*\\]");
  std::vector<Eigen::Vector3d> vectors;
  for (std::sregex_iterator it(array_text.begin(), array_text.end(), vector_pattern), end; it != end;
       ++it) {
    vectors.emplace_back(std::stod((*it)[1].str()), std::stod((*it)[2].str()),
                         std::stod((*it)[3].str()));
  }

  if (vectors.empty()) {
    throw std::runtime_error("Field '" + key + "' must contain at least one 3D vector.");
  }

  return vectors;
}

GraspPlan load_grasp_plan_from_file(const std::string& cable_id) {
  const std::filesystem::path grasp_plan_path = build_grasp_plan_path(cable_id);
  const std::string file_text = read_text_file(grasp_plan_path);
  const std::vector<double> grasp_point = extract_number_list(file_text, "grasp_point");
  const std::vector<double> plane_normal = extract_number_list(file_text, "local_tangent");
  if (grasp_point.size() != 3) {
    throw std::runtime_error("Field 'grasp_point' must contain exactly 3 values.");
  }
  if (plane_normal.size() != 3) {
    throw std::runtime_error("Field 'local_tangent' must contain exactly 3 values.");
  }

  return GraspPlan{
      Eigen::Vector3d(grasp_point[0], grasp_point[1], grasp_point[2]),
      Eigen::Vector3d(plane_normal[0], plane_normal[1], plane_normal[2]),
      extract_vector3_list(file_text, "gripper_directions"),
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

class CartesianCableInteractController : public controller_interface::ControllerInterface {
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
    config.names.push_back(arm_prefix_ + robot_type_ + "/robot_time");
    return config;
  }

  controller_interface::return_type update(const rclcpp::Time&,
                                           const rclcpp::Duration& period) override {
    if (initialization_flag_) {
      std::tie(initial_orientation_, initial_position_) =
          franka_cartesian_pose_->getCurrentOrientationAndTranslation();
      target_orientation_ = compute_target_orientation(initial_orientation_);
      commanded_position_ = initial_position_;
      commanded_orientation_ = initial_orientation_;
      update_joint_states();
      previous_tau_commanded_.setZero();
      initial_robot_time_ = state_interfaces_.at(robot_time_interface_index_).get_value();
      phase_start_time_ = initial_robot_time_;
      motion_phase_ = MotionPhase::kIdle;
      open_goal_sent_ = false;
      open_result_received_ = false;
      open_succeeded_ = false;
      grasp_goal_sent_ = false;
      grasp_result_received_ = false;
      grasp_succeeded_ = false;
      board_contact_force_baseline_valid_ = false;
      board_contact_settle_start_time_ = -1.0;
      interaction_recording_active_ = false;
      interaction_action_index_ = -1;
      last_interaction_publish_time_ = -1.0;
      initialization_flag_ = false;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Controller activated. Holding current pose until a new target is received.");
    }

    robot_time_ = state_interfaces_.at(robot_time_interface_index_).get_value();
    update_joint_states();
    update_external_force_state();
    current_joint7_position_ = joint_positions_current_[6];
    apply_pending_target_update();
    const auto [current_orientation, current_position] =
        franka_cartesian_pose_->getCurrentOrientationAndTranslation();
    update_motion_state(current_position, current_orientation);

    const Vector7d tau_d_calculated = compute_torque_command(
        current_position, current_orientation, vector_from_std(joint_velocities_current_));
    for (int i = 0; i < 7; ++i) {
      command_interfaces_[i].set_value(tau_d_calculated(i));
    }

    (void)period;
    return controller_interface::return_type::OK;
  }

  CallbackReturn on_init() override {
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("target_cable_id", "");
    auto_declare<std::vector<double>>("cartesian_stiffness",
                                      {800.0, 900.0, 1050.0, 35.0, 35.0, 30.0});
    auto_declare<std::vector<double>>("cartesian_damping",
                                      {56.7, 60.0, 64.8, 11.8, 11.8, 11.0});
    auto_declare<std::vector<double>>("max_torque_deltas",
                                      {0.25, 0.25, 0.22, 0.20, 0.15, 0.12, 0.10});
    auto_declare<double>("max_translation_error", 0.10);
    auto_declare<double>("max_rotation_error", 0.35);
    auto_declare<double>("motion_duration", 8.0);
    auto_declare<double>("align_y_duration", 8.0);
    auto_declare<double>("roll_stage1_duration", 12.0);
    auto_declare<double>("roll_stage2_duration", 12.0);
    auto_declare<double>("roll_midpoint_ratio", 0.5);
    auto_declare<double>("move_to_target_duration", 24.0);
    auto_declare<bool>("board_contact_probe_enabled", true);
    auto_declare<double>("board_contact_detection_min_progress", 0.85);
    auto_declare<double>("board_contact_probe_speed", 0.002);
    auto_declare<double>("board_contact_max_probe_distance", 0.03);
    auto_declare<double>("board_contact_max_probe_duration", 20.0);
    auto_declare<double>("board_contact_detection_force", 3.0);
    auto_declare<double>("board_contact_target_force", 1.0);
    auto_declare<double>("board_contact_force_tolerance", 0.4);
    auto_declare<double>("board_contact_force_gain", 0.0005);
    auto_declare<double>("board_contact_settle_time", 0.3);
    auto_declare<double>("external_force_filter_alpha", 0.2);
    auto_declare<double>("gripper_width", 0.004);
    auto_declare<double>("gripper_speed", 0.02);
    auto_declare<double>("gripper_force", 10.0);
    auto_declare<double>("gripper_epsilon_inner", 0.003);
    auto_declare<double>("gripper_epsilon_outer", 0.003);
    auto_declare<double>("open_width", 0.08);
    auto_declare<double>("open_speed", 0.05);
    auto_declare<bool>("interaction_enabled", true);
    auto_declare<double>("interaction_amplitude", 0.03);
    auto_declare<double>("interaction_move_duration", 3.0);
    auto_declare<double>("interaction_hold_duration", 0.5);
    auto_declare<double>("interaction_publish_rate", 30.0);
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
    align_y_duration_sec_ = get_node()->get_parameter("align_y_duration").as_double();
    roll_stage1_duration_sec_ =
        get_node()->get_parameter("roll_stage1_duration").as_double();
    roll_stage2_duration_sec_ =
        get_node()->get_parameter("roll_stage2_duration").as_double();
    roll_midpoint_ratio_ =
        get_node()->get_parameter("roll_midpoint_ratio").as_double();
    move_to_target_duration_sec_ =
        get_node()->get_parameter("move_to_target_duration").as_double();
    board_contact_probe_enabled_ =
        get_node()->get_parameter("board_contact_probe_enabled").as_bool();
    board_contact_detection_min_progress_ = std::clamp(
        get_node()->get_parameter("board_contact_detection_min_progress").as_double(), 0.0, 1.0);
    board_contact_probe_speed_mps_ =
        get_node()->get_parameter("board_contact_probe_speed").as_double();
    board_contact_max_probe_distance_m_ =
        get_node()->get_parameter("board_contact_max_probe_distance").as_double();
    board_contact_max_probe_duration_sec_ =
        get_node()->get_parameter("board_contact_max_probe_duration").as_double();
    board_contact_detection_force_n_ =
        get_node()->get_parameter("board_contact_detection_force").as_double();
    board_contact_target_force_n_ =
        get_node()->get_parameter("board_contact_target_force").as_double();
    board_contact_force_tolerance_n_ =
        get_node()->get_parameter("board_contact_force_tolerance").as_double();
    board_contact_force_gain_ =
        get_node()->get_parameter("board_contact_force_gain").as_double();
    board_contact_settle_time_sec_ =
        get_node()->get_parameter("board_contact_settle_time").as_double();
    external_force_filter_alpha_ =
        std::clamp(get_node()->get_parameter("external_force_filter_alpha").as_double(), 0.0, 1.0);
    if (motion_duration_sec_ <= 0.0 || align_y_duration_sec_ <= 0.0 ||
        roll_stage1_duration_sec_ <= 0.0 || roll_stage2_duration_sec_ <= 0.0 ||
        move_to_target_duration_sec_ <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Motion duration parameters must all be greater than 0.");
      return CallbackReturn::ERROR;
    }
    if (board_contact_probe_speed_mps_ <= 0.0 || board_contact_max_probe_distance_m_ <= 0.0 ||
        board_contact_max_probe_duration_sec_ <= 0.0 ||
        board_contact_detection_force_n_ <= 0.0 || board_contact_target_force_n_ < 0.0 ||
        board_contact_force_tolerance_n_ < 0.0 || board_contact_force_gain_ <= 0.0 ||
        board_contact_settle_time_sec_ < 0.0) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Board contact probe parameters must be positive, except target force/tolerance/settle "
          "time may be zero.");
      return CallbackReturn::ERROR;
    }
    roll_midpoint_ratio_ = std::clamp(roll_midpoint_ratio_, 0.1, 0.9);
    gripper_width_ = get_node()->get_parameter("gripper_width").as_double();
    gripper_speed_ = get_node()->get_parameter("gripper_speed").as_double();
    gripper_force_ = get_node()->get_parameter("gripper_force").as_double();
    gripper_epsilon_inner_ = get_node()->get_parameter("gripper_epsilon_inner").as_double();
    gripper_epsilon_outer_ = get_node()->get_parameter("gripper_epsilon_outer").as_double();
    open_width_ = get_node()->get_parameter("open_width").as_double();
    open_speed_ = get_node()->get_parameter("open_speed").as_double();
    interaction_enabled_ = get_node()->get_parameter("interaction_enabled").as_bool();
    interaction_amplitude_m_ = get_node()->get_parameter("interaction_amplitude").as_double();
    interaction_move_duration_sec_ =
        get_node()->get_parameter("interaction_move_duration").as_double();
    interaction_hold_duration_sec_ =
        get_node()->get_parameter("interaction_hold_duration").as_double();
    interaction_publish_rate_hz_ =
        get_node()->get_parameter("interaction_publish_rate").as_double();
    if (interaction_amplitude_m_ <= 0.0 || interaction_move_duration_sec_ <= 0.0 ||
        interaction_hold_duration_sec_ < 0.0 || interaction_publish_rate_hz_ <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Interaction amplitude/move duration/publish rate must be positive, "
                   "and hold duration must be non-negative.");
      return CallbackReturn::ERROR;
    }
    interaction_action_pub_ = get_node()->create_publisher<std_msgs::msg::Int32>(
        "cable_interaction/action_index", 10);
    interaction_active_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(
        "cable_interaction/recording_active", 10);
    interaction_ee_point_pub_ = get_node()->create_publisher<geometry_msgs::msg::PointStamped>(
        "cable_interaction/ee_point", 10);
    const auto cartesian_stiffness =
        get_node()->get_parameter("cartesian_stiffness").as_double_array();
    const auto cartesian_damping =
        get_node()->get_parameter("cartesian_damping").as_double_array();
    const auto max_torque_deltas = get_node()->get_parameter("max_torque_deltas").as_double_array();
    if (cartesian_stiffness.size() != 6 || cartesian_damping.size() != 6 ||
        max_torque_deltas.size() != 7) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "cartesian_stiffness and cartesian_damping must each have 6 values, "
                   "and max_torque_deltas must have 7 values.");
      return CallbackReturn::ERROR;
    }
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
    }

    robot_type_ =
        robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
    try {
      franka_robot_state_ = std::make_unique<franka_semantic_components::FrankaRobotState>(
          arm_prefix_ + robot_type_ + "/" + k_robot_state_interface_name, robot_description_);
      franka_robot_state_->initialize_robot_state_msg(franka_robot_state_msg_);
    } catch (const std::exception& exception) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to initialize franka robot state: %s",
                   exception.what());
      return CallbackReturn::ERROR;
    }
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
    franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);
    franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
    franka_robot_state_->assign_loaned_state_interfaces(state_interfaces_);
    external_force_initialized_ = false;
    external_force_valid_ = false;
    pose_state_interface_count_ = franka_cartesian_pose_->get_state_interface_names().size();
    joint_position_state_start_index_ = pose_state_interface_count_;
    joint_velocity_state_start_index_ = joint_position_state_start_index_ + 7;
    joint_effort_state_start_index_ = joint_velocity_state_start_index_ + 7;
    robot_time_interface_index_ =
        joint_effort_state_start_index_ + 7 + franka_robot_model_->get_state_interface_names().size();
    if (!gripper_homing_action_client_->wait_for_action_server(std::chrono::milliseconds(500))) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Homing Action server is not visible during activation. "
                  "The controller will stay active and wait again when a target is triggered.");
    }
    if (!gripper_move_action_client_->wait_for_action_server(std::chrono::milliseconds(500))) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Move Action server is not visible during activation. "
                  "The controller will stay active and wait again when a target is triggered.");
    }
    if (!gripper_grasp_action_client_->wait_for_action_server(std::chrono::milliseconds(500))) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Grasp Action server is not visible during activation. "
                  "The controller will stay active and wait again when a target is triggered.");
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override {
    franka_cartesian_pose_->release_interfaces();
    franka_robot_model_->release_interfaces();
    franka_robot_state_->release_interfaces();
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
    kAdjustBoardContact,
    kCloseGripper,
    kCableInteraction,
    kHold
  };

  static double compute_progress(double elapsed_time, double duration_sec) {
    const double normalized_time = std::clamp(elapsed_time / duration_sec, 0.0, 1.0);
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

  static Eigen::Quaterniond build_orientation_from_xy_axes(const Eigen::Vector3d& desired_x_axis,
                                                           const Eigen::Vector3d& desired_y_axis) {
    Eigen::Vector3d x_axis = desired_x_axis.normalized();
    Eigen::Vector3d y_axis = project_onto_plane(desired_y_axis, x_axis);
    if (y_axis.norm() < 1e-6) {
      y_axis = project_onto_plane(Eigen::Vector3d::UnitY(), x_axis);
    }
    if (y_axis.norm() < 1e-6) {
      y_axis = project_onto_plane(Eigen::Vector3d::UnitZ(), x_axis);
    }
    y_axis.normalize();
    Eigen::Vector3d z_axis = x_axis.cross(y_axis).normalized();
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

  static Vector7d vector_from_std(const std::vector<double>& values) {
    Vector7d vector = Vector7d::Zero();
    for (size_t i = 0; i < 7 && i < values.size(); ++i) {
      vector(static_cast<Eigen::Index>(i)) = values[i];
    }
    return vector;
  }

  void update_joint_states() {
    for (int i = 0; i < 7; ++i) {
      joint_positions_current_[i] =
          state_interfaces_.at(joint_position_state_start_index_ + i).get_value();
      joint_velocities_current_[i] =
          state_interfaces_.at(joint_velocity_state_start_index_ + i).get_value();
      joint_efforts_current_[i] =
          state_interfaces_.at(joint_effort_state_start_index_ + i).get_value();
    }
  }

  void update_external_force_state() {
    if (!franka_robot_state_ ||
        !franka_robot_state_->get_values_as_message(franka_robot_state_msg_)) {
      external_force_valid_ = false;
      return;
    }

    const double raw_force_y = franka_robot_state_msg_.o_f_ext_hat_k.wrench.force.y;
    if (!external_force_initialized_) {
      external_force_y_filtered_ = raw_force_y;
      external_force_initialized_ = true;
    } else {
      external_force_y_filtered_ =
          external_force_filter_alpha_ * raw_force_y +
          (1.0 - external_force_filter_alpha_) * external_force_y_filtered_;
    }
    external_force_y_raw_ = raw_force_y;
    external_force_valid_ = true;
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

  bool target_position_converged(const Eigen::Vector3d& current_position) const {
    const Eigen::Vector3d translation_error = target_position_ - current_position;
    const double position_error = translation_error.norm();
    return position_error <= target_position_convergence_threshold_m_;
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
    const Eigen::Vector3d translation_error = commanded_position_ - current_position;
    pose_error.head<3>() = translation_error;
    pose_error.tail<3>() =
        compute_orientation_error(current_orientation, commanded_orientation_);
    for (int i = 0; i < 3; ++i) {
      pose_error(i) = std::clamp(pose_error(i), -max_translation_error_, max_translation_error_);
      pose_error(i + 3) =
          std::clamp(pose_error(i + 3), -max_rotation_error_, max_rotation_error_);
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

  Eigen::Quaterniond compute_hover_orientation() const {
    const Eigen::Vector3d current_y_axis =
        (initial_orientation_ * Eigen::Vector3d::UnitY()).normalized();
    return build_orientation_from_yz_axes(current_y_axis, kHoverTargetZAxis);
  }

  SelectedDirectionDecision choose_gripper_direction_for_target() const {
    const auto [current_hover_orientation, current_hover_position] =
        franka_cartesian_pose_->getCurrentOrientationAndTranslation();
    (void)current_hover_position;
    const Eigen::Vector3d hover_y_axis =
        (current_hover_orientation * Eigen::Vector3d::UnitY()).normalized();
    const Eigen::Vector3d hover_x_axis =
        (current_hover_orientation * Eigen::Vector3d::UnitX()).normalized();

    Eigen::Vector3d plane_normal = selected_plane_normal_;
    if (plane_normal.norm() < 1e-6) {
      plane_normal = Eigen::Vector3d::UnitX();
    }
    plane_normal.normalize();

    auto evaluate_candidate = [&](const Eigen::Vector3d& direction, const Eigen::Vector3d& candidate_x_axis,
                                  std::size_t index) -> SelectedDirectionDecision {
      const Eigen::Vector3d normalized_direction = direction.normalized();
      const Eigen::Vector3d normalized_x_axis = candidate_x_axis.normalized();
      const Eigen::Vector3d normalized_z_axis = normalized_x_axis.cross(normalized_direction).normalized();
      const double y_axis_rotation_about_positive_y =
          signed_angle_about_axis(hover_y_axis, normalized_direction, Eigen::Vector3d::UnitY());
      Eigen::Vector3d hover_x_on_target_plane = project_onto_plane(hover_x_axis, normalized_direction);
      if (hover_x_on_target_plane.norm() < 1e-6) {
        hover_x_on_target_plane = project_onto_plane(hover_y_axis, normalized_direction);
      }
      if (hover_x_on_target_plane.norm() < 1e-6) {
        hover_x_on_target_plane = project_onto_plane(Eigen::Vector3d::UnitX(), normalized_direction);
      }
      hover_x_on_target_plane.normalize();
      const double x_axis_roll_about_selected_y = signed_angle_about_axis(
          hover_x_on_target_plane, normalized_x_axis, normalized_direction);
      const double estimated_joint7 =
          current_joint7_position_ + y_axis_rotation_about_positive_y + x_axis_roll_about_selected_y;
      const double limit_margin =
          std::min(estimated_joint7 - joint7_limits_.lower, joint7_limits_.upper - estimated_joint7);
      const double z_axis_cosine_to_link0_y =
          std::clamp(normalized_z_axis.dot(Eigen::Vector3d::UnitY()), -1.0, 1.0);
      const double z_axis_angle_to_link0_y_deg =
          std::acos(z_axis_cosine_to_link0_y) * 180.0 / M_PI;
      const bool within_tcp_z_constraint =
          z_axis_angle_to_link0_y_deg <= kMaxTcpZAxisAngleToLink0YAxisDeg;
      return SelectedDirectionDecision{
          normalized_direction,
          normalized_x_axis,
          index,
          y_axis_rotation_about_positive_y,
          x_axis_roll_about_selected_y,
          z_axis_angle_to_link0_y_deg,
          estimated_joint7,
          limit_margin,
          within_tcp_z_constraint,
          estimated_joint7 >= joint7_limits_.lower && estimated_joint7 <= joint7_limits_.upper};
    };

    if (gripper_directions_.empty()) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "No gripper directions available in the grasp plan. Falling back to UnitX.");
      return SelectedDirectionDecision{
          Eigen::Vector3d::UnitY(),
          plane_normal,
          0,
          0.0,
          0.0,
          90.0,
          current_joint7_position_,
          0.0,
          false,
          false};
    }

    auto candidate_is_better = [](const SelectedDirectionDecision& candidate,
                                  const SelectedDirectionDecision& current_best) -> bool {
      const double candidate_total_rotation =
          std::abs(candidate.y_axis_rotation_about_positive_y + candidate.x_axis_roll_about_selected_y);
      const double current_best_total_rotation = std::abs(
          current_best.y_axis_rotation_about_positive_y + current_best.x_axis_roll_about_selected_y);

      if (candidate.within_tcp_z_constraint != current_best.within_tcp_z_constraint) {
        return candidate.within_tcp_z_constraint;
      }
      if (candidate.within_limits != current_best.within_limits) {
        return candidate.within_limits;
      }
      if (candidate.limit_margin != current_best.limit_margin) {
        return candidate.limit_margin > current_best.limit_margin;
      }
      if (candidate_total_rotation != current_best_total_rotation) {
        return candidate_total_rotation < current_best_total_rotation;
      }
      return false;
    };

    bool has_selected_candidate = false;
    SelectedDirectionDecision selected_candidate{};
    const std::array<Eigen::Vector3d, 2> plane_normal_sides{{plane_normal, -plane_normal}};

    // Outer loop: choose hand_tcp.y from the 36 sampled in-plane grasp directions.
    for (std::size_t direction_index = 0; direction_index < gripper_directions_.size(); ++direction_index) {
      const Eigen::Vector3d& candidate_y_axis = gripper_directions_[direction_index];

      // Inner loop: choose which side of the grasp plane the hand_tcp.x axis should point to.
      for (const Eigen::Vector3d& candidate_x_axis : plane_normal_sides) {
        const SelectedDirectionDecision candidate =
            evaluate_candidate(candidate_y_axis, candidate_x_axis, direction_index);
        if (!has_selected_candidate || candidate_is_better(candidate, selected_candidate)) {
          selected_candidate = candidate;
          has_selected_candidate = true;
        }
      }
    }

    if (!(selected_candidate.within_tcp_z_constraint && selected_candidate.within_limits)) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "No sampled grasp orientation satisfies both tcp.z angle <= %.1f deg to link0 +Y and "
          "joint7 limits [%.4f, %.4f]. Choosing candidate %zu with x-axis [%.4f, %.4f, %.4f], "
          "tcp.z angle %.2f deg, total rotation %.4f rad and estimated joint7 %.4f rad.",
          kMaxTcpZAxisAngleToLink0YAxisDeg, joint7_limits_.lower, joint7_limits_.upper,
          selected_candidate.index, selected_candidate.plane_normal.x(),
          selected_candidate.plane_normal.y(), selected_candidate.plane_normal.z(),
          selected_candidate.z_axis_angle_to_link0_y_deg,
          selected_candidate.y_axis_rotation_about_positive_y +
              selected_candidate.x_axis_roll_about_selected_y,
          selected_candidate.estimated_joint7);
    } else {
      RCLCPP_INFO(
          get_node()->get_logger(),
          "Selected candidate %zu with x-axis [%.4f, %.4f, %.4f], tcp.z angle %.2f deg to "
          "link0 +Y, y-rotation %.4f rad, roll %.4f rad and estimated joint7 %.4f rad inside "
          "limits [%.4f, %.4f].",
          selected_candidate.index, selected_candidate.plane_normal.x(),
          selected_candidate.plane_normal.y(), selected_candidate.plane_normal.z(),
          selected_candidate.z_axis_angle_to_link0_y_deg,
          selected_candidate.y_axis_rotation_about_positive_y,
          selected_candidate.x_axis_roll_about_selected_y, selected_candidate.estimated_joint7,
          joint7_limits_.lower, joint7_limits_.upper);
    }

    return selected_candidate;
  }

  Eigen::Quaterniond compute_target_orientation(const Eigen::Quaterniond& reference_orientation) const {
    (void)reference_orientation;
    return build_orientation_from_xy_axes(selected_plane_normal_, selected_gripper_direction_);
  }

  void update_motion_state(const Eigen::Vector3d& current_position,
                           const Eigen::Quaterniond& current_orientation) {
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
      case MotionPhase::kPrepareMoveToTarget:
        update_prepare_move_to_target_phase();
        break;
      case MotionPhase::kMoveToTarget:
        update_move_to_target_phase(current_position, current_orientation);
        break;
      case MotionPhase::kAdjustBoardContact:
        update_adjust_board_contact_phase();
        break;
      case MotionPhase::kCloseGripper:
        update_close_gripper_phase();
        break;
      case MotionPhase::kCableInteraction:
        update_cable_interaction_phase(current_position);
        break;
      case MotionPhase::kHold:
        commanded_position_ = target_position_;
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
    Eigen::Vector3d requested_plane_normal = requested_plane_normal_;
    std::vector<Eigen::Vector3d> requested_gripper_directions = gripper_directions_;
    bool target_changed = false;

    for (const auto& parameter : parameters) {
      if (parameter.get_name() == "target_cable_id") {
        try {
          const auto grasp_plan = load_grasp_plan_from_file(parameter.as_string());
          current_target_cable_id_ = parameter.as_string();
          requested_target = grasp_plan.grasp_point;
          requested_plane_normal = grasp_plan.plane_normal;
          requested_gripper_directions = grasp_plan.gripper_directions;
          requested_gripper_direction = grasp_plan.gripper_directions.front();
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
        requested_plane_normal_ = requested_plane_normal;
        gripper_directions_ = requested_gripper_directions;
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
      selected_plane_normal_ = requested_plane_normal_;
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
    board_contact_force_baseline_valid_ = false;
    board_contact_settle_start_time_ = -1.0;
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
    const double alpha = compute_progress(elapsed_time, motion_duration_sec_);

    commanded_position_ = initial_position_ + alpha * (hover_position_ - initial_position_);
    commanded_orientation_ = initial_orientation_.slerp(alpha, hover_orientation_);

    if (elapsed_time >= motion_duration_sec_) {
      commanded_position_ = hover_position_;
      commanded_orientation_ = hover_orientation_;
      const SelectedDirectionDecision selected_direction_decision =
          choose_gripper_direction_for_target();
      selected_gripper_direction_ = selected_direction_decision.direction;
      selected_plane_normal_ = selected_direction_decision.plane_normal;
      try {
        persist_viewer_selection_asset(current_target_cable_id_, selected_direction_decision);
      } catch (const std::exception& exception) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "Failed to persist viewer selection asset: %s",
                    exception.what());
      }
      target_orientation_ = compute_target_orientation(hover_orientation_);
      phase_start_time_ = robot_time_;
      motion_phase_ = MotionPhase::kPrepareMoveToTarget;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Reached hover pose. Selected candidate %zu with gripper y-axis [%.4f, %.4f, %.4f] "
                  "and x-axis [%.4f, %.4f, %.4f].",
                  selected_direction_decision.index, selected_gripper_direction_.x(),
                  selected_gripper_direction_.y(), selected_gripper_direction_.z(),
                  selected_plane_normal_.x(), selected_plane_normal_.y(), selected_plane_normal_.z());
    }
  }

  void update_prepare_move_to_target_phase() {
    commanded_position_ = hover_position_;
    commanded_orientation_ = hover_orientation_;

    board_contact_force_baseline_valid_ = false;
    if (board_contact_probe_enabled_ && external_force_valid_) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Board contact detection will start after target approach progress %.2f. "
                  "Until then force_y baseline will track the filtered estimate.",
                  board_contact_detection_min_progress_);
    } else if (board_contact_probe_enabled_) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "External force estimate is unavailable before target approach. "
                  "Board contact adjustment may not be able to run.");
    }

    initial_robot_time_ = robot_time_;
    motion_phase_ = MotionPhase::kMoveToTarget;
    RCLCPP_INFO(get_node()->get_logger(),
                "Reached hover pose with downward orientation. Starting direct move to target with target orientation.");
  }

  void update_move_to_target_phase(const Eigen::Vector3d& current_position,
                                   const Eigen::Quaterniond& current_orientation) {
    const double elapsed_time = std::max(0.0, robot_time_ - initial_robot_time_);
    const double alpha =
        elapsed_time >= move_to_target_duration_sec_
            ? 1.0
            : compute_progress(elapsed_time, move_to_target_duration_sec_);

    commanded_position_ = hover_position_ + alpha * (target_position_ - hover_position_);
    commanded_orientation_ = hover_orientation_.slerp(alpha, target_orientation_);

    if (board_contact_probe_enabled_ && external_force_valid_) {
      if (alpha < board_contact_detection_min_progress_) {
        board_contact_force_baseline_y_ = external_force_y_filtered_;
        board_contact_force_baseline_valid_ = true;
      } else {
        if (!board_contact_force_baseline_valid_) {
          board_contact_force_baseline_y_ = external_force_y_filtered_;
          board_contact_force_baseline_valid_ = true;
        }
        const double force_y_delta = external_force_y_filtered_ - board_contact_force_baseline_y_;
        const double contact_force_n = std::abs(force_y_delta);
        if (contact_force_n >= board_contact_detection_force_n_) {
          start_board_contact_adjustment();
          RCLCPP_INFO(get_node()->get_logger(),
                      "Detected board contact during target approach. progress=%.3f, "
                      "force_y_delta=%.3f N, magnitude=%.3f N. Switching to light-contact regulation.",
                      alpha, force_y_delta, contact_force_n);
          return;
        }
      }
    }

    if (elapsed_time >= move_to_target_duration_sec_) {
      commanded_position_ = target_position_;
      commanded_orientation_ = target_orientation_;
      const Eigen::Vector3d translation_error = target_position_ - current_position;
      const double position_error = translation_error.norm();
      const Eigen::Vector3d orientation_error_vector =
          compute_orientation_error(current_orientation, target_orientation_);
      const double orientation_error = orientation_error_vector.norm();
      if (target_position_converged(current_position)) {
        start_gripper_close_phase();
        RCLCPP_INFO(
            get_node()->get_logger(),
            "Reached target pose. Starting gripper close. Position error [x=%.4f, y=%.4f, z=%.4f] m "
            "(norm %.4f m), orientation error [rx=%.4f, ry=%.4f, rz=%.4f] rad (norm %.4f rad).",
            translation_error.x(), translation_error.y(), translation_error.z(), position_error,
            orientation_error_vector.x(), orientation_error_vector.y(),
            orientation_error_vector.z(), orientation_error);
      } else {
        RCLCPP_INFO_THROTTLE(
            get_node()->get_logger(), *get_node()->get_clock(), 2000,
            "Waiting for target pose convergence. Position error [x=%.4f, y=%.4f, z=%.4f] m "
            "(norm %.4f m, threshold %.4f m), orientation error [rx=%.4f, ry=%.4f, rz=%.4f] rad "
            "(norm %.4f rad, threshold %.4f rad).",
            translation_error.x(), translation_error.y(), translation_error.z(), position_error,
            target_position_convergence_threshold_m_, orientation_error_vector.x(),
            orientation_error_vector.y(), orientation_error_vector.z(), orientation_error,
            target_orientation_convergence_threshold_rad_);
      }
    }
  }

  void start_gripper_close_phase() {
    grasp_goal_sent_ = false;
    grasp_result_received_ = false;
    grasp_succeeded_ = false;
    motion_phase_ = MotionPhase::kCloseGripper;
  }

  void start_board_contact_adjustment() {
    commanded_orientation_ = target_orientation_;

    if (!external_force_valid_) {
      target_position_ = commanded_position_;
      RCLCPP_ERROR(get_node()->get_logger(),
                   "External force estimate is unavailable. Holding current pose without closing.");
      motion_phase_ = MotionPhase::kHold;
      return;
    }

    if (!board_contact_force_baseline_valid_) {
      board_contact_force_baseline_y_ = external_force_y_filtered_;
      board_contact_force_baseline_valid_ = true;
    }

    board_contact_probe_start_time_ = robot_time_;
    board_contact_probe_last_update_time_ = robot_time_;
    board_contact_settle_start_time_ = -1.0;
    motion_phase_ = MotionPhase::kAdjustBoardContact;

    RCLCPP_INFO(
        get_node()->get_logger(),
        "Board contact adjustment started after force contact during target approach. baseline force_y=%.3f N "
        "(raw %.3f N), detection force %.3f N, target light force %.3f N.",
        board_contact_force_baseline_y_, external_force_y_raw_, board_contact_detection_force_n_,
        board_contact_target_force_n_);
  }

  void finish_board_contact_probe(double contact_force_n) {
    target_position_ = commanded_position_;
    requested_target_position_ = target_position_;
    start_gripper_close_phase();
    RCLCPP_INFO(get_node()->get_logger(),
                "Board contact adjustment finished. New target position [%.6f, %.6f, %.6f], "
                "relative contact force %.3f N. Starting gripper close.",
                target_position_.x(), target_position_.y(), target_position_.z(), contact_force_n);
  }

  void abort_board_contact_probe(const std::string& reason, double contact_force_n) {
    target_position_ = commanded_position_;
    requested_target_position_ = target_position_;
    motion_phase_ = MotionPhase::kHold;
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Board contact adjustment aborted: %s. Holding adjusted target [%.6f, %.6f, %.6f] "
                 "without closing. Relative contact force %.3f N.",
                 reason.c_str(), target_position_.x(), target_position_.y(), target_position_.z(),
                 contact_force_n);
  }

  void update_adjust_board_contact_phase() {
    commanded_orientation_ = target_orientation_;

    if (!external_force_valid_) {
      abort_board_contact_probe("external force estimate became unavailable", 0.0);
      return;
    }

    const double elapsed_time = std::max(0.0, robot_time_ - board_contact_probe_start_time_);
    const double dt = std::clamp(robot_time_ - board_contact_probe_last_update_time_, 0.0, 0.02);
    board_contact_probe_last_update_time_ = robot_time_;

    const double force_y_delta = external_force_y_filtered_ - board_contact_force_baseline_y_;
    const double contact_force_n = std::abs(force_y_delta);
    const double y_offset_from_original_target = target_position_.y() - commanded_position_.y();

    if (elapsed_time > board_contact_max_probe_duration_sec_) {
      abort_board_contact_probe("max probe duration reached", contact_force_n);
      return;
    }
    if (y_offset_from_original_target > board_contact_max_probe_distance_m_) {
      abort_board_contact_probe("max y adjustment distance reached", contact_force_n);
      return;
    }

    const double force_error_n = board_contact_target_force_n_ - contact_force_n;
    const double velocity_mps = std::clamp(board_contact_force_gain_ * force_error_n,
                                           -board_contact_probe_speed_mps_,
                                           board_contact_probe_speed_mps_);
    commanded_position_.y() =
        std::clamp(commanded_position_.y() + velocity_mps * dt,
                   target_position_.y() - board_contact_max_probe_distance_m_, target_position_.y());

    if (std::abs(force_error_n) <= board_contact_force_tolerance_n_) {
      if (board_contact_settle_start_time_ < 0.0) {
        board_contact_settle_start_time_ = robot_time_;
      }
      if (robot_time_ - board_contact_settle_start_time_ >= board_contact_settle_time_sec_) {
        finish_board_contact_probe(contact_force_n);
        return;
      }
    } else {
      board_contact_settle_start_time_ = -1.0;
    }

    RCLCPP_INFO_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "Regulating board contact by adjusting link0 Y. force_y_delta %.3f N, magnitude %.3f N, "
        "target %.3f +/- %.3f N, y %.6f (original target %.6f), velocity %.5f m/s.",
        force_y_delta, contact_force_n, board_contact_target_force_n_,
        board_contact_force_tolerance_n_, commanded_position_.y(), target_position_.y(),
        velocity_mps);
  }

  void update_close_gripper_phase() {
    commanded_position_ = target_position_;
    commanded_orientation_ = target_orientation_;

    if (!grasp_goal_sent_) {
      send_grasp_goal();
      return;
    }

    if (!grasp_result_received_) {
      return;
    }

    if (grasp_succeeded_) {
      if (interaction_enabled_) {
        start_cable_interaction_phase();
        return;
      }
      RCLCPP_INFO(get_node()->get_logger(), "Gripper closed. Sequence finished.");
    } else {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Gripper close finished without success. Ending while holding target pose.");
    }

    publish_interaction_state(commanded_position_, -1, false, true);
    motion_phase_ = MotionPhase::kHold;
  }

  static Eigen::Vector3d interpolate_position(const Eigen::Vector3d& start,
                                              const Eigen::Vector3d& goal,
                                              double alpha) {
    return start + alpha * (goal - start);
  }

  void start_cable_interaction_phase() {
    interaction_base_position_ = target_position_;
    interaction_base_orientation_ = target_orientation_;
    initial_robot_time_ = robot_time_;
    interaction_recording_active_ = true;
    interaction_action_index_ = -1;
    last_interaction_publish_time_ = -1.0;
    commanded_position_ = interaction_base_position_;
    commanded_orientation_ = interaction_base_orientation_;
    motion_phase_ = MotionPhase::kCableInteraction;
    RCLCPP_INFO(get_node()->get_logger(),
                "Gripper closed. Starting cable interaction recording motion: +/- %.3f m along "
                "target tcp.x then target tcp.y.",
                interaction_amplitude_m_);
  }

  void update_cable_interaction_phase(const Eigen::Vector3d& current_position) {
    commanded_orientation_ = interaction_base_orientation_;
    const double elapsed_time = std::max(0.0, robot_time_ - initial_robot_time_);
    const Eigen::Vector3d tcp_x =
        (interaction_base_orientation_ * Eigen::Vector3d::UnitX()).normalized();
    const Eigen::Vector3d tcp_y =
        (interaction_base_orientation_ * Eigen::Vector3d::UnitY()).normalized();
    const Eigen::Vector3d center = interaction_base_position_;
    const Eigen::Vector3d x_positive = center + interaction_amplitude_m_ * tcp_x;
    const Eigen::Vector3d x_negative = center - interaction_amplitude_m_ * tcp_x;
    const Eigen::Vector3d y_positive = center + interaction_amplitude_m_ * tcp_y;
    const Eigen::Vector3d y_negative = center - interaction_amplitude_m_ * tcp_y;
    const double hold = interaction_hold_duration_sec_;
    const double move = interaction_move_duration_sec_;
    double segment_start = 0.0;
    int action_index = -1;

    auto apply_hold = [&](double duration) -> bool {
      if (elapsed_time < segment_start + duration) {
        commanded_position_ = center;
        action_index = -1;
        return true;
      }
      segment_start += duration;
      return false;
    };
    auto apply_move = [&](const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                          int action, double duration) -> bool {
      if (elapsed_time < segment_start + duration) {
        const double alpha = compute_progress(elapsed_time - segment_start, duration);
        commanded_position_ = interpolate_position(start, goal, alpha);
        action_index = action;
        return true;
      }
      segment_start += duration;
      return false;
    };

    bool segment_active = false;
    segment_active = apply_hold(hold) ||
                     apply_move(center, x_positive, 0, move) ||
                     apply_move(x_positive, x_negative, 0, move) ||
                     apply_move(x_negative, center, 0, move) ||
                     apply_hold(hold) ||
                     apply_move(center, y_positive, 1, move) ||
                     apply_move(y_positive, y_negative, 1, move) ||
                     apply_move(y_negative, center, 1, move) ||
                     apply_hold(hold);

    if (!segment_active) {
      commanded_position_ = center;
      commanded_orientation_ = interaction_base_orientation_;
      interaction_recording_active_ = false;
      interaction_action_index_ = -1;
      publish_interaction_state(current_position, -1, false, true);
      motion_phase_ = MotionPhase::kHold;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Cable interaction motion finished. Holding the target pose.");
      return;
    }

    interaction_action_index_ = action_index;
    publish_interaction_state(current_position, action_index, true, false);
  }

  void publish_interaction_state(const Eigen::Vector3d& ee_position, int action_index,
                                 bool recording_active, bool force_publish = false) {
    if (!interaction_action_pub_ || !interaction_active_pub_ || !interaction_ee_point_pub_) {
      return;
    }
    if (!force_publish && last_interaction_publish_time_ >= 0.0 &&
        robot_time_ - last_interaction_publish_time_ < 1.0 / interaction_publish_rate_hz_) {
      return;
    }
    last_interaction_publish_time_ = robot_time_;

    std_msgs::msg::Bool active_msg;
    active_msg.data = recording_active;
    interaction_active_pub_->publish(active_msg);

    std_msgs::msg::Int32 action_msg;
    action_msg.data = action_index;
    interaction_action_pub_->publish(action_msg);

    geometry_msgs::msg::PointStamped ee_msg;
    ee_msg.header.stamp = get_node()->now();
    ee_msg.header.frame_id = "robot_base";
    ee_msg.point.x = ee_position.x();
    ee_msg.point.y = ee_position.y();
    ee_msg.point.z = ee_position.z();
    interaction_ee_point_pub_->publish(ee_msg);
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
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to submit gripper close goal.");
      grasp_result_received_ = true;
      grasp_succeeded_ = false;
      return;
    }

    RCLCPP_INFO(get_node()->get_logger(),
                "Submitted gripper close goal: width=%.4f speed=%.3f force=%.1f",
                gripper_width_, gripper_speed_, gripper_force_);
  }

  void send_homing_goal() {
    if (!gripper_homing_action_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Cannot submit homing goal: Homing Action server is not available.");
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
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to submit homing goal.");
      homing_result_received_ = true;
      homing_succeeded_ = false;
      return;
    }

    RCLCPP_INFO(get_node()->get_logger(), "Submitted gripper homing goal.");
  }

  void send_open_goal() {
    if (!gripper_move_action_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Cannot submit open goal: Move Action server is not available.");
      open_goal_sent_ = false;
      open_result_received_ = true;
      open_succeeded_ = false;
      return;
    }

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
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper close goal was rejected.");
          } else {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper close goal accepted.");
          }
        };

    grasp_goal_options_.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::WrappedResult&
                   result) {
          grasp_result_received_ = true;
          grasp_succeeded_ = result.code == rclcpp_action::ResultCode::SUCCEEDED &&
                             result.result && result.result->success;

          if (grasp_succeeded_) {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper close succeeded.");
          } else {
            const std::string error_message =
                result.result ? result.result->error : "unknown gripper close error";
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper close failed: %s",
                         error_message.c_str());
          }
        };
  }

  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  std::unique_ptr<franka_semantic_components::FrankaRobotState> franka_robot_state_;
  franka_msgs::msg::FrankaRobotState franka_robot_state_msg_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Homing>> gripper_homing_action_client_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Grasp>> gripper_grasp_action_client_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Move>> gripper_move_action_client_;
  rclcpp_action::Client<franka_msgs::action::Homing>::SendGoalOptions homing_goal_options_;
  rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions grasp_goal_options_;
  rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions move_goal_options_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr interaction_action_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr interaction_active_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr interaction_ee_point_pub_;

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
  double align_y_duration_sec_{8.0};
  double roll_stage1_duration_sec_{8.0};
  double roll_stage2_duration_sec_{8.0};
  double roll_midpoint_ratio_{0.5};
  double move_to_target_duration_sec_{24.0};
  double target_position_convergence_threshold_m_{0.01};
  double target_orientation_convergence_threshold_rad_{0.1};
  bool board_contact_probe_enabled_{true};
  double board_contact_detection_min_progress_{0.85};
  double board_contact_probe_speed_mps_{0.002};
  double board_contact_max_probe_distance_m_{0.03};
  double board_contact_max_probe_duration_sec_{20.0};
  double board_contact_detection_force_n_{3.0};
  double board_contact_target_force_n_{1.0};
  double board_contact_force_tolerance_n_{0.4};
  double board_contact_force_gain_{0.0005};
  double board_contact_settle_time_sec_{0.3};
  double external_force_filter_alpha_{0.2};
  double external_force_y_raw_{0.0};
  double external_force_y_filtered_{0.0};
  bool external_force_initialized_{false};
  bool external_force_valid_{false};
  bool board_contact_force_baseline_valid_{false};
  double board_contact_force_baseline_y_{0.0};
  double board_contact_probe_start_time_{0.0};
  double board_contact_probe_last_update_time_{0.0};
  double board_contact_settle_start_time_{-1.0};
  double gripper_width_{0.004};
  double gripper_speed_{0.02};
  double gripper_force_{10.0};
  double gripper_epsilon_inner_{0.001};
  double gripper_epsilon_outer_{0.001};
  double open_width_{0.08};
  double open_speed_{0.05};
  bool interaction_enabled_{true};
  bool interaction_recording_active_{false};
  int interaction_action_index_{-1};
  double interaction_amplitude_m_{0.03};
  double interaction_move_duration_sec_{3.0};
  double interaction_hold_duration_sec_{0.5};
  double interaction_publish_rate_hz_{30.0};
  double last_interaction_publish_time_{-1.0};
  double current_joint7_position_{0.0};

  Eigen::Quaterniond initial_orientation_;
  Eigen::Quaterniond hover_orientation_;
  Eigen::Quaterniond target_orientation_;
  Eigen::Quaterniond commanded_orientation_;
  Eigen::Quaterniond interaction_base_orientation_;
  Eigen::Vector3d initial_position_;
  Eigen::Vector3d hover_position_;
  Eigen::Vector3d commanded_position_;
  Eigen::Vector3d interaction_base_position_;
  Eigen::Vector3d target_position_{0.397664, 0.386541, 0.587742};
  Eigen::Vector3d requested_target_position_{0.397664, 0.386541, 0.587742};
  std::vector<Eigen::Vector3d> gripper_directions_{
      Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX()};
  Eigen::Vector3d selected_plane_normal_{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d selected_gripper_direction_{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d requested_plane_normal_{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d requested_gripper_direction_{Eigen::Vector3d::UnitX()};

  std::string robot_description_;
  std::string robot_type_{"fr3"};
  std::string arm_prefix_;
  std::string current_target_cable_id_;
  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};
  JointLimits joint7_limits_{-3.0159, 3.0159};
  std::atomic_bool target_update_requested_{false};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
  std::mutex target_update_mutex_;
  size_t pose_state_interface_count_{0};
  size_t joint_position_state_start_index_{0};
  size_t joint_velocity_state_start_index_{0};
  size_t joint_effort_state_start_index_{0};
  size_t robot_time_interface_index_{0};
  Vector7d previous_tau_commanded_{Vector7d::Zero()};
  Vector6d cartesian_stiffness_{Vector6d::Zero()};
  Vector6d cartesian_damping_{Vector6d::Zero()};
  Vector7d max_torque_deltas_{Vector7d::Zero()};
  double max_translation_error_{0.10};
  double max_rotation_error_{0.35};
  std::vector<double> joint_positions_current_{0, 0, 0, 0, 0, 0, 0};
  std::vector<double> joint_velocities_current_{0, 0, 0, 0, 0, 0, 0};
  std::vector<double> joint_efforts_current_{0, 0, 0, 0, 0, 0, 0};
};

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianCableInteractController,
                       controller_interface::ControllerInterface)
