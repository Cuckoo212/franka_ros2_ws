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

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_msgs/action/move.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

namespace {

constexpr const char* kCablePointRoot =
    "/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint";
const Eigen::Vector3d kHoverApproachOffset(0.0, -0.15, 0.0);
const Eigen::Vector3d kHoverTargetZAxis = Eigen::Vector3d::UnitY();
constexpr double kCableTrajectoryYOffsetM = -0.02;

std::string normalize_cable_id(const std::string& cable_id) {
  if (cable_id.empty()) {
    return cable_id;
  }

  bool all_digits = std::all_of(cable_id.begin(), cable_id.end(),
                                [](unsigned char c) { return std::isdigit(c) != 0; });
  if (!all_digits) {
    return cable_id;
  }

  std::ostringstream stream;
  stream << std::setw(3) << std::setfill('0') << std::stoi(cable_id);
  return stream.str();
}

std::filesystem::path build_cable_plan_path(const std::string& cable_id) {
  const std::string normalized_cable_id = normalize_cable_id(cable_id);
  const std::filesystem::path cable_dir =
      std::filesystem::path(kCablePointRoot) / ("cable_" + normalized_cable_id);
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
  const std::string key_token = "\"" + key + "\"";
  const std::size_t key_position = text.find(key_token);
  if (key_position == std::string::npos) {
    throw std::runtime_error("Failed to find list field '" + key + "' in cable plan file.");
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
  for (std::sregex_iterator it(array_text.begin(), array_text.end(), number_pattern), end;
       it != end; ++it) {
    values.push_back(std::stod(it->str()));
  }
  return values;
}

std::vector<Eigen::Vector3d> parse_vector3_list(const std::vector<double>& values,
                                                const std::string& key) {
  if (values.size() < 6 || values.size() % 3 != 0) {
    throw std::runtime_error("Field '" + key +
                             "' must contain at least two 3D points encoded as triples.");
  }

  std::vector<Eigen::Vector3d> points;
  points.reserve(values.size() / 3);
  for (std::size_t idx = 0; idx < values.size(); idx += 3) {
    points.emplace_back(values[idx], values[idx + 1], values[idx + 2]);
  }
  return points;
}

std::vector<Eigen::Vector3d> load_cable_skeleton_from_file(const std::string& cable_id) {
  const std::filesystem::path cable_plan_path = build_cable_plan_path(cable_id);
  const std::string file_text = read_text_file(cable_plan_path);
  std::vector<Eigen::Vector3d> skeleton =
      parse_vector3_list(extract_number_list(file_text, "skeleton"), "skeleton");
  for (auto& point : skeleton) {
    point.y() += kCableTrajectoryYOffsetM;
  }
  return skeleton;
}

}  // namespace

class CartesianCableTrajectoryController : public controller_interface::ControllerInterface {
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
      config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) +
                             "/position");
    }
    for (int i = 1; i <= 7; ++i) {
      config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) +
                             "/velocity");
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
      commanded_position_ = initial_position_;
      commanded_orientation_ = initial_orientation_;
      target_position_ = initial_position_;
      target_orientation_ = initial_orientation_;
      update_joint_states();
      previous_tau_commanded_.setZero();
      initial_robot_time_ = state_interfaces_.at(robot_time_interface_index_).get_value();
      motion_phase_ = MotionPhase::kIdle;
      initialization_flag_ = false;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Controller activated. Holding current pose until target_cable_id is set.");
    }

    robot_time_ = state_interfaces_.at(robot_time_interface_index_).get_value();
    update_joint_states();
    apply_pending_trajectory_update();
    const auto [current_orientation, current_position] =
        franka_cartesian_pose_->getCurrentOrientationAndTranslation();
    update_motion_state(current_position);

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
    auto_declare<std::vector<double>>("cartesian_damping", {56.7, 60.0, 64.8, 11.8, 11.8, 11.0});
    auto_declare<std::vector<double>>("max_torque_deltas",
                                      {0.25, 0.25, 0.22, 0.20, 0.15, 0.12, 0.10});
    auto_declare<double>("max_translation_error", 0.10);
    auto_declare<double>("max_rotation_error", 0.35);
    auto_declare<double>("motion_duration", 8.0);
    auto_declare<double>("skeleton_trajectory_duration", 24.0);
    auto_declare<double>("close_width", 0.004);
    auto_declare<double>("close_speed", 0.02);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State&) override {
    is_gazebo_ = get_node()->get_parameter("gazebo").as_bool();
    arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
    arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
    target_position_ = Eigen::Vector3d::Zero();
    requested_cable_trajectory_points_.clear();
    cable_trajectory_points_.clear();
    active_cable_trajectory_points_.clear();
    active_cable_trajectory_arc_lengths_.clear();
    active_cable_trajectory_length_m_ = 0.0;
    requested_target_cable_id_.clear();
    current_target_cable_id_.clear();

    motion_duration_sec_ = get_node()->get_parameter("motion_duration").as_double();
    skeleton_trajectory_duration_sec_ =
        get_node()->get_parameter("skeleton_trajectory_duration").as_double();
    if (motion_duration_sec_ <= 0.0 || skeleton_trajectory_duration_sec_ <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "Motion duration parameters must be greater than 0.");
      return CallbackReturn::ERROR;
    }
    close_width_ = get_node()->get_parameter("close_width").as_double();
    close_speed_ = get_node()->get_parameter("close_speed").as_double();
    if (close_width_ < 0.0 || close_speed_ <= 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "close_width must be non-negative and close_speed must be greater than 0.");
      return CallbackReturn::ERROR;
    }

    const auto cartesian_stiffness =
        get_node()->get_parameter("cartesian_stiffness").as_double_array();
    const auto cartesian_damping = get_node()->get_parameter("cartesian_damping").as_double_array();
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

    std::string action_namespace = get_node()->get_namespace();
    if (action_namespace == "/") {
      action_namespace.clear();
    }
    gripper_move_action_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
        get_node(), action_namespace + "/franka_gripper/move");
    assign_close_goal_options_callbacks();

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
    const size_t pose_state_interface_count =
        franka_cartesian_pose_->get_state_interface_names().size();
    joint_position_state_start_index_ = pose_state_interface_count;
    joint_velocity_state_start_index_ = joint_position_state_start_index_ + 7;
    joint_effort_state_start_index_ = joint_velocity_state_start_index_ + 7;
    robot_time_interface_index_ = joint_effort_state_start_index_ + 7 +
                                  franka_robot_model_->get_state_interface_names().size();
    if (!gripper_move_action_client_->wait_for_action_server(std::chrono::milliseconds(500))) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Gripper move action server is not visible during activation. "
                  "The controller will stay active and wait again when a target is triggered.");
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override {
    franka_cartesian_pose_->release_interfaces();
    franka_robot_model_->release_interfaces();
    return CallbackReturn::SUCCESS;
  }

 private:
  enum class MotionPhase {
    kIdle,
    kWaitForGripperClose,
    kPrepareMoveToHover,
    kMoveToHover,
    kPrepareMoveToTarget,
    kMoveToTarget,
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
      y_axis = project_onto_plane(Eigen::Vector3d::UnitX(), z_axis);
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

  static Vector7d vector_from_std(const std::vector<double>& values) {
    Vector7d vector = Vector7d::Zero();
    for (size_t i = 0; i < 7 && i < values.size(); ++i) {
      vector(static_cast<Eigen::Index>(i)) = values[i];
    }
    return vector;
  }

  void update_joint_states() {
    for (int i = 0; i < 7; ++i) {
      joint_velocities_current_[i] =
          state_interfaces_.at(joint_velocity_state_start_index_ + i).get_value();
    }
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

    const Vector6d desired_wrench = cartesian_stiffness_.cwiseProduct(pose_error) -
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

  void update_motion_state(const Eigen::Vector3d& current_position) {
    switch (motion_phase_) {
      case MotionPhase::kIdle:
        commanded_position_ = initial_position_;
        commanded_orientation_ = initial_orientation_;
        break;
      case MotionPhase::kWaitForGripperClose:
        update_wait_for_gripper_close_phase();
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
        update_move_to_target_phase(current_position);
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

    std::vector<Eigen::Vector3d> requested_cable_trajectory = requested_cable_trajectory_points_;
    std::string requested_target_cable_id = requested_target_cable_id_;
    bool trajectory_changed = false;
    bool close_params_changed = false;
    double requested_close_width = close_width_;
    double requested_close_speed = close_speed_;

    for (const auto& parameter : parameters) {
      if (parameter.get_name() == "target_cable_id") {
        try {
          const std::string cable_id = parameter.as_string();
          requested_cable_trajectory = load_cable_skeleton_from_file(cable_id);
          requested_target_cable_id = normalize_cable_id(cable_id);
          trajectory_changed = true;
        } catch (const std::exception& exception) {
          result.successful = false;
          result.reason = exception.what();
          return result;
        }
      } else if (parameter.get_name() == "close_width") {
        requested_close_width = parameter.as_double();
        if (requested_close_width < 0.0) {
          result.successful = false;
          result.reason = "close_width must be non-negative.";
          return result;
        }
        close_params_changed = true;
      } else if (parameter.get_name() == "close_speed") {
        requested_close_speed = parameter.as_double();
        if (requested_close_speed <= 0.0) {
          result.successful = false;
          result.reason = "close_speed must be greater than 0.";
          return result;
        }
        close_params_changed = true;
      }
    }

    if (close_params_changed) {
      close_width_ = requested_close_width;
      close_speed_ = requested_close_speed;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Updated gripper close parameters: width=%.4f speed=%.3f.",
                  close_width_, close_speed_);
    }

    if (trajectory_changed) {
      {
        std::lock_guard<std::mutex> lock(target_update_mutex_);
        requested_cable_trajectory_points_ = requested_cable_trajectory;
        requested_target_cable_id_ = requested_target_cable_id;
      }
      trajectory_update_requested_ = true;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Received cable %s skeleton trajectory: %zu points, y offset %.3f m.",
                  requested_target_cable_id.c_str(), requested_cable_trajectory.size(),
                  kCableTrajectoryYOffsetM);
    }

    return result;
  }

  void apply_pending_trajectory_update() {
    if (!trajectory_update_requested_.exchange(false)) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(target_update_mutex_);
      cable_trajectory_points_ = requested_cable_trajectory_points_;
      current_target_cable_id_ = requested_target_cable_id_;
    }
    if (initialization_flag_) {
      return;
    }

    if (cable_trajectory_points_.size() < 2) {
      target_position_ = commanded_position_;
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Cable skeleton trajectory needs at least two points. Holding current pose.");
      motion_phase_ = MotionPhase::kHold;
      return;
    }

    initial_position_ = commanded_position_;
    initial_orientation_ = commanded_orientation_;
    hover_position_ = cable_trajectory_points_.front() + kHoverApproachOffset;
    hover_orientation_ = compute_hover_orientation();
    target_position_ = cable_trajectory_points_.back();
    target_orientation_ = hover_orientation_;
    motion_phase_ = MotionPhase::kWaitForGripperClose;
    close_goal_sent_ = false;
    close_result_received_ = false;
    close_succeeded_ = false;
    send_close_goal();

    RCLCPP_INFO(get_node()->get_logger(),
                "Applying cable %s trajectory. Closing gripper before moving to hover point "
                "[%.6f, %.6f, %.6f].",
                current_target_cable_id_.c_str(), hover_position_.x(), hover_position_.y(),
                hover_position_.z());
  }

  void update_wait_for_gripper_close_phase() {
    commanded_position_ = initial_position_;
    commanded_orientation_ = initial_orientation_;

    if (!close_result_received_) {
      return;
    }

    if (!close_succeeded_) {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                            "Gripper close failed. Holding initial pose.");
      return;
    }

    initial_robot_time_ = robot_time_;
    motion_phase_ = MotionPhase::kPrepareMoveToHover;
    RCLCPP_INFO(get_node()->get_logger(),
                "Gripper closed. Keeping the last gripper width and starting motion to hover pose.");
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
      motion_phase_ = MotionPhase::kPrepareMoveToTarget;
      RCLCPP_INFO(get_node()->get_logger(), "Reached hover pose. Starting cable trajectory.");
    }
  }

  bool rebuild_active_cable_trajectory() {
    active_cable_trajectory_points_.clear();
    active_cable_trajectory_arc_lengths_.clear();
    active_cable_trajectory_length_m_ = 0.0;

    active_cable_trajectory_points_.push_back(hover_position_);
    for (const auto& point : cable_trajectory_points_) {
      if ((point - active_cable_trajectory_points_.back()).norm() > 1e-6) {
        active_cable_trajectory_points_.push_back(point);
      }
    }

    if (active_cable_trajectory_points_.size() < 2) {
      return false;
    }

    active_cable_trajectory_arc_lengths_.reserve(active_cable_trajectory_points_.size());
    active_cable_trajectory_arc_lengths_.push_back(0.0);
    for (std::size_t idx = 1; idx < active_cable_trajectory_points_.size(); ++idx) {
      const double segment_length =
          (active_cable_trajectory_points_[idx] - active_cable_trajectory_points_[idx - 1]).norm();
      active_cable_trajectory_length_m_ += segment_length;
      active_cable_trajectory_arc_lengths_.push_back(active_cable_trajectory_length_m_);
    }

    return active_cable_trajectory_length_m_ > 1e-6;
  }

  Eigen::Vector3d sample_active_cable_trajectory(double alpha) const {
    if (active_cable_trajectory_points_.empty()) {
      return target_position_;
    }

    const double target_distance = std::clamp(alpha, 0.0, 1.0) * active_cable_trajectory_length_m_;
    for (std::size_t idx = 1; idx < active_cable_trajectory_points_.size(); ++idx) {
      if (target_distance <= active_cable_trajectory_arc_lengths_[idx]) {
        const double segment_start_distance = active_cable_trajectory_arc_lengths_[idx - 1];
        const double segment_length =
            active_cable_trajectory_arc_lengths_[idx] - segment_start_distance;
        const double segment_alpha =
            segment_length > 1e-9 ? (target_distance - segment_start_distance) / segment_length
                                  : 0.0;
        return active_cable_trajectory_points_[idx - 1] +
               segment_alpha * (active_cable_trajectory_points_[idx] -
                                active_cable_trajectory_points_[idx - 1]);
      }
    }

    return active_cable_trajectory_points_.back();
  }

  void update_prepare_move_to_target_phase() {
    commanded_position_ = hover_position_;
    commanded_orientation_ = hover_orientation_;

    if (!rebuild_active_cable_trajectory()) {
      target_position_ = commanded_position_;
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Failed to build a non-empty cable skeleton trajectory. Holding hover pose.");
      motion_phase_ = MotionPhase::kHold;
      return;
    }

    initial_robot_time_ = robot_time_;
    motion_phase_ = MotionPhase::kMoveToTarget;
    RCLCPP_INFO(
        get_node()->get_logger(),
        "Starting cable skeleton trajectory with %zu points, %.4f m arc length, y offset %.3f m.",
        active_cable_trajectory_points_.size(), active_cable_trajectory_length_m_,
        kCableTrajectoryYOffsetM);
  }

  void update_move_to_target_phase(const Eigen::Vector3d& current_position) {
    const double elapsed_time = std::max(0.0, robot_time_ - initial_robot_time_);
    const double alpha = elapsed_time >= skeleton_trajectory_duration_sec_
                             ? 1.0
                             : compute_progress(elapsed_time, skeleton_trajectory_duration_sec_);

    commanded_position_ = sample_active_cable_trajectory(alpha);
    commanded_orientation_ = target_orientation_;

    if (elapsed_time >= skeleton_trajectory_duration_sec_) {
      const Eigen::Vector3d final_position = active_cable_trajectory_points_.back();
      commanded_position_ = final_position;
      const Eigen::Vector3d translation_error = final_position - current_position;
      const double position_error = translation_error.norm();
      if (position_error <= target_position_convergence_threshold_m_) {
        target_position_ = final_position;
        motion_phase_ = MotionPhase::kHold;
        RCLCPP_INFO(get_node()->get_logger(),
                    "Finished cable skeleton trajectory. Holding final point [%.6f, %.6f, %.6f]. "
                    "Position error [x=%.4f, y=%.4f, z=%.4f] m (norm %.4f m).",
                    final_position.x(), final_position.y(), final_position.z(),
                    translation_error.x(), translation_error.y(), translation_error.z(),
                    position_error);
      } else {
        RCLCPP_INFO_THROTTLE(
            get_node()->get_logger(), *get_node()->get_clock(), 2000,
            "Waiting for cable trajectory final-point convergence. Position error [x=%.4f, y=%.4f, "
            "z=%.4f] m (norm %.4f m, threshold %.4f m).",
            translation_error.x(), translation_error.y(), translation_error.z(), position_error,
            target_position_convergence_threshold_m_);
      }
    }
  }

  void send_close_goal() {
    if (!gripper_move_action_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Cannot submit gripper close goal: Move Action server is not available.");
      close_goal_sent_ = false;
      close_result_received_ = true;
      close_succeeded_ = false;
      return;
    }

    franka_msgs::action::Move::Goal move_goal;
    move_goal.width = close_width_;
    move_goal.speed = close_speed_;

    close_goal_sent_ = true;
    close_result_received_ = false;
    close_succeeded_ = false;

    auto goal_handle_future =
        gripper_move_action_client_->async_send_goal(move_goal, close_goal_options_);
    if (!goal_handle_future.valid()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to submit gripper close goal.");
      close_result_received_ = true;
      close_succeeded_ = false;
      return;
    }

    RCLCPP_INFO(get_node()->get_logger(),
                "Submitted gripper close move goal: width=%.4f speed=%.3f.",
                close_width_, close_speed_);
  }

  void assign_close_goal_options_callbacks() {
    close_goal_options_.goal_response_callback =
        [this](const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>>&
                   goal_handle) {
          if (!goal_handle) {
            close_result_received_ = true;
            close_succeeded_ = false;
            RCLCPP_ERROR(get_node()->get_logger(), "Gripper close goal was rejected.");
          } else {
            RCLCPP_INFO(get_node()->get_logger(), "Gripper close goal accepted.");
          }
        };

    close_goal_options_.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>::WrappedResult&
                   result) {
          close_result_received_ = true;
          close_succeeded_ = result.code == rclcpp_action::ResultCode::SUCCEEDED &&
                             result.result && result.result->success;

          if (close_succeeded_) {
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
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Move>> gripper_move_action_client_;
  rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions close_goal_options_;

  bool initialization_flag_{true};
  bool is_gazebo_{false};
  MotionPhase motion_phase_{MotionPhase::kIdle};
  std::atomic_bool close_goal_sent_{false};
  std::atomic_bool close_result_received_{false};
  std::atomic_bool close_succeeded_{false};

  double initial_robot_time_{0.0};
  double robot_time_{0.0};
  double motion_duration_sec_{8.0};
  double skeleton_trajectory_duration_sec_{24.0};
  double target_position_convergence_threshold_m_{0.01};
  double close_width_{0.004};
  double close_speed_{0.02};

  Eigen::Quaterniond initial_orientation_;
  Eigen::Quaterniond hover_orientation_;
  Eigen::Quaterniond target_orientation_;
  Eigen::Quaterniond commanded_orientation_;
  Eigen::Vector3d initial_position_;
  Eigen::Vector3d hover_position_;
  Eigen::Vector3d commanded_position_;
  Eigen::Vector3d target_position_{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> requested_cable_trajectory_points_;
  std::vector<Eigen::Vector3d> cable_trajectory_points_;
  std::vector<Eigen::Vector3d> active_cable_trajectory_points_;
  std::vector<double> active_cable_trajectory_arc_lengths_;
  double active_cable_trajectory_length_m_{0.0};

  std::string robot_description_;
  std::string robot_type_{"fr3"};
  std::string arm_prefix_;
  std::string requested_target_cable_id_;
  std::string current_target_cable_id_;
  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};
  std::atomic_bool trajectory_update_requested_{false};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
  std::mutex target_update_mutex_;
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
  std::vector<double> joint_velocities_current_{0, 0, 0, 0, 0, 0, 0};
};

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianCableTrajectoryController,
                       controller_interface::ControllerInterface)
