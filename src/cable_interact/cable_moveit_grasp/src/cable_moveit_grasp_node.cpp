#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <Eigen/SVD>
#include <cable_moveit_grasp/geometry.hpp>
#include <cable_moveit_grasp/path_validation.hpp>
#include <cable_moveit_grasp/platform.hpp>
#include <moveit/kinematic_constraints/utils.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <interactive_markers/menu_handler.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/move.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using namespace std::chrono_literals;
using MGI = moveit::planning_interface::MoveGroupInterface;
using Trigger = std_srvs::srv::Trigger;
using Pose = geometry_msgs::msg::PoseStamped;
using Marker = visualization_msgs::msg::Marker;

class CableMoveItGrasp : public rclcpp::Node {
  struct Candidate {
    int id, sign;
    double angle;
    Pose tcp, entry;
    bool valid{false};
    std::string reason{"not planned"};
    double score{0}, margin{0}, dexterity{0}, clearance{0}, length{0};
    MGI::Plan plan;
  };
  struct Busy {
    std::atomic_bool& value;
    explicit Busy(std::atomic_bool& v) : value(v) { value = true; }
    ~Busy() { value = false; }
  };
public:
  CableMoveItGrasp() : Node("cable_moveit_grasp") {
    declare_parameter("planning_group", "fr3_arm");
    declare_parameter("move_group_namespace", "/NS_1");
    declare_parameter("base_frame", "fr3_link0");
    declare_parameter("tcp_link", "fr3_hand_tcp");
    declare_parameter("end_effector_link", "fr3_hand_tcp");
    declare_parameter("cloud_topic", "/cable_camera/cable_points");
    declare_parameter("clicked_point_topic", "/clicked_point");
    declare_parameter("auto_generate", true);
    declare_parameter("local_radius", 0.025);
    declare_parameter("min_local_points", 12);
    declare_parameter("pca_ratio", 4.0);
    declare_parameter("max_line_offset", 0.01);
    declare_parameter("max_cloud_age", 2.0);
    declare_parameter("require_fresh_cloud", false);
    declare_parameter("roll_step_degrees", 10.0);
    declare_parameter("approach_distance", 0.04);
    declare_parameter("cartesian_step", 0.003);
    declare_parameter("jump_threshold", 2.0);
    declare_parameter("ik_attempts", 2);
    declare_parameter("planning_time", 1.0);
    declare_parameter("velocity_scaling", 0.1);
    declare_parameter("acceleration_scaling", 0.1);
    declare_parameter("validation_joint_step", 0.03);
    declare_parameter("min_joint_margin", 0.02);
    declare_parameter("min_dexterity", 0.005);
    declare_parameter("jacobian_length_scale", 0.3);
    declare_parameter("weight_margin", 3.0);
    declare_parameter("weight_dexterity", 1.0);
    declare_parameter("weight_clearance", 1.0);
    declare_parameter("weight_length", 0.1);
    declare_parameter("target_offset", std::vector<double>{0., 0., 0.});
    declare_parameter("tf_timeout", 1.0);
    declare_parameter("start_tolerance", 0.02);
    declare_parameter("gripper_enabled", true);
    declare_parameter("gripper_namespace", "/NS_1/franka_gripper");
    declare_parameter("gripper_joint_names", std::vector<std::string>{"fr3_finger_joint1", "fr3_finger_joint2"});
    declare_parameter("open_width", 0.08);
    declare_parameter("grasp_width", 0.004);
    declare_parameter("gripper_speed", 0.02);
    declare_parameter("gripper_force", 10.0);
    declare_parameter("grasp_epsilon", 0.003);
    declare_parameter("gripper_timeout", 15.0);
    declare_parameter("place_enabled", true);
    declare_parameter("place_joint_names", std::vector<std::string>{"fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"});
    declare_parameter("place_joint_positions", std::vector<double>{-0.5233661532402039, -1.5837199687957764, -0.40842023491859436, -1.467320442199707, -1.0411325693130493, 1.900314211845398, -2.7240424156188965});
    declare_parameter("place_planning_time", 5.0);
    declare_parameter("place_goal_tolerance", 0.01);
    declare_parameter("place_arrival_tolerance", 0.10);
    declare_parameter("platform_enabled", true);
    declare_parameter("platform_thickness", 0.01);
    declare_parameter("platform_points", std::vector<double>{-0.252, -0.510, 0.422, -0.234, -0.521, 0.936, 0.108, -0.530, 0.953, 0.380, -0.536, 0.907, 0.557, -0.537, 0.740, 0.644, -0.534, 0.573, 0.653, -0.531, 0.454});
    buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    listener_ = std::make_unique<tf2_ros::TransformListener>(*buffer_);
    operations_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions options;
    options.callback_group = operations_;
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(str("cloud_topic"),
        rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) {
          std::lock_guard<std::mutex> lock(input_mutex_); cloud_ = cloud; cloud_received_ = std::chrono::steady_clock::now();
        });
    point_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(str("clicked_point_topic"),
        10, [this](geometry_msgs::msg::PointStamped::ConstSharedPtr p) {
          if (busy_) { RCLCPP_WARN(get_logger(), "Operation active: selection ignored; click again when finished"); return; }
          std::lock_guard<std::mutex> lock(input_mutex_);
          point_ = p;
          selected_cloud_ = cloud_;
          selected_cloud_received_ = cloud_received_;
          ++input_revision_; pending_ = flag("auto_generate");
          RCLCPP_INFO(get_logger(), "Clicked target received; cloud snapshot %s, freshness policy %s",
              selected_cloud_ ? "captured" : "unavailable", flag("require_fresh_cloud") ? "strict" : "static scene");
          selected_id_ = -1; clear_pending_ = true;
        });
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/candidates", rclcpp::QoS(1).transient_local());
    report_pub_ = create_publisher<std_msgs::msg::String>("~/candidate_report", rclcpp::QoS(1).transient_local());
    target_pub_ = create_publisher<Pose>("~/grasp_pose", rclcpp::QoS(1).transient_local());
    entry_pub_ = create_publisher<Pose>("~/entry_pose", rclcpp::QoS(1).transient_local());
    preview_pub_ = create_publisher<moveit_msgs::msg::DisplayTrajectory>("~/display_trajectory", rclcpp::QoS(1).transient_local());
    select_pub_ = create_publisher<std_msgs::msg::Int32>("~/select_candidate", 10);
    select_sub_ = create_subscription<std_msgs::msg::Int32>("~/select_candidate", 10,
        [this](std_msgs::msg::Int32::ConstSharedPtr id) { try { select(id->data); } catch (const std::exception& e) { RCLCPP_ERROR(get_logger(), "%s", e.what()); } }, options);
    obstacle_sub_ = create_subscription<moveit_msgs::msg::CollisionObject>("~/obstacles", 10,
        [this](moveit_msgs::msg::CollisionObject::ConstSharedPtr object) {
          try {
            initialize(); invalidate();
            if (object->id.empty() || object->header.frame_id.empty()) throw std::runtime_error("Obstacle needs stable id and frame_id");
            if (!scene_interface_->applyCollisionObject(*object)) throw std::runtime_error("MoveIt rejected collision object");
            RCLCPP_INFO(get_logger(), "Applied obstacle %s; regenerate candidates", object->id.c_str());
          } catch (const std::exception& e) { RCLCPP_ERROR(get_logger(), "%s", e.what()); }
        }, options);
    generate_service_ = service("~/generate_candidates", [this] { generate(); return "Candidates evaluated and best feasible trajectory previewed"; });
    plan_service_ = service("~/plan", [this] { generate(); return "Candidates evaluated"; });
    execute_service_ = service("~/execute_selected", [this] { execute(); return flag("place_enabled") ? "Grasp, transport and release completed" : "Selected trajectory completed"; });
    // Feedback merely queues an ID; planning and execution are serialized in operations_.
    server_ = std::make_unique<interactive_markers::InteractiveMarkerServer>(
        get_namespace() + std::string("/cable_grasp_frames"), this);
    menu_.insert("Select and preview this candidate", [this](auto feedback) {
      try { std_msgs::msg::Int32 id; const auto split=feedback->marker_name.find('_',6);
        const auto revision=std::stoull(feedback->marker_name.substr(6,split-6));
        if (revision!=input_revision_) return;
        id.data = std::stoi(feedback->marker_name.substr(split+1)); select_pub_->publish(id); }
      catch (const std::exception&) { RCLCPP_ERROR(get_logger(), "Invalid candidate feedback"); }
    });
    timer_ = create_wall_timer(200ms, [this] {
      if (clear_pending_.exchange(false)) invalidate();
      bool run;
      { std::lock_guard<std::mutex> lock(input_mutex_); run = pending_; pending_ = false; }
      if (run) try { Busy guard(busy_); generate(); } catch (const std::exception& e) { RCLCPP_ERROR(get_logger(), "Generation failed: %s", e.what()); }
    }, operations_);
    parameter_callback_ = add_on_set_parameters_callback([this, keys=list_parameters({},10).names](const std::vector<rclcpp::Parameter>& updates) {
      std::vector<rclcpp::Parameter> parameters;
      for (const auto& p : updates)
        if (std::find(keys.begin(),keys.end(),p.get_name())!=keys.end() && p.get_name()!="use_sim_time" && p.get_name().find("robot_description")!=0)
          parameters.push_back(p);
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = parameters.empty() || !busy_;
      if (!result.successful) { result.reason = "Operation active; change parameters after it finishes"; return result; }
      for (const auto& p : parameters) {
        if (p.get_name() == "planning_group" || p.get_name() == "move_group_namespace" ||
            p.get_name() == "base_frame" || p.get_name() == "tcp_link" ||
            p.get_name() == "end_effector_link" || p.get_name() == "cloud_topic" ||
            p.get_name() == "clicked_point_topic" || p.get_name() == "gripper_namespace") {
          result.successful = false; result.reason = "Interface/model parameters require restart"; return result;
        }
      }
      if (!parameters.empty()) {
        ++input_revision_; selected_id_ = -1; clear_pending_ = true;
      }
      return result;
    });
    platform_timer_ = create_wall_timer(2s, [this] {
      try {
        initialize(); applyPlatform(); platform_timer_->cancel();
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "Waiting to load platform obstacle: %s",e.what());
      }
    }, operations_);
    RCLCPP_INFO(get_logger(), "Cable candidate planner ready; select and preview a candidate before calling execute_selected");
  }
private:
  std::string str(const char* key) const { return get_parameter(key).as_string(); }
  double number(const char* key) const { return get_parameter(key).as_double(); }
  int integer(const char* key) const { return get_parameter(key).as_int(); }
  bool flag(const char* key) const { return get_parameter(key).as_bool(); }
  template<class Function> rclcpp::Service<Trigger>::SharedPtr service(const char* name, Function function) {
    return create_service<Trigger>(name, [this, function](std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> response) {
      Busy guard(busy_);
      try { response->message = function(); response->success = true; }
      catch (const std::exception& e) { response->message = e.what(); response->success = false; RCLCPP_ERROR(get_logger(), "%s", e.what()); }
    }, rmw_qos_profile_services_default, operations_);
  }
  void invalidate() {
    selected_id_ = -1;
    candidates_.clear();
    server_->clear(); server_->applyChanges();
    visualization_msgs::msg::MarkerArray markers;
    Marker clear; clear.action = Marker::DELETEALL; markers.markers.push_back(clear); marker_pub_->publish(markers);
    moveit_msgs::msg::DisplayTrajectory empty; preview_pub_->publish(empty);
  }
  void initialize() {
    if (move_group_) return;
    auto interface = std::make_unique<MGI>(shared_from_this(), MGI::Options(str("planning_group"), "robot_description", str("move_group_namespace")));
    if (!interface->setEndEffectorLink(str("end_effector_link"))) throw std::runtime_error("Invalid end_effector_link");
    interface->setPoseReferenceFrame(str("base_frame"));
    const auto model = interface->getRobotModel();
    if (!model->hasLinkModel(str("tcp_link"))) throw std::runtime_error("Unknown tcp_link");
    // Only fixed tool offsets may be converted once and reused for candidate poses.
    auto link = model->getLinkModel(str("tcp_link"));
    while (link && link->getName() != str("end_effector_link")) {
      if (link->getParentJointModel()->getType() != moveit::core::JointModel::FIXED)
        throw std::runtime_error("TCP must equal or be a fixed descendant of end_effector_link");
      link = link->getParentLinkModel();
    }
    if (!link) throw std::runtime_error("TCP is not a descendant of end_effector_link");
    move_group_ = std::move(interface);
    scene_interface_ = std::make_unique<moveit::planning_interface::PlanningSceneInterface>(str("move_group_namespace"));
    scene_client_ = create_client<moveit_msgs::srv::GetPlanningScene>(str("move_group_namespace") + "/get_planning_scene");
    open_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(shared_from_this(), str("gripper_namespace") + "/move");
    grasp_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(shared_from_this(), str("gripper_namespace") + "/grasp");
  }
  void applyPlatform() {
    if (!flag("platform_enabled")) {
      scene_interface_->removeCollisionObjects({"cable_platform"});
      return;
    }
    const auto platform=cable_grasp::makePlatform(
        get_parameter("platform_points").as_double_array(),number("platform_thickness"));
    moveit_msgs::msg::CollisionObject object;
    object.header.frame_id=str("base_frame"); object.id="cable_platform";
    object.operation=moveit_msgs::msg::CollisionObject::ADD;
    object.meshes.push_back(platform.mesh);
    geometry_msgs::msg::Pose identity; identity.orientation.w=1.;
    object.mesh_poses.push_back(identity);
    if (!scene_interface_->applyCollisionObject(object))
      throw std::runtime_error("MoveIt rejected cable_platform obstacle");
    RCLCPP_INFO(get_logger(), "Platform obstacle loaded: %zu points, thickness %.4f m, max plane residual %.6f m",
        platform.mesh.vertices.size()/2,number("platform_thickness"),platform.max_residual);
  }
  planning_scene::PlanningScenePtr sceneSnapshot() {
    applyPlatform();
    if (!scene_client_->wait_for_service(3s)) throw std::runtime_error("get_planning_scene unavailable");
    auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    request->components.components = 1023; // all standard PlanningSceneComponents, including ACM and octomap
    auto future = scene_client_->async_send_request(request);
    if (future.wait_for(5s) != std::future_status::ready) throw std::runtime_error("Planning scene request timed out");
    auto scene = std::make_shared<planning_scene::PlanningScene>(move_group_->getRobotModel());
    if (!scene->setPlanningSceneMsg(future.get()->scene)) throw std::runtime_error("Failed to load planning scene");
    if (flag("platform_enabled") && !scene->getWorld()->hasObject("cable_platform"))
      throw std::runtime_error("Platform obstacle missing from PlanningScene; planning stopped");
    return scene;
  }
  Eigen::Isometry3d transform(const std::string& frame, const builtin_interfaces::msg::Time& stamp) {
    if (frame == str("base_frame")) return Eigen::Isometry3d::Identity();
    const auto tf = buffer_->lookupTransform(str("base_frame"), frame, rclcpp::Time(stamp), rclcpp::Duration::from_seconds(number("tf_timeout")));
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    const auto& q = tf.transform.rotation;
    result.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
    result.translation() = Eigen::Vector3d(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
    return result;
  }
  Pose pose(const Eigen::Isometry3d& t) {
    Pose p; p.header.frame_id = str("base_frame"); p.header.stamp = now();
    p.pose.position.x = t.translation().x(); p.pose.position.y = t.translation().y(); p.pose.position.z = t.translation().z();
    const Eigen::Quaterniond q(t.linear());
    p.pose.orientation.x = q.x(); p.pose.orientation.y = q.y(); p.pose.orientation.z = q.z(); p.pose.orientation.w = q.w();
    return p;
  }
  Eigen::Isometry3d matrix(const Pose& p) const {
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    const auto& q = p.pose.orientation;
    t.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix();
    t.translation() = Eigen::Vector3d(p.pose.position.x, p.pose.position.y, p.pose.position.z);
    return t;
  }
  void validateParameters() {
    for (const auto* key : {"local_radius", "max_cloud_age", "roll_step_degrees", "approach_distance", "cartesian_step", "jump_threshold", "planning_time", "validation_joint_step", "jacobian_length_scale", "tf_timeout", "start_tolerance"})
      if (!(number(key) > 0 && std::isfinite(number(key)))) throw std::runtime_error(std::string("Invalid positive parameter: ") + key);
    for (const auto* key : {"velocity_scaling", "acceleration_scaling"})
      if (!(number(key) > 0 && number(key) <= 1)) throw std::runtime_error(std::string("Invalid scaling: ") + key);
    if (number("roll_step_degrees") < 1 || number("roll_step_degrees") > 180 || integer("ik_attempts") < 1 || integer("min_local_points") < 6)
      throw std::runtime_error("Invalid sampling parameters");
  }
  void generate() {
    invalidate(); validateParameters();
    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
    geometry_msgs::msg::PointStamped::ConstSharedPtr clicked;
    std::chrono::steady_clock::time_point received;
    { std::lock_guard<std::mutex> lock(input_mutex_); cloud = selected_cloud_; received = selected_cloud_received_; clicked = point_; planned_revision_ = input_revision_; pending_ = false; }
    if (!cloud || !clicked) throw std::runtime_error("Need a cloud snapshot captured at selection; wait for the first cloud then click again");
    if (cloud->header.frame_id.empty() || clicked->header.frame_id.empty()) throw std::runtime_error("Missing input frame");
    const double age = (now() - rclcpp::Time(cloud->header.stamp)).seconds();
    const double separation = std::abs((rclcpp::Time(cloud->header.stamp)-rclcpp::Time(clicked->header.stamp)).seconds());
    const double receive_age = std::chrono::duration<double>(std::chrono::steady_clock::now()-received).count();
    std::ostringstream timing;
    timing << "cloud_age=" << age << "s, click_cloud_delta=" << separation
           << "s, since_last_receive=" << receive_age << "s, max_cloud_age="
           << number("max_cloud_age") << "s; cloud_stamp=" << cloud->header.stamp.sec
           << "." << cloud->header.stamp.nanosec << ", click_stamp="
           << clicked->header.stamp.sec << "." << clicked->header.stamp.nanosec;
    RCLCPP_INFO(get_logger(), "Selection timing: %s", timing.str().c_str());
    if (flag("require_fresh_cloud") && (age < -0.5 || age > number("max_cloud_age") || separation > number("max_cloud_age")))
      throw std::runtime_error("Cloud/selection timestamp check rejected: " + timing.str() +
          ". Verify live cloud updates and end-to-end latency before adjusting max_cloud_age; reselect after changes.");
    if (!std::isfinite(clicked->point.x) || !std::isfinite(clicked->point.y) || !std::isfinite(clicked->point.z)) throw std::runtime_error("Invalid clicked point");
    const Eigen::Vector3d selected = transform(clicked->header.frame_id, clicked->header.stamp) * Eigen::Vector3d(clicked->point.x, clicked->point.y, clicked->point.z);
    const Eigen::Isometry3d cloud_tf = transform(cloud->header.frame_id, cloud->header.stamp);
    for (const auto* field_name : {"x", "y", "z"}) {
      auto field = std::find_if(cloud->fields.begin(), cloud->fields.end(),
          [field_name](const auto& f) { return f.name == field_name; });
      if (field == cloud->fields.end() || field->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
          field->count != 1 || field->offset + sizeof(float) > cloud->point_step)
        throw std::runtime_error("Cloud requires FLOAT32 x/y/z fields");
    }
    if (cloud->row_step != cloud->width*cloud->point_step || cloud->data.size() != static_cast<size_t>(cloud->height)*cloud->row_step || cloud->is_bigendian)
      throw std::runtime_error("Unsupported padded/big-endian or malformed cloud");
    std::vector<Eigen::Vector3d> local;
    sensor_msgs::PointCloud2ConstIterator<float> x(*cloud,"x"), y(*cloud,"y"), z(*cloud,"z");
    for (; x != x.end(); ++x, ++y, ++z) {
      Eigen::Vector3d point(*x,*y,*z);
      if (!point.allFinite()) continue;
      point = cloud_tf*point;
      if ((point-selected).norm() <= number("local_radius")) local.push_back(point);
    }
    if (local.size() < static_cast<size_t>(integer("min_local_points"))) throw std::runtime_error("Too few local cable points");
    const auto tangent = cable_grasp::estimateTangent(local, number("pca_ratio"), number("max_line_offset"), selected);
    auto offset = get_parameter("target_offset").as_double_array();
    if (offset.size()!=3 || !std::all_of(offset.begin(),offset.end(),[](double v){return std::isfinite(v);})) throw std::runtime_error("Invalid target_offset");
    initialize();
    auto start = move_group_->getCurrentState(5.);
    if (!start) throw std::runtime_error("No current joint state");
    auto scene = sceneSnapshot();
    // Retain attached objects from the scene while using live joint positions.
    auto scene_start = scene->getCurrentState();
    scene_start.setVariablePositions(start->getVariablePositions());
    *start = scene_start;
    if (flag("gripper_enabled")) {
      // Own the array: as_string_array() returns a reference to a temporary Parameter.
      const auto gripper_names = get_parameter("gripper_joint_names").as_string_array();
      const auto& model_names = start->getRobotModel()->getVariableNames();
      if (gripper_names.empty()) throw std::runtime_error("gripper_joint_names must not be empty");
      for (const auto& name : gripper_names) {
        if (std::find(model_names.begin(), model_names.end(), name) == model_names.end())
          throw std::runtime_error("Gripper model joint unavailable: " + name + "; verify gripper_joint_names and loaded URDF prefix");
        start->setVariablePosition(name, number("open_width")/2);
      }
      start->update();
    }
    const Eigen::Isometry3d base_to_model = start->getGlobalLinkTransform(str("base_frame"));
    const Eigen::Isometry3d tcp_to_tip = start->getGlobalLinkTransform(str("tcp_link")).inverse()*start->getGlobalLinkTransform(str("end_effector_link"));
    move_group_->setPlanningTime(number("planning_time"));
    move_group_->setMaxVelocityScalingFactor(number("velocity_scaling"));
    move_group_->setMaxAccelerationScalingFactor(number("acceleration_scaling"));
    int id = 0;
    for (int sign : {1,-1}) for (double degrees=0; degrees<360.-1e-6; degrees+=number("roll_step_degrees")) {
      Candidate candidate; candidate.id = id++; candidate.sign = sign; candidate.angle = degrees;
      Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity();
      tcp.linear() = cable_grasp::candidateRotation(tangent,sign,degrees*M_PI/180.);
      tcp.translation() = selected+Eigen::Vector3d(offset[0],offset[1],offset[2]);
      Eigen::Isometry3d entry = tcp; entry.translation() -= number("approach_distance")*tcp.linear().col(2);
      candidate.tcp = pose(tcp); candidate.entry = pose(entry);
      try { planCandidate(candidate,*start,*scene,base_to_model,tcp_to_tip); }
      catch (const std::exception& e) { candidate.reason = e.what(); }
      candidates_.push_back(std::move(candidate));
      RCLCPP_INFO(get_logger(), "Candidate %d/%d: %s", id, 2*static_cast<int>(std::ceil(360./number("roll_step_degrees"))), candidates_.back().reason.c_str());
    }
    publishCandidates();
    const Candidate* best = nullptr;
    for (const auto& c : candidates_) if (c.valid && (!best || c.score > best->score)) best = &c;
    if (!best) throw std::runtime_error("No feasible candidate; inspect candidate_report and adjust selection/scene");
    select(best->id);
  }
  void planCandidate(Candidate& candidate, const moveit::core::RobotState& start,
                     const planning_scene::PlanningScene& scene, const Eigen::Isometry3d& base_to_model,
                     const Eigen::Isometry3d& tcp_to_tip) {
    const auto* group = start.getJointModelGroup(str("planning_group"));
    bool found = false;
    std::string failure = "No collision-free IK/trajectory";
    for (int attempt=0; attempt<integer("ik_attempts"); ++attempt) {
      try {
        moveit::core::RobotState ik(start);
        if (attempt) ik.setToRandomPositions(group);
        auto valid = [&scene](moveit::core::RobotState* state, const moveit::core::JointModelGroup* g, const double* q) {
          state->setJointGroupPositions(g,q); state->update();
          return state->satisfiesBounds(g) && scene.isStateValid(*state, g->getName());
        };
        if (!ik.setFromIK(group, base_to_model*matrix(candidate.entry)*tcp_to_tip, str("end_effector_link"), 0.1, valid)) { failure="MoveIt entry IK failed or state invalid"; continue; }
        move_group_->setStartState(start);
        move_group_->setJointValueTarget(ik);
        MGI::Plan first;
        const auto planning_result = move_group_->plan(first);
        if (planning_result != moveit::core::MoveItErrorCode::SUCCESS) { failure="MoveIt planning failed code="+std::to_string(planning_result.val); continue; }
        robot_trajectory::RobotTrajectory joined(move_group_->getRobotModel(),str("planning_group"));
        joined.setRobotTrajectoryMsg(start,first.trajectory_);
        if (!joined.getWayPointCount()) continue;
        move_group_->setStartState(joined.getLastWayPoint());
        const Pose goal = pose(matrix(candidate.tcp)*tcp_to_tip);
        moveit_msgs::msg::RobotTrajectory final_segment;
        const double fraction = move_group_->computeCartesianPath({goal.pose},number("cartesian_step"),number("jump_threshold"),final_segment,true);
        if (fraction < 0.999999) { failure="Incomplete Cartesian approach"; continue; }
        robot_trajectory::RobotTrajectory approach(move_group_->getRobotModel(),str("planning_group"));
        approach.setRobotTrajectoryMsg(joined.getLastWayPoint(), final_segment);
        for (size_t i=1;i<approach.getWayPointCount();++i) joined.addSuffixWayPoint(approach.getWayPoint(i),0.1);
        trajectory_processing::IterativeParabolicTimeParameterization timing;
        if (!timing.computeTimeStamps(joined,number("velocity_scaling"),number("acceleration_scaling"))) { failure="Time parameterization failed"; continue; }
        Candidate trial = candidate;
        evaluate(trial,joined,scene);
        trial.plan.start_state_ = first.start_state_;
        joined.getRobotTrajectoryMsg(trial.plan.trajectory_);
        trial.valid = true; trial.reason = "MoveIt standard checks passed";
        if (trial.margin<number("min_joint_margin")) trial.reason += "; low joint margin (advisory)";
        if (trial.dexterity<number("min_dexterity")) trial.reason += "; low dexterity (advisory)";
        if (!found || trial.score > candidate.score) candidate = std::move(trial);
        found = true;
      } catch (const std::exception& e) { failure=e.what(); }
    }
    if (!found) throw std::runtime_error(failure);
  }
  void evaluate(Candidate& candidate, const robot_trajectory::RobotTrajectory& trajectory,
                const planning_scene::PlanningScene& scene) {
    if (trajectory.getWayPointCount()<2) throw std::runtime_error("Empty trajectory");
    auto initial = trajectory.getFirstWayPoint(); initial.update();
    const Eigen::Isometry3d tcp_to_tip = initial.getGlobalLinkTransform(str("tcp_link")).inverse() *
                                       initial.getGlobalLinkTransform(str("end_effector_link"));
    const auto goal = kinematic_constraints::constructGoalConstraints(
        str("end_effector_link"), pose(matrix(candidate.tcp)*tcp_to_tip),
        move_group_->getGoalPositionTolerance(), move_group_->getGoalOrientationTolerance());
    const auto sampled = cable_grasp::validateTrajectory(trajectory, scene,
        number("validation_joint_step"), move_group_->getPathConstraints(), {goal});
    // Below this point, metrics rank valid paths; no quality threshold rejects them.
    candidate.margin = 0.5; candidate.dexterity = 1.; candidate.clearance = 0.1; candidate.length = 0.;
    const auto* group = sampled.getGroup();
    for (size_t i=0; i<sampled.getWayPointCount(); ++i) {
      const auto& state = sampled.getWayPoint(i);
      if (i) {
        double squared = 0.;
        const auto& previous = sampled.getWayPoint(i-1);
        for (const auto& name : group->getVariableNames()) {
          const double delta = state.getVariablePosition(name)-previous.getVariablePosition(name);
          squared += delta*delta;
        }
        candidate.length += std::sqrt(squared);
      }
      for (const auto& name : group->getVariableNames()) {
        const auto& bounds = state.getRobotModel()->getVariableBounds(name);
        if (bounds.position_bounded_) candidate.margin=std::min(candidate.margin,
            cable_grasp::normalizedMargin(state.getVariablePosition(name),bounds.min_position_,bounds.max_position_));
      }
      Eigen::MatrixXd jacobian;
      if (state.getJacobian(group,state.getLinkModel(str("tcp_link")),Eigen::Vector3d::Zero(),jacobian) && jacobian.allFinite()) {
        jacobian.topRows(3)/=number("jacobian_length_scale");
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian);
        const auto singular=svd.singularValues();
        candidate.dexterity=std::min(candidate.dexterity,
            singular.minCoeff()/std::max(1e-12,singular.maxCoeff()));
      } else {
        candidate.dexterity=0.;
        RCLCPP_WARN(get_logger(), "Jacobian metric unavailable; ranking uses dexterity=0, path validity unchanged");
      }
      const double distance=scene.distanceToCollision(state);
      if (std::isfinite(distance)) candidate.clearance=std::min(candidate.clearance,distance);
    }
    candidate.score=number("weight_margin")*candidate.margin+number("weight_dexterity")*candidate.dexterity+
        number("weight_clearance")*std::clamp(candidate.clearance/0.1,0.,1.)-number("weight_length")*candidate.length;
  }
  moveit::core::RobotState placeTarget(const moveit::core::RobotState& start) {
    if (!flag("gripper_enabled")) throw std::runtime_error("Placement requires gripper_enabled=true");
    auto goal = start;
    const auto names = get_parameter("place_joint_names").as_string_array();
    const auto positions = get_parameter("place_joint_positions").as_double_array();
    const auto* group = start.getJointModelGroup(str("planning_group"));
    auto expected = group->getVariableNames();
    auto sorted = names;
    std::sort(expected.begin(), expected.end()); std::sort(sorted.begin(), sorted.end());
    if (sorted != expected || names.size() != positions.size())
      throw std::runtime_error("Placement target must name every arm joint exactly once");
    for (size_t i=0; i<names.size(); ++i) {
      if (!std::isfinite(positions[i])) throw std::runtime_error("Non-finite placement target");
      goal.setVariablePosition(names[i], positions[i]);
    }
    goal.update();
    if (!goal.satisfiesBounds(group)) throw std::runtime_error("Placement target outside MoveIt joint bounds");
    for (const auto* key : {"place_planning_time", "place_goal_tolerance", "place_arrival_tolerance"})
      if (!std::isfinite(number(key)) || number(key)<=0.) throw std::runtime_error("Invalid placement timing/tolerance");
    return goal;
  }
  MGI::Plan planPlace(const moveit::core::RobotState& start,
                      const planning_scene::PlanningScene& scene) {
    const auto goal = placeTarget(start);
    move_group_->setStartState(start);
    move_group_->setJointValueTarget(goal);
    const double old_time = move_group_->getPlanningTime();
    const double old_tolerance = move_group_->getGoalJointTolerance();
    move_group_->setPlanningTime(number("place_planning_time"));
    move_group_->setGoalJointTolerance(number("place_goal_tolerance"));
    MGI::Plan plan;
    moveit::core::MoveItErrorCode result;
    try { result = move_group_->plan(plan); }
    catch (...) {
      move_group_->setPlanningTime(old_time); move_group_->setGoalJointTolerance(old_tolerance); throw;
    }
    move_group_->setPlanningTime(old_time); move_group_->setGoalJointTolerance(old_tolerance);
    if (result != moveit::core::MoveItErrorCode::SUCCESS)
      throw std::runtime_error("Placement planning failed code="+std::to_string(result.val));
    robot_trajectory::RobotTrajectory trajectory(move_group_->getRobotModel(),str("planning_group"));
    trajectory.setRobotTrajectoryMsg(start, plan.trajectory_);
    const auto constraints = kinematic_constraints::constructGoalConstraints(
        goal, start.getJointModelGroup(str("planning_group")), number("place_goal_tolerance"));
    cable_grasp::validateTrajectory(trajectory,scene,number("validation_joint_step"),
                                    move_group_->getPathConstraints(),{constraints});
    return plan;
  }
  void select(int id) {
    if (planned_revision_!=input_revision_) throw std::runtime_error("Selection changed; regenerate candidates");
    if (id<0 || static_cast<size_t>(id)>=candidates_.size() || !candidates_[id].valid) throw std::runtime_error("Candidate invalid; inspect report");
    selected_id_=-1;
    auto& candidate=candidates_[id]; target_pub_->publish(candidate.tcp); entry_pub_->publish(candidate.entry);
    moveit_msgs::msg::DisplayTrajectory preview; preview.model_id=move_group_->getRobotModel()->getName();
    preview.trajectory_start=candidate.plan.start_state_; preview.trajectory.push_back(candidate.plan.trajectory_);
    if (flag("place_enabled")) {
      robot_trajectory::RobotTrajectory grasp(move_group_->getRobotModel(),str("planning_group"));
      auto scene=sceneSnapshot();
      grasp.setRobotTrajectoryMsg(scene->getCurrentState(),candidate.plan.trajectory_);
      auto closed=grasp.getLastWayPoint();
      const auto fingers=get_parameter("gripper_joint_names").as_string_array();
      for (const auto& name:fingers) closed.setVariablePosition(name,number("grasp_width")/2.);
      closed.update();
      preview.trajectory.push_back(planPlace(closed,*scene).trajectory_);
    }
    preview_pub_->publish(preview);
    selected_id_=id;
    publishCandidates();
    RCLCPP_INFO(get_logger(), "Selected candidate %d score %.4f; preview only",id,candidate.score);
  }
  void publishCandidates() {
    visualization_msgs::msg::MarkerArray markers;
    Marker clear; clear.action=Marker::DELETEALL; markers.markers.push_back(clear);
    server_->clear();
    std::ostringstream report; report<<"id,sign,roll_deg,feasible,score,joint_margin,dexterity,clearance_m,joint_length,reason\n";
    for (const auto& c:candidates_) {
      report<<c.id<<','<<c.sign<<','<<c.angle<<','<<c.valid<<','<<c.score<<','<<c.margin<<','<<c.dexterity<<','<<c.clearance<<','<<c.length<<','<<c.reason<<'\n';
      // Axes originate at actual TCP. Numbered buttons fan out for distinct selection.
      const auto rotation=matrix(c.tcp).linear().eval();
      for (int axis=0;axis<3;++axis) {
        Marker m; m.header=c.tcp.header; m.ns="candidate_axes"; m.id=3*c.id+axis; m.type=Marker::ARROW;
        m.pose.orientation.w=1.; m.scale.x=0.0015; m.scale.y=0.003; m.scale.z=0.005;
        m.color.a=c.valid?0.8:0.12; m.color.r=(axis==0); m.color.g=(axis==1); m.color.b=(axis==2);
        geometry_msgs::msg::Point a=c.tcp.pose.position,b=a;
        const auto v=0.025*rotation.col(axis); b.x+=v.x(); b.y+=v.y(); b.z+=v.z(); m.points={a,b}; markers.markers.push_back(m);
      }
      if (!c.valid) continue;
      visualization_msgs::msg::InteractiveMarker button;
      button.header=c.tcp.header; button.name="frame_"+std::to_string(planned_revision_)+"_"+std::to_string(c.id);
      button.description="ID "+std::to_string(c.id)+" roll "+std::to_string(static_cast<int>(c.angle))+(c.id==selected_id_?" SELECTED":"");
      button.pose=c.tcp.pose; button.pose.orientation=geometry_msgs::msg::Quaternion(); button.pose.orientation.w=1.;
      button.pose.position.x+=0.12+0.025*(c.id%12);
      button.pose.position.z+=0.025*(c.id/12);
      button.scale=0.025;
      visualization_msgs::msg::InteractiveMarkerControl control; control.always_visible=true;
      control.interaction_mode=visualization_msgs::msg::InteractiveMarkerControl::MENU;
      Marker sphere; sphere.type=Marker::SPHERE; sphere.pose.orientation.w=1.; sphere.scale.x=sphere.scale.y=sphere.scale.z=0.012;
      sphere.color.a=1.; sphere.color.g=1.; sphere.color.r=c.id==selected_id_?1.:0.; control.markers.push_back(sphere); button.controls.push_back(control);
      server_->insert(button); menu_.apply(*server_,button.name);
      Marker line; line.header=c.tcp.header; line.ns="candidate_buttons"; line.id=c.id; line.type=Marker::LINE_STRIP;
      line.pose.orientation.w=1.; line.scale.x=0.0005; line.color.g=1.; line.color.a=0.3;
      line.points={c.tcp.pose.position,button.pose.position}; markers.markers.push_back(line);
    }
    server_->applyChanges(); marker_pub_->publish(markers);
    std_msgs::msg::String message; message.data=report.str(); report_pub_->publish(message);
  }
  template<class Action> void gripper(typename rclcpp_action::Client<Action>::SharedPtr client, const typename Action::Goal& goal) {
    if (!client->wait_for_action_server(3s)) throw std::runtime_error("Gripper action unavailable");
    auto handle_future=client->async_send_goal(goal);
    if (handle_future.wait_for(3s)!=std::future_status::ready) throw std::runtime_error("Gripper goal timeout; check action state before retry");
    auto handle=handle_future.get(); if (!handle) throw std::runtime_error("Gripper rejected goal");
    auto result=client->async_get_result(handle);
    if (result.wait_for(std::chrono::duration<double>(number("gripper_timeout")))!=std::future_status::ready) {
      client->async_cancel_goal(handle); throw std::runtime_error("Gripper timed out; cancel requested");
    }
    auto wrapped=result.get();
    if (wrapped.code!=rclcpp_action::ResultCode::SUCCEEDED || !wrapped.result->success) throw std::runtime_error("Gripper action failed");
  }
  void execute() {
    if (flag("place_enabled") && !flag("gripper_enabled")) throw std::runtime_error("Placement requires gripper_enabled=true");
    const int id=selected_id_;
    if (planned_revision_!=input_revision_ || id<0 || static_cast<size_t>(id)>=candidates_.size()) throw std::runtime_error("No current selected candidate");
    Candidate candidate=candidates_[id];
    if (!candidate.valid) throw std::runtime_error("Candidate did not pass MoveIt standard checks");
    // Consume authorization to prevent repeated execution of a cached trajectory.
    selected_id_=-1;
    if (flag("gripper_enabled")) {
      franka_msgs::action::Move::Goal goal; goal.width=number("open_width"); goal.speed=number("gripper_speed");
      gripper<franka_msgs::action::Move>(open_client_,goal);
    }
    auto current=move_group_->getCurrentState(5.);
    if (!current) throw std::runtime_error("Missing current state");
    auto scene=sceneSnapshot();
    auto execution_start=scene->getCurrentState();
    execution_start.setVariablePositions(current->getVariablePositions());
    robot_trajectory::RobotTrajectory trajectory(move_group_->getRobotModel(),str("planning_group"));
    trajectory.setRobotTrajectoryMsg(execution_start,candidate.plan.trajectory_);
    const auto& first=trajectory.getFirstWayPoint();
    for (const auto& name:trajectory.getGroup()->getVariableNames())
      if (std::abs(first.getVariablePosition(name)-current->getVariablePosition(name))>number("start_tolerance")) throw std::runtime_error("Robot moved since planning; regenerate");
    evaluate(candidate,trajectory,*scene);
    if (move_group_->execute(candidate.plan)!=moveit::core::MoveItErrorCode::SUCCESS) throw std::runtime_error("MoveIt execution failed; gripper not closed");
    if (flag("gripper_enabled")) {
      franka_msgs::action::Grasp::Goal goal;
      goal.width=number("grasp_width"); goal.speed=number("gripper_speed"); goal.force=number("gripper_force");
      goal.epsilon.inner=goal.epsilon.outer=number("grasp_epsilon");
      gripper<franka_msgs::action::Grasp>(grasp_client_,goal);
    }
    if (flag("place_enabled")) {
      try {
        RCLCPP_INFO(get_logger(), "Grasp succeeded; replanning transport from measured joint state (preview was approximate)");
        auto measured=move_group_->getCurrentState(5.);
        if (!measured) throw std::runtime_error("Missing post-grasp joint state");
        auto place_scene=sceneSnapshot();
        const auto place_plan=planPlace(*measured,*place_scene);
        moveit_msgs::msg::DisplayTrajectory preview;
        preview.model_id=move_group_->getRobotModel()->getName();
        preview.trajectory_start=place_plan.start_state_; preview.trajectory.push_back(place_plan.trajectory_);
        preview_pub_->publish(preview);
        if (move_group_->execute(place_plan)!=moveit::core::MoveItErrorCode::SUCCESS)
          throw std::runtime_error("Placement trajectory execution failed");
        auto arrived=move_group_->getCurrentState(5.);
        if (!arrived) throw std::runtime_error("Missing arrival joint state");
        const auto target=placeTarget(*arrived);
        for (const auto& name:arrived->getJointModelGroup(str("planning_group"))->getVariableNames())
          if (std::abs(arrived->getVariablePosition(name)-target.getVariablePosition(name))>number("place_arrival_tolerance")) {
            std::ostringstream detail;
            detail << "Placement arrival tolerance exceeded: " << name
                   << " actual=" << arrived->getVariablePosition(name)
                   << " target=" << target.getVariablePosition(name)
                   << " error=" << std::abs(arrived->getVariablePosition(name)-target.getVariablePosition(name))
                   << " tolerance=" << number("place_arrival_tolerance") << " rad";
            throw std::runtime_error(detail.str());
          }
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Transport failed; release not commanded: ")+e.what());
      }
      RCLCPP_INFO(get_logger(), "Placement reached; opening gripper to release Cable");
      franka_msgs::action::Move::Goal release;
      release.width=number("open_width"); release.speed=number("gripper_speed");
      try { gripper<franka_msgs::action::Move>(open_client_,release); }
      catch (const std::exception& e) { throw std::runtime_error(std::string("Placement reached but release failed: ")+e.what()); }
      RCLCPP_INFO(get_logger(), "Grasp, transport and release complete");
    } else RCLCPP_INFO(get_logger(), "Grasp complete; placement disabled");
  }
  std::mutex input_mutex_;
  std::atomic_bool busy_{false}, clear_pending_{false};
  OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
  std::atomic<int> selected_id_{-1};
  std::atomic<uint64_t> input_revision_{0};
  uint64_t planned_revision_{0};
  bool pending_{false};
  std::vector<Candidate> candidates_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_;
  std::chrono::steady_clock::time_point cloud_received_, selected_cloud_received_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr selected_cloud_;
  geometry_msgs::msg::PointStamped::ConstSharedPtr point_;
  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::unique_ptr<tf2_ros::TransformListener> listener_;
  std::unique_ptr<MGI> move_group_;
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> scene_interface_;
  std::unique_ptr<interactive_markers::InteractiveMarkerServer> server_;
  interactive_markers::MenuHandler menu_;
  rclcpp::CallbackGroup::SharedPtr operations_;
  rclcpp::TimerBase::SharedPtr timer_, platform_timer_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr point_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr select_sub_;
  rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr obstacle_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr report_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr select_pub_;
  rclcpp::Publisher<Pose>::SharedPtr target_pub_,entry_pub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr preview_pub_;
  rclcpp::Service<Trigger>::SharedPtr generate_service_,plan_service_,execute_service_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr scene_client_;
  rclcpp_action::Client<franka_msgs::action::Move>::SharedPtr open_client_;
  rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr grasp_client_;
};
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);
  auto node=std::make_shared<CableMoveItGrasp>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),4);
  executor.add_node(node); executor.spin(); rclcpp::shutdown(); return 0;
}
