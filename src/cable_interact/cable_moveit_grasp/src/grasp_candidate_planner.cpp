#include <cable_moveit_grasp/continuous_cable_selection.hpp>
#include <optional>
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
#include <thread>
#include <Eigen/SVD>
#include <cable_moveit_grasp/geometry.hpp>
#include <cable_moveit_grasp/fixed_y_selection.hpp>
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
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/srv/get_motion_plan.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using namespace std::chrono_literals;
using MGI = moveit::planning_interface::MoveGroupInterface;
using Trigger = std_srvs::srv::Trigger;
using Pose = geometry_msgs::msg::PoseStamped;
using Marker = visualization_msgs::msg::Marker;

class GraspCandidatePlanner : public rclcpp::Node {
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
  GraspCandidatePlanner() : Node("grasp_candidate_planner") {
    for(const auto& entry:get_node_parameters_interface()->get_parameter_overrides())
      if(entry.first.rfind("ompl.",0)==0 && !has_parameter(entry.first)) declare_parameter(entry.first,entry.second);
    declare_parameter("candidate_arm", "");
    declare_parameter("require_arm_selection",true);
    arm_selected_=!get_parameter("require_arm_selection").as_bool();
    declare_parameter("candidate_plan_max_age", 60.0);
    declare_parameter("a_from_b_enabled",false);
    declare_parameter("a_from_b_use_current_tcp",true);
    declare_parameter("a_from_b_x_offset",0.10);
    declare_parameter("a_from_b_x_tolerance",0.01);
    declare_parameter("a_recovery_from_b_x_offset",0.15);
    declare_parameter("b_recovery_x_offset",0.05);
    declare_parameter("recovery_x_tolerance",0.01);
    declare_parameter("b_auto_enabled",false);
    declare_parameter("b_search_x_min",-0.30);
    declare_parameter("b_min_board_clearance",0.03);
    declare_parameter("b_search_a_tcp_clearance",0.10);
    declare_parameter("a_search_x_length",0.30);
    declare_parameter("quality_min_length",0.05);
    declare_parameter("quality_max_gap",0.006);
    declare_parameter("quality_voxel_size",0.002);
    declare_parameter("planning_group", "fr3_arm");
    declare_parameter("move_group_namespace", "/dual_arm");
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
    declare_parameter("candidate_worker_threads", 1);
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
    declare_parameter("gripper_joint_names", std::vector<std::string>{"fr3_finger_joint1", "fr3_finger_joint2"});
    declare_parameter("open_width", 0.08);
    declare_parameter("near_board_height_threshold", 0.01);
    declare_parameter("near_board_raise_threshold", 0.008);
    declare_parameter("near_board_target_raise", 0.005);
    declare_parameter("platform_enabled", true);
    declare_parameter("platform_thickness", 0.01);
    declare_parameter("platform_rectangle", true);
    declare_parameter("platform_points", std::vector<double>{-0.252, -0.504, 0.422, -0.234, -0.515, 0.936, 0.108, -0.524, 0.953, 0.380, -0.530, 0.907, 0.557, -0.531, 0.740, 0.644, -0.528, 0.573, 0.653, -0.525, 0.454});
    buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    listener_ = std::make_unique<tf2_ros::TransformListener>(*buffer_);
    operations_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions options;
    options.callback_group = operations_;
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(str("cloud_topic"),
        rclcpp::QoS(1).reliable().durability_volatile(), [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) {
          std::lock_guard<std::mutex> lock(input_mutex_); cloud_ = cloud; cloud_received_ = std::chrono::steady_clock::now();
        });
    point_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(str("clicked_point_topic"),
        10, [this](geometry_msgs::msg::PointStamped::ConstSharedPtr p) {
          if (flag("b_auto_enabled") && !manual_fallback_) {
            RCLCPP_WARN(get_logger(),"B automatic mode: call grasp_b first; after automatic failure, RViz clicks are enabled");return;
          }
          if (!arm_selected_) return;
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
    if(str("candidate_arm")=="a") {
      b_grasp_sub_=create_subscription<Pose>("/dual_arm/dual_arm_motion/last_successful_b_grasp",rclcpp::QoS(1).transient_local(),
        [this](Pose::ConstSharedPtr message) {
          b_grasp_reference_=*message;
          if(!flag("a_from_b_enabled")) return;
          manual_fallback_=false;
          ++input_revision_;selected_id_=-1;clear_pending_=true;
        },options);
    }

      arm_selection_sub_ = create_subscription<std_msgs::msg::String>("/dual_arm/selected_grasp_arm",
          rclcpp::QoS(1).transient_local(), [this](std_msgs::msg::String::ConstSharedPtr message) {
            if(!flag("require_arm_selection")) return;
            const bool enabled=message->data==str("candidate_arm");
            if (arm_selected_.exchange(enabled)!=enabled) {
              ++input_revision_; selected_id_=-1; clear_pending_=true;
              std::lock_guard<std::mutex> lock(input_mutex_); pending_=false; point_.reset(); selected_cloud_.reset();
            }
          });
      selected_plan_service_ = create_service<moveit_msgs::srv::GetMotionPlan>("~/get_selected_grasp_plan",
          [this](moveit_msgs::srv::GetMotionPlan::Request::SharedPtr request,
                 moveit_msgs::srv::GetMotionPlan::Response::SharedPtr response) {
            Busy guard(busy_);
            auto& result=response->motion_plan_response;
            result.error_code.val=moveit_msgs::msg::MoveItErrorCodes::FAILURE;
            try {
              last_plan_error_.clear();
              if(request->motion_plan_request.planner_id=="cable_grasp_reselect") {
                try {generate(str("candidate_arm")=="a",true);}
                catch(...) {manual_fallback_=true;throw;}
              } else if(flag("a_from_b_enabled") && !manual_fallback_) {
                try {generate(true);}
                catch(const std::exception& error) {
                  manual_fallback_=true;
                  throw std::runtime_error(std::string("A automatic selection/planning failed: ")+error.what()+
                      "; select A LEFT, click cable in RViz, wait for preview, then call grasp_a again");
                }
              } else if(flag("b_auto_enabled") && !manual_fallback_) {
                try {generate();}
                catch(const std::exception& error) {
                  manual_fallback_=true;
                  throw std::runtime_error(std::string("B automatic selection/planning failed: ")+error.what()+
                      "; select B RIGHT, click cable in RViz, wait for preview, then call grasp_b again");
                }
              }
              else if (!arm_selected_) throw std::runtime_error("Select this arm in RViz first");
              const int id=selected_id_;
              if (planned_revision_!=input_revision_ || id<0 || static_cast<size_t>(id)>=candidates_.size())
                throw std::runtime_error("No selected RViz candidate; click cloud and wait for preview");
              if ((now()-selected_plan_time_).seconds()>number("candidate_plan_max_age"))
                throw std::runtime_error("Candidate preview expired; select the cloud again");
              const auto& candidate=candidates_[id];
              if (!candidate.valid) throw std::runtime_error("Selected candidate is invalid");
              last_handed_cloud_point_=generated_cloud_point_;
              result.trajectory_start=candidate.plan.start_state_;
              result.trajectory=candidate.plan.trajectory_;
              result.group_name=str("planning_group");
              result.error_code.val=moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
              manual_fallback_=false;
              if(request->motion_plan_request.planner_id!="preview_only") {
                selected_id_=-1;publishPlanReady(false); // Consume only actual execution handoffs.
              }
            } catch(const std::exception& error) {last_plan_error_=error.what();RCLCPP_ERROR(get_logger(), "Candidate handoff rejected: %s",error.what());}
          },rmw_qos_profile_services_default,operations_);
      last_error_service_=create_service<Trigger>("~/get_last_plan_error",
        [this](Trigger::Request::SharedPtr,Trigger::Response::SharedPtr response) {
          response->success=true;response->message=last_plan_error_;
        },rmw_qos_profile_services_default,operations_);
    
    plan_ready_pub_=create_publisher<std_msgs::msg::String>("~/plan_ready",rclcpp::QoS(1).transient_local());
    prepare_manual_service_=create_service<Trigger>("~/prepare_manual_grasp",
      [this](Trigger::Request::SharedPtr,Trigger::Response::SharedPtr response) {
        manual_fallback_=true;++input_revision_;clear_pending_=false;
        {std::lock_guard<std::mutex> lock(input_mutex_);pending_=false;point_.reset();selected_cloud_.reset();}
        invalidate();last_plan_error_.clear();
        response->success=true;response->message=std::to_string(input_revision_.load());
        RCLCPP_INFO(get_logger(),"Manual grasp armed; select this arm and publish a fresh RViz point");
      },rmw_qos_profile_services_default,operations_);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/candidates", rclcpp::QoS(1).transient_local());
    report_pub_ = create_publisher<std_msgs::msg::String>("~/candidate_report", rclcpp::QoS(1).transient_local());
    target_pub_ = create_publisher<Pose>("~/grasp_pose", rclcpp::QoS(1).transient_local());
    entry_pub_ = create_publisher<Pose>("~/entry_pose", rclcpp::QoS(1).transient_local());
    preview_pub_ = create_publisher<moveit_msgs::msg::DisplayTrajectory>("~/display_trajectory", rclcpp::QoS(1).transient_local());
    select_pub_ = create_publisher<std_msgs::msg::Int32>("~/select_candidate", 10);
    select_sub_ = create_subscription<std_msgs::msg::Int32>("~/select_candidate", 10,
        [this](std_msgs::msg::Int32::ConstSharedPtr id) { try { select(id->data); } catch (const std::exception& e) { RCLCPP_ERROR(get_logger(), "%s", e.what()); } }, options);
    generate_service_ = service("~/generate_candidates", [this] { generate(); return "Candidates evaluated and best feasible trajectory previewed"; });
    plan_service_ = service("~/plan", [this] { generate(); return "Candidates evaluated"; });
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
        if (p.get_name() == "require_arm_selection" || p.get_name() == "candidate_arm" || p.get_name() == "planning_group" || p.get_name() == "move_group_namespace" ||
            p.get_name() == "base_frame" || p.get_name() == "tcp_link" ||
            p.get_name() == "end_effector_link" || p.get_name() == "cloud_topic" ||
            p.get_name() == "clicked_point_topic") {
          result.successful = false; result.reason = "Interface/model parameters require restart"; return result;
        }
      }
      for(const auto& parameter:parameters)
        if(parameter.get_name().rfind("b_",0)==0 || parameter.get_name().rfind("quality_",0)==0 || parameter.get_name()=="a_search_x_length" || parameter.get_name().rfind("a_from_b_",0)==0)
          manual_fallback_=false;
      if (!parameters.empty()) {
        ++input_revision_; selected_id_ = -1; clear_pending_ = true;
      }
      return result;
    });
    init_timer_ = create_wall_timer(2s, [this] {
      try {
        initialize(); init_timer_->cancel();
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "Waiting for candidate planner model: %s",e.what());
      }
    }, operations_);
    RCLCPP_INFO(get_logger(), "Agentic grasp candidate planner ready; skills backend owns all physical execution");
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
  void publishPlanReady(bool ready) {
    if(!plan_ready_pub_) return;
    std_msgs::msg::String message;
    message.data="{\"ready\":"+std::string(ready ? "true" : "false")+",\"revision\":"+
      std::to_string(input_revision_.load())+",\"selected_id\":"+std::to_string(selected_id_.load())+"}";
    plan_ready_pub_->publish(message);
  }
  void invalidate() {
    publishPlanReady(false);
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
    scene_client_ = create_client<moveit_msgs::srv::GetPlanningScene>(str("move_group_namespace") + "/get_planning_scene");
  }

  planning_scene::PlanningScenePtr sceneSnapshot() {
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
  void generate(bool from_b=false,bool recovery=false) {
    const bool automatic=(flag("b_auto_enabled") && !manual_fallback_) || from_b || recovery;
    if (!arm_selected_ && !automatic) throw std::runtime_error("Arm is not selected in RViz");
    clear_pending_=false; // Consume prior parameter/selection invalidation before publishing a fresh plan.
    invalidate(); validateParameters();
    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
    geometry_msgs::msg::PointStamped::ConstSharedPtr clicked;
    std::chrono::steady_clock::time_point received;
    { std::lock_guard<std::mutex> lock(input_mutex_);
      if(automatic) {
        cloud=cloud_;received=cloud_received_;++input_revision_;
        if(cloud) {auto point=std::make_shared<geometry_msgs::msg::PointStamped>();point->header=cloud->header;clicked=point;}
      } else {cloud=selected_cloud_;received=selected_cloud_received_;clicked=point_;}
      planned_revision_=input_revision_;pending_=false;
    }
    if(from_b && (str("candidate_arm")!="a" || str("base_frame")!="fr3_link0"))
      throw std::runtime_error("A automatic point selection requires shared B fr3_link0 frame");
    if(automatic && !from_b && (str("candidate_arm")!="b" || str("base_frame")!="fr3_link0"))
      throw std::runtime_error("Automatic X search is supported only for B in fr3_link0");
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
    Eigen::Vector3d selected = transform(clicked->header.frame_id, clicked->header.stamp) * Eigen::Vector3d(clicked->point.x, clicked->point.y, clicked->point.z);
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
    std::vector<Eigen::Vector3d> local,points;
    sensor_msgs::PointCloud2ConstIterator<float> x(*cloud,"x"), y(*cloud,"y"), z(*cloud,"z");
    for (; x != x.end(); ++x, ++y, ++z) {
      Eigen::Vector3d point(*x,*y,*z);
      if (!point.allFinite()) continue;
      point = cloud_tf*point;
      points.push_back(point);
    }
    auto continuous=[&](double lower,double upper,bool require_board_clearance=false) {
      std::function<bool(const Eigen::Vector3d&)> acceptable;
      if(require_board_clearance) {
        if(!flag("platform_enabled")) throw std::runtime_error("B automatic board clearance requires platform model enabled");
        const auto platform=cable_grasp::makePlatform(get_parameter("platform_points").as_double_array(),number("platform_thickness"),flag("platform_rectangle"));
        const auto center=platform.center,normal=platform.normal;
        const double thickness=number("platform_thickness");
        const double threshold=std::max(number("b_min_board_clearance"),number("near_board_height_threshold"));
        if(!std::isfinite(number("b_min_board_clearance")) || number("b_min_board_clearance")<0.)
          throw std::runtime_error("Invalid B minimum board clearance");
        // Use the exact inverse of the downstream near-board test, including its
        // numerical boundary tolerance, to guarantee normal roll sampling.
        acceptable=[center,normal,thickness,threshold](const Eigen::Vector3d& point) {
          return !cable_grasp::needsBoardNormalGrasp(cable_grasp::boardHeightGap(point,center,normal,thickness),threshold);
        };
        RCLCPP_INFO(get_logger(),"B automatic X scan additionally requires board surface gap > %.6f m",threshold);
      }
      const auto choice=cable_grasp::selectContinuousCablePoint(points,lower,upper,
          number("quality_min_length"),number("quality_max_gap"),number("quality_voxel_size"),
          number("local_radius"),integer("min_local_points"),number("pca_ratio"),number("max_line_offset"),acceptable);
      RCLCPP_INFO(get_logger(),"%s continuous cable X scan [%.6f, %.6f], selected [%.6f, %.6f, %.6f], connected span=%.6f m, support voxels=%zu",
          str("candidate_arm").c_str(),lower,upper,choice.point.x(),choice.point.y(),choice.point.z(),choice.span,choice.support_points);
      return choice.point;
    };
    if(recovery && str("candidate_arm")=="b") {
      if(str("base_frame")!="fr3_link0" || !last_handed_cloud_point_)
        throw std::runtime_error("B recovery needs the previous handed cloud point in fr3_link0");
      selected=cable_grasp::selectPositiveXPoint(points,*last_handed_cloud_point_,number("b_recovery_x_offset"),number("recovery_x_tolerance"));
      RCLCPP_INFO(get_logger(),"B recovery point [%.6f, %.6f, %.6f], previous point X=%.6f + %.3f m",selected.x(),selected.y(),selected.z(),last_handed_cloud_point_->x(),number("b_recovery_x_offset"));
    } else if(from_b) {
      if(b_grasp_reference_.header.frame_id!="fr3_link0") throw std::runtime_error("No successful B grasp reference available");
      const auto& position=b_grasp_reference_.pose.position;
      Eigen::Vector3d reference(position.x,position.y,position.z);
      if(flag("a_from_b_use_current_tcp")) {
        initialize();auto state=move_group_->getCurrentState(3.);
        if(!state) throw std::runtime_error("Current B TCP state unavailable");
        state->update();
        reference=(state->getGlobalLinkTransform("fr3_link0").inverse()*state->getGlobalLinkTransform("fr3_hand_tcp")).translation();
      }
      const double x_offset=number(recovery ? "a_recovery_from_b_x_offset" : "a_from_b_x_offset");
      RCLCPP_INFO(get_logger(),"A auto reference=%s, B TCP [%.6f, %.6f, %.6f], target X=%.6f +/- %.6f m; saved grasp X=%.6f",
          flag("a_from_b_use_current_tcp") ? "current TCP" : "saved grasp",reference.x(),reference.y(),reference.z(),
          reference.x()+x_offset,number("a_from_b_x_tolerance"),position.x);
      if(recovery) selected=cable_grasp::selectPositiveXPoint(points,reference,x_offset,number("a_from_b_x_tolerance"));
      else {
        const double length=number("a_search_x_length");
        if(!std::isfinite(length) || length<=0.) throw std::runtime_error("Invalid A X search length");
        selected=continuous(reference.x()+x_offset,reference.x()+x_offset+length);
      }
      RCLCPP_INFO(get_logger(),"Automatic A cable point in B fr3_link0: [%.6f, %.6f, %.6f], B reference X=%.6f, +X offset=%.3f m",
          selected.x(),selected.y(),selected.z(),reference.x(),x_offset);
    } else if(automatic) {
      initialize();
      auto state=move_group_->getCurrentState(3.);
      if(!state) throw std::runtime_error("Current A TCP unavailable for B X search");
      state->update();
      const Eigen::Vector3d a_tcp=(state->getGlobalLinkTransform("fr3_link0").inverse()*state->getGlobalLinkTransform("a_fr3_hand_tcp")).translation();
      const double clearance=number("b_search_a_tcp_clearance");
      if(!std::isfinite(clearance) || clearance<0.) throw std::runtime_error("Invalid B search clearance");
      selected=continuous(number("b_search_x_min"),a_tcp.x()-clearance,true);
    }
    generated_cloud_point_=selected;
    for(const auto& point:points)
      if((point-selected).norm()<=number("local_radius")) local.push_back(point);
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
    Eigen::Vector3d grasp_target=selected;
    bool near_board=false;
    if (flag("platform_enabled")) {
      const auto platform=cable_grasp::makePlatform(
          get_parameter("platform_points").as_double_array(),number("platform_thickness"),flag("platform_rectangle"));
      const double gap=cable_grasp::boardHeightGap(selected,platform.center,platform.normal,number("platform_thickness"));
      near_board=cable_grasp::needsBoardNormalGrasp(gap,number("near_board_height_threshold"));
      grasp_target=cable_grasp::raisedBoardTarget(selected,gap,
          number("near_board_raise_threshold"),number("near_board_target_raise"));
      RCLCPP_INFO(get_logger(), "Near-board target correction: original [%.6f, %.6f, %.6f], target [%.6f, %.6f, %.6f], +Y shift %.6f m (before target_offset)",
          selected.x(),selected.y(),selected.z(),grasp_target.x(),grasp_target.y(),grasp_target.z(),grasp_target.y()-selected.y());
      RCLCPP_INFO(get_logger(), "Selected point board +Y surface height gap %.6f m, threshold %.6f m; %s",
          gap,number("near_board_height_threshold"),near_board ? "TCP Z fixed to base -Y; two tangent-sign candidates" : "normal roll sampling");
    }
    const int total=near_board ? 2 : 2*static_cast<int>(std::ceil(360./number("roll_step_degrees")));
    int id = 0;
    for (int sign : {1,-1}) for (double degrees=0; degrees<(near_board ? 1. : 360.-1e-6); degrees+=number("roll_step_degrees")) {
      Candidate candidate; candidate.id = id++; candidate.sign = sign; candidate.angle = degrees;
      Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity();
      tcp.linear() = near_board ? cable_grasp::boardNormalRotation(tangent,sign) :
          cable_grasp::candidateRotation(tangent,sign,degrees*M_PI/180.);
      tcp.translation() = grasp_target+Eigen::Vector3d(offset[0],offset[1],offset[2]);
      Eigen::Isometry3d entry = tcp; entry.translation() -= number("approach_distance")*tcp.linear().col(2);
      candidate.tcp = pose(tcp); candidate.entry = pose(entry);
      candidates_.push_back(std::move(candidate));
    }
    const int requested=integer("candidate_worker_threads");
    if(requested<1 || requested>10) throw std::runtime_error("candidate_worker_threads must be 1..10");
    const int workers=std::min(requested,total);
    const auto began=std::chrono::steady_clock::now();
    if(workers==1) {
      for(auto& candidate:candidates_) {
        try {planCandidate(candidate,*start,*scene,base_to_model,tcp_to_tip);}
        catch(const std::exception& e) {candidate.reason=e.what();}
      }
    } else {
      // Each worker owns its OMPL manager and scene. Never share mutable MGI goals.
      while(static_cast<int>(candidate_pipelines_.size())<workers) {
        auto pipeline=std::make_unique<planning_pipeline::PlanningPipeline>(move_group_->getRobotModel(),shared_from_this(),
            "ompl","ompl_interface/OMPLPlanner",std::vector<std::string>{});
        pipeline->displayComputedMotionPlans(false);pipeline->publishReceivedRequests(false);
        candidate_pipelines_.push_back(std::move(pipeline));
      }
      moveit_msgs::msg::PlanningScene snapshot;scene->getPlanningSceneMsg(snapshot);
      std::vector<planning_scene::PlanningScenePtr> scenes;
      for(int i=0;i<workers;++i) {
        auto copy=std::make_shared<planning_scene::PlanningScene>(move_group_->getRobotModel());
        if(!copy->setPlanningSceneMsg(snapshot)) throw std::runtime_error("Cannot copy candidate planning scene");
        scenes.push_back(std::move(copy));
      }
      std::atomic<size_t> next{0};std::vector<std::future<void>> jobs;
      for(int i=0;i<workers;++i) jobs.push_back(std::async(std::launch::async,[&,i] {
        for(size_t index;(index=next.fetch_add(1))<candidates_.size();) {
          auto& candidate=candidates_[index];
          try {planCandidate(candidate,*start,*scenes[i],base_to_model,tcp_to_tip,candidate_pipelines_[i].get());}
          catch(const std::exception& e) {candidate.reason=e.what();}
        }
      }));
      for(auto& job:jobs) job.get();
    }
    for(const auto& candidate:candidates_)
      RCLCPP_INFO(get_logger(),"Candidate %d/%d: %s",candidate.id+1,total,candidate.reason.c_str());
    RCLCPP_INFO(get_logger(),"Evaluated %d candidates with %d workers in %.3f s",total,workers,
      std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count());
    publishCandidates();
    const Candidate* best = nullptr;
    for (const auto& c : candidates_) if (c.valid && (!best || c.score > best->score)) best = &c;
    if (!best) throw std::runtime_error("No feasible candidate; inspect candidate_report and adjust selection/scene");
    select(best->id);
  }
  void planCandidate(Candidate& candidate, const moveit::core::RobotState& start,
                     const planning_scene::PlanningScene& scene, const Eigen::Isometry3d& base_to_model,
                     const Eigen::Isometry3d& tcp_to_tip,planning_pipeline::PlanningPipeline* pipeline=nullptr) {
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
        bool ik_ok;
        {std::unique_lock<std::mutex> lock(candidate_ik_mutex_,std::defer_lock);if(pipeline) lock.lock();
          ik_ok=ik.setFromIK(group,base_to_model*matrix(candidate.entry)*tcp_to_tip,str("end_effector_link"),0.1,valid);}
        if(!ik_ok) {failure="MoveIt entry IK failed or state invalid";continue;}
        MGI::Plan first;
        robot_trajectory::RobotTrajectory joined(move_group_->getRobotModel(),str("planning_group"));
        if(pipeline) {
          planning_interface::MotionPlanRequest request;planning_interface::MotionPlanResponse response;
          request.group_name=str("planning_group");request.planner_id="RRTConnectkConfigDefault";
          request.allowed_planning_time=number("planning_time");request.num_planning_attempts=1;
          moveit::core::robotStateToRobotStateMsg(start,request.start_state);
          request.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(ik,group));
          const planning_scene::PlanningSceneConstPtr view(&scene,[](const auto*){});
          if(!pipeline->generatePlan(view,request,response) || response.error_code_.val!=moveit_msgs::msg::MoveItErrorCodes::SUCCESS || !response.trajectory_) {
            failure="Local MoveIt planning failed";continue;
          }
          joined=*response.trajectory_;first.start_state_=request.start_state;
        } else {
          move_group_->setStartState(start);move_group_->setJointValueTarget(ik);
          const auto result=move_group_->plan(first);
          if(result!=moveit::core::MoveItErrorCode::SUCCESS) {failure="MoveIt planning failed code="+std::to_string(result.val);continue;}
          joined.setRobotTrajectoryMsg(start,first.trajectory_);
        }
        if(!joined.getWayPointCount()) continue;
        robot_trajectory::RobotTrajectory approach(move_group_->getRobotModel(),str("planning_group"));
        if(pipeline) {
          auto state=joined.getLastWayPoint();std::vector<moveit::core::RobotStatePtr> path;
          double fraction;
          {std::lock_guard<std::mutex> lock(candidate_ik_mutex_);
            fraction=state.computeCartesianPath(group,path,state.getLinkModel(str("end_effector_link")),
              EigenSTL::vector_Isometry3d{base_to_model*matrix(candidate.tcp)*tcp_to_tip},true,
              number("cartesian_step"),number("jump_threshold"),valid);}
          if(fraction<0.999999) {failure="Incomplete Cartesian approach";continue;}
          for(const auto& waypoint:path) approach.addSuffixWayPoint(*waypoint,0.1);
        } else {
          move_group_->setStartState(joined.getLastWayPoint());
          moveit_msgs::msg::RobotTrajectory message;
          const Pose goal=pose(matrix(candidate.tcp)*tcp_to_tip);
          const double fraction=move_group_->computeCartesianPath({goal.pose},number("cartesian_step"),number("jump_threshold"),message,true);
          if(fraction<0.999999) {failure="Incomplete Cartesian approach";continue;}
          approach.setRobotTrajectoryMsg(joined.getLastWayPoint(),message);
        }
        for(size_t i=1;i<approach.getWayPointCount();++i) joined.addSuffixWayPoint(approach.getWayPoint(i),0.1);
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



  void select(int id) {
    if (planned_revision_!=input_revision_) throw std::runtime_error("Selection changed; regenerate candidates");
    if (id<0 || static_cast<size_t>(id)>=candidates_.size() || !candidates_[id].valid) throw std::runtime_error("Candidate invalid; inspect report");
    selected_id_=-1;
    auto& candidate=candidates_[id]; target_pub_->publish(candidate.tcp); entry_pub_->publish(candidate.entry);
    moveit_msgs::msg::DisplayTrajectory preview; preview.model_id=move_group_->getRobotModel()->getName();
    preview.trajectory_start=candidate.plan.start_state_; preview.trajectory.push_back(candidate.plan.trajectory_);

    preview_pub_->publish(preview);
    selected_id_=id;
    selected_plan_time_=now();
    publishCandidates();publishPlanReady(true);
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


  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr plan_ready_pub_;
  rclcpp::Service<Trigger>::SharedPtr prepare_manual_service_;
  std::optional<Eigen::Vector3d> last_handed_cloud_point_;
  Eigen::Vector3d generated_cloud_point_=Eigen::Vector3d::Zero();
  std::string last_plan_error_;
  rclcpp::Service<Trigger>::SharedPtr last_error_service_;
  std::atomic_bool manual_fallback_{false};
  Pose b_grasp_reference_;
  rclcpp::Subscription<Pose>::SharedPtr b_grasp_sub_;
  std::atomic_bool arm_selected_{false};
  rclcpp::Time selected_plan_time_{0,0,RCL_ROS_TIME};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr arm_selection_sub_;
  rclcpp::Service<moveit_msgs::srv::GetMotionPlan>::SharedPtr selected_plan_service_;
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
  std::mutex candidate_ik_mutex_;
  std::vector<std::unique_ptr<planning_pipeline::PlanningPipeline>> candidate_pipelines_;
  std::unique_ptr<MGI> move_group_;
  std::unique_ptr<interactive_markers::InteractiveMarkerServer> server_;
  interactive_markers::MenuHandler menu_;
  rclcpp::CallbackGroup::SharedPtr operations_;
  rclcpp::TimerBase::SharedPtr timer_, init_timer_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr point_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr select_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr report_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr select_pub_;
  rclcpp::Publisher<Pose>::SharedPtr target_pub_,entry_pub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr preview_pub_;
  rclcpp::Service<Trigger>::SharedPtr generate_service_,plan_service_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr scene_client_;
};
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);
  auto node=std::make_shared<GraspCandidatePlanner>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),4);
  executor.add_node(node); executor.spin(); rclcpp::shutdown(); return 0;
}
