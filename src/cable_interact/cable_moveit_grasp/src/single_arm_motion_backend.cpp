#include <cable_moveit_grasp/single_arm_motion.hpp>
using namespace std::chrono_literals;
SingleArmMotion::SingleArmMotion():Node("single_arm_motion") {
  declare_parameter("execution_enabled",false);
  declare_parameter("include_arm_a_collision",true);
  declare_parameter("planning_group","fr3_arm");declare_parameter("arm_prefix","");
  declare_parameter("move_group_namespace","/single_arm");declare_parameter("robot_namespace","/NS_1");
  declare_parameter("arm_a_namespace","/NS_2");declare_parameter("base_frame","fr3_link0");declare_parameter("tcp_link","fr3_hand_tcp");
  declare_parameter("gripper_namespace","/NS_1/franka_gripper");
  declare_parameter("planner_service","/single_arm/planner/candidate_planner/get_selected_grasp_plan");
  declare_parameter("candidate_plan_timeout",120.0);declare_parameter("gripper_timeout",15.0);
  declare_parameter("velocity_scaling",.05);declare_parameter("acceleration_scaling",.05);
  declare_parameter("open_width",.08);declare_parameter("grasp_width",.004);declare_parameter("gripper_speed",.02);
  declare_parameter("gripper_force",10.);declare_parameter("grasp_epsilon",.003);
  declare_parameter("lift_distance",.40);declare_parameter("lift_position_tolerance",.01);
  declare_parameter("place_joint_positions",std::vector<double>{-.5233661532402039,-1.5837199687957764,-.40842023491859436,-1.467320442199707,-1.0411325693130493,1.900314211845398,-2.7240424156188965});
  declare_parameter("place_planning_time",5.);declare_parameter("place_goal_tolerance",.01);
  declare_parameter("place_arrival_tolerance",.10);declare_parameter("arrival_timeout",3.);
  declare_parameter("platform_thickness",.01);
  declare_parameter("platform_points",std::vector<double>{-.252,-.504,.422,-.234,-.515,.936,.108,-.524,.953,.380,-.530,.907,.557,-.531,.740,.644,-.528,.573,.653,-.525,.454});
  declare_parameter("camera_obstacle_enabled",true);declare_parameter("camera_obstacle_size",std::vector<double>{.12,.05,.04});
  declare_parameter("camera_obstacle_pose",std::vector<double>{});
  operations_=create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  status_=create_publisher<std_msgs::msg::String>("~/status",rclcpp::QoS(1).transient_local());
  preview_=create_publisher<moveit_msgs::msg::DisplayTrajectory>("~/recycling_preview",rclcpp::QoS(1).transient_local());
  auto source=[&](const std::string& ns,const std::string& prefix) {
    std::vector<std::string> joints,fingers;
    for(int i=1;i<=7;++i) joints.push_back(prefix+"fr3_joint"+std::to_string(i));
    for(int i=1;i<=2;++i) fingers.push_back(prefix+"fr3_finger_joint"+std::to_string(i));
    required_.insert(required_.end(),joints.begin(),joints.end());required_.insert(required_.end(),fingers.begin(),fingers.end());
    arm_groups_.push_back(prefix+"fr3_arm");
    subscriptions_.push_back(create_subscription<sensor_msgs::msg::JointState>(ns+"/franka/joint_states",rclcpp::SensorDataQoS(),
       [this,joints](sensor_msgs::msg::JointState::ConstSharedPtr m){receive(*m,joints);}));
    subscriptions_.push_back(create_subscription<sensor_msgs::msg::JointState>(ns+"/franka_gripper/joint_states",rclcpp::SensorDataQoS(),
       [this,fingers](sensor_msgs::msg::JointState::ConstSharedPtr m){receive(*m,fingers);}));
  };
  source(get_parameter("robot_namespace").as_string(),get_parameter("arm_prefix").as_string());
  if(get_parameter("include_arm_a_collision").as_bool()) source(get_parameter("arm_a_namespace").as_string(),"a_");
  auto service=[&](const std::string& name,std::function<void()> function,bool physical) {
    services_.push_back(create_service<Trigger>("~/"+name,[this,function,physical](Trigger::Request::SharedPtr,Trigger::Response::SharedPtr r) {
      physical_sent_=false;
      try {
        if(physical && !get_parameter("execution_enabled").as_bool()) throw std::runtime_error("execution_enabled=false");
        initialize();function();r->success=true;r->message="completed";
      } catch(const std::exception& error) {
        if(physical_sent_) fault_=true;
        r->message=(fault_ ? "PHYSICAL_FAILURE: " : "PRECHECK_REJECTED: ")+std::string(error.what());report(r->message);
      }
    },rmw_qos_profile_services_default,operations_));
  };
  service("preview_selected",[this]{preview();},false);
  service("execute_selected",[this]{executeSelected();},true);
  service("release",[this]{open();holding_=false;uncertain_=false;report("released; reset fault explicitly before a new cycle");},true);
  service("reset",[this]{
    const auto state=measured();const auto prefix=get_parameter("arm_prefix").as_string();
    if(holding_ || uncertain_ || state.getVariablePosition(prefix+"fr3_finger_joint1")+state.getVariablePosition(prefix+"fr3_finger_joint2")<.06)
      throw std::runtime_error("Reset requires explicit release and open measured hand");
    fault_=false;report("idle");
  },false);
  timer_=create_wall_timer(2s,[this]{try {initialize();} catch(const std::exception& e){RCLCPP_WARN(get_logger(),"Waiting for single-arm scene: %s",e.what());}},operations_);
  report("idle; RViz point -> preview_selected -> execute_selected");
}
void SingleArmMotion::receive(const sensor_msgs::msg::JointState& msg,const std::vector<std::string>& allowed) {
  const auto stamp=rclcpp::Time(msg.header.stamp).nanoseconds();const double age=(now().nanoseconds()-stamp)*1e-9;
  if(stamp<=0 || age<-.5 || age>1. || msg.name.size()!=msg.position.size()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  for(size_t i=0;i<msg.name.size();++i) if(std::find(allowed.begin(),allowed.end(),msg.name[i])!=allowed.end() && std::isfinite(msg.position[i])) {
    if(stamps_.count(msg.name[i]) && stamp<stamps_.at(msg.name[i])) continue;
    positions_[msg.name[i]]=msg.position[i];stamps_[msg.name[i]]=stamp;
  }
}
void SingleArmMotion::report(const std::string& message) {
  phase_=message;std_msgs::msg::String msg;msg.data=message;status_->publish(msg);RCLCPP_INFO(get_logger(),"%s",message.c_str());
}
void SingleArmMotion::initialize() {
  if(!group_) {
    const auto ns=get_parameter("move_group_namespace").as_string();
    group_=std::make_unique<MGI>(shared_from_this(),MGI::Options(get_parameter("planning_group").as_string(),"robot_description",ns));
    group_->setPoseReferenceFrame(get_parameter("base_frame").as_string());group_->setEndEffectorLink(get_parameter("tcp_link").as_string());
    group_->setMaxVelocityScalingFactor(get_parameter("velocity_scaling").as_double());group_->setMaxAccelerationScalingFactor(get_parameter("acceleration_scaling").as_double());
    world_=std::make_unique<moveit::planning_interface::PlanningSceneInterface>(ns);
    scene_client_=create_client<moveit_msgs::srv::GetPlanningScene>(ns+"/get_planning_scene");
  }
  if(ready_) return;
  const auto platform=cable_grasp::makePlatform(get_parameter("platform_points").as_double_array(),get_parameter("platform_thickness").as_double(),true);
  moveit_msgs::msg::CollisionObject table;table.id="cable_platform";table.header.frame_id=get_parameter("base_frame").as_string();table.operation=table.ADD;
  table.meshes.push_back(platform.mesh);geometry_msgs::msg::Pose pose;pose.orientation.w=1.;table.mesh_poses.push_back(pose);
  if(!world_->applyCollisionObject(table)) throw std::runtime_error("Cannot apply table");
  if(get_parameter("camera_obstacle_enabled").as_bool()) {
    const auto size=get_parameter("camera_obstacle_size").as_double_array(),camera=get_parameter("camera_obstacle_pose").as_double_array();
    if(size.size()!=3 || camera.size()!=7) throw std::runtime_error("Invalid camera obstacle parameters");
    for(double x:size) if(!std::isfinite(x) || x<=0.) throw std::runtime_error("Invalid camera size");
    for(double x:camera) if(!std::isfinite(x)) throw std::runtime_error("Invalid camera pose");
    if(std::abs(Eigen::Quaterniond(camera[6],camera[3],camera[4],camera[5]).norm()-1.)>1e-6) throw std::runtime_error("Invalid camera quaternion");
    moveit_msgs::msg::CollisionObject box;box.id="realsense_camera";box.header=table.header;box.operation=box.ADD;
    shape_msgs::msg::SolidPrimitive shape;shape.type=shape.BOX;shape.dimensions.assign(size.begin(),size.end());box.primitives.push_back(shape);
    pose.position.x=camera[0];pose.position.y=camera[1];pose.position.z=camera[2];pose.orientation.x=camera[3];pose.orientation.y=camera[4];pose.orientation.z=camera[5];pose.orientation.w=camera[6];box.primitive_poses.push_back(pose);
    if(!world_->applyCollisionObject(box)) throw std::runtime_error("Cannot apply camera");
  }
  ready_=true;
}
moveit::core::RobotState SingleArmMotion::measured() {
  auto current=group_->getCurrentState(3.);if(!current) throw std::runtime_error("Current model state unavailable");
  std::lock_guard<std::mutex> lock(mutex_);
  for(const auto& name:required_) {
    if(!stamps_.count(name) || (now().nanoseconds()-stamps_.at(name))*1e-9>1. || (now().nanoseconds()-stamps_.at(name))*1e-9<-.5)
      throw std::runtime_error("Missing/stale raw measurement: "+name);
    const auto value=positions_.at(name);current->setVariablePosition(name,name.find("finger")!=std::string::npos ? cable_grasp::boundedFingerPosition(*current,name,value) : value);
  }
  current->update();return *current;
}
planning_scene::PlanningScenePtr SingleArmMotion::scene() {
  return cable_grasp::requestMotionScene(scene_client_,group_->getRobotModel(),get_parameter("camera_obstacle_enabled").as_bool());
}
SingleArmMotion::MGI::Plan SingleArmMotion::selected(bool preview) {
  auto client=create_client<moveit_msgs::srv::GetMotionPlan>(get_parameter("planner_service").as_string());
  if(!client->wait_for_service(3s)) throw std::runtime_error("Candidate planner unavailable");
  auto request=std::make_shared<moveit_msgs::srv::GetMotionPlan::Request>();if(preview) request->motion_plan_request.planner_id="preview_only";
  auto future=client->async_send_request(request);
  if(future.wait_for(std::chrono::duration<double>(get_parameter("candidate_plan_timeout").as_double()))!=std::future_status::ready) throw std::runtime_error("Candidate request timeout");
  auto data=cable_grasp::takeSelectedGraspPlan(future.get(),group_->getName());MGI::Plan plan;plan.start_state_=data.trajectory_start;plan.trajectory_=data.trajectory;return plan;
}
robot_trajectory::RobotTrajectory SingleArmMotion::validate(const MGI::Plan& plan,bool assume_open) {
  auto current=measured();auto world=scene();auto planned=current;moveit::core::robotStateMsgToRobotState(plan.start_state_,planned);
  cable_grasp::requireUnchangedArms(planned,current,arm_groups_,.02);
  cable_grasp::requireCandidateJointNames(plan.trajectory_,current.getJointModelGroup(group_->getName())->getVariableNames());
  if(assume_open) {
    const auto prefix=get_parameter("arm_prefix").as_string();for(int i=1;i<=2;++i) current.setVariablePosition(prefix+"fr3_finger_joint"+std::to_string(i),get_parameter("open_width").as_double()/2.);
    current.update();
  }
  robot_trajectory::RobotTrajectory path(group_->getRobotModel(),group_->getName());path.setRobotTrajectoryMsg(current,plan.trajectory_);
  for(const auto& name:path.getGroup()->getVariableNames())
    if(std::abs(path.getFirstWayPoint().getVariablePosition(name)-current.getVariablePosition(name))>.02) throw std::runtime_error("Robot moved since planning");
  return cable_grasp::validateTrajectory(path,*world,.01);
}
void SingleArmMotion::executePlan(const MGI::Plan& plan) {
  validate(plan);physical_sent_=true;
  if(group_->execute(plan)!=moveit::core::MoveItErrorCode::SUCCESS) {group_->stop();throw std::runtime_error("MoveIt trajectory execution failed; keep grasp");}
}
void SingleArmMotion::open() {franka_msgs::action::Move::Goal goal;goal.width=get_parameter("open_width").as_double();goal.speed=get_parameter("gripper_speed").as_double();gripper<franka_msgs::action::Move>(goal);}
void SingleArmMotion::publishPreview(const MGI::Plan& grasp,const MGI::Plan& lift,const MGI::Plan& place) {
  moveit_msgs::msg::DisplayTrajectory msg;msg.model_id=group_->getRobotModel()->getName();msg.trajectory_start=grasp.start_state_;
  msg.trajectory={grasp.trajectory_,lift.trajectory_,place.trajectory_};preview_->publish(msg);
}
void SingleArmMotion::preview() {
  if(fault_ || holding_ || uncertain_) throw std::runtime_error("Recovery required before new preview");
  const auto grasp=selected(true);const auto checked=validate(grasp,true);auto closed=checked.getLastWayPoint();
  const auto prefix=get_parameter("arm_prefix").as_string();for(int i=1;i<=2;++i) closed.setVariablePosition(prefix+"fr3_finger_joint"+std::to_string(i),get_parameter("grasp_width").as_double()/2.);closed.update();
  const auto lift=planLift(closed);robot_trajectory::RobotTrajectory lifted(group_->getRobotModel(),group_->getName());lifted.setRobotTrajectoryMsg(closed,lift.trajectory_);
  publishPreview(grasp,lift,planPlace(lifted.getLastWayPoint()));previewed_=grasp;report("preview_ready: grasp + link0 +Y lift + place; execution replans lift/place from measured state");
}
void SingleArmMotion::executeSelected() {
  if(fault_ || holding_ || uncertain_) throw std::runtime_error("Recovery required; no implicit release of held cable");
  // Preview also checks that a hypothetical complete lift/place route exists,
  // but actual post-grasp trajectories always use new measured state and scene.
  preview();const auto grasp=selected(false);
  if(!previewed_ || grasp.trajectory_!=previewed_->trajectory_) throw std::runtime_error("Selection changed during full preview; preview again");
  validate(grasp,true);
  report("opening");open();report("grasp_trajectory");executePlan(grasp);
  franka_msgs::action::Grasp::Goal close;close.width=get_parameter("grasp_width").as_double();close.speed=get_parameter("gripper_speed").as_double();close.force=get_parameter("gripper_force").as_double();close.epsilon.inner=close.epsilon.outer=get_parameter("grasp_epsilon").as_double();
  report("closing");uncertain_=true;gripper<franka_msgs::action::Grasp>(close);holding_=true;uncertain_=false;
  executeLiftSkill();executePlaceSkill();report("release_at_place");open();holding_=false;report("completed: cable released at recycling target");
}
