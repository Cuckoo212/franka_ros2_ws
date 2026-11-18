#include <cable_moveit_grasp/dual_motion.hpp>
#include <cable_moveit_grasp/fix_projection.hpp>

DualMotion::DualMotion(): Node("dual_arm_motion") {
    b_grasp_pub_=create_publisher<geometry_msgs::msg::PoseStamped>("~/last_successful_b_grasp",rclcpp::QoS(1).transient_local());
    declare_parameter("execution_enabled",false);
    declare_parameter("use_rviz_grasp_plan",true);
    declare_parameter("candidate_plan_timeout",120.0);
    declare_parameter("grasp_max_reselections",1);
    for (const auto* name:{"homeA","homeB","fixA","fixB"})
      declare_parameter(name,std::vector<double>{});
    declare_parameter("planning_time",5.0);
    declare_parameter("fix_tcp_max_total_rotation",3.141592653589793);
    declare_parameter("arrival_joint_tolerance",0.02);
    declare_parameter("arrival_joint_tolerances_a",std::vector<double>{.02,.02,.02,.02,.02,.02,0.17453292519943295});
    declare_parameter("arrival_joint_tolerances_b",std::vector<double>{});
    declare_parameter("arrival_timeout",3.0);
    declare_parameter("arrival_stable_duration",0.1);
    declare_parameter("velocity_scaling",0.05);
    declare_parameter("acceleration_scaling",0.05);
    declare_parameter("camera_obstacle_enabled",true);
    declare_parameter("camera_obstacle_size",std::vector<double>{.12,.05,.04});
    declare_parameter("camera_obstacle_center_optical",std::vector<double>{0.,0.,0.});
    declare_parameter("camera_obstacle_pose",std::vector<double>{});
    declare_parameter("platform_thickness",0.01);
    declare_parameter("platform_points",std::vector<double>{-.252,-.504,.422,-.234,-.515,.936,.108,-.524,.953,.380,-.530,.907,.557,-.531,.740,.644,-.528,.573,.653,-.525,.454});
    status_pub_=create_publisher<std_msgs::msg::String>("~/status",rclcpp::QoS(1).transient_local());
    override_pub_=create_publisher<std_msgs::msg::String>("~/manual_override",10);
    execution_state_pub_=create_publisher<std_msgs::msg::String>("~/execution_state",rclcpp::QoS(1).transient_local());
    operations_=create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    for (std::string arm:{"a","b"}) {
      fix_preview_pub_[arm]=create_publisher<moveit_msgs::msg::DisplayTrajectory>(
          "~/rejected_fix_"+arm,rclcpp::QoS(1).transient_local());
      auto ns=arm=="a" ? "/NS_2" : "/NS_1";
      subscriptions_.push_back(create_subscription<sensor_msgs::msg::JointState>(std::string(ns)+"/franka/joint_states",10,
        [this,arm](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
          const rclcpp::Time stamp(msg->header.stamp);
          const double age=(now()-stamp).seconds();
          if(age<-.5 || age>1.) return;
          std::lock_guard<std::mutex> lock(mutex_);
          for (size_t i=0;i<msg->name.size() && i<msg->position.size();++i)
            if (std::isfinite(msg->position[i]) && msg->name[i].find("finger")==std::string::npos) {
              joint_times_[msg->name[i]]=stamp;
              joint_positions_[msg->name[i]]=msg->position[i];
            }
        }));
      subscriptions_.push_back(create_subscription<sensor_msgs::msg::JointState>(std::string(ns)+"/franka_gripper/joint_states",rclcpp::SensorDataQoS(),
        [this,arm](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
          const rclcpp::Time stamp(msg->header.stamp);
          const double age=(now()-stamp).seconds();if(age<-.5 || age>1.) return;
          const std::string prefix=arm=="a" ? "a_" : "";
          std::lock_guard<std::mutex> lock(mutex_);
          for(size_t i=0;i<msg->name.size() && i<msg->position.size();++i) {
            const auto& name=msg->name[i];
            if((name==prefix+"fr3_finger_joint1" || name==prefix+"fr3_finger_joint2") && std::isfinite(msg->position[i])) {
              if(joint_times_.count(name) && stamp<joint_times_.at(name)) continue;
              joint_times_[name]=stamp;joint_positions_[name]=msg->position[i];
            }
          }
        }));
      targets_.push_back(create_subscription<geometry_msgs::msg::PoseStamped>("~/"+arm+"/grasp_target",10,
        [this,arm](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex_); goals_[arm]=*msg; goal_times_[arm]=now();
        }));
      for (std::string op:{"home","fix","grasp","verify","release"})
        services_.push_back(create_service<Trigger>("~/"+arm+"/"+op,
          [this,arm,op](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response) {
            performRequest(arm,op,false,response);
          },rmw_qos_profile_services_default,operations_));
      for(std::string command:{"grasp","fix","home"})
        services_.push_back(create_service<Trigger>("~/"+command+"_"+arm,
          [this,arm,command](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response) {
            performRequest(arm,command,true,response);
          },rmw_qos_profile_services_default,operations_));
    }
    for(std::string arm:{"a","b"}) for(std::string command:{"grasp","fix","home"})
      services_.push_back(create_service<Trigger>("~/auto/"+command+"_"+arm,
        [this,arm,command](Trigger::Request::SharedPtr,Trigger::Response::SharedPtr response) {
          performRequest(arm,command,true,response,true);
        },rmw_qos_profile_services_default,operations_));
    services_.push_back(create_service<Trigger>("~/get_execution_state",
      [this](Trigger::Request::SharedPtr,Trigger::Response::SharedPtr response) {
        response->success=true;response->message=executionState();
      },rmw_qos_profile_services_default,operations_));
    publishExecutionState();
    timer_=create_wall_timer(2s,[this] {
      try {initialize();} catch(const std::exception& e) {RCLCPP_WARN(get_logger(),"Waiting for scene: %s",e.what());}
    },operations_);
  }

void DualMotion::performRequest(const std::string& arm,const std::string& command,bool manual,
                      Trigger::Response::SharedPtr response,bool coordinated) {
    physical_command_sent_=false;
    recovering_home_=manual && command=="home" && !coordinated;
    const bool previous_fault=fault_;
    if(!coordinated) {std_msgs::msg::String event;event.data=command+"_"+arm;override_pub_->publish(event);}
    active_command_=command+"_"+arm;
    report("request received");
    try {
      if(!get_parameter("execution_enabled").as_bool()) throw std::runtime_error("execution_enabled=false");
      report("checking robot state and shared scene");
      initialize();freshState();
      if(coordinated && command=="home" && holding_[arm] && !holding_[arm=="a" ? "b" : "a"])
        throw std::runtime_error("Automatic HOME cannot release the only confirmed holder");
      if(manual) {
        const auto other=arm=="a" ? "b" : "a";
        const auto operations=cable_grasp::manualOperations(command,holding_[arm],holding_[other]);
        for(const auto& op:operations) {freshState();run(arm,op);}
      } else run(arm,command);
      if(recovering_home_ && previous_fault && (fault_arm_.empty() || fault_arm_==arm)) {fault_=false;fault_arm_.clear();}
      response->success=true;response->message=command+"_"+arm+" completed";
      report("completed");
    } catch(const cable_grasp::GraspRecovered& error) {
      response->message="PRECHECK_REJECTED: GRASP_PLAN_REJECTED: "+std::string(error.what());
      report(response->message); // HOME succeeded, no physical fault: allow fresh RViz input.
    } catch(const std::exception& error) {
      // A compound command may have completed its release before a later check failed.
      if(physical_command_sent_) fault_arm_=arm;
      fault_=fault_ || physical_command_sent_;
      response->message=(fault_ ? "PHYSICAL_FAILURE: " : "PRECHECK_REJECTED: ")+std::string(error.what());
      report(response->message);
      RCLCPP_ERROR(get_logger(),"%s",response->message.c_str());
    }
    recovering_home_=false;publishExecutionState();
  }

std::string DualMotion::executionState() const {
    std::ostringstream result;result<<std::boolalpha<<"{\"fault\":"<<fault_<<",\"fault_arm\":\""<<fault_arm_
      <<"\",\"holding_a\":"<<holding_.at("a")<<",\"holding_b\":"<<holding_.at("b")
      <<",\"home_a\":"<<home_.at("a")<<",\"home_b\":"<<home_.at("b")<<"}";
    return result.str();
  }

void DualMotion::publishExecutionState() {std_msgs::msg::String message;message.data=executionState();execution_state_pub_->publish(message);}

void DualMotion::report(const std::string& phase) {
    std_msgs::msg::String message;message.data=active_command_+": "+phase;
    status_pub_->publish(message);RCLCPP_INFO(get_logger(),"%s",message.data.c_str());
  }

void DualMotion::initialize() {
    if (!a_) {
      a_=std::make_unique<MGI>(shared_from_this(),MGI::Options("a_fr3_arm","robot_description","/dual_arm"));
      b_=std::make_unique<MGI>(shared_from_this(),MGI::Options("fr3_arm","robot_description","/dual_arm"));
      for (auto* group:{a_.get(),b_.get()}) {
        group->setPlanningTime(get_parameter("planning_time").as_double());
        group->setMaxVelocityScalingFactor(get_parameter("velocity_scaling").as_double());
        group->setMaxAccelerationScalingFactor(get_parameter("acceleration_scaling").as_double());
        group->setPoseReferenceFrame("fr3_link0");
      }
      a_->setEndEffectorLink("a_fr3_hand_tcp"); b_->setEndEffectorLink("fr3_hand_tcp");
      scene_=std::make_unique<moveit::planning_interface::PlanningSceneInterface>("/dual_arm");
      scene_client_=create_client<moveit_msgs::srv::GetPlanningScene>("/dual_arm/get_planning_scene");
    }
    if (!platform_ready_) {
      auto platform=cable_grasp::makePlatform(get_parameter("platform_points").as_double_array(),get_parameter("platform_thickness").as_double(),true);
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id="fr3_link0"; object.id="cable_platform"; object.operation=object.ADD;
      object.meshes.push_back(platform.mesh); geometry_msgs::msg::Pose pose; pose.orientation.w=1.; object.mesh_poses.push_back(pose);
      if (!scene_->applyCollisionObject(object)) throw std::runtime_error("Cannot load rectangular table");
      platform_ready_=true;
    }
    if (get_parameter("camera_obstacle_enabled").as_bool() && !camera_ready_) {
      const auto size=get_parameter("camera_obstacle_size").as_double_array();
      const auto pose=get_parameter("camera_obstacle_pose").as_double_array();
      if(size.size()!=3 || pose.size()!=7) throw std::runtime_error("Camera obstacle requires calibrated pose and XYZ size");
      for(double v:size) if(!std::isfinite(v) || v<=0.) throw std::runtime_error("Invalid camera obstacle size");
      for(double v:pose) if(!std::isfinite(v)) throw std::runtime_error("Invalid camera obstacle pose");
      const double norm=std::sqrt(pose[3]*pose[3]+pose[4]*pose[4]+pose[5]*pose[5]+pose[6]*pose[6]);
      if(std::abs(norm-1.)>1e-6) throw std::runtime_error("Invalid camera obstacle quaternion");
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id="fr3_link0"; object.id="realsense_camera"; object.operation=object.ADD;
      shape_msgs::msg::SolidPrimitive box; box.type=box.BOX; box.dimensions.assign(size.begin(),size.end());
      geometry_msgs::msg::Pose center;
      center.position.x=pose[0];center.position.y=pose[1];center.position.z=pose[2];
      center.orientation.x=pose[3];center.orientation.y=pose[4];center.orientation.z=pose[5];center.orientation.w=pose[6];
      object.primitives.push_back(box);object.primitive_poses.push_back(center);
      if(!scene_->applyCollisionObject(object)) throw std::runtime_error("Cannot load calibrated RealSense obstacle");
      camera_ready_=true;
      RCLCPP_INFO(get_logger(),"RealSense collision box loaded in fr3_link0 at [%.6f, %.6f, %.6f]",pose[0],pose[1],pose[2]);
    }
  }

void DualMotion::freshState() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::string prefix:{"","a_"}) for(int i=1;i<=7;++i) {
      auto found=joint_times_.find(prefix+"fr3_joint"+std::to_string(i));
      if(found==joint_times_.end() || (now()-found->second).seconds()>1.)
        throw std::runtime_error("Missing or stale joint state for either arm");
    }
  }

MGI::Plan DualMotion::selectedGraspPlan(const std::string& arm, MGI& group,bool reselect) {
    report("requesting grasp candidate for "+arm);
    auto client=create_client<moveit_msgs::srv::GetMotionPlan>("/dual_arm/planner_"+arm+"/candidate_planner/get_selected_grasp_plan");
    if(!client->wait_for_service(3s)) throw std::runtime_error("RViz candidate planner unavailable");
    auto request=std::make_shared<moveit_msgs::srv::GetMotionPlan::Request>();
    if(reselect) request->motion_plan_request.planner_id="cable_grasp_reselect";
    auto future=client->async_send_request(request);
    const double timeout=get_parameter("candidate_plan_timeout").as_double();
    if(!std::isfinite(timeout) || timeout<=0.) throw std::runtime_error("Invalid candidate_plan_timeout");
    if(future.wait_for(std::chrono::duration<double>(timeout))!=std::future_status::ready)
      throw std::runtime_error("Candidate planning timed out; no motion sent");
    auto response=future.get();
    if(response && response->motion_plan_response.error_code.val!=moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      auto error_client=create_client<Trigger>("/dual_arm/planner_"+arm+"/candidate_planner/get_last_plan_error");
      if(error_client->wait_for_service(1s)) {
        auto error_future=error_client->async_send_request(std::make_shared<Trigger::Request>());
        if(error_future.wait_for(2s)==std::future_status::ready) {
          const auto error=error_future.get();
          if(error && !error->message.empty()) throw std::runtime_error("GRASP_PLAN_REJECTED: "+error->message);
        }
      }
    }
    const auto result=cable_grasp::takeSelectedGraspPlan(std::move(response),group.getName());
    MGI::Plan plan; plan.start_state_=result.trajectory_start; plan.trajectory_=result.trajectory;
    return plan;
  }

robot_trajectory::RobotTrajectory DualMotion::checkPlan(MGI& group,const MGI::Plan& plan,bool assume_open) {
    report("validating trajectory against live arms and collision scene");
    freshState();
    const auto* jmg=group.getRobotModel()->getJointModelGroup(group.getName());
    cable_grasp::requireCandidateJointNames(plan.trajectory_,jmg->getVariableNames());
    auto request=std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    request->components.components=1023;
    auto future=scene_client_->async_send_request(request);
    if(future.wait_for(3s)!=std::future_status::ready) throw std::runtime_error("Scene unavailable");
    planning_scene::PlanningScene scene(group.getRobotModel());
    if(!scene.setPlanningSceneMsg(future.get()->scene)) throw std::runtime_error("Invalid scene response");
    if(!scene.getWorld()->hasObject("cable_platform")) throw std::runtime_error("Table missing from scene");
    if(get_parameter("camera_obstacle_enabled").as_bool() && !scene.getWorld()->hasObject("realsense_camera"))
      throw std::runtime_error("RealSense camera obstacle missing from scene");
    auto measured=group.getCurrentState(3.);
    if(!measured) throw std::runtime_error("Missing live robot state");
    auto start=scene.getCurrentState();start.setVariablePositions(measured->getVariablePositions());start.update();
    auto planned=start;moveit::core::robotStateMsgToRobotState(plan.start_state_,planned);
    cable_grasp::requireUnchangedArms(planned,start,{"a_fr3_arm","fr3_arm"},0.02);
    if(assume_open) {
      const std::string prefix=group.getName()=="a_fr3_arm" ? "a_" : "";
      start.setVariablePosition(prefix+"fr3_finger_joint1",.04);
      start.setVariablePosition(prefix+"fr3_finger_joint2",.04);start.update();
    }
    robot_trajectory::RobotTrajectory trajectory(group.getRobotModel(),group.getName());
    trajectory.setRobotTrajectoryMsg(start,plan.trajectory_);
    for(const auto& joint:jmg->getVariableNames())
      if(std::abs(trajectory.getFirstWayPoint().getVariablePosition(joint)-start.getVariablePosition(joint))>0.02)
        throw std::runtime_error("Candidate trajectory start differs from current state");
    const auto constraints=group.getPathConstraints();
    const bool fix=constraints.name=="fix_projection_terminal_guard";
    const auto sampled=cable_grasp::validateTrajectory(trajectory,scene,fix ? .01 : .03,constraints);
    if(fix) {
      const bool left=group.getName()=="a_fr3_arm";
      const std::string frame=left ? "a_fr3_link0" : "fr3_link0",tcp=left ? "a_fr3_hand_tcp" : "fr3_hand_tcp";
      const double limit=get_parameter("fix_tcp_max_total_rotation").as_double();
      const auto data=cable_grasp::fixTCPVectorRotation(sampled,frame,tcp,limit);
      report("FIX terminal signed TCP-Y XZ accumulated angle="+std::to_string(data.accumulated.y()*180./M_PI)+
             " deg; diagnostic peak="+std::to_string(data.peak*180./M_PI)+" deg (not limited)");
      if(std::abs(data.accumulated.y())>limit+1e-6) {
        moveit_msgs::msg::DisplayTrajectory preview;preview.model_id=group.getRobotModel()->getName();
        moveit::core::robotStateToRobotStateMsg(sampled.getFirstWayPoint(),preview.trajectory_start);
        moveit_msgs::msg::RobotTrajectory message;sampled.getRobotTrajectoryMsg(message);preview.trajectory.push_back(message);
        fix_preview_pub_.at(left ? "a" : "b")->publish(preview);
        report("FIX rejected path preview published; NOT executable");
      }
      cable_grasp::requireFixTCPRotationTotal(sampled,frame,tcp,limit);
    }
    return fix ? sampled : trajectory;
  }

void DualMotion::execute(MGI& group,const MGI::Plan* cached,bool verify_arrival) {
    MGI::Plan plan;
    if(cached) plan=*cached;
    else {
      report("planning joint-space motion");
      if(group.plan(plan)!=moveit::core::MoveItErrorCode::SUCCESS) throw std::runtime_error("Planning failed");
    }
    const auto trajectory=checkPlan(group,plan); // Pre-execution state, bounds and collision checks always apply.
    report("executing trajectory for "+group.getName());
    physical_command_sent_=true;
    if(group.execute(plan)!=moveit::core::MoveItErrorCode::SUCCESS) throw std::runtime_error("Execution failed; no automatic release");
    if(!verify_arrival) {
      report("trajectory execution succeeded; no extra joint-arrival gate requested");return;
    }
    waitForArrival(group,trajectory,now().nanoseconds());
  }

void DualMotion::waitForArrival(MGI& group,const robot_trajectory::RobotTrajectory& trajectory,std::int64_t finished) {
    const double tolerance=get_parameter("arrival_joint_tolerance").as_double();
    const double timeout=get_parameter("arrival_timeout").as_double();
    if(!std::isfinite(timeout) || timeout<=0. || !std::isfinite(tolerance) || tolerance<=0.)
      throw std::runtime_error("Invalid arrival-check parameters");
    cable_grasp::ArrivalSettling settling(get_parameter("arrival_stable_duration").as_double());
    const bool is_a=group.getName()=="a_fr3_arm";
    std::map<std::string,double> joint_tolerances;
    const auto configured=get_parameter(is_a ? "arrival_joint_tolerances_a" : "arrival_joint_tolerances_b").as_double_array();
    if(!configured.empty()) {
      if(configured.size()!=7) throw std::runtime_error("Expected seven joint arrival tolerances");
      for(size_t i=0;i<7;++i) {
        if(!std::isfinite(configured[i]) || configured[i]<=0.) throw std::runtime_error("Invalid per-joint arrival tolerance");
        joint_tolerances[(is_a ? "a_" : "")+std::string("fr3_joint")+std::to_string(i+1)]=configured[i];
      }
    }
    const auto& names=trajectory.getGroup()->getVariableNames();
    std::vector<double> target;
    for(const auto& joint:names) target.push_back(trajectory.getLastWayPoint().getVariablePosition(joint));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(timeout);
    cable_grasp::ArrivalReport latest;
    report("waiting for fresh settled raw joint positions for HOME/FIX");
    while(rclcpp::ok() && std::chrono::steady_clock::now()<deadline) {
      std::map<std::string,std::int64_t> stamps;
      std::map<std::string,double> positions;
      {std::lock_guard<std::mutex> lock(mutex_);
        positions=joint_positions_;
        for(const auto& entry:joint_times_) stamps[entry.first]=entry.second.nanoseconds();
      }
      latest=cable_grasp::checkArrival(names,target,positions,stamps,finished,now().nanoseconds(),tolerance,joint_tolerances);
      if(settling.update(latest)) {
        report("arrival confirmed; max joint error="+std::to_string(latest.worst_error)+" rad");return;
      }
      std::this_thread::sleep_for(20ms);
    }
    throw std::runtime_error("Arrival check failed after "+std::to_string(timeout)+"s; fresh="+
        std::to_string(latest.fresh)+" within_tolerance="+std::to_string(latest.within)+
        " max_error="+std::to_string(latest.worst_error)+" tolerance="+std::to_string(tolerance)+
        " rad; "+latest.details);
  }

void DualMotion::recordBGrasp(const std::string& arm,MGI& group) {
    if(arm!="b") return;
    geometry_msgs::msg::PoseStamped message;
    try {
      moveit::core::RobotState state(group.getRobotModel());state.setToDefaultValues();
      {std::lock_guard<std::mutex> lock(mutex_);
        for(const auto& name:state.getJointModelGroup(group.getName())->getVariableNames()) {
          if(!joint_positions_.count(name) || !joint_times_.count(name) || (now()-joint_times_.at(name)).seconds()>1.)
            throw std::runtime_error("B measured joints unavailable at grasp success");
          state.setVariablePosition(name,joint_positions_.at(name));
        }
      }
      state.update();const Eigen::Isometry3d tcp=state.getGlobalLinkTransform("fr3_link0").inverse()*state.getGlobalLinkTransform("fr3_hand_tcp");
      const Eigen::Quaterniond q(tcp.linear());message.header.frame_id="fr3_link0";message.header.stamp=now();
      message.pose.position.x=tcp.translation().x();message.pose.position.y=tcp.translation().y();message.pose.position.z=tcp.translation().z();
      message.pose.orientation.x=q.x();message.pose.orientation.y=q.y();message.pose.orientation.z=q.z();message.pose.orientation.w=q.w();
      report("recorded B grasp-success TCP in fr3_link0");
    } catch(const std::exception& error) {RCLCPP_WARN(get_logger(),"A automatic reference unavailable: %s",error.what());}
    b_grasp_pub_->publish(message); // Empty frame invalidates any previous reference on failure.
  }

void DualMotion::run(const std::string& arm,const std::string& op) {
    const auto other=arm=="a" ? "b" : "a";
    if(fault_ && !recovering_home_) throw std::runtime_error("Previous physical failure; manually HOME the failed arm before resuming");
    auto& group=arm=="a" ? *a_ : *b_;
    if(op=="verify") {executeVerifySkill(arm);return;}
    if(op=="release" && !holding_[other]) throw std::runtime_error("Other arm has no confirmed grasp");
    if((op=="home" || op=="grasp") && holding_[arm]) throw std::runtime_error("Arm already holds cable");
    if(op=="fix" && (!holding_[arm] || holding_[other] || !home_[other])) throw std::runtime_error("Fix requires this arm holding and other arm released at home");
    try {
      if(op=="home") executeHomeSkill(arm,group);
      else if(op=="fix") executeFixSkill(arm,group);
      else if(op=="grasp") executeGraspSkill(arm,group);
      else if(op=="release" || op=="open_for_home") executeReleaseSkill(arm,op);
      else throw std::runtime_error("Unknown skill: "+op);
    } catch(const cable_grasp::GraspRecovered&) {throw;}
    catch(...) {fault_=fault_ || physical_command_sent_;throw;}
  }
