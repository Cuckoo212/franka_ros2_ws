"""Combined planning scene; two isolated drivers, explicit physical execution gate."""
import os
import sys
import yaml
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    p = lambda key: LaunchConfiguration(key).perform(context)
    execution_enabled = p('execution_enabled').lower() == 'true'
    share = get_package_share_directory('cable_moveit_grasp')
    sys.path.insert(0, os.path.join(get_package_prefix('cable_moveit_grasp'), 'lib', 'cable_moveit_grasp'))
    from demo_launch_common import acquire_robot_leases
    if p('start_drivers').lower() == 'true' or execution_enabled:
        acquire_robot_leases([p('left_robot_ip'), p('right_robot_ip')])
    from dual_arm_model import build_model
    model = build_model()
    with open(os.path.join(share, 'config/candidate_planner.yaml'), encoding='utf-8') as stream:
        grasp_parameters = yaml.safe_load(stream)['/**']['ros__parameters']
    with open(p('motion_config'), encoding='utf-8') as stream:
        motion_parameters = dict(grasp_parameters)
        motion_parameters.update(yaml.safe_load(stream)['/**']['ros__parameters'])
    if motion_parameters.get('camera_obstacle_enabled', True):
        from calibration_math import camera_obstacle_pose
        motion_parameters['camera_obstacle_pose'] = camera_obstacle_pose(
            p('calibration_file'), motion_parameters.get('camera_obstacle_center_optical', [0., 0., 0.]))
    config = get_package_share_directory('franka_fr3_moveit_config')
    def load(file):
        with open(os.path.join(config, 'config', file), encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    kin = load('kinematics.yaml')
    kin['a_fr3_arm'] = dict(kin['fr3_arm'])
    model['robot_description_kinematics'] = kin
    ompl = load('ompl_planning.yaml')
    ompl['a_fr3_arm'] = dict(ompl.get('fr3_arm', {}))
    ompl.update(planning_plugin='ompl_interface/OMPLPlanner',
                request_adapters='default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
                start_state_max_bounds_error=0.1)
    controllers = {'controller_names': []}
    for ns, prefix in [('NS_1', ''), ('NS_2', 'a_')]:
        name = '/' + ns + '/fr3_arm_controller'
        controllers['controller_names'].append(name)
        controllers[name] = dict(action_ns='follow_joint_trajectory', type='FollowJointTrajectory',
                                 default=True, joints=[prefix + 'fr3_joint' + str(i) for i in range(1, 8)])
    nodes = [LogInfo(msg=('Dual-arm execution enabled: command controllers require EXEC/FCI readiness.'
                         if execution_enabled else
                         'Dual-arm READ ONLY: state acquisition and planning; no arm command controller activation.'))]
    if p('start_cloud_calibration').lower() == 'true':
        nodes.append(Node(package='cable_moveit_grasp', executable='cloud_calibration_node',
                          name='right_arm_cloud_calibration', output='screen', parameters=[{
                              'calibration_file': p('calibration_file'),
                              'input_topic': '/cable_camera/cable_points',
                              'output_topic': '/cable_camera/cable_points_base',
                              'input_frame': 'cable_camera_color_optical_frame',
                              'input_is_optical': True, 'expected_base_frame': 'fr3_link0',
                              'publish_calibration_tf': True,
                          }]))
    for ns, prefix, ip, file in [('NS_1', '', p('right_robot_ip'), 'arm_b_controllers.yaml'),
                                  ('NS_2', 'a', p('left_robot_ip'), 'arm_a_controllers.yaml')]:
        if p('start_drivers').lower() == 'true':
            nodes.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('franka_bringup'), 'launch', 'franka.launch.py')),
                launch_arguments=dict(namespace=ns, robot_type='fr3', robot_ip=ip, arm_prefix=prefix,
                                      load_gripper='true', start_gripper_server='false', use_fake_hardware=p('use_fake_hardware'),
                                      publish_robot_state='false', publish_joint_states='false', spawn_state_broadcasters='false', controllers_yaml=os.path.join(share, 'config', file)).items()))
            if p('start_grippers').lower() == 'true':
                # Same explicit server launch as the known-good standalone command. Scoped per arm.
                nodes.append(LogInfo(msg='Starting standalone gripper server: namespace=/' + ns + ', robot_ip=' + ip + ', arm_prefix=' + prefix))
                nodes.append(GroupAction(actions=[IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
                    get_package_share_directory('franka_gripper'), 'launch', 'gripper.launch.py')),
                    launch_arguments=dict(namespace='/' + ns, arm_prefix=prefix, robot_type='fr3', robot_ip=ip,
                                          use_fake_hardware=p('use_fake_hardware')).items())], scoped=True))
            nodes.append(Node(package='cable_moveit_grasp', executable='prepare_dual_controllers',
                              name='prepare_controllers', namespace=ns, output='screen', parameters=[{
                                  'controller_manager': '/'+ns+'/controller_manager',
                                  'use_fake_hardware': p('use_fake_hardware').lower() == 'true',
                                  'execution_enabled': execution_enabled,
                              }]))
    nodes += [
        Node(package='cable_moveit_grasp', executable='measured_joint_state_mux', name='dual_joint_states',
             output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher', name='dual_robot_state',
             parameters=[{'robot_description': model['robot_description']}],
             remappings=[('joint_states', '/dual_arm/joint_states')]),
        Node(package='moveit_ros_move_group', executable='move_group', namespace='dual_arm', output='screen',
             parameters=[model, {'planning_pipelines': ['ompl'], 'default_planning_pipeline': 'ompl', 'ompl': ompl,
                                'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
                                'moveit_simple_controller_manager': controllers,
                                'moveit_manage_controllers': False,
                                'allow_trajectory_execution': execution_enabled,
                                'publish_robot_description_semantic': True, 'publish_planning_scene': True,
                                'publish_geometry_updates': True, 'publish_state_updates': True,
                                'publish_transforms_updates': True}],
             remappings=[('joint_states', '/dual_arm/joint_states')]),
        Node(package='cable_moveit_grasp', executable='dual_arm_motion_node', namespace='dual_arm',
             parameters=[model, motion_parameters,
                         {'execution_enabled': execution_enabled}],
             remappings=[('joint_states', '/dual_arm/joint_states')], output='screen'),
        Node(package='cable_moveit_grasp', executable='arm_base_cloud_node',
             parameters=[os.path.join(share, 'config/arm_a_cloud.yaml'), {
                 'publish_base_tf': False, 'input_is_camera': True,
                 'input_topic': '/cable_camera/cable_points',
                 'calibration_file': p('calibration_file'),
             }]),
    ]
    for arm, prefix, group in [('a', 'a_', 'a_fr3_arm'), ('b', '', 'fr3_arm')]:
        nodes.append(Node(package='cable_moveit_grasp', executable='grasp_candidate_planner',
                          name='candidate_planner', namespace='dual_arm/planner_' + arm, output='screen',
                          parameters=[model, {"ompl": ompl}, {**grasp_parameters,
                              'candidate_arm': arm,
                              'a_from_b_enabled': arm == 'a' and motion_parameters.get('grasp_a_from_b_enabled', True),
                              'a_from_b_use_current_tcp': motion_parameters.get('grasp_a_use_current_b_tcp', True),
                              'a_from_b_x_offset': motion_parameters.get('grasp_a_from_b_x_offset', 0.10),
                              'a_from_b_x_tolerance': motion_parameters.get('grasp_a_from_b_x_tolerance', 0.01),
                              'a_recovery_from_b_x_offset': motion_parameters.get('grasp_a_recovery_x_offset', 0.15),
                              'b_recovery_x_offset': motion_parameters.get('grasp_b_recovery_x_offset', 0.05),
                              'recovery_x_tolerance': motion_parameters.get('grasp_recovery_x_tolerance', 0.01),
                              'b_auto_enabled': arm == 'b' and motion_parameters.get('grasp_b_auto_enabled', True),
                              'b_search_x_min': motion_parameters.get('grasp_b_search_x_min', -0.30),
                              'b_min_board_clearance': motion_parameters.get('grasp_b_min_board_clearance', 0.03),
                              'b_search_a_tcp_clearance': motion_parameters.get('grasp_b_search_a_tcp_clearance', 0.10),
                              'a_search_x_length': motion_parameters.get('grasp_a_search_x_length', 0.30),
                              'quality_min_length': motion_parameters.get('grasp_quality_min_length', 0.05),
                              'quality_max_gap': motion_parameters.get('grasp_quality_max_gap', 0.006),
                              'quality_voxel_size': motion_parameters.get('grasp_quality_voxel_size', 0.002),
                              'roll_step_degrees': 30.0, 'candidate_worker_threads': 10,
                              'planning_group': group, 'move_group_namespace': '/dual_arm',
                              # Task geometry is expressed in the shared B/table frame, even for arm A.
                              'base_frame': 'fr3_link0', 'tcp_link': prefix + 'fr3_hand_tcp',
                              'end_effector_link': prefix + 'fr3_hand_tcp',
                              'cloud_topic': '/cable_camera/cable_points_base',
                              'clicked_point_topic': '/dual_arm/planner_' + arm + '/clicked_point',
                              'gripper_joint_names': [prefix + 'fr3_finger_joint1', prefix + 'fr3_finger_joint2'],
                              'candidate_plan_max_age': 60.0,
                              'velocity_scaling': 0.05, 'acceleration_scaling': 0.05,
                          }], remappings=[('joint_states', '/dual_arm/joint_states')]))
    nodes.append(Node(package='cable_moveit_grasp', executable='dual_arm_grasp_selector',
                      namespace='dual_arm', output='screen'))
    if p('start_coordinator').lower() == 'true':
        nodes.append(Node(package='cable_moveit_grasp', executable='cable_moveit_alternate_grasp',
                          name='cable_moveit_alternate_grasp', output='screen',
                          parameters=[os.path.join(share, 'config/alternate_grasp.yaml'),
                                      {'physical_backend': p('execution_enabled').lower() == 'true',
                                       'skill_catalog_file': os.path.join(share, 'skills/catalog.json'),
                                       'task_design_file': os.path.join(share, 'skills/alternate_cable_task.json')}]))
    if p('rviz').lower() == 'true':
        nodes.append(Node(package='rviz2', executable='rviz2', namespace='dual_arm',
                          parameters=[model], arguments=['-d', os.path.join(share, 'config/dual_arm.rviz'), '-f', 'fr3_link0']))
    return nodes


def generate_launch_description():
    share = get_package_share_directory('cable_moveit_grasp')
    defaults = dict(start_coordinator='true', start_drivers='true', start_grippers='true', use_fake_hardware='true', execution_enabled='false',
                    left_robot_ip='192.168.3.102', right_robot_ip='192.168.3.101', rviz='true',
                    motion_config=os.path.join(share, 'config/dual_arm_motion.yaml'),
                    start_cloud_calibration='true',
                    calibration_file=os.path.join(share, 'config/calibration/fr3_calibration_apriltag_depth_003.calib'))
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
                              OpaqueFunction(function=setup)])
