"""Single-arm RViz grasp -> fixed-orientation +Y lift -> recorded recycling pose."""
import os
import sys
import yaml
import xacro
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    p = lambda key: LaunchConfiguration(key).perform(context)
    flag = lambda key: p(key).lower() == 'true'
    share = get_package_share_directory('cable_moveit_grasp')
    sys.path.insert(0, os.path.join(get_package_prefix('cable_moveit_grasp'), 'lib/cable_moveit_grasp'))
    from calibration_math import camera_obstacle_pose
    from demo_launch_common import acquire_robot_leases
    task = p('task_namespace').strip('/')
    hardware = p('robot_namespace').strip('/')
    if not task or not hardware or task == hardware:
        raise ValueError('Use distinct nonempty task_namespace and robot_namespace')
    include_a = flag('include_arm_a_collision')
    if hardware == 'NS_2' and include_a:
        raise ValueError('B hardware namespace cannot equal the optional A namespace /NS_2')
    if flag('start_drivers') or flag('execution_enabled'):
        acquire_robot_leases([p('robot_ip')] + ([p('arm_a_robot_ip')] if include_a else []))
    if include_a:
        from dual_arm_model import build_model
        model = build_model()
    else:
        description = get_package_share_directory('franka_description')
        mappings = dict(arm_prefix='', hand='true', ros2_control='false')
        model = {'robot_description': xacro.process_file(os.path.join(description, 'robots/fr3/fr3.urdf.xacro'), mappings=mappings).toxml(),
                 'robot_description_semantic': xacro.process_file(os.path.join(description, 'robots/fr3/fr3.srdf.xacro'), mappings=mappings).toxml()}
    config = get_package_share_directory('franka_fr3_moveit_config')
    def load(file):
        with open(os.path.join(config, 'config', file), encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    kin = load('kinematics.yaml')
    ompl = load('ompl_planning.yaml')
    if include_a:
        kin['a_fr3_arm'] = dict(kin['fr3_arm'])
        ompl['a_fr3_arm'] = dict(ompl.get('fr3_arm', {}))
    model['robot_description_kinematics'] = kin
    ompl.update(planning_plugin='ompl_interface/OMPLPlanner',
                request_adapters='default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
                start_state_max_bounds_error=0.1)
    with open(os.path.join(share, 'config/candidate_planner.yaml')) as stream:
        candidates = yaml.safe_load(stream)['/**']['ros__parameters']
    with open(p('motion_config')) as stream:
        motion = yaml.safe_load(stream)['/**']['ros__parameters']
    if motion.get('camera_obstacle_enabled', True):
        motion['camera_obstacle_pose'] = camera_obstacle_pose(p('calibration_file'), motion.get('camera_obstacle_center_optical', [0., 0., 0.]))
    controller = '/' + hardware + '/fr3_arm_controller'
    controllers = {'controller_names': [controller], controller: dict(action_ns='follow_joint_trajectory', type='FollowJointTrajectory',
                   default=True, joints=['fr3_joint' + str(i) for i in range(1, 8)])}
    joint_topic = '/' + task + '/joint_states'
    cloud_topic = '/' + task + '/cable_points_base'
    nodes = [LogInfo(msg='Single-arm recycling: B only executes trajectories; A collision model=' + str(include_a))]
    if flag('start_drivers'):
        for ns, prefix, ip, file, execution in [(hardware, '', p('robot_ip'), 'arm_b_controllers.yaml', flag('execution_enabled'))] + (
                [('NS_2', 'a', p('arm_a_robot_ip'), 'arm_a_controllers.yaml', False)] if include_a else []):
            nodes.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('franka_bringup'), 'launch/franka.launch.py')),
                launch_arguments=dict(namespace=ns, robot_type='fr3', robot_ip=ip, arm_prefix=prefix,
                    load_gripper='true', start_gripper_server='false', use_fake_hardware=p('use_fake_hardware'),
                    publish_robot_state='false', publish_joint_states='false', spawn_state_broadcasters='false',
                    controllers_yaml=os.path.join(share, 'config', file)).items()))
            if flag('start_grippers'):
                nodes.append(GroupAction(actions=[IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('franka_gripper'), 'launch/gripper.launch.py')),
                    launch_arguments=dict(namespace='/' + ns, robot_type='fr3', robot_ip=ip, arm_prefix=prefix, use_fake_hardware=p('use_fake_hardware')).items())], scoped=True))
            nodes.append(Node(package='cable_moveit_grasp', executable='prepare_dual_controllers', namespace=ns, name='prepare_controllers',
                parameters=[dict(controller_manager='/' + ns + '/controller_manager', use_fake_hardware=flag('use_fake_hardware'), execution_enabled=execution)], output='screen'))
    nodes += [
        Node(package='cable_moveit_grasp', executable='measured_joint_state_mux', namespace=task, name='measured_joint_states',
             parameters=[dict(required_arms=['a', 'b'] if include_a else ['b'], arm_a_namespace='/NS_2', arm_b_namespace='/' + hardware, output_topic=joint_topic)]),
        Node(package='robot_state_publisher', executable='robot_state_publisher', namespace=task, name='robot_state', parameters=[{'robot_description': model['robot_description']}], remappings=[('joint_states', joint_topic)]),
        Node(package='moveit_ros_move_group', executable='move_group', namespace=task, parameters=[model, {
            'planning_pipelines': ['ompl'], 'default_planning_pipeline': 'ompl', 'ompl': ompl,
            'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
            'moveit_simple_controller_manager': controllers, 'moveit_manage_controllers': False,
            'allow_trajectory_execution': flag('execution_enabled'), 'publish_robot_description_semantic': True,
            'publish_planning_scene': True, 'publish_geometry_updates': True, 'publish_state_updates': True, 'publish_transforms_updates': True}],
            remappings=[('joint_states', joint_topic)], output='screen'),
        Node(package='cable_moveit_grasp', executable='single_arm_motion_node', namespace=task, parameters=[model, motion, {
            'execution_enabled': flag('execution_enabled'), 'include_arm_a_collision': include_a,
            'move_group_namespace': '/' + task, 'robot_namespace': '/' + hardware,
            'gripper_namespace': '/' + hardware + '/franka_gripper', 'planner_service': '/' + task + '/planner/candidate_planner/get_selected_grasp_plan'}],
            remappings=[('joint_states', joint_topic)], output='screen'),
        Node(package='cable_moveit_grasp', executable='grasp_candidate_planner', namespace=task + '/planner', name='candidate_planner',
             parameters=[model, {'ompl': ompl}, {**candidates, 'candidate_arm': 'b', 'require_arm_selection': False,
                'a_from_b_enabled': False, 'b_auto_enabled': False, 'move_group_namespace': '/' + task,
                'cloud_topic': cloud_topic, 'clicked_point_topic': '/' + task + '/clicked_point',
                'roll_step_degrees': 30., 'candidate_worker_threads': 10}], remappings=[('joint_states', joint_topic)], output='screen'),
    ]
    if flag('start_cloud_calibration'):
        nodes.append(Node(package='cable_moveit_grasp', executable='cloud_calibration_node', namespace=task, name='cloud_calibration',
            parameters=[dict(calibration_file=p('calibration_file'), input_topic=p('cloud_topic'), output_topic=cloud_topic,
                input_frame='cable_camera_color_optical_frame', expected_base_frame='fr3_link0', input_is_optical=True, publish_calibration_tf=True)]))
    if flag('rviz'):
        # Remap the shipped default demo namespace, including interactive-marker updates.
        names=['clicked_point', 'monitored_planning_scene', 'planner/candidate_planner/candidates',
               'planner/candidate_planner/grasp_pose', 'planner/candidate_planner/entry_pose',
               'planner/candidate_planner/display_trajectory', 'planner/cable_grasp_frames/update',
               'planner/cable_grasp_frames/feedback', 'single_arm_motion/recycling_preview', 'cable_points_base']
        nodes.append(Node(package='rviz2', executable='rviz2', namespace=task, parameters=[model],
            arguments=['-d', os.path.join(share, 'config/single_arm.rviz'), '-f', 'fr3_link0'],
            remappings=[('/single_arm/' + name, '/' + task + '/' + name) for name in names]))
    return nodes


def generate_launch_description():
    share = get_package_share_directory('cable_moveit_grasp')
    defaults = dict(task_namespace='single_arm', robot_namespace='NS_1', robot_ip='192.168.3.101',
        arm_a_robot_ip='192.168.3.102', include_arm_a_collision='true', start_drivers='true',
        start_grippers='false', use_fake_hardware='true', execution_enabled='false', rviz='true',
        start_cloud_calibration='true', cloud_topic='/cable_camera/cable_points',
        motion_config=os.path.join(share, 'config/single_arm_motion.yaml'),
        calibration_file=os.path.join(share, 'config/calibration/fr3_calibration_apriltag_depth_003.calib'))
    return LaunchDescription([*[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()], OpaqueFunction(function=setup)])
