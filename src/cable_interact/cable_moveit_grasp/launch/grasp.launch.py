"""Application + combined RViz, optionally planning-only move_group. No hardware bringup."""
import os
import yaml
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    value = lambda key: LaunchConfiguration(key).perform(context)
    share = get_package_share_directory('cable_moveit_grasp')
    description = get_package_share_directory('franka_description')
    config = get_package_share_directory('franka_fr3_moveit_config')
    namespace = value('namespace').strip('/')
    prefix = value('arm_prefix')
    mappings = {'arm_prefix': prefix, 'hand': 'true', 'ros2_control': 'false'}
    # These must match the model used by the existing Franka bringup.
    for key in ('tcp_xyz', 'tcp_rpy', 'xyz_ee', 'rpy_ee'):
        mappings[key] = value(key)
    urdf = xacro.process_file(os.path.join(description, 'robots/fr3/fr3.urdf.xacro'),
                              mappings=mappings).toxml()
    srdf = xacro.process_file(os.path.join(description, 'robots/fr3/fr3.srdf.xacro'),
                              mappings=mappings).toxml()

    def load(name):
        with open(os.path.join(config, 'config', name), encoding='utf-8') as stream:
            text = stream.read()
        if prefix:
            text = text.replace('fr3', prefix + '_fr3')
        return yaml.safe_load(text)

    model = {'robot_description': urdf, 'robot_description_semantic': srdf,
             'robot_description_kinematics': load('kinematics.yaml')}
    root = '/' + namespace if namespace else ''
    calibrated = value('transform_cloud').lower() == 'true'
    displayed_cloud = value('base_cloud_topic') if calibrated else value('cloud_topic')
    nodes = [Node(package='cable_moveit_grasp', executable='cable_moveit_grasp_node',
                  name='cable_moveit_grasp', namespace=namespace, output='screen',
                  parameters=[model, value('config'),
                              {'move_group_namespace': root,
                               'cloud_topic': displayed_cloud}],
                  remappings=[('joint_states', value('joint_states_topic'))])]
    if calibrated:
        nodes.append(Node(package='cable_moveit_grasp', executable='cloud_calibration_node',
                          name='cloud_calibration', namespace=namespace, output='screen',
                          parameters=[{'calibration_file': value('calibration_file'),
                                       'input_topic': value('cloud_topic'),
                                       'output_topic': value('base_cloud_topic'),
                                       'input_frame': value('cloud_frame'),
                                       'input_is_optical': True,
                                       'expected_base_frame': value('fixed_frame'),
                                       'publish_calibration_tf': value('publish_calibration_tf').lower() == 'true'}]))
    if value('start_move_group').lower() == 'true':
        ompl = load('ompl_planning.yaml')
        ompl.update({'planning_plugin': 'ompl_interface/OMPLPlanner',
                     'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                         'default_planner_request_adapters/ResolveConstraintFrames '
                                         'default_planner_request_adapters/FixWorkspaceBounds '
                                         'default_planner_request_adapters/FixStartStateBounds '
                                         'default_planner_request_adapters/FixStartStateCollision '
                                         'default_planner_request_adapters/FixStartStatePathConstraints',
                     'start_state_max_bounds_error': 0.1})
        nodes.append(Node(package='moveit_ros_move_group', executable='move_group',
                          namespace=namespace, output='screen', parameters=[model, {
                              'planning_pipelines': ['ompl'], 'default_planning_pipeline': 'ompl',
                              'ompl': ompl,
                              'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
                              'moveit_simple_controller_manager': load('fr3_controllers.yaml'),
                              'moveit_manage_controllers': False,
                              'publish_robot_description': True,
                              'publish_robot_description_semantic': True,
                              'publish_planning_scene': True,
                              'publish_geometry_updates': True,
                              'publish_state_updates': True,
                              'publish_transforms_updates': True,
                              'trajectory_execution.allowed_start_tolerance': 0.01,
                          }], remappings=[('joint_states', value('joint_states_topic'))]))
    if value('rviz').lower() == 'true':
        nodes.append(Node(package='rviz2', executable='rviz2', name='cable_grasp_rviz',
                          namespace=namespace, output='screen',
                          parameters=[model],
                          arguments=['-d', os.path.join(share, 'config/grasp.rviz'),
                                     '-f', value('fixed_frame')],
                          remappings=[('/cable_camera/cable_points', displayed_cloud),
                                      ('/NS_1/cable_moveit_grasp/candidates', root+'/cable_moveit_grasp/candidates'),
                                      ('/NS_1/cable_moveit_grasp/grasp_pose', root+'/cable_moveit_grasp/grasp_pose'),
                                      ('/NS_1/cable_moveit_grasp/entry_pose', root+'/cable_moveit_grasp/entry_pose'),
                                      ('/NS_1/cable_moveit_grasp/display_trajectory', root+'/cable_moveit_grasp/display_trajectory'),
                                      ('/NS_1/cable_grasp_frames/update', root+'/cable_grasp_frames/update'),
                                      ('/NS_1/cable_grasp_frames/feedback', root+'/cable_grasp_frames/feedback'),
                                      ('/NS_1/monitored_planning_scene', root+'/monitored_planning_scene')]))
    return nodes


def generate_launch_description():
    share = get_package_share_directory('cable_moveit_grasp')
    defaults = {
        'config': os.path.join(share, 'config/grasp.yaml'), 'namespace': 'NS_1',
        'arm_prefix': '', 'joint_states_topic': '/NS_1/fr3_joint_states',
        'cloud_topic': '/cable_camera/cable_points', 'fixed_frame': 'fr3_link0',
        'transform_cloud': 'true', 'base_cloud_topic': '/cable_camera/cable_points_base',
        'cloud_frame': 'cable_camera_color_optical_frame', 'publish_calibration_tf': 'true',
        'calibration_file': os.path.expanduser('~/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint/samples_calibration/hand_eye_calibration/fr3_calibration_apriltag_depth_003.calib'),
        'start_move_group': 'false', 'rviz': 'true',
        'tcp_xyz': '0 0 0.1034', 'tcp_rpy': '0 0 0',
        'xyz_ee': '0 0 0', 'rpy_ee': '0 0 -0.7853981633974483',
    }
    return LaunchDescription([DeclareLaunchArgument(key, default_value=val)
                              for key, val in defaults.items()] + [OpaqueFunction(function=setup)])
