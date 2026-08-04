#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    arm_prefix = LaunchConfiguration('arm_prefix')
    namespace = LaunchConfiguration('namespace')
    robot_ip = LaunchConfiguration('robot_ip')
    launch_robot = LaunchConfiguration('launch_robot')
    load_gripper = LaunchConfiguration('load_gripper')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')
    joint_state_rate = LaunchConfiguration('joint_state_rate')
    controllers_yaml = LaunchConfiguration('controllers_yaml')
    controller_name = LaunchConfiguration('controller_name')
    spawn_controller = LaunchConfiguration('spawn_controller')
    use_rviz = LaunchConfiguration('use_rviz')
    launch_realsense = LaunchConfiguration('launch_realsense')
    launch_apriltag_detector = LaunchConfiguration('launch_apriltag_detector')
    launch_pose_publisher = LaunchConfiguration('launch_pose_publisher')
    calibration_path = LaunchConfiguration('calibration_path')

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])
        ),
        launch_arguments={
            'robot_type': robot_type,
            'arm_prefix': arm_prefix,
            'namespace': namespace,
            'robot_ip': robot_ip,
            'load_gripper': load_gripper,
            'use_fake_hardware': use_fake_hardware,
            'fake_sensor_commands': fake_sensor_commands,
            'joint_state_rate': joint_state_rate,
            'controllers_yaml': controllers_yaml,
        }.items(),
        condition=IfCondition(launch_robot),
    )

    apriltag_controller = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=[controller_name, '--controller-manager-timeout', '30'],
        parameters=[controllers_yaml],
        output='screen',
        condition=IfCondition(spawn_controller),
    )

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])
        ),
        launch_arguments={
            'camera_namespace': 'camera',
            'camera_name': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
        }.items(),
        condition=IfCondition(launch_realsense),
    )

    apriltag_detector = Node(
        package='franka_handeye_calibration_ros2',
        executable='calibration_apriltag_publisher',
        name='calibration_apriltag_publisher',
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare('franka_handeye_calibration_ros2'), 'config', 'apriltag_parameters.yaml']
            )
        ],
        output='screen',
        condition=IfCondition(launch_apriltag_detector),
    )

    robot_pose_publisher = Node(
        package='franka_handeye_calibration_ros2',
        executable='apriltag_robot_pose_publisher',
        name='apriltag_robot_pose_publisher',
        parameters=[
            {
                'calibration_path': calibration_path,
                'camera_tag_frame': 'calibration_apriltag',
                'robot_tag_frame': 'apriltag_robot_frame',
                'gripper_target_frame': 'apriltag_gripper_target_frame',
                'publish_gripper_target_frame': True,
                'hover_frame': 'apriltag_hover_frame',
                'publish_hover_frame': True,
                'hover_offset': [0.0, 0.0, -0.15],
                'target_pose_topic': '/apriltag_target_pose',
                'camera_link_frame': 'camera_link',
                'publish_camera_link_frame': True,
                'publish_rate': 30.0,
            }
        ],
        output='screen',
        condition=IfCondition(launch_pose_publisher),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '--display-config',
            PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'rviz', 'apriltag_position_controller.rviz']
            ),
        ],
        condition=IfCondition(use_rviz),
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('robot_type', default_value='fr3'),
            DeclareLaunchArgument('arm_prefix', default_value=''),
            DeclareLaunchArgument('namespace', default_value=''),
            DeclareLaunchArgument('robot_ip', default_value='192.168.3.102', description='Hostname or IP address of the robot.'),
            DeclareLaunchArgument('launch_robot', default_value='true'),
            DeclareLaunchArgument('load_gripper', default_value='true'),
            DeclareLaunchArgument('use_fake_hardware', default_value='false'),
            DeclareLaunchArgument('fake_sensor_commands', default_value='false'),
            DeclareLaunchArgument('joint_state_rate', default_value='30'),
            DeclareLaunchArgument('controller_name', default_value='cartesian_apriltag_position_controller'),
            DeclareLaunchArgument('spawn_controller', default_value='false'),
            DeclareLaunchArgument('use_rviz', default_value='true'),
            DeclareLaunchArgument('launch_realsense', default_value='true'),
            DeclareLaunchArgument('launch_apriltag_detector', default_value='true'),
            DeclareLaunchArgument('launch_pose_publisher', default_value='true'),
            DeclareLaunchArgument(
                'calibration_path',
                #default_value='/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration_apriltag.calib',
                #default_value='/home/flexcycle/franka_ros2_ws/src/cable_interact/pointcloud_tools/info_for_3Dpoint/samples_calibration/calibration_001/fr3_calibration_integrated.calib',
            ),
            DeclareLaunchArgument(
                'controllers_yaml',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('franka_bringup'), 'config', 'controllers.yaml']
                ),
            ),
            bringup,
            apriltag_controller,
            realsense,
            apriltag_detector,
            robot_pose_publisher,
            rviz,
        ]
    )
