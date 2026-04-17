#!/usr/bin/env python3

# Copyright 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch manual hand-guiding setup for calibration pose collection.

This launch file starts the standard Franka bringup stack, publishes TF through
robot_state_publisher, and spawns the gravity compensation controller so the
robot can be hand-guided while remaining in Program / external control mode.
"""

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
    load_gripper = LaunchConfiguration('load_gripper')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')
    joint_state_rate = LaunchConfiguration('joint_state_rate')
    use_rviz = LaunchConfiguration('use_rviz')
    controllers_yaml = LaunchConfiguration('controllers_yaml')
    controller_name = LaunchConfiguration('controller_name')
    spawn_manual_controller = LaunchConfiguration('spawn_manual_controller')

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py']
            )
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
    )

    manual_controller = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=[
            controller_name,
            '--controller-manager-timeout',
            '30',
        ],
        parameters=[controllers_yaml],
        output='screen',
        condition=IfCondition(spawn_manual_controller),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '--display-config',
            PathJoinSubstitution(
                [
                    FindPackageShare('franka_description'),
                    'rviz',
                    'visualize_franka.rviz',
                ]
            ),
        ],
        condition=IfCondition(use_rviz),
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'robot_type',
                default_value='fr3',
                description='Robot type, for example fr3.',
            ),
            DeclareLaunchArgument(
                'arm_prefix',
                default_value='',
                description='Optional arm prefix.',
            ),
            DeclareLaunchArgument(
                'namespace',
                default_value='',
                description='Namespace for the robot nodes.',
            ),
            DeclareLaunchArgument(
                'robot_ip',
                description='Hostname or IP address of the robot.',
            ),
            DeclareLaunchArgument(
                'load_gripper',
                default_value='false',
                description='Whether to launch the Franka gripper.',
            ),
            DeclareLaunchArgument(
                'use_fake_hardware',
                default_value='false',
                description='Use fake hardware instead of the robot.',
            ),
            DeclareLaunchArgument(
                'fake_sensor_commands',
                default_value='false',
                description='Fake command interfaces when using fake hardware.',
            ),
            DeclareLaunchArgument(
                'joint_state_rate',
                default_value='30',
                description='Aggregated joint state publication rate in Hz.',
            ),
            DeclareLaunchArgument(
                'use_rviz',
                default_value='false',
                description='Launch RViz for monitoring TF and robot state.',
            ),
            DeclareLaunchArgument(
                'controller_name',
                default_value='gravity_compensation_example_controller',
                description='Controller used during manual calibration.',
            ),
            DeclareLaunchArgument(
                'spawn_manual_controller',
                default_value='true',
                description='Spawn the manual guiding controller on startup.',
            ),
            DeclareLaunchArgument(
                'controllers_yaml',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('franka_bringup'), 'config', 'controllers.yaml']
                ),
                description='Controller configuration file.',
            ),
            bringup,
            manual_controller,
            rviz,
        ]
    )
