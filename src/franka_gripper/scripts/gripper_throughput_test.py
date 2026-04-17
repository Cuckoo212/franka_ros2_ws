#!/usr/bin/env python3
# Copyright (c) 2026
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

import argparse
import sys
import time

from franka_msgs.action import Grasp
from franka_msgs.action import Move
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState


class GripperThroughputTest(Node):

    def __init__(self, args: argparse.Namespace):
        super().__init__('gripper_throughput_test')
        self._action_name = args.action_name
        self._duration = args.duration
        self._mode = args.mode
        self._result_timeout = args.result_timeout
        self._vary_goal = args.vary_goal
        self._width_min = args.width_min
        self._width_max = args.width_max
        self._width_step = args.width_step
        self._joint_states_topic = args.joint_states_topic
        self._state_timeout = args.state_timeout
        self._current_width = None
        self._direction = -1 if args.decrement_first else 1
        self._last_feedback_width = None
        self._joint_state_subscription = None

        if self._mode == 'grasp':
            self._goal = Grasp.Goal()
            self._goal.width = 0.0
            self._goal.speed = args.speed
            self._goal.force = args.force
            self._goal.epsilon.inner = args.epsilon_inner
            self._goal.epsilon.outer = args.epsilon_outer
            self._client = ActionClient(self, Grasp, self._action_name)
        else:
            self._goal = Move.Goal()
            self._goal.width = 0.0
            self._goal.speed = args.speed
            self._client = ActionClient(self, Move, self._action_name)

    def _joint_state_callback(self, msg: JointState) -> None:
        if len(msg.position) >= 2:
            self._current_width = msg.position[0] + msg.position[1]

    def _feedback_callback(self, feedback_msg) -> None:
        self._last_feedback_width = feedback_msg.feedback.current_width

    def initialize_current_width(self) -> None:
        self._joint_state_subscription = self.create_subscription(
            JointState,
            self._joint_states_topic,
            self._joint_state_callback,
            10,
        )

        deadline = time.monotonic() + self._state_timeout
        while self._current_width is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.destroy_subscription(self._joint_state_subscription)
        self._joint_state_subscription = None

        if self._current_width is None:
            raise RuntimeError(
                f'Failed to read current gripper width from {self._joint_states_topic} '
                f'within {self._state_timeout:.1f}s'
            )

        self._current_width = max(self._width_min, min(self._current_width, self._width_max))
        self.get_logger().info(
            f'Initialized current width from joint states: {self._current_width:.4f} m'
        )

    def _compute_next_width(self) -> float:
        if self._current_width is None:
            raise RuntimeError('Current width has not been initialized yet.')

        next_width = self._current_width + self._direction * self._width_step

        if next_width >= self._width_max:
            next_width = self._width_max
            self._direction = -1
        elif next_width <= self._width_min:
            next_width = self._width_min
            self._direction = 1

        return next_width

    def update_goal_width(self) -> None:
        if not self._vary_goal:
            if self._current_width is None:
                raise RuntimeError('Current width has not been initialized yet.')
            self._goal.width = self._current_width
            return

        self._goal.width = self._compute_next_width()

    def send_once(self) -> bool:
        self.update_goal_width()
        self._last_feedback_width = None
        send_goal_future = self._client.send_goal_async(
            self._goal, feedback_callback=self._feedback_callback
        )
        rclpy.spin_until_future_complete(
            self, send_goal_future, timeout_sec=self._result_timeout
        )

        if not send_goal_future.done():
            self.get_logger().error('Timed out while sending grasp goal.')
            return False

        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                f'{self._mode.capitalize()} goal was rejected by the action server.'
            )
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=self._result_timeout
        )

        if not result_future.done():
            self.get_logger().error('Timed out while waiting for grasp result.')
            return False

        result = result_future.result()
        if result is None:
            self.get_logger().error('No result returned by the action server.')
            return False

        if result.status != 4:
            self.get_logger().error(
                f'{self._mode.capitalize()} finished with non-success status {result.status}.'
            )
            return False

        if not result.result.success:
            self.get_logger().warning(
                f'{self._mode.capitalize()} action completed but reported success=false: '
                f'{result.result.error}'
            )
            return False

        if self._last_feedback_width is not None:
            self._current_width = self._last_feedback_width
        else:
            self._current_width = self._goal.width

        return True

    def run(self) -> int:
        self.get_logger().info(f'Waiting for action server {self._action_name} ...')
        self._client.wait_for_server()
        self.initialize_current_width()

        success_count = 0
        total_count = 0
        start_time = time.monotonic()

        while time.monotonic() - start_time < self._duration:
            total_count += 1
            self.get_logger().info(f'current time: {time.monotonic() - start_time:.2f} s ')
            if self.send_once():
                success_count += 1

        elapsed = time.monotonic() - start_time
        throughput_hz = success_count / elapsed if elapsed > 0.0 else 0.0

        self.get_logger().info(
            f'Throughput: {throughput_hz:.2f} Hz '
            f'({success_count} successful commands out of {total_count} in {elapsed:.2f}s)'
        )
        return 0 if total_count > 0 else 1


def parse_args(cli_args):
    parser = argparse.ArgumentParser(
        description='Measure Franka gripper throughput by sending sequential actions.'
    )
    parser.add_argument(
        '--mode',
        choices=['grasp', 'move'],
        default='grasp',
        help='Action type to measure.',
    )
    parser.add_argument(
        '--action-name',
        default='/franka_gripper/grasp',
        help='Fully qualified action name, e.g. /franka_gripper/grasp or /fr3/franka_gripper/move.',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=5.0,
        help='Test duration in seconds.',
    )
    parser.add_argument(
        '--vary-goal',
        action='store_true',
        help='Vary the target width over time instead of using a fixed width.',
    )
    parser.add_argument(
        '--decrement-first',
        action='store_true',
        help='Start by decreasing from the current width instead of increasing.',
    )
    parser.add_argument(
        '--width-min',
        type=float,
        default=0.01,
        help='Minimum width in meters when --vary-goal is enabled.',
    )
    parser.add_argument(
        '--width-max',
        type=float,
        default=0.05,
        help='Maximum width in meters when --vary-goal is enabled.',
    )
    parser.add_argument(
        '--width-step',
        type=float,
        default=0.002,
        help='Width increment/decrement in meters for each command when --vary-goal is enabled.',
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=0.03,
        help='Gripper closing speed in meters per second.',
    )
    parser.add_argument(
        '--force',
        type=float,
        default=20.0,
        help='Grasp force in Newton. Only used in grasp mode.',
    )
    parser.add_argument(
        '--epsilon-inner',
        type=float,
        default=0.005,
        help='Inner grasp epsilon in meters. Only used in grasp mode.',
    )
    parser.add_argument(
        '--epsilon-outer',
        type=float,
        default=0.005,
        help='Outer grasp epsilon in meters. Only used in grasp mode.',
    )
    parser.add_argument(
        '--result-timeout',
        type=float,
        default=30.0,
        help='Timeout in seconds for goal acceptance and result waiting.',
    )
    parser.add_argument(
        '--joint-states-topic',
        default='',
        help='Joint states topic used to read the current gripper width once at startup.',
    )
    parser.add_argument(
        '--state-timeout',
        type=float,
        default=5.0,
        help='Timeout in seconds while waiting for the initial joint state.',
    )
    return parser.parse_args(cli_args)


def main(args=None):
    cli_args = sys.argv[1:] if args is None else args
    parsed_args = parse_args(cli_args)

    if parsed_args.width_min > parsed_args.width_max:
        raise ValueError('--width-min must be less than or equal to --width-max')

    if parsed_args.width_step <= 0.0:
        raise ValueError('--width-step must be greater than 0')

    if '--action-name' not in cli_args:
        if parsed_args.mode == 'move':
            parsed_args.action_name = '/franka_gripper/move'
        else:
            parsed_args.action_name = '/franka_gripper/grasp'

    if not parsed_args.joint_states_topic:
        action_prefix = parsed_args.action_name.rsplit('/', 1)[0]
        parsed_args.joint_states_topic = f'{action_prefix}/joint_states'

    rclpy.init(args=args)
    node = GripperThroughputTest(parsed_args)

    try:
        return_code = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return return_code


if __name__ == '__main__':
    raise SystemExit(main())
