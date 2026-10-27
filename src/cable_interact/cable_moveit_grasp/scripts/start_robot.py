#!/usr/bin/python3
"""Own one driver process group; prepare its arm controller without sending goals."""
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from controller_manager_msgs.srv import (
    ListControllers, LoadController, ConfigureController, SwitchController,
)

MANAGER = '/NS_1/controller_manager'
ARM = 'fr3_arm_controller'
BROADCASTERS = ('joint_state_broadcaster', 'franka_robot_state_broadcaster')


def controller_steps(state):
    if state is None:
        return ('load', 'configure', 'activate')
    if state == 'unconfigured':
        return ('configure', 'activate')
    if state == 'inactive':
        return ('activate',)
    if state == 'active':
        return ()
    raise RuntimeError(f'Unexpected {ARM} state: {state}')


def stop_driver(driver):
    # Signal only the process group created by this script, never other ROS nodes.
    for sig, seconds in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 1)):
        try:
            os.killpg(driver.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            driver.poll()  # Reap the group leader if it has exited.
            try:
                os.killpg(driver.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
    try:
        driver.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def main():
    driver = None
    node = None
    # Keep signal handling in this supervisor, including during startup/service waits.
    def interrupt(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = rclpy.create_node('cable_robot_startup')
        clients = {
            'list': node.create_client(ListControllers, MANAGER + '/list_controllers'),
            'load': node.create_client(LoadController, MANAGER + '/load_controller'),
            'configure': node.create_client(ConfigureController, MANAGER + '/configure_controller'),
            'activate': node.create_client(SwitchController, MANAGER + '/switch_controller'),
        }
        if clients['list'].wait_for_service(timeout_sec=1.):
            raise RuntimeError('A controller_manager already exists at /NS_1; stop the old driver before starting this script')
        config = Path('/home/flexcycle/franka_ros2_ws/install/franka_fr3_moveit_config/share/franka_fr3_moveit_config/config/fr3_ros_controllers.yaml')
        if not config.is_file():
            raise RuntimeError(f'Controller configuration missing: {config}')
        driver = subprocess.Popen([
            'ros2', 'launch', 'franka_bringup', 'franka.launch.py',
            'robot_type:=fr3', 'robot_ip:=192.168.3.101', 'namespace:=NS_1',
            'load_gripper:=true', 'use_fake_hardware:=false',
            'controllers_yaml:=' + str(config),
        ], start_new_session=True)

        def check_driver():
            if driver.poll() is not None:
                raise RuntimeError(f'Robot driver exited with code {driver.returncode}; inspect driver output above')

        def call(key, request, timeout=15.):
            check_driver()
            client = clients[key]
            deadline = time.monotonic() + timeout
            while not client.wait_for_service(timeout_sec=0.2):
                check_driver()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Timed out waiting for {key} service')
            future = client.call_async(request)
            while not future.done():
                check_driver()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Timed out calling {key} service')
                rclpy.spin_once(node, timeout_sec=0.1)
            result = future.result()
            if result is None:
                raise RuntimeError(f'{key} service returned no result')
            if key != 'list' and not result.ok:
                raise RuntimeError(f'{key} failed for {ARM}; check conflicting controllers and driver output')
            return result

        print('Waiting for robot driver and state broadcasters...', flush=True)
        deadline = time.monotonic() + 60.
        while not clients['list'].wait_for_service(timeout_sec=0.2):
            check_driver()
            if time.monotonic() >= deadline:
                raise RuntimeError('Controller manager did not start within 60 seconds')
        while True:
            states = {c.name: c.state for c in call('list', ListControllers.Request()).controller}
            if all(states.get(name) == 'active' for name in BROADCASTERS):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('State broadcasters did not become active within 60 seconds')
            time.sleep(0.2)
        for step in controller_steps(states.get(ARM)):
            print(f'{ARM}: {step}', flush=True)
            if step == 'activate':
                request = SwitchController.Request()
                request.activate_controllers = [ARM]
                request.strictness = SwitchController.Request.STRICT
                request.activate_asap = True
                request.timeout.sec = 10
            else:
                request = (LoadController if step == 'load' else ConfigureController).Request()
                request.name = ARM
            call(step, request)
        states = {c.name: c.state for c in call('list', ListControllers.Request()).controller}
        if states.get(ARM) != 'active':
            raise RuntimeError(f'{ARM} did not become active')
        print('READY: FR3 driver, state broadcasters and fr3_arm_controller active. Start terminal 3; Ctrl+C here stops the driver.', flush=True)
        while driver.poll() is None:
            time.sleep(0.2)
        raise RuntimeError(f'Robot driver exited with code {driver.returncode}')
    except KeyboardInterrupt:
        print('\nStopping robot driver...', flush=True)
        return 130
    except Exception as error:
        print(f'Robot startup failed: {error}', flush=True)
        return 1
    finally:
        # A second Ctrl+C must not interrupt cleanup and leave a driver behind.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if driver is not None:
            stop_driver(driver)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
