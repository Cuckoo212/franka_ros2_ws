#!/usr/bin/env python3
"""Prepare one arm's controllers sequentially; reconcile lost responses via state."""
import time
import rclpy
from controller_manager_msgs.srv import ListControllers, LoadController, ConfigureController, SwitchController


def next_action(state):
    if state is None:
        return 'load'
    if state == 'unconfigured':
        return 'configure'
    if state == 'inactive':
        return 'activate'
    if state == 'active':
        return None
    raise RuntimeError('Unexpected controller state: ' + str(state))


def ensure_active(name, get_state, transition):
    for _ in range(6):
        before = get_state(name)
        action = next_action(before)
        if action is None:
            return
        error = None
        try:
            transition(name, action)
        except (TimeoutError, RuntimeError) as caught:
            error = caught
        # A lost response does not mean the mutation failed. Never blindly repeat it.
        after = get_state(name)
        if after == before:
            raise RuntimeError(f'{name}: {action} produced no confirmed state change; '
                               f'state={after}; {error or "check controller manager logs"}')
    raise RuntimeError(name + ': lifecycle did not converge')


def prepare_controllers(use_fake_hardware, execution_enabled, get_state, transition):
    """Only claim hardware command interfaces when execution was requested."""
    if not execution_enabled and get_state('fr3_arm_controller') == 'active':
        raise RuntimeError('Read-only startup found an active fr3_arm_controller; '
                           'stop the previous execution launch before manual recording')
    names = ['joint_state_broadcaster']
    if not use_fake_hardware:
        names.append('franka_robot_state_broadcaster')
    if execution_enabled:
        names.append('fr3_arm_controller')
    for name in names:
        ensure_active(name, get_state, transition)


def main():
    rclpy.init()
    node = rclpy.create_node('prepare_dual_controllers')
    node.declare_parameter('controller_manager', '/NS_1/controller_manager')
    node.declare_parameter('use_fake_hardware', False)
    node.declare_parameter('execution_enabled', False)
    manager = node.get_parameter('controller_manager').value
    clients = {
        'list': node.create_client(ListControllers, manager + '/list_controllers'),
        'load': node.create_client(LoadController, manager + '/load_controller'),
        'configure': node.create_client(ConfigureController, manager + '/configure_controller'),
        'activate': node.create_client(SwitchController, manager + '/switch_controller'),
    }

    def call(action, request):
        client = clients[action]
        if not client.wait_for_service(timeout_sec=60.):
            raise TimeoutError(manager + ': service unavailable: ' + action)
        future = client.call_async(request)
        deadline = time.monotonic() + 30.
        while not future.done() and rclpy.ok():
            if time.monotonic() >= deadline:
                raise TimeoutError(manager + ': response timeout: ' + action)
            rclpy.spin_once(node, timeout_sec=.1)
        if not future.done():
            raise RuntimeError('Shutdown during controller preparation')
        result = future.result()
        if result is None:
            raise RuntimeError('No response from ' + action)
        if action != 'list' and not result.ok:
            raise RuntimeError(manager + ': service rejected ' + action)
        return result

    def get_state(name):
        result = call('list', ListControllers.Request())
        return next((item.state for item in result.controller if item.name == name), None)

    def transition(name, action):
        node.get_logger().info(f'{manager}: {name} -> {action}')
        if action == 'activate':
            request = SwitchController.Request()
            request.activate_controllers = [name]
            request.strictness = SwitchController.Request.STRICT
            request.activate_asap = True
            request.timeout.sec = 20
        else:
            request = (LoadController if action == 'load' else ConfigureController).Request()
            request.name = name
        call(action, request)

    try:
        execution_enabled = node.get_parameter('execution_enabled').value
        prepare_controllers(node.get_parameter('use_fake_hardware').value,
                            execution_enabled, get_state, transition)
        mode = ('execution controllers active; no motion goals sent' if execution_enabled else
                'READ ONLY: state broadcasters active; arm command controller not activated')
        node.get_logger().info('READY: ' + manager + ' ' + mode)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        node.get_logger().error('Controller preparation failed: ' + str(error))
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
