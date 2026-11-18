#!/usr/bin/env python3
"""Automatic serialized dual-arm sequence, with fresh manual plans and explicit recovery."""
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from std_srvs.srv import Trigger
from alternate_grasp_state_machine import AlternateGraspStateMachine, State, Mode, COMMAND, GRASPS, validate_resume


class CableMoveItAlternateGrasp(Node):
    def __init__(self):
        super().__init__('cable_moveit_alternate_grasp')
        self.declare_parameter('physical_backend', False)
        self.declare_parameter('backend_timeout', 240.0)
        self.declare_parameter('perception_settle_time', 0.5)
        self.declare_parameter('max_cycles', 5)
        self.machine = AlternateGraspStateMachine(self.get_parameter('max_cycles').value)
        self.declare_parameter('skill_catalog_file', '')
        self.declare_parameter('task_design_file', '')
        self.skill_descriptions = {}
        catalog_path = self.get_parameter('skill_catalog_file').value
        task_path = self.get_parameter('task_design_file').value
        if bool(catalog_path) != bool(task_path):
            raise ValueError('Both skill_catalog_file and task_design_file are required together')
        if catalog_path:
            from skill_catalog import load_skill_task
            from alternate_grasp_state_machine import State, COMMAND, NEXT
            self.skill_descriptions, self.task_design = load_skill_task(catalog_path, task_path, list(State), COMMAND, NEXT)
            self.get_logger().info('Validated JSON skill catalog and task design; existing state-machine transitions retained')
        self.pending = None
        self.pending_kind = None
        self.pending_token = None
        self.pending_path = ''
        self.pending_deadline = 0.
        self.timed_out = False
        self.rpc_clients = {}
        self.client_group = ReentrantCallbackGroup()
        self.control_busy = False
        self.wait_arm = None
        self.min_revision = None
        self.prepare_stage = None
        self.plan_states = {}
        self.last_motion_time = 0.
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(String, '~/state', qos)
        self.input_subscriptions = []
        for arm in ('a', 'b'):
            self.input_subscriptions.append(self.create_subscription(String,
                '/dual_arm/planner_' + arm + '/candidate_planner/plan_ready',
                lambda msg, arm=arm: self.plan_update(arm, msg), qos))
        self.input_subscriptions.append(self.create_subscription(String, '/dual_arm/dual_arm_motion/manual_override', self.manual_override, 10))
        self.task_services = [self.create_service(Trigger, '~/start', self.start),
            self.create_service(Trigger, '~/pause', self.pause), self.create_service(Trigger, '~/stop', self.pause),
            self.create_service(Trigger, '~/resume', self.resume), self.create_service(Trigger, '~/reset', self.reset)]
        self.timer = self.create_timer(0.1, self.tick)
        self.publish()
        self.get_logger().info('Automatic coordinator idle; start -> HOME both -> first B RViz point -> automatic cycle')

    def publish(self):
        m = self.machine
        self.publisher.publish(String(data=json.dumps(dict(state=m.state.value, mode=m.mode.value,
            completed_cycles=m.completed_cycles, max_cycles=m.max_cycles, reason=m.reason, step_running=self.pending is not None,
            pending_command=self.pending_path if self.pending else '', waiting_arm=self.wait_arm,
            manual_plan_min_revision=self.min_revision,
            ready_for_manual_point=m.mode == Mode.WAIT_MANUAL and self.min_revision is not None and self.prepare_stage is None))))

    def plan_update(self, arm, message):
        try:
            self.plan_states[arm] = json.loads(message.data)
        except (ValueError, TypeError):
            pass

    def manual_override(self, message):
        if self.machine.mode not in (Mode.IDLE, Mode.COMPLETE):
            self.machine.pause('External manual command ' + message.data + '; scheduling paused; use resume after recovery')
            self.publish()

    def client(self, path):
        if path not in self.rpc_clients:
            self.rpc_clients[path] = self.create_client(Trigger, path, callback_group=self.client_group)
        return self.rpc_clients[path]

    def start(self, request, response):
        try:
            if not self.get_parameter('physical_backend').value:
                raise ValueError('physical_backend=false; launch dual_arm with execution_enabled:=true')
            if self.pending is not None or self.control_busy:
                raise ValueError('A request is still pending')
            cycles=int(self.get_parameter('max_cycles').value)
            if cycles<0: raise ValueError('max_cycles must be nonnegative')
            self.machine.max_cycles=cycles
            self.machine.start()
            response.success = True
            response.message = 'Started: HOME both, then wait for a fresh first B RViz point; selection executes automatically; max_cycles=' + str(cycles)
        except ValueError as error:
            response.message = str(error)
        self.publish()
        return response

    def pause(self, request, response):
        self.machine.pause('Scheduling paused; any in-flight physical action is still tracked, not cancelled')
        response.success = True
        response.message = self.machine.reason or 'Idle'
        self.publish()
        return response

    async def snapshot(self):
        client = self.client('/dual_arm/dual_arm_motion/get_execution_state')
        if not client.service_is_ready():
            raise ValueError('Execution state service unavailable')
        result = await client.call_async(Trigger.Request())
        if not result.success:
            raise ValueError(result.message)
        return json.loads(result.message)

    async def resume(self, request, response):
        if self.pending is not None or self.control_busy:
            response.message = 'A request is still pending; unknown results must be resolved before resuming'
            return response
        if self.machine.mode not in (Mode.PAUSED, Mode.FAILED):
            response.message = 'Resume requires paused/failed_hold'
            return response
        self.control_busy = True
        try:
            backend = await self.snapshot()
            validate_resume(self.machine.state, backend)
            was_manual = self.wait_arm is not None or self.machine.state in (State.FIRST_B_POINT, State.FIRST_B_GRASP)
            self.machine.resume()
            self.timed_out = False
            if was_manual:
                self.begin_manual('b' if self.machine.state in (State.FIRST_B_POINT, State.FIRST_B_GRASP) else GRASPS[self.machine.state])
            response.success = True
            response.message = 'Resumed; see state topic'
        except (ValueError, TypeError) as error:
            response.message = str(error)
        finally:
            self.control_busy = False
        self.publish()
        return response

    async def reset(self, request, response):
        if self.pending is not None or self.control_busy or self.machine.mode in (Mode.RUNNING, Mode.WAIT_MANUAL):
            response.message = 'Pause and resolve the pending action before reset'
            return response
        self.control_busy = True
        try:
            backend = await self.snapshot()
            if backend.get('fault', True) or any(backend.get('holding_' + a, True) for a in ('a', 'b')) or not all(backend.get('home_' + a, False) for a in ('a', 'b')):
                raise ValueError('Reset requires both arms empty at HOME and no backend fault')
            self.machine = AlternateGraspStateMachine(self.get_parameter('max_cycles').value)
            self.wait_arm = self.min_revision = self.prepare_stage = None
            self.timed_out = False
            response.success = True
            response.message = 'Reset to idle; start begins a new first-B run'
        except (ValueError, TypeError) as error:
            response.message = str(error)
        finally:
            self.control_busy = False
        self.publish()
        return response

    def begin_manual(self, arm):
        self.machine.wait_manual('Preparing fresh manual input for ' + arm.upper())
        self.wait_arm, self.min_revision, self.prepare_stage = arm, None, 'selector'

    def send(self, path, kind):
        client = self.client(path)
        if not client.service_is_ready():
            self.machine.fail('Service unavailable: ' + path)
            return
        self.pending_kind, self.pending_token, self.pending_path = kind, self.machine.state.value, path
        self.pending = client.call_async(Trigger.Request())
        self.pending_deadline = time.monotonic() + float(self.get_parameter('backend_timeout').value)
        self.timed_out = False

    def handle_result(self, kind, token, result):
        m = self.machine
        if self.timed_out or token != m.state.value:
            return  # Keep the failed phase; a late/unknown result never advances it.
        if kind == 'motion':
            if result.success:
                self.last_motion_time = time.monotonic()
                if m.mode in (Mode.RUNNING, Mode.PAUSED):
                    m.complete(token)
            elif (m.mode == Mode.RUNNING and m.state in GRASPS and
                  result.message.startswith('PRECHECK_REJECTED: GRASP_PLAN_REJECTED:')):
                self.begin_manual(GRASPS[m.state])
            else:
                m.fail(result.message)
        elif not result.success:
            m.fail(result.message)
        elif kind == 'selector':
            self.prepare_stage = 'planner'
        elif kind == 'prepare':
            self.min_revision = int(result.message)
            self.prepare_stage = None
            self.machine.reason = 'Select ' + self.wait_arm.upper() + ' and publish a cable point; a valid preview executes automatically'
            self.get_logger().info(self.machine.reason)

    def tick(self):
        if self.control_busy:
            return
        if self.pending is not None:
            if self.pending.done():
                future, kind, token = self.pending, self.pending_kind, self.pending_token
                self.pending = None
                try:
                    self.handle_result(kind, token, future.result())
                except Exception as error:
                    self.machine.fail(str(error))
            elif time.monotonic() > self.pending_deadline and not self.timed_out:
                self.timed_out = True
                self.machine.fail('Backend timeout; physical result unknown. Tracking the request; no retry or handover')
            self.publish()
            return  # Never enqueue another motion in the same tick as a completion.
        m = self.machine
        if m.mode == Mode.RUNNING:
            if m.state == State.FIRST_B_POINT:
                self.begin_manual('b')
            elif m.state in GRASPS and time.monotonic() - self.last_motion_time < float(self.get_parameter('perception_settle_time').value):
                pass
            else:
                self.send('/dual_arm/dual_arm_motion/auto/' + COMMAND[m.state], 'motion')
        elif m.mode == Mode.WAIT_MANUAL:
            if self.prepare_stage == 'selector':
                self.send('/dual_arm/dual_arm_grasp_selector/select_' + self.wait_arm, 'selector')
            elif self.prepare_stage == 'planner':
                self.send('/dual_arm/planner_' + self.wait_arm + '/candidate_planner/prepare_manual_grasp', 'prepare')
            elif self.min_revision is not None:
                plan = self.plan_states.get(self.wait_arm, {})
                if plan.get('ready') and int(plan.get('revision', -1)) > self.min_revision:
                    m.manual_ready()
                    self.wait_arm = self.min_revision = None
        self.publish()


def main(args=None):
    rclpy.init(args=args)
    node = CableMoveItAlternateGrasp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
