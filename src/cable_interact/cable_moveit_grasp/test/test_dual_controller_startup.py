import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_dual_controllers import ensure_active, prepare_controllers


class StartupTests(unittest.TestCase):
    def prepare(self, fake, execution, initial=None):
        states = dict(initial or {})
        calls = []
        def change(name, action):
            calls.append((name, action))
            states[name] = {'load': 'unconfigured', 'configure': 'inactive', 'activate': 'active'}[action]
        prepare_controllers(fake, execution, states.get, change)
        return states, calls

    def test_real_read_only_never_loads_or_activates_arm_controller(self):
        states, calls = self.prepare(False, False)
        self.assertEqual(set(states), {'joint_state_broadcaster', 'franka_robot_state_broadcaster'})
        self.assertFalse(any(name == 'fr3_arm_controller' for name, _ in calls))

    def test_fake_read_only_only_activates_joint_broadcaster(self):
        states, _ = self.prepare(True, False)
        self.assertEqual(states, {'joint_state_broadcaster': 'active'})

    def test_execution_preserves_arm_activation_after_broadcasters(self):
        states, calls = self.prepare(False, True)
        self.assertEqual(states['fr3_arm_controller'], 'active')
        self.assertEqual(calls[-1], ('fr3_arm_controller', 'activate'))

    def test_read_only_leaves_inactive_arm_controller_alone(self):
        states, calls = self.prepare(False, False, {'fr3_arm_controller': 'inactive'})
        self.assertEqual(states['fr3_arm_controller'], 'inactive')
        self.assertFalse(any(name == 'fr3_arm_controller' for name, _ in calls))

    def test_read_only_rejects_already_active_arm_without_mutating_it(self):
        calls = []
        with self.assertRaisesRegex(RuntimeError, 'previous execution launch'):
            prepare_controllers(False, False, lambda name: 'active' if name == 'fr3_arm_controller' else None,
                                lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def run_startup(self, initial, lose_response=None):
        state = [initial]
        calls = []
        def change(name, action):
            calls.append(action)
            state[0] = {'load': 'unconfigured', 'configure': 'inactive', 'activate': 'active'}[action]
            if action == lose_response:
                raise TimeoutError('Simulated DDS response loss after successful mutation')
        ensure_active('controller', lambda name: state[0], change)
        self.assertEqual(state[0], 'active')
        return calls

    def test_lost_load_response_does_not_reload(self):
        self.assertEqual(self.run_startup(None, 'load'), ['load', 'configure', 'activate'])

    def test_lost_configure_and_activation_response(self):
        for action in ('configure', 'activate'):
            self.assertEqual(self.run_startup(None, action), ['load', 'configure', 'activate'])

    def test_resume_already_loaded_controller(self):
        self.assertEqual(self.run_startup('unconfigured'), ['configure', 'activate'])
        self.assertEqual(self.run_startup('inactive'), ['activate'])
        self.assertEqual(self.run_startup('active'), [])

    def test_unconfirmed_mutation_never_retried(self):
        calls = []
        def fail(name, action):
            calls.append(action)
            raise TimeoutError('Unknown state; no confirmed transition')
        with self.assertRaises(RuntimeError):
            ensure_active('controller', lambda name: None, fail)
        self.assertEqual(calls, ['load'])


if __name__ == '__main__':
    unittest.main()
