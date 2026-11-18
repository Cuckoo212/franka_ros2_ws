import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from alternate_grasp_state_machine import AlternateGraspStateMachine, State, Mode, COMMAND, validate_resume


class StateMachineTests(unittest.TestCase):
    def test_exact_first_b_sequence_and_loop(self):
        m = AlternateGraspStateMachine(max_cycles=2)
        m.start()
        commands = []
        for _ in range(30):
            if m.mode == Mode.COMPLETE:
                break
            if m.state == State.FIRST_B_POINT:
                m.wait_manual('first point');m.manual_ready()
            else:
                commands.append(COMMAND[m.state]);m.complete(m.state.value)
        self.assertEqual(commands, ['home_a', 'home_b', 'grasp_b', 'fix_b',
            'grasp_a', 'home_b', 'fix_a', 'grasp_b', 'home_a',
            'fix_b', 'grasp_a', 'home_b', 'fix_a', 'grasp_b', 'home_a'])
        self.assertEqual(m.completed_cycles, 2)
        self.assertEqual(m.mode, Mode.COMPLETE)

    def test_default_five_cycles_stop_without_sixth_fix_or_release(self):
        from collections import Counter
        m=AlternateGraspStateMachine();m.start();commands=[]
        for _ in range(60):
            if m.mode==Mode.COMPLETE:break
            if m.state==State.FIRST_B_POINT:m.wait_manual('first');m.manual_ready()
            else:commands.append(COMMAND[m.state]);m.complete(m.state.value)
        self.assertEqual(m.max_cycles,5)
        self.assertEqual(m.completed_cycles,5)
        self.assertEqual(m.mode,Mode.COMPLETE)
        self.assertEqual(m.state,State.A_HOME)
        self.assertEqual(commands,['home_a','home_b','grasp_b']+
                         ['fix_b','grasp_a','home_b','fix_a','grasp_b','home_a']*5)
        self.assertEqual(Counter(commands)['fix_b'],5)
        with self.assertRaises(ValueError):m.complete(State.A_HOME.value)
        self.assertEqual(m.completed_cycles,5)

    def test_failure_in_fifth_round_does_not_count_until_a_home_success(self):
        m=AlternateGraspStateMachine();m.start();m.completed_cycles=4;m.state=State.B_GRASP
        m.fail('grasp failed')
        self.assertEqual(m.completed_cycles,4)
        self.assertEqual(m.mode,Mode.FAILED)
        m.resume();m.complete(State.B_GRASP.value)
        self.assertEqual(m.completed_cycles,4)
        m.complete(State.A_HOME.value)
        self.assertEqual(m.completed_cycles,5)
        self.assertEqual(m.mode,Mode.COMPLETE)

    def test_explicit_unlimited_mode_remains_available(self):
        m=AlternateGraspStateMachine(max_cycles=0);m.start()
        for _ in range(50):
            if m.state==State.FIRST_B_POINT:m.wait_manual('first');m.manual_ready()
            else:m.complete(m.state.value)
        self.assertGreater(m.completed_cycles,5)
        self.assertEqual(m.mode,Mode.RUNNING)

    def test_each_completed_handover_returns_to_b_fix_not_b_regrasp(self):
        m = AlternateGraspStateMachine();m.start();m.state=State.A_HOME
        m.complete(State.A_HOME.value)
        self.assertEqual(m.state, State.B_FIX)
        self.assertEqual(COMMAND[m.state], 'fix_b')
        self.assertEqual(m.completed_cycles, 1)
        m.fail('FIX rotation limit')
        self.assertEqual(m.state, State.B_FIX)
        with self.assertRaises(ValueError):m.complete(State.B_FIX.value)
        validate_resume(State.B_FIX, dict(fault=False,home_a=True,home_b=False,holding_a=False,holding_b=True))
        m.resume();m.complete(State.B_FIX.value)
        self.assertEqual(m.state, State.A_GRASP)

    def test_failed_grasp_never_advances_to_release(self):
        for phase in (State.FIRST_B_GRASP, State.A_GRASP, State.B_GRASP):
            m = AlternateGraspStateMachine();m.start();m.state = phase
            m.fail('gripper failed')
            self.assertEqual(m.state, phase)
            with self.assertRaises(ValueError):m.complete(phase.value)
            with self.assertRaises(ValueError):m.start()

    def test_stale_completion_and_pause_track_inflight_success(self):
        m = AlternateGraspStateMachine();m.start();m.pause('manual control')
        m.complete(State.HOME_A.value)
        self.assertEqual(m.state, State.HOME_B)
        self.assertEqual(m.mode, Mode.PAUSED)
        with self.assertRaises(ValueError):m.complete(State.HOME_A.value)

    def test_manual_plan_does_not_count_as_a_successful_grasp(self):
        m = AlternateGraspStateMachine();m.start();m.state = State.FIRST_B_POINT
        m.wait_manual('pick');m.manual_ready()
        self.assertEqual(m.state, State.FIRST_B_GRASP)
        self.assertEqual(m.completed_cycles, 0)

    def test_resume_requires_failed_arm_home_and_other_holder(self):
        state = dict(fault=False, home_a=True, home_b=False, holding_a=False, holding_b=True)
        validate_resume(State.A_GRASP, state)
        for change in ({'fault':True}, {'home_a':False}, {'holding_b':False}, {'holding_a':True}):
            with self.assertRaises(ValueError):validate_resume(State.A_GRASP, dict(state, **change))
        with self.assertRaises(ValueError):validate_resume(State.A_FIX, state)

    def test_pause_cannot_clear_failure_or_create_a_hold(self):
        m = AlternateGraspStateMachine();m.start();m.state=State.A_GRASP;m.fail('failed')
        m.pause('external home')
        self.assertEqual(m.mode, Mode.FAILED)
        self.assertEqual(m.state, State.A_GRASP)


if __name__ == '__main__':unittest.main()
