"""Authoritative completion, stale manual plan gating and late timeout handling."""
import pathlib
import sys
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from alternate_grasp_state_machine import AlternateGraspStateMachine, State, Mode
from cable_moveit_alternate_grasp import CableMoveItAlternateGrasp as C


class BackendTests(unittest.TestCase):
    def fake(self, success=True, message='done'):
        m=AlternateGraspStateMachine();m.start();m.state=State.A_GRASP
        result=SimpleNamespace(success=success, message=message)
        f=SimpleNamespace(machine=m, pending=SimpleNamespace(done=lambda:True,result=lambda:result),
            pending_kind='motion', pending_token=m.state.value, pending_path='grasp_a', timed_out=False,
            control_busy=False, last_motion_time=0.,wait_arm=None,min_revision=None,prepare_stage=None,
            publish=lambda:None,plan_states={},get_parameter=lambda _:SimpleNamespace(value=.0))
        f.handle_result=lambda k,t,r:C.handle_result(f,k,t,r)
        f.begin_manual=lambda arm:(setattr(f,'wait_arm',arm),m.wait_manual('manual'))
        return f

    def test_physical_failure_never_releases_b(self):
        f=self.fake(False,'PHYSICAL_FAILURE: grasp aborted');C.tick(f)
        self.assertEqual(f.machine.state,State.A_GRASP)
        self.assertEqual(f.machine.mode,Mode.FAILED)

    def test_only_planner_rejection_enters_manual_wait(self):
        f=self.fake(False,'PRECHECK_REJECTED: GRASP_PLAN_REJECTED: no point');C.tick(f)
        self.assertEqual(f.machine.mode,Mode.WAIT_MANUAL)
        self.assertEqual(f.wait_arm,'a')
        f=self.fake(False,'PRECHECK_REJECTED: Missing or stale joint state');C.tick(f)
        self.assertEqual(f.machine.mode,Mode.FAILED)

    def test_recovered_misses_wait_for_new_point_without_handover(self):
        f=self.fake(False,'PRECHECK_REJECTED: GRASP_PLAN_REJECTED: All bounded grasp retries exhausted; failed arm is open at HOME')
        C.tick(f)
        self.assertEqual(f.machine.state,State.A_GRASP)
        self.assertEqual(f.machine.mode,Mode.WAIT_MANUAL)
        self.assertEqual(f.wait_arm,'a')
        self.assertEqual(f.machine.completed_cycles,0)

    def test_success_advances_but_does_not_send_next_action_in_same_tick(self):
        f=self.fake();C.tick(f)
        self.assertEqual(f.machine.state,State.B_HOME)
        self.assertIsNone(f.pending)

    def test_pause_does_not_lose_result_or_resume_automatically(self):
        f=self.fake();f.machine.pause('manual home');C.tick(f)
        self.assertEqual(f.machine.state,State.B_HOME)
        self.assertEqual(f.machine.mode,Mode.PAUSED)

    def test_late_success_after_timeout_never_advances(self):
        f=self.fake();f.timed_out=True;f.machine.fail('unknown result');C.tick(f)
        self.assertEqual(f.machine.state,State.A_GRASP)
        self.assertEqual(f.machine.mode,Mode.FAILED)

    def test_completed_fifth_cycle_never_dispatches_sixth_fix(self):
        f=self.fake();f.pending=None;f.machine.state=State.A_HOME
        f.machine.completed_cycles=5;f.machine.mode=Mode.COMPLETE
        sent=[];f.send=lambda *args:sent.append(args)
        C.tick(f)
        self.assertEqual(sent,[])
        self.assertEqual(f.machine.completed_cycles,5)

    def test_old_manual_preview_cannot_execute_first_grasp(self):
        f=self.fake();f.pending=None;f.machine.state=State.FIRST_B_POINT
        f.machine.wait_manual('first');f.wait_arm='b';f.min_revision=10
        f.plan_states={'b':dict(ready=True,revision=10)}
        C.tick(f)
        self.assertEqual(f.machine.mode,Mode.WAIT_MANUAL)
        f.plan_states['b']['revision']=11;C.tick(f)
        self.assertEqual(f.machine.state,State.FIRST_B_GRASP)
        self.assertEqual(f.machine.mode,Mode.RUNNING)


if __name__ == '__main__':unittest.main()
