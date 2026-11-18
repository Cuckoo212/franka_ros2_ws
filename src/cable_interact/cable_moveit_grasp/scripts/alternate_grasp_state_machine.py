"""Pure, event-driven sequence. Only authoritative motion results advance a phase."""
from enum import Enum


class State(str, Enum):
    HOME_A = 'home_a_initial'
    HOME_B = 'home_b_initial'
    FIRST_B_POINT = 'wait_first_b_point'
    FIRST_B_GRASP = 'first_b_grasp'
    B_FIX = 'b_to_fixB'
    A_GRASP = 'a_grasp'
    B_HOME = 'b_release_and_home'
    A_FIX = 'a_fix'
    B_GRASP = 'b_grasp'
    A_HOME = 'a_release_and_home'


class Mode(str, Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    WAIT_MANUAL = 'waiting_manual_point'
    PAUSED = 'paused'
    FAILED = 'failed_hold'
    COMPLETE = 'completed'


ORDER = list(State)
NEXT = dict(zip(ORDER, ORDER[1:] + [State.B_FIX]))
COMMAND = {
    State.HOME_A: 'home_a', State.HOME_B: 'home_b',
    State.FIRST_B_GRASP: 'grasp_b', State.B_FIX: 'fix_b',
    State.A_GRASP: 'grasp_a', State.B_HOME: 'home_b', State.A_FIX: 'fix_a',
    State.B_GRASP: 'grasp_b', State.A_HOME: 'home_a',
}
GRASPS = {State.FIRST_B_GRASP: 'b', State.A_GRASP: 'a', State.B_GRASP: 'b'}


class AlternateGraspStateMachine:
    def __init__(self, max_cycles=5):
        self.state = State.HOME_A
        self.mode = Mode.IDLE
        self.completed_cycles = 0
        self.max_cycles = max_cycles
        self.reason = ''

    def start(self):
        if self.mode != Mode.IDLE:
            raise ValueError('Start requires idle; use resume for a paused run or reset after recovery')
        self.mode = Mode.RUNNING
        self.reason = ''

    def complete(self, expected_state):
        if self.mode not in (Mode.RUNNING, Mode.PAUSED) or expected_state != self.state.value:
            raise ValueError('Stale or unauthorized completion')
        if self.state == State.FIRST_B_POINT:
            raise ValueError('First point requires a fresh manual plan, not a motion completion')
        if self.state == State.A_HOME:
            self.completed_cycles += 1
        if self.max_cycles and self.completed_cycles >= self.max_cycles:
            self.mode = Mode.COMPLETE
            self.reason = 'Completed ' + str(self.completed_cycles) + ' cycles; A at HOME, B remains holding; no further actions scheduled'
            return
        self.state = NEXT[self.state]
        self.reason = '' if self.mode == Mode.RUNNING else self.reason

    def wait_manual(self, reason):
        if self.state not in GRASPS and self.state != State.FIRST_B_POINT:
            raise ValueError('This phase cannot wait for a grasp point')
        self.mode = Mode.WAIT_MANUAL
        self.reason = reason

    def manual_ready(self):
        if self.mode != Mode.WAIT_MANUAL:
            raise ValueError('Not waiting for a manual point')
        if self.state == State.FIRST_B_POINT:
            self.state = State.FIRST_B_GRASP
        self.mode = Mode.RUNNING
        self.reason = ''

    def pause(self, reason):
        if self.mode in (Mode.IDLE, Mode.COMPLETE):
            return
        if self.mode != Mode.FAILED:
            self.mode = Mode.PAUSED
        self.reason = reason

    def fail(self, reason):
        self.mode = Mode.FAILED
        self.reason = reason

    def resume(self):
        if self.mode not in (Mode.PAUSED, Mode.FAILED):
            raise ValueError('Resume requires a paused/failed run')
        self.mode = Mode.RUNNING
        self.reason = ''


def validate_resume(state, backend):
    """Never infer a successful hold from a failed grasp or skip a handover."""
    if backend.get('fault', True):
        raise ValueError('Backend fault remains; run home_a/home_b for the failed arm first')
    ha, hb = backend.get('holding_a', False), backend.get('holding_b', False)
    a, b = backend.get('home_a', False), backend.get('home_b', False)
    valid = {
        State.HOME_A: True, State.HOME_B: True, State.FIRST_B_POINT: a and b and not ha and not hb,
        State.FIRST_B_GRASP: a and b and not ha and not hb,
        State.B_FIX: hb and a and not ha,
        State.A_GRASP: a and hb and not ha,
        State.B_HOME: ha,
        State.A_FIX: ha and b and not hb,
        State.B_GRASP: b and ha and not hb,
        State.A_HOME: hb,
    }
    if not valid[state]:
        raise ValueError('Holding/HOME records do not permit resuming ' + state.value +
                         '; recover with manual HOME/grasp commands or reset when both arms are empty at HOME')
