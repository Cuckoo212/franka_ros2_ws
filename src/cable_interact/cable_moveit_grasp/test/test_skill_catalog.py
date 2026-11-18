import json
import pathlib
import sys
import tempfile
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from skill_catalog import load_skill_task
from alternate_grasp_state_machine import State, COMMAND, NEXT

class CatalogTests(unittest.TestCase):
    def test_current_task_preserves_state_machine(self):
        skills, task = load_skill_task(ROOT/'skills/catalog.json', ROOT/'skills/alternate_cable_task.json', list(State), COMMAND, NEXT)
        self.assertEqual(set(skills), {'home', 'grasp', 'fix', 'release', 'verify'})
        self.assertEqual(task['default_cycles'], 5)
        for skill in skills.values():
            self.assertTrue((ROOT / skill['implementation']).is_file())
        self.assertEqual(skills['fix']['backend'], 'projected_fix_trajectory_selection')

    def test_reject_mismatched_transition(self):
        task = json.loads((ROOT/'skills/alternate_cable_task.json').read_text())
        task['states'][0]['next'] = State.B_FIX.value
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(task, f); f.flush()
            with self.assertRaises(ValueError):
                load_skill_task(ROOT/'skills/catalog.json', f.name, list(State), COMMAND, NEXT)

if __name__ == '__main__': unittest.main()
