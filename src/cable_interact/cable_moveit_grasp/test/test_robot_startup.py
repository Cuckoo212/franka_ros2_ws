#!/usr/bin/python3
"""No ROS initialization or hardware connections in these tests."""
import importlib.util
from pathlib import Path
import subprocess
import unittest

script = Path(__file__).resolve().parents[1] / 'scripts' / 'start_robot.py'
if not script.exists():
    script = Path(__file__).with_name('start_robot.py')
spec = importlib.util.spec_from_file_location('robot_start', script)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class StartupTests(unittest.TestCase):
    def test_controller_lifecycle(self):
        self.assertEqual(startup.controller_steps(None), ('load', 'configure', 'activate'))
        self.assertEqual(startup.controller_steps('unconfigured'), ('configure', 'activate'))
        self.assertEqual(startup.controller_steps('inactive'), ('activate',))
        self.assertEqual(startup.controller_steps('active'), ())
        with self.assertRaises(RuntimeError):
            startup.controller_steps('finalized')

    def test_cleanup_owned_process_group(self):
        child = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True)
        try:
            startup.stop_driver(child)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == '__main__':
    unittest.main()
