"""Structural contracts between combined model, hardware and MoveIt routing."""
import pathlib
import sys
import unittest
import xml.etree.ElementTree as ET
import yaml
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from dual_arm_model import build_model


class DualModelTests(unittest.TestCase):
    def test_model_names_and_mount(self):
        model = build_model()
        urdf = ET.fromstring(model['robot_description'])
        links = {x.get('name') for x in urdf.findall('link')}
        for tag in ('link', 'joint'):
            names = [x.get('name') for x in urdf.findall(tag)]
            self.assertEqual(len(names), len(set(names)))
        children = {x.find('child').get('link') for x in urdf.findall('joint')}
        self.assertEqual(links - children, {'base'})
        for joint in urdf.findall('joint'):
            self.assertIn(joint.find('parent').get('link'), links)
            self.assertIn(joint.find('child').get('link'), links)
        mount = next(j for j in urdf.findall('joint') if j.find('child').get('link') == 'a_fr3_link0')
        self.assertEqual(mount.find('parent').get('link'), 'fr3_link0')
        self.assertEqual(mount.find('origin').get('xyz'), '0.47 0 0')
        self.assertAlmostEqual(float(mount.find('origin').get('rpy').split()[2]), 3.141592653589793)
        semantic = ET.fromstring(model['robot_description_semantic'])
        self.assertTrue({'fr3_arm', 'a_fr3_arm', 'both_arms'} <= {g.get('name') for g in semantic.findall('group')})
        for pair in semantic.findall('disable_collisions'):
            # Vendor intra-arm exemptions only; never exempt A from B collisions.
            self.assertEqual(pair.get('link1').startswith('a_'), pair.get('link2').startswith('a_'))
        for prefix, file in [('', 'arm_b_controllers.yaml'), ('a_', 'arm_a_controllers.yaml')]:
            config = pathlib.Path(__file__).resolve().parents[1] / 'config' / file
            parameters = yaml.safe_load(config.read_text())['/**']['fr3_arm_controller']['ros__parameters']
            names = [prefix + 'fr3_joint' + str(i) for i in range(1, 8)]
            self.assertEqual(parameters['joints'], names)
            self.assertEqual(set(parameters['gains']), set(names))
            model_names = {j.get('name') for j in urdf.findall('joint')}
            self.assertTrue(set(names) <= model_names)
            self.assertIn(prefix + 'fr3_hand_tcp', links)


if __name__ == '__main__':
    unittest.main()
