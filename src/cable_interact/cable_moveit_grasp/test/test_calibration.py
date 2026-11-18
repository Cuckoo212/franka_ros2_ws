import pathlib
import struct
import sys
import tempfile
import unittest
import numpy as np
import yaml
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from calibration_math import camera_obstacle_pose, load_calibration, transform_xyz_buffer, rotation_quaternion, OPTICAL_TO_CAMERA, invert_rigid_transform, camera_to_left_transform


class CalibrationTests(unittest.TestCase):
    def test_optical_and_quaternion(self):
        doc = {'parameters': {'calibration_type': 'eye_on_base', 'tracking_base_frame': 'camera_color_frame',
                              'robot_base_frame': 'fr3_link0'},
               'transform': {'translation': dict(x=1., y=2., z=3.),
                             'rotation': dict(x=0., y=0., z=0., w=1.)}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.calib') as f:
            yaml.safe_dump(doc, f); f.flush()
            r,t,source,target = load_calibration(f.name)
        np.testing.assert_allclose(r @ [1.,2.,3.] + t, [4.,1.,1.])
        np.testing.assert_allclose(r, OPTICAL_TO_CAMERA)
        x,y,z,w=rotation_quaternion(r)
        # Reconstruct and compare the TF rotation with the point-cloud rotation.
        reconstructed=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        np.testing.assert_allclose(reconstructed,r,atol=1e-12)
        self.assertEqual(target,'fr3_link0')

    def test_camera_collision_pose_matches_cloud(self):
        calib = pathlib.Path(__file__).resolve().parents[2] / 'pointcloud_tools/info_for_3Dpoint/samples_calibration/hand_eye_calibration/fr3_calibration_apriltag_depth_003.calib'
        rotation, translation, _, _ = load_calibration(calib)
        center = [.02, -.03, -.04]
        pose = camera_obstacle_pose(calib, center)
        np.testing.assert_allclose(pose[:3], rotation @ center + translation)
        np.testing.assert_allclose(pose[3:], rotation_quaternion(rotation))
        np.testing.assert_allclose(camera_obstacle_pose(calib, [0.,0.,0.])[:3], translation)
        with self.assertRaises(ValueError):
            camera_obstacle_pose(calib, [float('nan'), 0., 0.])

    def test_colors_padding_and_invalid_points(self):
        for endian in ('<','>'):
            # Two organized rows, RGB and extra per-point bytes, plus row padding.
            data=bytearray(48)
            struct.pack_into(endian+'fffI',data,0,1.,2.,3.,0x00aabbcc)
            struct.pack_into(endian+'fffI',data,24,float('nan'),2.,3.,0x00112233)
            data[16:24]=b'padding!'; data[40:48]=b'padding!'
            result=transform_xyz_buffer(data,1,2,16,24,[0,4,8],endian=='>',OPTICAL_TO_CAMERA,np.array([1.,2.,3.]))
            self.assertEqual(struct.unpack_from(endian+'fff',result),(4.,1.,1.))
            self.assertEqual(result[12:24],data[12:24])
            self.assertEqual(result[24:],data[24:])
            self.assertEqual(len(result),len(data))

    def test_arm_a_camera_composition(self):
        r_ba = np.diag([-1., -1., 1.])
        t_ba = np.array([.47, 0., 0.])
        r_ab, t_ab = invert_rigid_transform(r_ba, t_ba)
        np.testing.assert_allclose(r_ab @ t_ba + t_ab, [0., 0., 0.])
        np.testing.assert_allclose(t_ab, [.47, 0., 0.])
        p_b = np.array([.6, .2, .3])
        p_a = r_ab @ p_b + t_ab
        np.testing.assert_allclose(p_a, [-.13, -.2, .3])
        np.testing.assert_allclose(r_ba @ p_a + t_ba, p_b)
        # Existing eye-on-base calibration plus optical conversion, composed once.
        calib = pathlib.Path(__file__).resolve().parents[2] / 'pointcloud_tools/info_for_3Dpoint/samples_calibration/hand_eye_calibration/fr3_calibration_apriltag_depth_003.calib'
        r_bc, t_bc, _, _ = load_calibration(calib)
        r_ac, t_ac = r_ab @ r_bc, r_ab @ t_bc + t_ab
        p_c = np.array([.1, -.2, .8])
        np.testing.assert_allclose(r_ac @ p_c + t_ac, r_ab @ (r_bc @ p_c + t_bc) + t_ab)
        np.testing.assert_allclose(t_ac, [.2549470060408407, -.3309618702166004, .5218691240613372])

    def test_direct_camera_to_a_matches_composed_path(self):
        calib = pathlib.Path(__file__).resolve().parents[2] / 'pointcloud_tools/info_for_3Dpoint/samples_calibration/hand_eye_calibration/fr3_calibration_apriltag_depth_003.calib'
        r_bc,t_bc,_,_ = load_calibration(calib)
        origin = np.array([.47,0.,0.])
        r_ac,t_ac = camera_to_left_transform(r_bc,t_bc,origin)
        points = np.array([[0.,0.,0.],[.1,-.2,.8],[-.3,.1,1.2]])
        expected_b = points @ r_bc.T + t_bc
        expected_a = (expected_b-origin) @ np.diag([-1.,-1.,1.])
        np.testing.assert_allclose(points @ r_ac.T + t_ac,expected_a,atol=1e-12)
        # One float32 conversion instead of re-reading rounded B-frame XYZ.
        source = struct.pack('<fffI',.1,-.2,.8,0xaabbcc)
        result = transform_xyz_buffer(source,1,1,16,16,[0,4,8],False,r_ac,t_ac)
        xyz = np.array(struct.unpack_from('<fff',source),dtype=float)
        np.testing.assert_allclose(struct.unpack_from('<fff',result),r_ac@xyz+t_ac,atol=1e-7)
        self.assertEqual(result[12:],source[12:])
        with self.assertRaises(ValueError):
            camera_to_left_transform(r_bc,t_bc,[float('nan'),0.,0.])

    def test_arm_a_cloud_preserves_metadata(self):
        from types import SimpleNamespace
        from sensor_msgs.msg import PointCloud2, PointField
        from arm_base_cloud_node import ArmBaseCloud
        output = []
        fake = SimpleNamespace(input_frame='fr3_link0', target='a_fr3_link0',
                               rotation=np.diag([-1., -1., 1.]), translation=np.array([.47, 0., 0.]),
                               publisher=SimpleNamespace(publish=output.append))
        msg = PointCloud2()
        msg.header.frame_id = 'fr3_link0'
        msg.header.stamp.sec = 123
        msg.height = msg.width = 1
        msg.point_step = msg.row_step = 16
        msg.fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(['x', 'y', 'z', 'rgb'])]
        msg.data = struct.pack('<fffI', .6, .2, .3, 0xaabbcc)
        ArmBaseCloud.receive(fake, msg)
        self.assertEqual(output[0].header.frame_id, 'a_fr3_link0')
        self.assertEqual(output[0].header.stamp, msg.header.stamp)
        np.testing.assert_allclose(struct.unpack_from('<fff', output[0].data), [-.13, -.2, .3], atol=1e-7)
        self.assertEqual(output[0].data[12:], msg.data[12:])
        self.assertEqual(msg.header.frame_id, 'fr3_link0')

    def test_malformed_layout(self):
        with self.assertRaises(ValueError):
            transform_xyz_buffer(b'bad',1,1,16,16,[0,4,8],False,np.eye(3),np.zeros(3))


if __name__ == '__main__':
    unittest.main()
