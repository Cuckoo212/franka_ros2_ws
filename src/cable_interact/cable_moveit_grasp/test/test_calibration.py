import pathlib
import struct
import sys
import tempfile
import unittest
import numpy as np
import yaml
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from calibration_math import load_calibration, transform_xyz_buffer, rotation_quaternion, OPTICAL_TO_CAMERA


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

    def test_malformed_layout(self):
        with self.assertRaises(ValueError):
            transform_xyz_buffer(b'bad',1,1,16,16,[0,4,8],False,np.eye(3),np.zeros(3))


if __name__ == '__main__':
    unittest.main()
