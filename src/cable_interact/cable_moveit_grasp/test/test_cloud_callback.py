import pathlib
import struct
import sys
import unittest
from types import SimpleNamespace
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from cloud_calibration_node import CalibratedCloud


class CallbackTests(unittest.TestCase):
    def test_publish_preserves_stamp_color_and_input(self):
        published=[]
        fake=SimpleNamespace(input_frame='camera_optical',target='fr3_link0',
                             rotation=np.eye(3),translation=np.array([1.,2.,3.]),
                             publisher=SimpleNamespace(publish=published.append))
        message=PointCloud2()
        message.header.frame_id='camera_optical'; message.header.stamp.sec=123; message.header.stamp.nanosec=456
        message.height=1; message.width=1; message.point_step=16; message.row_step=16
        message.fields=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1)
                        for i,name in enumerate(['x','y','z','rgb'])]
        message.data=struct.pack('<fffI',.1,.2,.3,0xaabbcc)
        CalibratedCloud.receive(fake,message)
        self.assertEqual(len(published),1)
        result=published[0]
        self.assertEqual(result.header.frame_id,'fr3_link0')
        self.assertEqual(result.header.stamp,message.header.stamp)
        np.testing.assert_allclose(struct.unpack_from('<fff',result.data),[1.1,2.2,3.3],atol=1e-6)
        self.assertEqual(result.data[12:],message.data[12:])
        self.assertEqual(message.header.frame_id,'camera_optical')
        self.assertEqual(bytes(message.data),struct.pack('<fffI',.1,.2,.3,0xaabbcc))

    def test_reject_wrong_frame(self):
        warnings=[]
        fake=SimpleNamespace(input_frame='camera_optical',get_logger=lambda:SimpleNamespace(warning=lambda text,**kw:warnings.append(text)))
        message=PointCloud2(); message.header.frame_id='fr3_link0'
        CalibratedCloud.receive(fake,message)
        self.assertEqual(len(warnings),1)


if __name__=='__main__':
    unittest.main()
