#!/usr/bin/env python3
"""Publish left-arm cloud directly from camera or, optionally, from right-arm cloud."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from cloud_calibration_node import CalibratedCloud
from calibration_math import rotation_quaternion, invert_rigid_transform, load_calibration, camera_to_left_transform
from pathlib import Path


class ArmBaseCloud(Node):
    # Reuse validated XYZ conversion, preserving timestamp, color and layout.
    receive = CalibratedCloud.receive

    def __init__(self):
        super().__init__('arm_base_cloud')
        defaults = dict(input_topic='/cable_camera/cable_points_base',
                        output_topic='/cable_camera/cable_points_arm_a',
                        right_base_frame='fr3_link0', left_base_frame='a_fr3_link0',
                        left_origin_in_right=[0.47, 0.0, 0.0],
                        publish_base_tf=True, input_is_camera=False, calibration_file='',
                        camera_frame='cable_camera_color_optical_frame', input_is_optical=True)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        p = lambda name: self.get_parameter(name).value
        self.input_frame, self.target = p('right_base_frame'), p('left_base_frame')
        if not self.input_frame or not self.target or self.input_frame == self.target:
            raise ValueError('Left/right base frame names must be distinct and nonempty')
        if p('input_topic') == p('output_topic'):
            raise ValueError('Input/output topics must differ')
        origin = np.asarray(p('left_origin_in_right'), dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('left_origin_in_right must contain three finite meters')
        # B<-A: A origin at B +X, A X/Y reversed, common Z (180 deg about Z).
        rotation_b_a = np.diag([-1., -1., 1.])
        self.rotation, self.translation = invert_rigid_transform(rotation_b_a, origin)
        if p('input_is_camera'):
            rotation_b_c, translation_b_c, _, reference = load_calibration(
                Path(p('calibration_file')).expanduser(), p('input_is_optical'))
            if reference != p('right_base_frame'):
                raise ValueError('Calibration base frame differs from right_base_frame')
            self.rotation, self.translation = camera_to_left_transform(rotation_b_c, translation_b_c, origin)
            self.input_frame = p('camera_frame')
            if not self.input_frame or self.input_frame in (self.target, p('right_base_frame')):
                raise ValueError('Camera frame must differ from both arm bases')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.create_publisher(PointCloud2, p('output_topic'), qos)
        self.subscription = self.create_subscription(PointCloud2, p('input_topic'), self.receive, qos)
        if p('publish_base_tf'):
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id, tf.child_frame_id = p('right_base_frame'), self.target
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, origin)
            q = rotation_quaternion(rotation_b_a)
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
            self.broadcaster = StaticTransformBroadcaster(self)
            self.broadcaster.sendTransform(tf)
        self.get_logger().info(f'Cloud {self.input_frame} -> A cloud {self.target}; direct camera mode={p("input_is_camera")}; no robot commands')


def main():
    rclpy.init()
    node = None
    try:
        node = ArmBaseCloud()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
