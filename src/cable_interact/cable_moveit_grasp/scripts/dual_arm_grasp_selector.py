#!/usr/bin/env python3
"""RViz arm buttons route each cloud click to exactly one planning-only node."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer


class GraspSelector(Node):
    def __init__(self):
        super().__init__('dual_arm_grasp_selector')
        self.arm = 'a'
        self.declare_parameter('button_frame', 'fr3_link0')
        self.declare_parameter('button_position', [0.23, -0.12, 1.05])
        self.selected_pub = self.create_publisher(String, '/dual_arm/selected_grasp_arm',
                                                  QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.click_pubs = {arm: self.create_publisher(PointStamped, '/dual_arm/planner_' + arm + '/clicked_point', 10)
                           for arm in ('a', 'b')}
        self.click_sub = self.create_subscription(PointStamped, '/clicked_point', self.clicked, 10)
        self.server = InteractiveMarkerServer(self, '/dual_arm/grasp_arm_selector')
        self.selection_services = [self.create_service(Trigger, '~/select_' + arm, self.select_service(arm))
                                   for arm in ('a', 'b')]
        self.refresh()

    def select_service(self, arm):
        def callback(request, response):
            self.select(arm)
            response.success = True
            response.message = 'Selected ' + arm.upper() + '; click a fresh point in RViz'
            return response
        return callback

    def select(self, arm):
        if self.arm != arm:
            self.arm = arm
            self.refresh()
        self.get_logger().info('Selected ' + arm.upper() + '; next cloud click routes to this arm (no motion)')

    def clicked(self, message):
        self.click_pubs[self.arm].publish(message)
        self.get_logger().info('Cloud point forwarded to ' + self.arm.upper() + ' candidate planner')

    def refresh(self):
        self.selected_pub.publish(String(data=self.arm))
        position = self.get_parameter('button_position').value
        self.server.clear()
        for index, arm in enumerate(('a', 'b')):
            button = InteractiveMarker()
            button.header.frame_id = self.get_parameter('button_frame').value
            button.name = 'select_' + arm
            button.description = ('A LEFT' if arm == 'a' else 'B RIGHT') + (' [SELECTED]' if arm == self.arm else '')
            # Swap the two existing slots: +X is screen-left in the configured RViz view.
            button.pose.position.x = float(position[0]) + (1 - index) * 0.16
            button.pose.position.y = float(position[1])
            button.pose.position.z = float(position[2])
            button.pose.orientation.w = 1.0
            button.scale = 0.1
            control = InteractiveMarkerControl()
            control.interaction_mode = InteractiveMarkerControl.BUTTON
            control.always_visible = True
            marker = Marker()
            marker.type = Marker.CUBE
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = 0.12, 0.04, 0.04
            marker.color.a = 1.0
            marker.color.r = 0.1 if arm == self.arm else 0.4
            marker.color.g = 0.9 if arm == self.arm else 0.4
            marker.color.b = 0.3 if arm == self.arm else 0.4
            control.markers.append(marker)
            button.controls.append(control)
            self.server.insert(button, feedback_callback=self.feedback)
        self.server.applyChanges()

    def feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            self.select(feedback.marker_name.removeprefix('select_'))


def main():
    rclpy.init()
    node = GraspSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
