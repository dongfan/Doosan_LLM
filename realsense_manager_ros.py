#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
from sensor_msgs.msg import CameraInfo

class RealSenseManagerROS(Node):
    def __init__(self):
        super().__init__('realsense_manager_ros')
        self.bridge = CvBridge()
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.profile = self.pipeline.start(self.config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        info = CameraInfo()
        info.header.frame_id = "camera_link"
        info.height = 480
        info.width = 640
        info.k = [600.0, 0.0, 320.0,
                0.0, 600.0, 240.0,
                0.0, 0.0, 1.0]
        pub_info = self.create_publisher(CameraInfo, "/fuel/camera_info", 10)
        pub_info.publish(info)

        self.publisher_color = self.create_publisher(Image, '/fuel/realsense_color', 10)
        self.publisher_depth = self.create_publisher(Image, '/fuel/realsense_depth', 10)

        self.get_logger().info("✅ RealSense pipeline started")
        self.timer = self.create_timer(0.05, self.capture_and_publish)

    def capture_and_publish(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            self.get_logger().warn("⚠️ RealSense frame missing")
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        msg_color = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        msg_depth = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')

        self.publisher_color.publish(msg_color)
        self.publisher_depth.publish(msg_depth)

        return color_image, depth_image

    def release(self):
        self.pipeline.stop()
        self.get_logger().info("📷 RealSense pipeline stopped")

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = RealSenseManagerROS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 RealSenseManagerROS 종료됨 (Ctrl+C)")
    finally:
        node.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
