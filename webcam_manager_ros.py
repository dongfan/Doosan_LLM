import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import glob
import cv2

class WebcamManagerROS(Node):
    def __init__(self, ):
        super().__init__('webcam_manager_ros')
        
        device_index="/dev/v4l/by-id/usb-Generic_HD_camera-video-index0"
        # 🔍 장치 확인
        if not device_index:
            devices = glob.glob('/dev/video*')
            if not devices:
                raise RuntimeError("❌ No /dev/video* devices found.")
            device_index = devices[0]

        self.cap = cv2.VideoCapture(device_index)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/fuel/webcam_color', 10)

        # ✅ 해상도 직접 지정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Webcam (index={device_index}) not opened")

        # ✅ 20Hz 주기로 capture_and_publish() 실행
        self.timer = self.create_timer(0.05, self.capture_and_publish)
        self.get_logger().info(f"✅ Webcam opened (device={device_index})")

    def capture_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ Webcam frame not received")
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

    def release(self):
        self.cap.release()
        self.get_logger().info("📷 Webcam released")

def main(args=None):
    rclpy.init(args=args)
    node = WebcamManagerROS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 WebcamManagerROS 종료됨 (Ctrl+C)")
    finally:
        node.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
