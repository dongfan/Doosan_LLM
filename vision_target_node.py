#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import json

from geometry_msgs.msg import PointStamped
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np

class VisionTargetNode(Node):
    def __init__(self, camera_source="realsense"):
        super().__init__('vision_target_node')
        self.bridge = CvBridge()
        self.depth_image = None  # 최근 프레임 저장용가
        self.camera_source = camera_source
        
        package_share = get_package_share_directory("dsr_example")
        # ✅ 각각 다른 모델 파일 경로 설정
        webcam_model_path = os.path.join(package_share, "weights", "car_detect.pt")
        realsense_model_path = os.path.join(package_share, "weights", "new_best_model.pt")

        # ✅ 모델 개별 로드
        self.model_webcam = YOLO(webcam_model_path)
        self.model_realsense = YOLO(realsense_model_path)
        
        # model_path = os.path.join(package_share, "weights", "nozzel_detect.pt")
        # self.model = YOLO(model_path)

        self.sub_webcam = self.create_subscription(Image, '/fuel/webcam_color', self.webcam_callback, 10)
        self.sub_realsense = self.create_subscription(Image, '/fuel/realsense_color', self.realsense_callback, 10)
        self.get_logger().info("✅ YOLO webcam detection node started")

        # ✅ RealSense 관련 구독 추
        self.sub_depth = self.create_subscription(Image, '/fuel/realsense_depth', self.depth_callback, 10)
        
        self.pub_result = self.create_publisher(Image, '/fuel/image_result', 10)
        self.pub_webcam_yolo = self.create_publisher(String, '/fuel/webcam_detections', 10)
        self.pub_realsense_yolo = self.create_publisher(String, '/fuel/realsense_detections', 10)
        self.pub_3d = self.create_publisher(PointStamped, '/fuel/object_3d', 10)
        self.pub_car_detected = self.create_publisher(String, '/car_detected', 10)

        self.car_detected_once = False

        # ✅ 카메라 intrinsic (640x480 기준, 필요시 캘리브레이션)
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

    def depth_callback(self, msg):
        """RealSense depth 이미지 최신 프레임 저장"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def webcam_callback(self, msg):
        self.process_frame(msg, "webcam")

    def realsense_callback(self, msg):
        self.process_frame(msg, "realsense")

    def process_frame(self, msg, source):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8').copy()

        # ✅ 입력 소스에 따라 다른 모델 선택
        model = self.model_webcam if source == "webcam" else self.model_realsense
        results = model.predict(frame, conf=0.6, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]
                detections.append({'cls': label, 'conf': conf, 'bbox': (x1, y1, x2, y2), 'source': source})

                # ✅ 차량 감지 여부 (웹캠일 때만)
                if source == "webcam":
                    car_found = any(det["cls"] in ["green_car", "yellow_car", "orange_car"] for det in detections)
                    if car_found:
                        self.car_detected_once = True
                        car_msg = String()
                        car_msg.data = "detected"
                        self.pub_car_detected.publish(car_msg)
                        self.get_logger().info("🚘 차량 감지됨 → /car_detected 퍼블리시")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # ✅ 중심 픽셀 좌표 계산
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # ✅ 깊이값 가져오기 (RealSense depth)
                depth_z = None
                if self.depth_image is not None:
                    if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                        depth_z = float(self.depth_image[cy, cx]) / 1000.0  # mm → m

                # ✅ 깊이 → 3D 좌표 변환 (realsense 전용)
                if source == "realsense" and self.depth_image is not None:
                    if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                        depth_z = float(self.depth_image[cy, cx]) / 1000.0
                        if depth_z > 0:
                            # 카메라 좌표계 기준 3D 좌표
                            X = (cx - self.cx) * depth_z / self.fx
                            Y = (cy - self.cy) * depth_z / self.fy
                            Z = depth_z

                            pt = PointStamped()
                            pt.header.stamp = self.get_clock().now().to_msg()
                            pt.header.frame_id = "camera_color_optical_frame"
                            pt.point.x, pt.point.y, pt.point.z = X, Y, Z
                            self.pub_3d.publish(pt)
                            self.get_logger().info(f"📍 Detected : ({X:.3f}, {Y:.3f}, {Z:.3f}) m")

                detections.append({
                    'cls': label, 'conf': conf, 'bbox': (x1, y1, x2, y2)
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                
        # ✅ YOLO가 그린 시각화(십자선 등)를 사용하지 않고, 직접 그린 frame만 퍼블리시
        self.pub_result.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        # YOLO 실행 결과 분기
        detections_json = json.dumps(detections)
        if self.camera_source == "webcam":
            self.pub_webcam_yolo.publish(String(data=detections_json))
        elif self.camera_source == "realsense":
            self.pub_realsense_yolo.publish(String(data=detections_json))
        else:
            self.get_logger().warn(f"⚠️ Unknown camera source: {self.camera_source}")   

    def overlay_heatmap(base_img, bbox, intensity=0.8, radius=60):
        """YOLO bbox 중심을 기준으로 heatmap 오버레이"""
        overlay = base_img.copy()
        heat = np.zeros_like(base_img, dtype=np.float32)
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        # 🔥 중심 기준 가우시안 생성
        for y in range(max(0, cy-radius), min(base_img.shape[0], cy+radius)):
            for x in range(max(0, cx-radius), min(base_img.shape[1], cx+radius)):
                d = np.sqrt((x-cx)**2 + (y-cy)**2)
                val = np.exp(- (d**2) / (2*(radius/2)**2))
                heat[y, x, 1] = val  # G 채널 강화 (시각적 집중)

        # normalize & 컬러맵 적용
        heatmap = (255 * heat / np.max(heat)).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap[:, :, 1], cv2.COLORMAP_JET)

        # 오버레이 합성
        cv2.addWeighted(heatmap, intensity, overlay, 1 - intensity, 0, overlay)
        return overlay

def main(args=None):
    rclpy.init(args=args)
    node = VisionTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 VisionTargetNode stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
