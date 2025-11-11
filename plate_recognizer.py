#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, json, time
from collections import Counter
import pytesseract
import threading

class PlateRecognizer(Node):
    def __init__(self):
        super().__init__('plate_recognizer')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # 구독: 웹캠 영상 + YOLO 감지 결과
        self.sub_img = self.create_subscription(Image, '/fuel/webcam_color', self.cb_image, 10)
        self.sub_yolo = self.create_subscription(String, '/fuel/webcam_detections', self.cb_yolo, 10)
        
        # 퍼블리시: 인식된 차량번호
        self.pub_plate = self.create_publisher(String, '/fuel/license_plate', 10)

        # 설정값
        self.capture_count = 6        # 감지 시 캡처할 프레임 수
        self.capture_interval = 0.1   # 초 단위 (100ms)
        self.min_box_area = 6000      # 차량 최소 영역 제한

        self.get_logger().info("✅ PlateRecognizer node started (webcam mode)")

    # ----------------------------------------------------------------------
    def cb_image(self, msg):
        """웹캠 영상 갱신"""
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.lock:
            self.latest_frame = frame

    # ----------------------------------------------------------------------
    def cb_yolo(self, msg):
        """YOLO 감지 결과 구독 → 차량 감지되면 OCR 스레드 실행"""
        try:
            dets = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"⚠️ YOLO JSON 파싱 오류: {e}")
            return
        
        for det in dets:
            label = det.get('cls', '')
            bbox = det.get('bbox', None)
            if not bbox or label not in ["car", "green_car", "yellow_car", "orange_car"]:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            if (x2-x1)*(y2-y1) < self.min_box_area:
                continue
            
            # 차량 1대 감지되면 OCR 실행 (비동기)
            threading.Thread(target=self.capture_and_ocr, args=((x1, y1, x2, y2),), daemon=True).start()
            break  # 한 프레임당 한 차량만 처리

    # ----------------------------------------------------------------------
    def capture_and_ocr(self, bbox):
        """차량 bbox 기준으로 여러 프레임 캡처 후 OCR"""
        x1, y1, x2, y2 = bbox
        results = []

        for i in range(self.capture_count):
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(self.capture_interval)
                continue

            crop = frame[y1:y2, x1:x2]
            plate_crop = self.extract_plate_region(crop)
            text = self.run_ocr(plate_crop)
            if text:
                results.append(text)
            time.sleep(self.capture_interval)

        final_text = self.aggregate_results(results)
        if final_text:
            msg = String()
            msg.data = final_text
            self.pub_plate.publish(msg)
            self.get_logger().info(f"📛 차량 번호 인식: {final_text}")

    # ----------------------------------------------------------------------
    def extract_plate_region(self, car_crop):
        """차량 이미지에서 번호판 후보 추출 (단순히 아래쪽 ROI)"""
        h, w = car_crop.shape[:2]
        roi_y1 = int(h * 0.6)
        plate_crop = car_crop[roi_y1:h, 0:w]
        return plate_crop

    # ----------------------------------------------------------------------
    def run_ocr(self, img):
        """Tesseract OCR 실행"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (400, 150))
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = '--psm 7 -l kor+eng --oem 3'
        text = pytesseract.image_to_string(binary, config=config)
        text = text.strip().replace(" ", "").replace("\n", "")
        if len(text) > 2:
            return text
        return None

    # ----------------------------------------------------------------------
    def aggregate_results(self, results):
        """다수결로 최종 문자열 선택"""
        if not results:
            return None
        cnt = Counter(results)
        return cnt.most_common(1)[0][0]


def main(args=None):
    rclpy.init(args=args)
    node = PlateRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 PlateRecognizer 종료됨")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
