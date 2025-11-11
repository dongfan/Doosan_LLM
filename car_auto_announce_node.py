#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gtts import gTTS
import tempfile, os, time
import torch, cv2
from PIL import Image as PILImage
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import pytesseract
from std_msgs.msg import String


class CarAutoAnnounceNode(Node):
    def __init__(self):
        super().__init__("car_auto_announce_node")
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"🧠 BLIP-2 로드 중 ({self.device})…")

        # BLIP-2 모델 로드
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-base")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-base"
        ).to(self.device)
        self.get_logger().info("✅ BLIP-2 로드 완료!")

        # 웹캠 이미지 구독
        self.sub_cam = self.create_subscription(
            Image, "/fuel/webcam_color", self.on_frame, 10
        )

        self.last_announce_time = 0.0
        self.cooldown = 5.0  # N초 간격마다 한 번만 안내
        self.get_logger().info("📸 차량 자동 인식·안내 모드 시작됨")

        self.pub_done = self.create_publisher(String, "/fuel/car_announce_done", 10)

    # ---------- 음성 출력 ----------
    def speak(self, text):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts = gTTS(text=text, lang="ko")
                tts.save(fp.name)
                os.system(f"mpg123 -q {fp.name}")
                os.unlink(fp.name)
        except Exception as e:
            self.get_logger().error(f"TTS 오류: {e}")

    # ---------- 프레임 수신 ----------
    def on_frame(self, msg):
        self.get_logger().info("📸 Frame received from /fuel/webcam_color")

        now = time.time()
        # ✅ 첫 프레임 무조건 통과
        if self.last_announce_time == 0.0:
            self.last_announce_time = now - self.cooldown - 1.0

        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        # ---------- 1️⃣ 번호판 OCR ----------
        ocr_text = pytesseract.image_to_string(cv_img, lang="eng+kor").strip()
        ocr_text = "".join(ocr_text.split())
        if len(ocr_text) < 3:
            ocr_text = None

        # ---------- 2️⃣ BLIP-2 차종 설명 ----------
        prompt = (
            "이 사진은 장난감 자동차입니다. "
            "이 장난감의 종류를 간단히 설명하고, 마지막에 어떤 차종인지 알려주세요."
        )
        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=40)
        desc = self.processor.decode(out[0], skip_special_tokens=True)

        # ---------- 3️⃣ 결과 조합 ----------
        if ocr_text:
            message = f"차량번호 {ocr_text}, {desc}"
        else:
            message = desc

        # ---------- 4️⃣ 안내 ----------
        self.last_announce_time = now
        self.get_logger().info(f"🚗 {message}")
        self.speak(message)
        
        # ✅ FuelTaskManager로 알림 퍼블리시
        msg = String()
        msg.data = message
        self.pub_done.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CarAutoAnnounceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
