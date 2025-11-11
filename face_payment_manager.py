#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FacePaymentManager (RealSense-only, no Flutter)
- 사용자가 말한 유종/금액을 인식 → Face ID(얼굴 인증) → 결제 승인 → 주문 정보 퍼블리시

토픽/흐름
- sub : /car_detected (String "detected")  ← 차량이 들어오면 트리거
- pub : /fuel_task/order_info (String JSON: {"fuel_type","amount","user_id","auth"})
- pub : /fuel_status (String: waiting/face_auth/paid/failed)
- (선택) /payment_status (String: approved/declined)

필수 패키지
pip install deepface opencv-python pyrealsense2 SpeechRecognition gTTS
sudo apt-get install mpg123  # gTTS 재생기
"""

import os
import re
import json
import time
import queue
import tempfile
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ====== RealSense ======
import pyrealsense2 as rs

# ====== Face ======
from deepface import DeepFace

# ====== Speech & TTS ======
try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:
    HAS_SR = False

from gtts import gTTS

# ---------- 설정 ----------
FACES_DIR = os.path.expanduser("~/faces")     # 등록된 얼굴 임베딩 저장 경로 (*.npy)
VERIFY_MODEL = "Facenet"                      # DeepFace 모델
DIST_THR = 0.90                               # 임계값 (낮을수록 엄격)
DEPTH_MIN_MM = 300                            # 라이브니스: 최소 거리
DEPTH_MAX_MM = 1500                           # 라이브니스: 최대 거리
MIC_TIMEOUT = 10                              # 음성 입력 타임아웃(초)
MIC_PHRASE_LIMIT = 10                         # 음성 구간 최대 길이(초)

# ---------- 유틸 ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_face_embeddings():
    ensure_dir(FACES_DIR)
    return [f for f in os.listdir(FACES_DIR) if f.endswith(".npy")]

# ---------- 노드 ----------
class FacePaymentManager(Node):
    def __init__(self):
        super().__init__("face_payment_manager")
        self.get_logger().info("🧾 FacePaymentManager started (RealSense-only)")

        # pubs
        self.order_pub = self.create_publisher(String, "/fuel_task/order_info", 10)
        self.status_pub = self.create_publisher(String, "/fuel_status", 10)
        self.payment_pub = self.create_publisher(String, "/payment_status", 10)

        # subs
        self.sub_car = self.create_subscription(String, "/car_detected", self.on_car_detected, 10)

        # TTS lock
        self.tts_lock = threading.Lock()

        # Speech recognizer
        self.recognizer = sr.Recognizer() if HAS_SR else None

        # pre-check
        if not HAS_SR:
            self.get_logger().warn("⚠️ SpeechRecognition 미설치 → 음성 명령 불가")
        if len(list_face_embeddings()) == 0:
            self.get_logger().warn(f"⚠️ 등록된 얼굴이 없습니다. 먼저 ~/faces/*.npy를 생성하세요.")

        self.get_logger().info("✅ 준비 완료 — /car_detected 수신 시 절차 시작")

    # ---------- 메인 플로우 ----------
    def on_car_detected(self, msg: String):
        self.status_pub.publish(String(data="waiting"))
        self.speak("차량이 감지되었습니다. 유종과 금액을 말씀해주세요. 예: 휘발유 5만원")

        # 1) 음성 → 텍스트
        text = self.recognize_speech(max_retries=3)
        if not text:
            self.speak("음성을 인식하지 못했습니다. 다시 시도해주세요.")
            self.status_pub.publish(String(data="failed"))
            return

        # 2) 파싱
        fuel_type, amount = self.parse_fuel_order(text)
        if not (fuel_type and amount):
            self.speak("유종과 금액을 정확히 말씀해주세요. 예: 휘발유 5만원, 경유 10만")
            self.status_pub.publish(String(data="failed"))
            return

        self.get_logger().info(f"🧠 파싱 성공: fuel_type={fuel_type}, amount={amount}")
        self.speak(f"{fuel_type} {amount//10000}만원으로 결제를 진행합니다. 잠시만요.")
        self.status_pub.publish(String(data="face_auth"))

        # 3) Face ID (RealSense RGB + Depth)
        verified, user_id = self.verify_face_with_liveness()
        if not verified:
            self.speak("얼굴 인증에 실패했습니다. 직원에게 문의해주세요.")
            self.status_pub.publish(String(data="failed"))
            self.payment_pub.publish(String(data="declined"))
            return

        # 4) 결제 승인 (모의)
        self.speak(f"{user_id} 님, 얼굴 인증이 완료되었습니다. 결제를 승인합니다.")
        self.payment_pub.publish(String(data="approved"))
        self.status_pub.publish(String(data="paid"))

        # 5) 주문 정보 퍼블리시 → 기존 파이프라인 연결
        order_info = {
            "fuel_type": fuel_type,
            "amount": amount,
            "user_id": user_id,
            "auth": "face_verified"
        }
        out = String()
        out.data = json.dumps(order_info, ensure_ascii=False)
        self.order_pub.publish(out)
        self.get_logger().info(f"🛰️ /fuel_task/order_info 퍼블리시: {out.data}")

        # (선택) 이후 자동 시퀀스 트리거가 필요하면 /fuel_task/start 퍼블리시 추가 가능
        # start_msg = String()
        # start_msg.data = json.dumps({"orderId": "AUTO", **order_info}, ensure_ascii=False)
        # self.create_publisher(String, "/fuel_task/start", 10).publish(start_msg)

    # ---------- 음성 ----------
    def speak(self, text: str):
        try:
            with self.tts_lock:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    gTTS(text=text, lang="ko").save(fp.name)
                    os.system(f"mpg123 -q {fp.name} 2>/dev/null")
                    os.unlink(fp.name)
                time.sleep(0.3)
        except Exception as e:
            self.get_logger().error(f"TTS 오류: {e}")

    def recognize_speech(self, max_retries=3):
        if not HAS_SR or self.recognizer is None:
            return None

        for attempt in range(1, max_retries+1):
            self.get_logger().info(f"🎤 ({attempt}/{max_retries}) 유종과 금액 발화 대기...")
            self.speak("유종과 금액을 말씀해주세요.")
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                    audio = self.recognizer.listen(source, timeout=MIC_TIMEOUT, phrase_time_limit=MIC_PHRASE_LIMIT)
                text = self.recognizer.recognize_google(audio, language="ko-KR").strip()
                self.get_logger().info(f"✅ 인식: {text}")
                return text
            except sr.WaitTimeoutError:
                self.get_logger().warn("🕒 시간 초과")
            except sr.UnknownValueError:
                self.get_logger().warn("🤷 음성을 인식하지 못했습니다.")
            except sr.RequestError as e:
                self.get_logger().error(f"Speech API 오류: {e}")
                break
            except Exception as e:
                self.get_logger().error(f"예상치 못한 오류: {e}")
                break

            self.speak("다시 말씀해주세요.")
            time.sleep(0.5)

        return None

    # ---------- 파싱 (정규식) ----------
    def parse_fuel_order(self, text: str):
        if not text:
            return None, None

        fuel_type = None
        amount_10k = None

        # 유종
        if "휘발유" in text:
            fuel_type = "휘발유"
        elif "경유" in text or "디젤" in text:
            fuel_type = "경유"

        # 금액: 숫자/만/만원/원
        m = re.search(r'(\d+)\s*(?:만|만원|원)?', text)
        if m:
            val = int(m.group(1))
            amount_10k = val if val < 10000 else val // 10000

        # 한국어 수사
        if amount_10k is None:
            kr = {"일":1,"이":2,"삼":3,"사":4,"오":5,"육":6,"륙":6,"칠":7,"팔":8,"구":9,"영":0,"공":0}
            if "십" in text or "열" in text:
                tail = 0
                m2 = re.search(r'(십|열)\s*([일이삼사오육륙칠팔구])?', text)
                if m2 and m2.group(2): tail = kr.get(m2.group(2), 0)
                amount_10k = 10 + tail
            else:
                m3 = re.search(r'([일이삼사오육륙칠팔구])\s*만', text)
                if m3: amount_10k = kr.get(m3.group(1))

        # 검증
        if fuel_type is None:
            return None, None
        if amount_10k is None or amount_10k < 1 or amount_10k > 15:
            return None, None

        return fuel_type, amount_10k * 10000

    # ---------- Face + Liveness ----------
    def verify_face_with_liveness(self):
        """RealSense로 RGB/Depth 동시 캡처 → 얼굴 임베딩 비교 + Depth 범위 확인"""
        # 등록된 임베딩 확인
        embeds = list_face_embeddings()
        if len(embeds) == 0:
            self.get_logger().error("등록된 얼굴이 없습니다. ~/faces/ 에 *.npy 파일을 추가하세요.")
            return False, None

        # RealSense 파이프라인
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            pipeline.start(config)
            # 버퍼 프레임 소거
            for _ in range(10):
                pipeline.wait_for_frames()

            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                self.get_logger().error("RealSense 프레임 획득 실패")
                return False, None

            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())

            # 중심부 평균 깊이로 간단한 라이브니스 체크
            h, w = depth.shape
            region = depth[h//3:2*h//3, w//3:2*w//3]
            center_depth = float(np.mean(region))
            self.get_logger().info(f"📏 얼굴 거리(평균, mm): {center_depth:.1f}")
            if not (DEPTH_MIN_MM <= center_depth <= DEPTH_MAX_MM):
                self.get_logger().warn("⚠️ 라이브니스 실패 (거리 비정상)")
                return False, None

            # DeepFace 임베딩
            emb = DeepFace.represent(color, model_name=VERIFY_MODEL)[0]["embedding"]
            emb = np.array(emb, dtype=np.float32)

            # 최적 매칭
            best_id, best_dist = None, 999.0
            for fname in embeds:
                uid = os.path.splitext(fname)[0]
                stored = np.load(os.path.join(FACES_DIR, fname)).astype(np.float32)
                dist = np.linalg.norm(emb - stored)
                if dist < best_dist:
                    best_dist, best_id = dist, uid

            self.get_logger().info(f"👤 best={best_id}, dist={best_dist:.4f}")
            if best_dist < DIST_THR:
                return True, best_id
            else:
                return False, None

        except Exception as e:
            self.get_logger().error(f"RealSense/Face 오류: {e}")
            return False, None
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = FacePaymentManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
