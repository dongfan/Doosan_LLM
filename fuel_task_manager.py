import cv2
import rclpy
from rclpy.node import Node
import time
import math

from std_msgs.msg import String
import json

from enum import Enum

import DR_init
from dsr_example.gripper_drl_controller import GripperController
from cv_bridge import CvBridge
from dsr_example.webcam_manager_ros import WebcamManagerROS
from dsr_example.realsense_manager_ros import RealSenseManagerROS

from gtts import gTTS
import tempfile
import os

# === [ADD] 음성인식/파싱 ===
import numpy as np
from openai import OpenAI
import threading

import re
try:
    import speech_recognition as sr  # 음성 → 텍스트
    HAS_SR = True
except Exception:
    HAS_SR = False

ROBOT_STATE = Enum('ROBOT_STATE',
                   ['IDLE',
                    'PARKING_CAR',
                    'MOVE_TO_FUEL_POS',
                    'APPROACH_FUEL_NOZZLE',
                    'GRIP_NOZZLE',
                    'LIFT_NOZZLE',
                    'MOVE_TO_CAR_FUEL_PORT',
                    'FUELING',
                    'RETURN_NOZZLE',
                    'RELEASE_NOZZLE',
                    'MOVE_TO_HOME_POS'])

CAR_TYPE = Enum('CAR_TYPE',
                ['orange_car',
                 'truck',
                 'yellow_car'])

detected_car_list = []

import DR_init
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

g_vel_move = 80
g_vel_rotate = 120

g_force_lift = 20.0

# --------------------- 주유 위치 좌표 (더미값) ---------------------#
# 1. 주유구 위치 : 더미 좌표
g_car1_posj = [-14, 20, 83, -47, 65, 28]
g_car2_posj = [500, 0, 300, 0, 0, 0]

# 2. 주유건 잡기 전 위치
# g_oil1_ready_posj = [-10, 63, 32, 91, 85, -104] 기존좌표
g_oil1_ready_posj = [22, 17, 91,-109, 29, -152] 
oil_middle_posj = [13, 7, 68,-29, 34, -250]

# 주유시작 위치 : 더미 좌표
g_fuel_car1_posx = [497, -333, 308, 49, 140, 159]
g_fuel_car2_posx = [496, -352, 293, 49, 140, 160]

# 주유 완료후 주유건 위치 : 더미 좌표
g_oil1_go_posj = [-9, 68, 22, 91, 88, -88]
g_oil2_go_posj = [-14, 65, 48, 87, 86, -123]
g_oil1_end_posj = [-9, 68, 22, 91, 88, -88]
g_oil2_end_posj = [-14, 65, 48, 87, 86, -123]
# -----------------------------------------------------------------#

g_find_car_posj = [0, 0, 0, 0, 0, 0]
g_find_nozzle_posj = [0, 0, 0, 0, 0, 0]

grip_shot = 600
grip_gun = 600

g_Cap_Grip_Off = 100
g_Cap_Grip_On = 300

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image as PILImage
import torch

class FuelTaskManager(Node):
    def __init__(self):
        super().__init__("fuel_task_manager")
        self.get_logger().info("🦾 로봇 제어 노드 초기화 중...")

        self.status_pub = self.create_publisher(String, '/fuel_status', 10)
        self.get_logger().info("🦾 FuelTaskManager started — waiting for /fuel_task/start")

        # 멀티모달 구현부분 ----------------------------
        self.pub_voice = self.create_publisher(String, '/speech_recognition/result', 10)

        self.tts_lock = threading.Lock()  # 🔒 음성 재생 중 동시 녹음 방지용 락
        self.first_car_detected = False
        # ✅ 차량 감지 → 바로 안내 및 음성인식
        self.sub_car_detected = self.create_subscription(
            String, '/car_detected', self.on_car_detected, 10
        )

        self.order_pub = self.create_publisher(String, '/fuel_task/order_info', 10)
        
        # --- Gripper 초기화 ---
        self.gripper = None
        try:
            from DSR_ROBOT2 import wait
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
        
            if not self.gripper.initialize():
                self.get_logger().error("Gripper initialization failed. Exiting.")
                raise Exception("Gripper initialization failed")
            
            self.get_logger().info("그리퍼를 활성화합니다...")
            self.gripper_is_open = True
            self.gripper.move(0)
            wait(2)
            
        except Exception as e:
            self.get_logger().error(f"An error occurred during gripper setup: {e}")
            rclpy.shutdown()

        # --- YOLO 객체 인식기 생성 ---
        self.bridge = CvBridge()
        # 상태 저장용
        self.tracked_objects = {}
        self.detected_car_list = []

        # === [ADD] BLIP 이미지 설명 모델 ===
        try:
            self.get_logger().info("🧠 BLIP 모델 로드 중...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.get_logger().info("✅ BLIP 모델 로드 완료")
        except Exception as e:
            self.get_logger().error(f"BLIP 모델 로드 실패: {e}")
            self.blip_processor = None
            self.blip_model = None

        # 20Hz 루프
        # self.timer = self.create_timer(0.05, self.timer_callback)
        # self.get_logger().info("📸 FuelTaskManager with YOLO initialized")

        self.current_state = ROBOT_STATE.IDLE

    def timer_callback(self):
        # YOLO 감지 결과 저장용
        webcam_detections = []
        realsense_detections = []
        realsense_masks = []

        # -------------------------------------------------------
        # 🎥 1️⃣ 웹캠 (Detection)
        # -------------------------------------------------------
        webcam_frame = self.webcam.capture_and_publish()  # 내부에서 /fuel/image_result 퍼블리시
        if webcam_frame is not None:
            webcam_detections = self.yolo.detect_objects(webcam_frame)
            if webcam_detections:
                annotated_webcam = self.yolo.draw_detections(webcam_frame, webcam_detections)
                self.get_logger().info(f"🎯 [Webcam] Detections: {webcam_detections}")

                # 웹캠 YOLO 결과 ROS로 퍼블리시
                img_msg = self.bridge.cv2_to_imgmsg(annotated_webcam, encoding="bgr8")
                self.pub_webcam_result.publish(img_msg)  # ✅ 퍼블리셔 이름 구분 추천 (/fuel/webcam_result)

        # -------------------------------------------------------
        # 🤖 2️⃣ 리얼센스 (Segmentation)
        # -------------------------------------------------------
        rs_color, rs_depth = self.realsense.get_latest_frames()
        if rs_color is not None:
            realsense_detections, realsense_masks = self.yolo.segment_objects(rs_color)
            # === [ADD] BLIP으로 장난감 차량 종류 감지 ===
            toy_type = self.describe_toy_car_with_blip(rs_color)
            self.get_logger().info(f"🧩 장난감 차량 종류 탐지: {toy_type}")
            
            if realsense_masks:
                annotated_rs = self.yolo.draw_masks(rs_color, realsense_masks)
                self.get_logger().info(f"🟦 [RealSense] Segmentations: {len(realsense_masks)} masks")

                # 세그멘테이션 결과 퍼블리시
                img_msg = self.bridge.cv2_to_imgmsg(annotated_rs, encoding="bgr8")
                self.pub_realsense_result.publish(img_msg)  # ✅ /fuel/realsense_result

            # 중심 깊이 정보 (거리)
            depth_mm = self.realsense.get_center_depth()
            if depth_mm:
                self.get_logger().info(f"📏 Center depth: {depth_mm:.1f} mm")

        # -------------------------------------------------------
        # 🧠 3️⃣ 웹캠에서 감지된 차량 추적 및 정지 판정
        # -------------------------------------------------------
        current_time = time.time()
        for det in webcam_detections:
            cls = det["cls"]
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if not hasattr(self, "tracked_objects"):
                self.tracked_objects = {}
            
            if cls not in self.tracked_objects:
                # 첫 감지
                self.tracked_objects[cls] = {
                    "cx": cx,
                    "cy": cy,
                    "start_time": current_time,
                }
            else:
                prev = self.tracked_objects[cls]
                dist = math.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                elapsed = current_time - prev["start_time"]

                # ✅ 3초 이상 고정 상태면 "차량 정지"로 판단
                if dist < 10 and elapsed >= 3.0:
                    self.get_logger().info(f"🟩 [Webcam] {cls} 정지 상태로 판단됨 (3초 이상 고정)")
                    # 👉 차량 정지 상태에서만 로봇 시퀀스 실행 가능
                    detected_car_list.append(det)

                    # 중복 실행 방지
                    self.tracked_objects[cls]["start_time"] = current_time

                # 위치 갱신
                self.tracked_objects[cls]["cx"] = cx
                self.tracked_objects[cls]["cy"] = cy

        for det in webcam_detections:
            cls = det["cls"]
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["conf"]

            # ROI 잘라내기
            car_crop = webcam_frame[y1:y2, x1:x2]
            if car_crop.size == 0:
                continue

            # BLIP 분석
            toy_desc = self.describe_toy_car(car_crop)
            self.get_logger().info(f"🚗 장난감 차량 종류: {toy_desc}")

            # (선택) 결과를 퍼블리시
            msg = String()
            msg.data = json.dumps({"label": cls, "type": toy_desc}, ensure_ascii=False)
            self.status_pub.publish(msg)
            
    def describe_toy_car_with_blip(self, frame_bgr):
        """리얼센스 프레임을 BLIP으로 분석"""
        if self.blip_processor is None or self.blip_model is None:
            return "unknown"

        try:
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(image_rgb)

            prompt = "A photo of a toy car. Describe its type (e.g., sports car, truck, SUV)."
            inputs = self.blip_processor(pil_img, prompt, return_tensors="pt")

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_new_tokens=50)

            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            self.get_logger().info(f"🚗 BLIP 결과: {caption}")

            msg = String()
            msg.data = caption
            self.pub_toycar.publish(msg)
            return caption
        except Exception as e:
            self.get_logger().warn(f"BLIP 분석 오류: {e}")
            return "error"
        
    def terminate_gripper(self):
        if self.gripper:
            try:
                print("🧹 Gripper 연결 종료 중...")
                if rclpy.ok():
                    self.gripper.terminate()
                    print("✅ Gripper 종료 완료")
                else:
                    print("⚠️ ROS context 종료됨 — terminate() 생략")
            except Exception as e:
                print(f"⚠️ 그리퍼 종료 중 오류: {e}")

    #--------------------- 초기화 부분 ---------------------#
    def robot_init(self):
        self.pos_init()
        self.grip_init()

        self.get_logger().info("Robot 초기화 완료.")

    def pos_init(self):
        from DSR_ROBOT2 import movej, posj, wait
        p_start = posj(0, 0, 90, 0, 90, 0)
        movej(p_start, 70, 70)
        wait(2.0)

    def grip_init(self):
        from DSR_ROBOT2 import wait
        self.gripper.move(0)
        wait(2.0)

    def orient_z_down(self):
        from DSR_ROBOT2 import (get_current_posx, movel, wait, DR_MV_MOD_ABS)
        from DR_common2 import posx

        c_pos = get_current_posx()
        x, y, z = c_pos[0][0:3]
        target_pos = posx(x, y, z, 0, 180, 0)

        movel(target_pos, v=50, a=50, mod=DR_MV_MOD_ABS)
        wait(3)

    # 음성, 멀티모달 관련 부분 --------------------------------
    def interpret_with_llm(self, text: str):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # ✅ 직접 LLaMA 클래스 로드 (버전 호환 문제 해결)
            try:
                import transformers.models.llama.modeling_llama
            except ImportError:
                self.get_logger().warn("LLaMA 모듈 수동 등록 시도 중...")
                import importlib
                importlib.import_module("transformers.models.llama.modeling_llama")

            import torch, transformers
            self.get_logger().info(f"🤖 transformers v{transformers.__version__} @ {transformers.__file__}")

            if not hasattr(self, "llama_model"):
                model_name = "distilgpt2"
                self.get_logger().info("🦙 LLaMA 모델 로드 중... (최초 1회)")
                self.llama_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=torch.float16, device_map="auto"
                )
                self.get_logger().info("✅ LLaMA 모델 로드 완료")

            prompt = (
                "사용자의 문장에서 유종(fuel_type: 휘발유 또는 경유)과 금액(amount)을 추출해줘.\n"
                "금액은 원 단위 정수로. JSON만 출력. 예: {\"fuel_type\":\"휘발유\",\"amount\":50000}\n"
                f'문장: "{text}"\n'
            )

            inputs = self.llama_tokenizer(prompt, return_tensors="pt").to(self.llama_model.device)
            output = self.llama_model.generate(**inputs, max_new_tokens=100)
            resp = self.llama_tokenizer.decode(output[0], skip_special_tokens=True)

            # JSON 부분만 추출
            import json
            js = "{" + resp.split("{", 1)[-1].split("}", 1)[0] + "}"
            data = json.loads(js)

            fuel_type = data.get("fuel_type")
            amount = data.get("amount")
            self.get_logger().info(f"🤖 LLaMA 해석 결과: fuel_type={fuel_type}, amount={amount}")
            return fuel_type, amount

        except Exception as e:
            # 어떤 이유로든 실패하면 바로 None 반환 -> 상위에서 정규식 파싱 fallback
            self.get_logger().warn(f"LLaMA 해석 실패 (파싱으로 fallback): {e}")
            return None, None
        
    def speak_text(self, text):
        try:
            with self.tts_lock:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    tts = gTTS(text=text, lang='ko')
                    tts.save(fp.name)
                    os.system(f'mpg123 -q {fp.name} 2>/dev/null')
                    os.unlink(fp.name)
                time.sleep(1.0)  # 🔹 TTS 버퍼가 완전히 끝난 뒤 STT 시작 대기
        except Exception as e:
            self.get_logger().error(f"TTS 오류: {e}")
    
    def describe_toy_car(self, image_bgr):
        """YOLO로 감지된 ROI 이미지를 BLIP으로 분석"""
        if self.blip_processor is None or self.blip_model is None:
            return "unknown"

        import torch
        from PIL import Image

        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            prompt = "A toy car type, for example sports car, truck, or SUV:"
            inputs = self.blip_processor(pil_image, prompt, return_tensors="pt")

            out = self.blip_model.generate(**inputs)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

            self.get_logger().info(f"🚙 BLIP 결과: {caption}")
            return caption
        except Exception as e:
            self.get_logger().warn(f"BLIP 분석 오류: {e}")
            return "unknown"
        
    # === [ADD] 번호판을 낱자로 읽히게 가공 (예: "69오 6665" → "6 9 오 6 6 6 5")
    def _spellout_plate(self, plate_raw: str) -> str:
        return " ".join(plate_raw.replace(" ", ""))
    
    # === [ADD] 음성 인식 (Google Speech API)
    def recognize_speech(self, max_retries=3):
        if not HAS_SR:
            self.get_logger().warn("SpeechRecognition(pyaudio) 미설치 → 음성입력 생략")
            return None

        recognizer = sr.Recognizer()

        for attempt in range(max_retries):
            self.get_logger().info(f"🎤 ({attempt+1}/{max_retries}) 유종과 금액을 말씀해주세요. (10초 제한)")

            # 🔇 음성 안내 후 잠시 대기 (TTS 완료 대기용)
            self.speak_text("유종과 금액을 말씀해주세요.")
            time.sleep(1.0)

            try:
                with sr.Microphone() as source:
                    self.get_logger().info("🎧 음성 대기 중... (10초 제한)")
                    recognizer.adjust_for_ambient_noise(source, duration=0.6)
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                
                # 🔹 Google API로 인식
                text = recognizer.recognize_google(audio, language='ko-KR')
                text = text.strip()
                self.get_logger().info(f"✅ 인식 결과: {text}")
                return text

            except sr.WaitTimeoutError:
                self.get_logger().warn("🕒 10초 동안 음성이 감지되지 않았습니다.")
            except sr.UnknownValueError:
                self.get_logger().warn("음성을 인식하지 못했습니다.")
            except sr.RequestError as e:
                self.get_logger().error(f"Speech API 오류: {e}")
                break
            except Exception as e:
                self.get_logger().error(f"예상치 못한 오류: {e}")
                break

            # 🔁 실패 시 한 번만 안내 멘트
            self.speak_text("다시 말씀해주세요.")
            time.sleep(1.0)

        self.speak_text("음성을 인식하지 못했습니다. 수동으로 입력해주세요.")
        return None

    # === [ADD] 유종/금액 파싱 (휘발유/경유만, 1~15만 원, 1만 단위)
    def parse_fuel_order(self, text: str):
        if not text:
            return None, None

        fuel_type = None
        amount_10k = None

        # ---------- 1️⃣ 유종 판별 ----------
        if "휘발유" in text:
            fuel_type = "휘발유"
        elif "경유" in text:
            fuel_type = "경유"

        # ---------- 2️⃣ 금액 판별 ----------
        # (a) 숫자 표기: "5만원", "10만", "12만 원"
        m = re.search(r'(\d+)\s*(?:만|만원|원)', text)
        if m:
            value = int(m.group(1))
            # 10,000 단위면 만원단위로 처리
            if value >= 10000:
                amount_10k = value // 10000
            else:
                amount_10k = value  # 나중에 범위 검증에서 걸러짐

        # (b) 한국어 표기: "오만원", "십이만원", "십오만"
        if amount_10k is None:
            kr_map = {"일":1,"이":2,"삼":3,"사":4,"오":5,"육":6,"륙":6,"칠":7,"팔":8,"구":9,"영":0,"공":0}
            if "십" in text or "열" in text:
                base = 10
                tail = 0
                m2 = re.search(r'(십|열)\s*([일이삼사오육륙칠팔구])?', text)
                if m2 and m2.group(2):
                    tail = kr_map.get(m2.group(2), 0)
                amount_10k = base + tail
            else:
                m3 = re.search(r'([일이삼사오육륙칠팔구])\s*만', text)
                if m3:
                    amount_10k = kr_map.get(m3.group(1), None)

        # ---------- 3️⃣ 개별 유효성 검사 ----------
        if fuel_type is None and amount_10k is None:
            self.speak_text("유종과 금액을 다시 말씀해주세요. 휘발유 또는 경유, 그리고 만원 단위로 말씀해주세요.")
            return None, None

        if fuel_type is None:
            self.speak_text("유종을 다시 말씀해주세요. 휘발유 또는 경유만 가능합니다.")
            return None, None

        # 금액 미검출 또는 잘못된 단위(예: 300, 5000, 300원 등)
        if amount_10k is None or amount_10k < 1:
            self.speak_text("금액을 만원 단위로 말씀해주세요.")
            return None, None

        # 1~15만 원 범위 검증
        if not (1 <= amount_10k <= 15):
            self.speak_text("금액은 1만원 이상 15만원 이하만 가능합니다.")
            return None, None

        # ---------- 4️⃣ 결과 반환 ----------
        amount = amount_10k * 10000
        return fuel_type, amount

    # 콜백 차량인식 부분 -------------------------------------------
    def on_car_detected(self, msg: String):
        self.get_logger().info("🎤 차량이 인식 되었 습 니 다")

        total_attempts = 5
        for attempt in range(total_attempts):
            self.get_logger().info(f"🎙️ 전체 시도 ({attempt+1}/{total_attempts})")

            text = self.recognize_speech()
            if not text:
                self.speak_text("다시 말씀해주세요.")
                continue

            # 1️⃣ LLM 기반 파싱 시도
            fuel_type, amount = self.interpret_with_llm(text)

            # 2️⃣ LLM이 항상 기본값을 반환하는 문제 → 실제 발화 내용 검증
            if fuel_type == "휘발유" and "휘발유" not in text:
                self.get_logger().warn("⚠️ LLM이 기본값(휘발유)을 반환 — 실제 텍스트에 없음 → 정규식 파싱 재시도")
                fuel_type, amount = self.parse_fuel_order(text)
            elif fuel_type == "경유" and "경유" not in text:
                self.get_logger().warn("⚠️ LLM이 기본값(경유)을 반환 — 실제 텍스트에 없음 → 정규식 파싱 재시도")

            # 3️⃣ 금액 검증 (만원 단위 단어 또는 숫자 확인)
            if not amount or (("만" not in text) and not re.search(r'\d+', text)):
                self.get_logger().warn("⚠️ LLM이 잘못된 금액을 반환 — 정규식 파싱 재시도")
                fuel_type, amount = self.parse_fuel_order(text)

            # 4️⃣ 정규식 + LLM 결과 확인
            if not (fuel_type and amount):
                self.speak_text("휘발유 또는 경유, 그리고 금액을 다시 말씀해주세요.")
                return
            
            # ✅ 성공 시 로직
            confirm_msg = f"주문이 확인되었습니다. {fuel_type} {amount//10000}만원 주유를 시작하겠습니다."
            self.get_logger().info(f"✅ {confirm_msg}")
            self.speak_text(confirm_msg)

            self.status_pub.publish(String(data="fueling_started"))
            self.start_simulation_fuel(amount)

            order_info = {"fuel_type": fuel_type, "amount": amount}
            msg = String()
            msg.data = json.dumps(order_info, ensure_ascii=False)
            self.order_pub.publish(msg)
            self.get_logger().info(f"🛰️ 주문 정보 퍼블리시됨 → /fuel_task/order_info : {msg.data}")
            break  # ✅ 성공 시 루프 탈출

        else:
            self.speak_text("음성을 인식하지 못했습니다. 수동 입력으로 전환합니다.")

    #--------------------- Set Position Callback ---------------------#
    # 주유건이 충돌했는지 확인하고 대응하는 함수
    def check_crash(self):
        from DSR_ROBOT2 import (task_compliance_ctrl, set_desired_force, get_tool_force,
            release_force, release_compliance_ctrl, amovel, wait, DR_MV_MOD_REL)
        from DR_common2 import posx
        
        k_d = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
        task_compliance_ctrl(k_d)
        # 강성 제어
        f_d = [0.0, 0.0, -20, 0.0, 0.0, 0.0]
        f_dir = [0, 0, 1, 0, 0, 0]
        set_desired_force(f_d, f_dir)
        wait(2.0)

        # 외력감지
        while True:
            force_ext = get_tool_force()
            # c_pos = get_current_posx()
            # x, y, z = c_pos[0]
            if force_ext[2] > 4:
                release_force()
                release_compliance_ctrl()

                self.gripper.move(0)
                wait(1.5)
                break
    
    # 주유구를 오픈하기 위해 그리퍼를 회전시키는 함수
    def rotate_grip(self, cnt: int, b_open: bool = True):
        from DSR_ROBOT2 import (amovel, DR_MV_MOD_REL,
            movel, movej, wait)
        from DR_common2 import posx, posj
        count = 0
        open_angle = -120 if b_open else 120

        if b_open:
            while count < cnt :
                self.gripper.move(g_Cap_Grip_On)
                wait(1.5)
                
                movej(posj(0, 0, 0, 0, 0, open_angle), v=g_vel_rotate, a=g_vel_rotate, mod=DR_MV_MOD_REL)
                wait(1.0)
                count = count + 1

                if count < cnt:
                    self.gripper.move(g_Cap_Grip_Off)
                    wait(1.5)
                    movej(posj(0, 0, 0, 0, 0, -open_angle), v=g_vel_rotate, a=g_vel_rotate, mod=DR_MV_MOD_REL)
                    wait(1.0)

        else:
            while count < cnt :
                self.gripper.move(g_Cap_Grip_On)
                wait(1.0)
                
                movej(posj(0, 0, 0, 0, 0, open_angle), v=g_vel_rotate, a=g_vel_rotate, mod=DR_MV_MOD_REL)
                wait(1.0)
                count = count + 1

                if count < cnt:
                    self.gripper.move(g_Cap_Grip_Off)
                    wait(1.5)
                    movej(posj(0, 0, 0, 0, 0, -open_angle), v=g_vel_rotate, a=g_vel_move, mod=DR_MV_MOD_REL)
                    wait(1.0)
    
    # 반복적으로 그리퍼를 열고 닫는 작업을 수행 : 주유 시작       
    def run_fuel_task(self, cnt):
        from DSR_ROBOT2 import wait
        try:
            for i in range(cnt):
                self.gripper.move(grip_shot)
                wait(1.0)
                self.gripper.move(grip_gun)
                wait(1.0)

        except Exception as e:
            self.get_logger().error(f"❌ Gripper 반복 동작 중 오류: {e}")  

    def start_simulation_fuel(self, amount):
        try:
            from DSR_ROBOT2 import get_current_posx, get_current_posj, movej, movel, wait, DR_MV_MOD_REL, DR_MV_MOD_ABS
            from DR_common2 import posx, posj
        except ImportError as e:
            print(f"DSR_ROBOT2 라이브러리를 임포트할 수 없습니다: {e}")
            rclpy.shutdown()
            exit(1)

        # 🔧 실제 로봇 주유 동작 시퀀스 작성
        self.get_logger().info(f"⛽ Gasoline fueling sequence for {amount}원 started...")
        
        if amount < 20000:
            amount = 20000 # 최소 1회 주유
        
        m_count = amount // 20000  # 30000원 단위로 주유 횟수 결정

        # gun_posj = get_current_posj()

        # 로봇 위치, 그리퍼 초기화
        self.robot_init()

        #--------------------- 차량 진입 후 작업 시작 ---------------------#
        # 주유구 위치로 이동 
        movej(g_car1_posj, 80, 80, DR_MV_MOD_ABS)
        wait(2.0)

        # 주유구 뚜껑 잡으러 이동 -> 오픈을 위한 그리퍼 회전
        movel(posx(0, -28, -18, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        self.rotate_grip(2, True)
        
        movel(posx(0, 30, 30, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(1.0)
        movel(posx(0, 100, -50, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(1.0)
        self.orient_z_down()
        cap_pick_posj = get_current_posj()
        self.check_crash()

        # 주유건 위치로 이동 후 그리퍼 닫기
        movej(g_oil1_ready_posj, 100, 100)
        wait(2.0)

        movel(posx(100, -10, 0, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(2.0)
        
        g_oil1_end_posj = get_current_posj()
        self.gripper.move(grip_gun)
        wait(2.5)

        # 주유건 그립 이후 주유건 뽑아 가기
        movel(posx(0, 0, 120, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(2.0)
        g_oil1_go_posj = get_current_posj()
        movel(posx(-100, -120, 150, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(2.0)

        # 주유구 위치로 이동 
        movel(g_fuel_car1_posx, 80, 80)
        wait(3.0)
        movel(g_fuel_car2_posx, 80, 80)
        # movel(posx(0, -75, -65, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(2.0)

        #--------------------- 직접 주유 작업 시작 ---------------------#
        # 주유 작업 반복 수행
        # self.run_fuel_task(m_count)
        wait(4.0)
        
        # (car)주유구에서 주유건 빼기
        movel(posx(0, 75, 65, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(2.0)

        movel(posx(0, 55, 150, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(1.0)

        movej(g_oil1_go_posj, 80, 80)
        wait(2.0)
        movel(posx(0, 0, -80, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        # movej(g_oil1_end_posj, 80, 80)
        wait(2.0)

        self.grip_init()
        wait(2.0)

        movej(g_oil1_ready_posj, 80, 80)
        wait(2.0)

        movej(cap_pick_posj, 80, 80)
        wait(2.0)
        self.gripper.move(g_Cap_Grip_On)
        wait(2.0)
        self.check_crash()

        movel(posx(0, 0, -10, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        wait(2.0)
        self.gripper.move(g_Cap_Grip_On)
        wait(2.0)

        movej(g_car1_posj, 80, 80)
        wait(2.0)

        # # 주유구 뚜껑 닫으러 이동 
        movel(posx(-10, -38, -25, 0, 0, 0), v=g_vel_move, a=g_vel_move, mod=DR_MV_MOD_REL)
        self.rotate_grip(2, False)
        self.grip_init()

        self.pos_init()

        # 주유 완료 시:
        self.current_state = ROBOT_STATE.IDLE
        self.status_pub.publish(String(data="completed"))
        self.get_logger().info("✅ Fueling completed.")
    
    def start_diesel_fuel(self, amount):
        self.get_logger().info(f"⛽ Diesel fueling sequence for {amount}원 started...")

    def start_window_cleaning(self):
        print("🚿 Starting window cleaning process...")

def main(args=None):
    # ✅ 1️⃣ ROS 초기화 먼저
    rclpy.init(args=args)

    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    # ✅ 3️⃣ FuelTaskManager 생성 (이제 Node 생성 가능)
    fuel_controller = FuelTaskManager()
    # fuel_controller.start_simulation_fuel(20000)

    try:
        while rclpy.ok():
            rclpy.spin_once(fuel_controller, timeout_sec=0.05)

    except KeyboardInterrupt:
        print("🛑 Keyboard Interrupt 감지됨, 로봇 정지 중...")
        fuel_controller.terminate_gripper() 
        pass

    finally:
        try:
            fuel_controller.terminate_gripper()
            fuel_controller.destroy_node()
            dsr_node.destroy_node()
        except Exception:
            print("⚠️ Node 종료 중 오류 무시")

        # 3️⃣ ROS context 마지막에 shutdown
        if rclpy.ok():
            rclpy.shutdown()

        print("✅ 종료 완료.")

if __name__ == '__main__':
    main()