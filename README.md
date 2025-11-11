# doosan_llm
음성, 멀티모달, LLM 추가 버젼

🧠 SmartFuel Robot – ROS2 자동 주유 시스템 (BLIP)
📘 개요

이 프로젝트는 ROS 2 기반 자동 주유 로봇 시스템으로,
카메라 인식 → 차량 감지 → 번호판 인식 → 음성 결제 인증 → 로봇 주유 제어까지 완전 자동화 파이프라인을 구성합니다.

🔄 동작 흐름

차량 감지 → YOLO가 웹캠 프레임에서 차량을 인식 후 /car_detected 발행

자동 안내 → car_auto_announce_node 가 차량 번호와 차종을 음성으로 안내

결제 절차 → face_payment_manager 가 음성 입력으로 유종/금액 파악 및 얼굴 인증 수행

주유 시퀀스 → fuel_task_manager 가 로봇팔을 이용해 주유구 개방 → 주유 → 노즐 복귀

완료 안내 → 상태 토픽 (/fuel_status) 갱신 및 TTS 출력

⚙️ 실행 방법
# ROS 2 Humble 환경 설정
source /opt/ros/humble/setup.bash
source ~/xyz_ws/install/setup.bash

# 카메라 노드 실행
ros2 run dsr_example webcam_manager_ros
ros2 run dsr_example realsense_manager_ros

# 인식 및 제어 노드 실행
ros2 run dsr_example vision_target_node
ros2 run dsr_example plate_recognizer
ros2 run dsr_example car_auto_announce_node
ros2 run dsr_example face_payment_manager
ros2 run dsr_example fuel_task_manager

🧩 기술 포인트

BLIP-2 : 시각 + 언어 추론 결합, 차량 종류·명령 인식에 활용

YOLOv8 Detections: 차량·주유구 실시간 탐지 및 3D 좌표 계산

RealSense Depth Fusion: 픽셀 깊이 → 월드좌표 변환

SpeechRecognition + gTTS: 음성 명령 입력 및 안내 출력

DeepFace Auth: 얼굴 인증을 이용한 결제 승인

Doosan E0509 Control: ROS2 SDK 기반 movej/movel 로 정밀 동작 제어

🧠 향후 확장

CoT-VLA 기반 Reasoning 강화 및 LLaVA, Qwen-VL 연동

강화학습 (RL) 을 통한 자동 최적화 시퀀스 학습

Unity 또는 Isaac Sim 기반 시뮬레이션 연동