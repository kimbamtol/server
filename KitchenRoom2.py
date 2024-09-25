import datetime
import cv2
import os
import time
import re
import numpy as np

cv2.setNumThreads(0)

import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage
from firebase_admin import messaging

# Firebase 서비스 계정 키 파일 경로
cred = credentials.Certificate('/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/aidetection-d68f6-firebase-adminsdk-hq597-416fcb3c4b.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aidetection-d68f6-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Firebase에 저장된 FCM 토큰을 가져오는 함수
def get_fcm_token(uid):
    fcm_token_ref = db.reference(f'Users/{uid}/fcmToken')
    fcm_token = fcm_token_ref.get()
    return fcm_token

# FCM 데이터 메시지를 보내는 함수
def send_fcm_data_message(fcm_token, data):
    message = messaging.Message(
        data=data,
        token=fcm_token
    )
    try:
        response = messaging.send(message)
        print(f'Successfully sent data message: {response}')
        print(f'FCM token used: {fcm_token}')
    except Exception as e:
        print(f'Failed to send message: {e}')
        print(f'FCM token used: {fcm_token}')

# 화재 감지 시 FCM 알림 전송 및 Firebase에 데이터 저장
def on_fire_detected(image_path):
    fcm_token = get_fcm_token(uid)

    if fcm_token:
        storage_file_path = f"Users/{uid}/{os.path.basename(image_path)}"
        upload_to_storage(image_path, storage_file_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"image_name is {image_name}")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        send_fcm_data_message(
            fcm_token,
            {
                "title": "화재가 감지되었습니다.",
                "body": f"발생한 시간: {current_time}\n앱 내의 캘린더를 확인해주세요.",
                "date": f"{image_name}"
            }
        )
    else:
        print("Cannot find a valid FCM token value")

# Firebase Storage에 파일 업로드 함수
def upload_to_storage(local_file_path, storage_file_path):
    bucket = storage.bucket('aidetection-d68f6.appspot.com')
    blob = bucket.blob(storage_file_path)
    blob.upload_from_filename(local_file_path)
    blob.make_public()
    return blob.public_url

# Firebase에 이벤트 데이터 저장 함수
def save_to_firebase(event_type, data):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    date = timestamp.split('_')[0]
    time_only = timestamp.split('_')[1]
    time_formatted = f"{time_only[:2]}:{time_only[2:4]}:{time_only[4:]}"
    ref = db.reference(f'Users/{uid}/OccurrenceData/{date}/{time_formatted}')
    ref.set(data)

# 사용자 UID 설정
uid = "wR5dbiFCl3OGCffPbsdWXh6Nf9G3"

# 모델 가동 상태를 제어하는 플래그
model_running = False

# 화재 감지 모델 로드
weights_path = "/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/My_Models/KitChenRoom_Model/yolov4-tiny-custom_final.weights"
config_path = "/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/My_Models/KitChenRoom_Model/yolov4-tiny-custom.cfg"
names_path = "/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/My_Models/KitChenRoom_Model/_darknet.labels"
fire_model = cv2.dnn.readNet(weights_path, config_path)

# 클래스 이름 불러오기
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

# 비디오 파일 경로 또는 RTSP 스트림
video_path = "/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/Test_Vid/fire/fire_vid0.mp4"

# VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)

# FPS 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 20.0  # 기본 FPS 설정
delay = int(1000 / fps)

if not cap.isOpened():
    print("Error -> Could not open video.")
    exit()

# 디렉토리 생성
if not os.path.exists('detected'):
    os.makedirs('detected')
if not os.path.exists('detected/fire'):
    os.makedirs('detected/fire')

# 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 화재 감지 관련 변수 초기화
fire_detection_count = 0
fire_detection_threshold = 3  # 화재 감지가 연속으로 발생하는 프레임 수
recording = False
video_writer = None
recorded_frames = 0
total_record_frames = int(fps * 5)  # 5초 동안 녹화

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream end or failed to capture.")
        break

    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    fire_model.setInput(blob)

    output_layers_names = fire_model.getUnconnectedOutLayersNames()
    layer_outputs = fire_model.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    height, width = frame.shape[:2]

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.6, nms_threshold=0.6)

    fire_detected_in_frame = False

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'fire':  # 화재 클래스 이름에 맞게 수정
                fire_detected_in_frame = True
                color = [0, 0, 255]  # 빨간색
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{label}: {confidences[i]:.2f}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if fire_detected_in_frame:
        fire_detection_count += 1
        if fire_detection_count >= fire_detection_threshold and not recording:
            print("Fire detected!!")
            # 녹화 시작
            recording = True
            recorded_frames = 0
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            image_path = f"detected/fire/{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            event_data = {'kind': '화재'}
            save_to_firebase('fire', event_data)
            video_output_path = f"detected/fire/{timestamp}.mp4"
            video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            fire_detection_count = 0  # 카운트 리셋
    else:
        fire_detection_count = 0  # 화재가 감지되지 않으면 카운트 리셋

    # 녹화 중이면 프레임 저장
    if recording:
        video_writer.write(frame)
        recorded_frames += 1
        if recorded_frames >= total_record_frames:
            # 녹화 종료
            recording = False
            video_writer.release()
            # 비디오 업로드 및 알림 전송
            upload_to_storage(video_output_path, f"Users/{uid}/{timestamp}.mp4")
            on_fire_detected(image_path)

    # 영상 표시
    cv2.imshow('Kitchen Stream', frame)

    # 실시간처럼 재생
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
