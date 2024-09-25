from ultralytics import YOLO
import datetime
import cv2
import os
import time
import threading
#rtsp값 검사
import re
import numpy as np

cv2.setNumThreads(0)
#
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage
from firebase_admin import messaging

# Firebase 서비스 계정 키 파일 경로
cred = credentials.Certificate('/Users/taehungim/Jupyter_Lab/Team_Workspace/aidetection-d68f6-firebase-adminsdk-hq597-416fcb3c4b.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aidetection-d68f6-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Firebase의 데이터 변경 감지 코드
def rtsp_changed(event):
    global model_running, video_path, cap

    new_rtsp_url = event.data
    print("RTSP Value Changed\n")
    print(f"{new_rtsp_url}\n")

    if not is_valid_rtsp_url(new_rtsp_url):
        print("Invalid RTSP Value !!! Model Stop")
        if model_running:
            # 모델 가동 중지 (비디오 캡처 해제 등)
            cap.release()
            model_running = False
    else:
        print("Valid RSTP Value. Model Restart")
        video_path = new_rtsp_url
        if model_running:
            # 이미 모델이 실행 중이면 재시작
            cap.release()
        cap = cv2.VideoCapture(video_path)
        model_running = True

uid = "wR5dbiFCl3OGCffPbsdWXh6Nf9G3"  # 감지하고자 하는 사용자의 uid
rtsp_ref = db.reference(f'Users/{uid}/CameraData/0/rtspAddress')
# 이벤트 리스너 설정(일단은 꺼놓을 것))
#rtsp_ref.listen(rtsp_changed)

# Firebase에 저장된 rtsp값이 유효한 값인가?
def is_valid_rtsp_url(url):
    rtsp_regex = re.compile(r"^rtsp:\/\/[^\s]+$")
    return rtsp_regex.match(url) is not None

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Firebase에 저장 함수
def save_to_firebase(event_type, data):
    timestamp = get_timestamp()  # 전체 타임스탬프 가져오기
    year_month_day = timestamp.split('_')[0]  # 날짜 부분 추출
    hour_minute_second = timestamp.split('_')[1]  # 시, 분, 초 부분 추출
    time_only = f"{hour_minute_second[:2]}:{hour_minute_second[2:4]}:{hour_minute_second[4:]}"  # 'HH:MM:SS'

    # Firebase 경로 설정
    ref = db.reference(f'Users/wR5dbiFCl3OGCffPbsdWXh6Nf9G3/OccurrenceData/{year_month_day}/{time_only}')
    ref.set({
        **data
    })

# Firebase Storage에 파일 업로드 함수
def upload_to_storage(local_file_path, storage_file_path):
    # Firebase Storage 버킷 참조
    bucket = storage.bucket('aidetection-d68f6.appspot.com')
    # 로컬 파일을 Firebase Storage에 업로드
    blob = bucket.blob(storage_file_path)
    blob.upload_from_filename(local_file_path)

    # 공개 URL 생성 (필요한 경우)
    blob.make_public()
    return blob.public_url


# Firebase에 저장된 FCM 토큰을 가져오는 함수
def get_fcm_token(uid):
    fcm_token_ref = db.reference(f'Users/{uid}/fcmToken')
    fcm_token = fcm_token_ref.get()
    return fcm_token

# FCM 데이터 메시지를 보내는 함수
def send_fcm_data_message(fcm_token, data):
    # 메시지 내용 설정
    message = messaging.Message(
        data=data,  # 데이터 메시지로 전송
        token=fcm_token
    )
    try:
        # 메시지 전송
        response = messaging.send(message)
        print(f'Successfully sent data message: {response}')
        print(f'FCM token used: {fcm_token}')  # 전송에 사용된 FCM 토큰 출력
    except Exception as e:
        print(f'Failed to send message: {e}')
        print(f'FCM token used: {fcm_token}')  # 오류 발생 시에도 FCM 토큰 출력

# 낙상 감지 시 FCM 알림 전송 코드
def on_fall_detected(image_path):
    # FCM 토큰 가져오기
    fcm_token = get_fcm_token(uid)

    if fcm_token:
        # 이미지 파일을 Firebase Storage에 업로드하고 공개 URL 생성
        storage_file_path = f"Users/wR5dbiFCl3OGCffPbsdWXh6Nf9G3/{os.path.basename(image_path)}"
        upload_to_storage(image_path, storage_file_path)
        #image_url = upload_to_storage(image_path, storage_file_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"image_name is {image_name}")
        # 현재 시간을 'YYYY-MM-DD HH:MM:SS' 형식으로 포맷팅
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # FCM 데이터 메시지 보내기
        send_fcm_data_message(
            fcm_token,
            {
                "title": "화재가 감지되었습니다.",
                "body": f"발생한 시간: {current_time}\n앱 내의 캘린더를 확인해주세요.",
                "date" : f"{image_name}"
            }
        )
    else:
        print("Cannot Found Valid Fcm Token Value")

# 여기서부터 모델 코드
# 모델 가동 상태를 제어하는 플래그
model_running = False

detect_model = YOLO('/Users/taehungim/Jupyter_Lab/Team_Workspace/best_5.pt')

pose_model = YOLO('yolov8s-pose.pt', task="pose")  # Load pretrained YOLOv8 pose model

#터미널에서 웹캠 스트림을 rtsp로 변경하는 코드
#video_path = "rtsp://admin:admin@172.100.1.16:1936"

cap = cv2.VideoCapture(video_path)
frame = 1
fall_detection_count = 0
fire_detection_count = 0
smoke_detection_count = 0
save_frames_fall = False
save_frames_fire = False
save_frames_smoke = False
recording = False

if not os.path.exists('detected'):
    os.makedirs('detected')
if not os.path.exists('detected/fire'):
    os.makedirs('detected/fire')
if not os.path.exists('detected/smoke'):
    os.makedirs('detected/smoke')
if not os.path.exists('detected/fall'):
    os.makedirs('detected/fall')
if not os.path.exists('detected/spoon'):
    os.makedirs('detected/spoon')
if not os.path.exists('detected/chopstick'):
    os.makedirs('detected/chopstick')

fourcc = cv2.VideoWriter_fourcc(*'X264')

def record_video(out, duration=5):
    global recording
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    out.release()
    recording = False



weights_path = "/Users/taehunkim/Downloads/Eating Person Models/Model/yolov4-tiny-custom_final.weights"
config_path = "/Users/taehunkim/Downloads/Eating Person Models/Model/yolov4-tiny-custom.cfg"
names_path = "/Users/taehunkim/Downloads/Eating Person Models/Model/_darknet.labels"
fire_model = cv2.dnn.readNet(weights_path, config_path)
# 클래스 이름 불러오기
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")


# 비디오 파일 경로 또는 RTSP 스트림
rtsp_url = "/Users/taehunkim/Downloads/Eating Person Models/Model/test_vid/fire_test_vid.mp4"

# VideoCapture 객체를 한 번만 생성
cap = cv2.VideoCapture(rtsp_url)

# FPS 정보를 얻어서 실시간처럼 재생
fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 파일의 FPS를 얻음
delay = int(1000 / fps)  # 각 프레임 간의 대기 시간(ms)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or failed to capture.")
        break

    print(f"frame.shape -> {frame.shape}")

    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    fire_model.setInput(blob)

    output_layers_names = fire_model.getUnconnectedOutLayersNames()
    layer_outputs = fire_model.forward(output_layers_names)

    boxes = []  # 객체의 박스 좌표
    confidences = []  # 탐지된 객체의 신뢰도
    class_ids = []  # 탐지된 객체의 클래스 ID

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:  # 신뢰도가 0.6보다 큰 경우만 객체로 인식
                # 탐지된 객체의 BOX를 이미지 크기에 맞게 조정
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])  # 탐지된 객체의 BOX 좌표와 크기를 저장
                confidences.append(float(confidence))  # 신뢰도 저장
                class_ids.append(class_id)  # 클래스 ID 저장

    # 겹치는 객체의 BOX 중 가장 신뢰도가 높은 BOX만 남김
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.6, nms_threshold=0.6)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            color = [0, 0, 255]  # 화재일 경우 빨간색 박스
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 영상 표시
    cv2.imshow('Kitchen Stream', frame)

    # 실시간처럼 재생: FPS에 맞춰 딜레이를 추가
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
