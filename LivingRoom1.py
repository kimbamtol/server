import datetime
import cv2
import os
import time
import threading
import re
import numpy as np
import torch

cv2.setNumThreads(0)

import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage
from firebase_admin import messaging

from ultralytics import YOLO
from Detection.Utils import ResizePadding
from PoseEstimateLoader import SPPE_FastPose
from ActionsEstLoader import TSSTG
from Track.Tracker import Detection, Tracker

# draw_single 함수 추가 (fn.py에서 가져옴)
def draw_single(image, keypoints, person_id=None, bbox=None, action=None, clr=(0, 255, 0)):
    '''
    이미지에 단일 사람의 키포인트와 스켈레톤을 그림.

    Parameters:
    - image: 입력 이미지
    - keypoints: 사람의 키포인트
    - person_id: 사람 ID
    - bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
    - action: 행동 이름 
    - clr: 색상 (BGR 튜플)
    '''
    # 스켈레톤 연결관계 (COCO 모델 기준)
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # 오른쪽 팔
        (0, 5), (5, 6), (6, 7), (7, 8),       # 왼쪽 팔
        (0, 9), (9, 10), (10, 11), (11, 12),  # 오른쪽 다리
        (0, 13), (13, 14), (14, 15), (15, 16) # 왼쪽 다리
    ]

    # 키포인트 그리기
    for i, (x, y, score) in enumerate(keypoints):
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

    # 스켈레톤 그리기
    for (start, end) in skeleton:
        if start < len(keypoints) and end < len(keypoints):
            x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
            x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # 바운딩 박스 그리기
    if bbox is not None:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr, 2)

    # 사람 ID 및 행동 표시
    if person_id is not None:
        cv2.putText(image, f'ID: {person_id}', (bbox[0], bbox[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
    if action is not None:
        cv2.putText(image, action, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
    return image

# Firebase 서비스 계정 키 파일 경로
cred = credentials.Certificate('/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/aidetection-d68f6-firebase-adminsdk-hq597-416fcb3c4b.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aidetection-d68f6-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Firebase의 데이터 변경 감지 코드 (필요 시 사용)
def rtsp_changed(event):
    global model_running, video_path, cap

    new_rtsp_url = event.data
    print("RTSP Value Changed\n")
    print(f"{new_rtsp_url}\n")

    if not is_valid_rtsp_url(new_rtsp_url):
        print("Invalid RTSP Value !!! Model Stop")
        if model_running:
            cap.release()
            model_running = False
    else:
        print("Valid RTSP Value. Model Restart")
        video_path = new_rtsp_url
        if model_running:
            cap.release()
        cap = cv2.VideoCapture(video_path)
        model_running = True

uid = "wR5dbiFCl3OGCffPbsdWXh6Nf9G3"  # 감지하고자 하는 사용자의 uid
rtsp_ref = db.reference(f'Users/{uid}/CameraData/0/rtspAddress')
# 이벤트 리스너 설정 (필요 시 사용)
# rtsp_ref.listen(rtsp_changed)

# Firebase에 저장된 rtsp 값이 유효한 값인지 확인
def is_valid_rtsp_url(url):
    rtsp_regex = re.compile(r"^rtsp:\/\/[^\s]+$")
    return rtsp_regex.match(url) is not None

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Firebase에 저장 함수
def save_to_firebase(event_type, data):
    timestamp = get_timestamp()
    year_month_day = timestamp.split('_')[0]
    hour_minute_second = timestamp.split('_')[1]
    time_only = f"{hour_minute_second[:2]}:{hour_minute_second[2:4]}:{hour_minute_second[4:]}"
    ref = db.reference(f'Users/{uid}/OccurrenceData/{year_month_day}/{time_only}')
    ref.set({**data})

# Firebase Storage에 파일 업로드 함수
def upload_to_storage(local_file_path, storage_file_path):
    bucket = storage.bucket('aidetection-d68f6.appspot.com')
    blob = bucket.blob(storage_file_path)
    blob.upload_from_filename(local_file_path)
    blob.make_public()
    return blob.public_url

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

# 낙상 감지 시 FCM 알림 전송 코드
def on_fall_detected(image_path):
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
                "title": "낙상이 감지되었습니다.",
                "body": f"발생한 시간: {current_time}\n앱 내의 캘린더를 확인해주세요.",
                "date": f"{image_name}"
            }
        )
    else:
        print("Cannot Found Valid Fcm Token Value")

# 모델 가동 상태를 제어하는 플래그
model_running = False

# Ultralytics YOLO 모델 로드 (best_5.pt)
device = torch.device('cpu')  # 필요에 따라 'cuda'로 변경.. 근데 aws에서 빌리려면 엄청 비쌀거라... 
#free tier에서도 gpu로 모델 가동이 가능 -> cuda 
#불가능하면 -> cpu
detect_model = YOLO('/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/My_Models/LivingRoom_Model/best_5.pt')

# SPPE FastPose 모델 로드
pose_input_size = '224x160'  # 모델의 입력 크기에 맞게 조정
inp_pose = pose_input_size.split('x')
inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
pose_backbone = 'resnet50'
pose_model = SPPE_FastPose(pose_backbone, inp_pose[0], inp_pose[1], device=device)

# Action Estimation Model
action_model = TSSTG()

# Tracker
max_age = 30
tracker = Tracker(max_age=max_age, n_init=3)

# 비디오 소스 설정 (웹캠 또는 RTSP 스트림)
# video_path = 0  # 웹캠의 경우 0 또는 카메라 번호, RTSP 스트림의 경우 RTSP URL
video_path = '/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/Test_Vid/fall/test_vid0.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("video capture failed !")
    exit()

frame_num = 1

fall_detection_count = 0
fall_detection_threshold = 3  # 연속으로 'Fall Down'이 감지되는 횟수
save_frames_fall = False

if not os.path.exists('detected'):
    os.makedirs('detected')
if not os.path.exists('detected/fall'):
    os.makedirs('detected/fall')

#fourcc = cv2.VideoWriter_fourcc(*'X264') # 무난한 코덱이라는데... 잘 안되는 것 같음.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # avi

# 녹화 관련 변수 초기화
recording = False
video_writer = None
recorded_frames = 0
total_record_frames = 60  # 녹화할 프레임 수

def preproc(image):
    # YOLO 모델은 내부적으로 전처리를 수행하므로 별도의 전처리 불필요
    # 하지만 필요하다면 여기에 나중에 코드를 추가해주자.
    return image

def kpt2bbox(kpt, ex=20):
    # 키포인트로부터 바운딩 박스 생성
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

while True:
    ret, img = cap.read()
    if not ret:
        break

    frame = preproc(img)
    image = frame.copy()

    # 사람 탐지
    results = detect_model(frame)
    detections = []
    detected = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0:  # Class 0: Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected.append([x1, y1, x2, y2, conf])

    if len(detected) > 0:
        detected = torch.tensor(detected, dtype=torch.float32)
        # 포즈 추정
        poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

        # Detections 객체 생성
        detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                np.concatenate((ps['keypoints'].numpy(),
                                                ps['kp_score'].numpy()), axis=1),
                                ps['kp_score'].mean().numpy()) for ps in poses]

    else:
        detected = None

    # 추적기 예측 업데이트
    tracker.predict()
    # 기존 추적 결과와 현재 감지 결과 병합
    for track in tracker.tracks:
        if track.is_confirmed():
            det = torch.tensor([track.to_tlbr().tolist() + [0.5]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

    # 추적기 업데이트
    tracker.update(detections)

    # 동작 인식 및 키포인트 표시
    for i, track in enumerate(tracker.tracks):
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_tlbr().astype(int)
        center = track.get_center().astype(int)

        action = 'Person'
        clr = (0, 255, 0)

        # 30프레임 단위예측
        if len(track.keypoints_list) == 30:
            pts = np.array(track.keypoints_list, dtype=np.float32)
            out = action_model.predict(pts, frame.shape[:2])
            action_name = action_model.class_names[out[0].argmax()]
            action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
            if action_name == 'Fall Down':
                fall_detection_count += 1
                print(f"Fall Down detected ({fall_detection_count}/{fall_detection_threshold})")
                if fall_detection_count >= fall_detection_threshold and not recording:
                    clr = (255, 0, 0)
                    print(f"Fall detected at frame {frame_num}")
                    # 녹화 시작
                    recording = True
                    recorded_frames = 0
                    timestamp = get_timestamp()
                    image_path = f"detected/fall/{timestamp}.jpg"
                    cv2.imwrite(image_path, img)
                    event_data = {
                        'kind': '낙상'
                    }
                    save_to_firebase('fall', event_data)
                    video_path = f"detected/fall/{timestamp}.mp4"
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps == 0:
                        fps = 24.0  # 테스트 영상은 24fps, 25fps 두개 . 우선은 24로 설정.
                    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img.shape[1], img.shape[0]))
                    fall_detection_count = 0  # 카운트 리셋
            else:
                # Fall Down 이 아니면 카운트 리셋
                fall_detection_count = 0

        # 바운딩 박스와 행동 표시
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr, 2)
        cv2.putText(img, action, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        # 키포인트 및 스켈레톤 그리기
        if track.time_since_update == 0:
            if len(track.keypoints_list) > 0:
                keypoints = track.keypoints_list[-1]
                # img = draw_single(img, keypoints, person_id=track_id, bbox=bbox, action=action, clr=clr)

    # 녹화 중이면 프레임 저장
    if recording:
        video_writer.write(img)
        recorded_frames += 1
        if recorded_frames >= total_record_frames:
            # 녹화 종료
            recording = False
            video_writer.release()
            # 비디오 업로드 및 푸시 알림 전송
            upload_to_storage(video_path, f"Users/{uid}/{timestamp}.mp4")
            on_fall_detected(image_path)

    # 프레임 표시
    cv2.imshow('Webcam Feed', img)

    frame_num += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
