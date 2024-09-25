import datetime
import cv2
import os
import time
import threading
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

# Firebase 서비스 계정 키 파일 경로
cred = credentials.Certificate('/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/aidetection-d68f6-firebase-adminsdk-hq597-416fcb3c4b.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aidetection-d68f6-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# 유틸리티 함수들
def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

def save_to_firebase(uid, data):
    timestamp = get_timestamp()
    year_month_day = timestamp.split('_')[0]
    hour_minute_second = timestamp.split('_')[1]
    time_only = f"{hour_minute_second[:2]}:{hour_minute_second[2:4]}:{hour_minute_second[4:]}"
    ref = db.reference(f'Users/{uid}/OccurrenceData/{year_month_day}/{time_only}')
    ref.set(data)

def upload_to_storage(local_file_path, storage_file_path):
    bucket = storage.bucket('aidetection-d68f6.appspot.com')
    blob = bucket.blob(storage_file_path)
    blob.upload_from_filename(local_file_path)
    blob.make_public()
    return blob.public_url

def get_fcm_token(uid):
    fcm_token_ref = db.reference(f'Users/{uid}/fcmToken')
    fcm_token = fcm_token_ref.get()
    return fcm_token

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

def on_lying_down_detected(uid, image_path):
    fcm_token = get_fcm_token(uid)
    if fcm_token:
        storage_file_path = f"Users/{uid}/{os.path.basename(image_path)}"
        upload_to_storage(image_path, storage_file_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
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
        print("Cannot find valid FCM token")

def preproc(image):
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kpt2bbox(kpt, ex=20):
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

# 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inp_dets = 384
detect_model = YOLO('/Users/taehungim/Jupyter_Lab/Team_Workspace/New_Model/Final/My_Models/LivingRoom_Model/best_5.pt')  # Replace with your model path

pose_input_size = '224x160'
inp_pose = (224, 160)
pose_backbone = 'resnet50'
pose_model = SPPE_FastPose(pose_backbone, inp_pose[0], inp_pose[1], device=device)

# Tracker
max_age = 30
tracker = Tracker(max_age=max_age, n_init=3)

# 행동 평가 모델 load 
action_model = TSSTG()

# Resize 함수(padding)
resize_fn = ResizePadding(inp_dets, inp_dets)

# Codec 설정
#fourcc = cv2.VideoWriter_fourcc(*'X264') # 무난한 코덱이라는데... 잘 안되는 것 같음.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # avi

# 이벤트 발생 시 비디오 녹화 함수
def record_video(cap, video_writer, total_frames):
    recorded_frames = 0
    while recorded_frames < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        video_writer.write(frame)
        recorded_frames += 1
    video_writer.release()

# 낙상 감지 카운트 변수 초기화
lying_down_detection_count = {}
lying_down_detection_threshold = 3  # 연속으로 'Fall Down'이 감지되는 횟수
recording = {}

# 사용자별로 카메라 스트림을 처리하는 함수
def process_camera_stream(rtsp_url, uid):
    global lying_down_detection_count
    global recording

    cap = cv2.VideoCapture(rtsp_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 20.0  # 기본 FPS 설정

    if not cap.isOpened():
        print(f"Error in video stream -> {uid}")
        return

    lying_down_detection_count[uid] = 0
    recording[uid] = False

    while True:
        ret, img = cap.read()
        if not ret:
            print(f"Stream ended or failed to capture for user {uid}")
            break

        frame = preproc(img)
        image = frame.copy()

        # 사람 탐지
        results = detect_model(frame)
        detected = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected.append([x1, y1, x2, y2, conf])

        detections = []
        if len(detected) > 0:
            detected = torch.tensor(detected, dtype=torch.float32)
            # 자세 추정
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Detections 객체 생성
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

        # 추적기 업데이트
        tracker.predict()
        tracker.update(detections)

        # 동작 인식
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'Person'
            clr = (0, 255, 0)
            # 30 프레임 시퀀스를 사용하여 예측
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    lying_down_detection_count[uid] += 1
                    print(f"(Test)Fall Down detected -> {uid} ({lying_down_detection_count[uid]}/{lying_down_detection_threshold})")
                    if lying_down_detection_count[uid] >= lying_down_detection_threshold and not recording[uid]:
                        clr = (255, 0, 0)
                        print(f"Fall Down user -> {uid}")
                        # 이벤트 발생 시 동작 수행
                        timestamp = get_timestamp()
                        image_path = f"detected/lying_down/{uid}_{timestamp}.jpg"
                        if not os.path.exists('detected/lying_down'):
                            os.makedirs('detected/lying_down')
                        cv2.imwrite(image_path, img)
                        event_data = {'kind': '낙상'}
                        save_to_firebase(uid, event_data)
                        video_path = f"detected/lying_down/{uid}_{timestamp}.mp4"
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img.shape[1], img.shape[0]))
                        total_frames = int(fps * 5)  # 5초 동안 녹화
                        recording[uid] = True

                        # 비디오 녹화 시작
                        recording_thread = threading.Thread(target=record_video, args=(cap, video_writer, total_frames))
                        recording_thread.start()
                        recording_thread.join()

                        # 비디오 녹화 완료 후 업로드 및 알림 전송
                        upload_to_storage(video_path, f"Users/{uid}/{timestamp}.mp4")
                        on_lying_down_detected(uid, image_path)
                        lying_down_detection_count[uid] = 0  # 카운트 리셋
                        recording[uid] = False
                else:
                    # Fall Down 이 아니면 카운트 리셋
                    lying_down_detection_count[uid] = 0

        # 프레임 표시 (메인 스레드에서 모델을 돌리는게 아니라 그런지 cv2 프레임이 표시되지않음...)
        # cv2.imshow(f'User {uid} Stream', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

# 모든 사용자(회원)의 카메라를 처리하는 스레드를 생성
def start_processing_for_all_users():
    uids = get_all_user_uids()
    for uid in uids:
        if uid is not None:
            thread = threading.Thread(target=process_user_cameras, args=(uid,))
            thread.start()

def get_all_user_uids():
    users_ref = db.reference('Users')
    users_data = users_ref.get()
    if users_data:
        uids = list(users_data.keys())
        return uids
    else:
        return []

def process_user_cameras(uid):
    rtsp1 = get_rtsp_address(uid, camera_index=0)
    rtsp2 = get_rtsp_address(uid, camera_index=1)

    # 두 개의 카메라를 각각 처리
    if rtsp1:
        print(f"Starting stream -> {uid} camera 0")
        thread1 = threading.Thread(target=process_camera_stream, args=(rtsp1, uid))
        thread1.start()
    if rtsp2:
        print(f"Starting stream -> {uid} camera 1")
        thread2 = threading.Thread(target=process_camera_stream, args=(rtsp2, uid))
        thread2.start()

def get_rtsp_address(uid, camera_index=0):
    rtsp_ref = db.reference(f'Users/{uid}/CameraData/{camera_index}/rtspAddress')
    rtsp_address = rtsp_ref.get()
    if rtsp_address:
        print(f"Connected to rtspAddress({camera_index}) user -> {uid}")
        return rtsp_address
    else:
        print(f"User -> {uid} -> invalid rstpAddres -> {camera_index}")
        return None

# 프로그램 시작 시 모든 사용자의 웹캠 스트림을 처리
start_processing_for_all_users()

# 메인 스레드에서 키 입력 감지
try:
    while True:
        user_input = input("Press 'q' to stop all streams: ")
        if user_input.strip().lower() == 'q':
            print("Stopping all streams...")
            break
except KeyboardInterrupt:
    print("Program interrupted. Stopping all streams...")
