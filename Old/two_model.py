from ultralytics import YOLO
import datetime
import cv2
import os
import time
import threading
import re
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage
from firebase_admin import messaging

# Firebase 서비스 계정 키 파일 경로
cred = credentials.Certificate('/Users/taehunkim/Downloads/Eating Person Models/Model/aidetection-d68f6-firebase-adminsdk-hq597-ce797e162e.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aidetection-d68f6-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Firebase에 저장 함수
def save_to_firebase(uid, data):
    timestamp = get_timestamp()  # 전체 타임스탬프 가져오기
    year_month_day = timestamp.split('_')[0]  # 날짜 부분 추출
    hour_minute_second = timestamp.split('_')[1]  # 시, 분, 초 부분 추출
    time_only = f"{hour_minute_second[:2]}:{hour_minute_second[2:4]}:{hour_minute_second[4:]}"  # 'HH:MM:SS'

    # Firebase 경로 설정 (uid를 포함)
    ref = db.reference(f'Users/{uid}/OccurrenceData/{year_month_day}/{time_only}')
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

# 회원이 여러명인 경우, 회원들의 uid를 반환하기 위한 리스트 선언.
uids = []
# 사용자 추가/삭제 감지 함수
def listen_for_user_changes():
    ref = db.reference('Users')

    # 사용자가 추가되거나 삭제될 때 on_value_change 호출
    ref.listen(on_value_change)

# 사용자가 추가되거나 삭제될 때 호출되는 함수
def on_value_change(event):
    global user_uids
    print(f'Change detected in Users: {event.event_type}')

    # 사용자 리스트를 새로 갱신
    uids = get_all_user_uids()
    print(f'Updated user list: {uids}')

def get_all_user_uids():
    # Users 경로에서 모든 데이터를 가져옴
    users_ref = db.reference('Users')
    users_data = users_ref.get()

    # Users 아래에 있는 모든 키 (UID)를 리스트로 반환
    if users_data:
        uids = list(users_data.keys())
        return uids
    elif(uids==None):
        return [0]


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

uid = "wR5dbiFCl3OGCffPbsdWXh6Nf9G3"  # 감지하고자 하는 사용자의 uid

def on_fall_detected(uid,image_path):
    # FCM 토큰 가져오기
    fcm_token = get_fcm_token(uid)

    if fcm_token:
        # 이미지 파일을 Firebase Storage에 업로드하고 공개 URL 생성
        storage_file_path = f"Users/{uid}/{os.path.basename(image_path)}"
        upload_to_storage(image_path, storage_file_path)
        #image_url = upload_to_storage(image_path, storage_file_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 현재 시간을 'YYYY-MM-DD HH:MM:SS' 형식으로 포맷팅
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # FCM 데이터 메시지 보내기
        send_fcm_data_message(
            fcm_token,
            {
                "title": "낙상이 감지되었습니다.",
                "body": f"발생한 시간: {current_time}\n앱 내의 캘린더를 확인해주세요.",
                "date" : f"{image_name}"
            }
        )
    else:
        print("Cannot Found Valid Fcm Token Value")

def on_fire_detected(uid, image_path):
    # FCM 토큰 가져오기
    fcm_token = get_fcm_token(uid)

    if fcm_token:
        # 이미지 파일을 Firebase Storage에 업로드하고 공개 URL 생성
        storage_file_path = f"Users/{uid}/{os.path.basename(image_path)}"
        image_url = upload_to_storage(image_path, storage_file_path)

        # 현재 시간을 'YYYY-MM-DD HH:MM:SS' 형식으로 포맷팅
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # FCM 데이터 메시지 보내기
        send_fcm_data_message(
            fcm_token,
            {
                "title": "화재가 감지되었습니다.",
                "body": f"발생한 시간: {current_time}\n앱 내의 캘린더를 확인해주세요.",
                "image_url": image_url  # 이미지 URL을 메시지에 포함
            }
        )
    else:
        print("Cannot Found Valid Fcm Token Value")

#내장 웹캠으로 테스트하고 싶다면, 0
#아니라면 rtsp_url에 정상적인 rtsp 값을 넣어주면 됨.
rtsp_url=0

cap = cv2.VideoCapture(rtsp_url)
frame = 1
fall_detection_count = 0
save_frames_fall = False
recording = False

# Codec 설정, X264 쓰고는 있는데, 다른 코덱 괜찮은거 있으면 바꾸셔도 OK
fourcc = cv2.VideoWriter_fourcc(*'X264')

# 이벤트 발생시, 이벤트 발생 프레임부터 잠깐 녹화하는 함수
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

# 사용자 추가/삭제 감지 함수
def listen_for_user_changes():
    ref = db.reference('Users')

    # 새로 추가된 uid를 감지
    ref.listen(on_user_added)

    # 삭제된 uid를 감지
    ref.listen(on_user_removed)

# 사용자가 추가되었을 때 호출되는 함수
def on_user_added(event):
    # event.key에는 추가된 uid에 대한 정보가 들어있음.
    print(f'New user added: {event.key}')


# 사용자가 삭제되었을 때 호출되는 함수
def on_user_removed(event):
    # event.key에는 삭제된 uid가 들어있음
    print(f'User removed: {event.key}')



# 모델 가동 상태를 제어하는 플래그
model_running = False

# 객체 감지 모델 Load
detect_model = YOLO('/Users/taehunkim/VSCode/Python/Practice/yolo/best_5.pt')

# detect_model로부터 객체를 넘겨받고, 넘겨 받은 객체의 포즈를 추정하는 모델
pose_model = YOLO('yolov8s-pose.pt', task="pose")  # Load pretrained YOLOv8 pose model

def process_user_cameras(uid):
    # 각 사용자의 두 개의 웹캠 RTSP 주소를 가져옴
    rtsp1 = get_rtsp_address1(uid, '0')
    rtsp2 = get_rtsp_address2(uid, '1')

    # 두 개의 카메라를 각각 처리
    if rtsp1:
        process_camera_stream0(rtsp1, uid)
    if rtsp2:
        process_camera_stream1(rtsp2, uid)

#거실 웹캠 처리 코드
def process_camera_stream0(rtsp_url, uid):
    global model_running
    global fall_detection_count
    global save_frames_fall
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, img = cap.read()
        if not ret:
            print(f"Error occurred while accessing stream for user {uid}")
            break

        # 영상 처리,분석 시작
        results = detect_model(frame, stream=True, save=False, device="cpu", imgsz=640)
        # 낙상 감지
        for result in results:
            try:
                boxes = result.boxes

                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    label = f"{detect_model.names[cls]} {conf:.2f}"

                    if cls == 0: # class number 0 = person
                        person_detected = True
                        # Drawing Box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        person_img = img[y1:y2, x1:x2]

                        pose_results = pose_model(person_img, stream=True, save=False, device="cpu", imgsz=640)

                        fall_detected = False

                        for pose_result in pose_results:
                            kpts = pose_result.keypoints
                            nk = kpts.shape[1]

                            # 키포인트의 개수를 출력
                            print(f"Number of keypoints: {nk}")
                            for i in range(nk):
                                keypoint = kpts.xy[0, i]
                                kx, ky = int(keypoint[0].item()), int(keypoint[1].item())
                                cv2.circle(img, (x1 + kx, y1 + ky), 5, (0, 0, 255), -1)
                            w = box.xywh[0][2]
                            h = box.xywh[0][3]

                            if w / h > 1.4:
                                fall_detected = True
                                fall_detection_count += 1
                                print(f"Fall detected at frame {frame}")

                                cv2.putText(img, "Fallen", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            else:
                                cv2.putText(img, "Stable", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if fall_detected:
                            if fall_detection_count == 3:
                                save_frames_fall = True
                                timestamp = get_timestamp()
                                image_path = f"detected/fall/{timestamp}.jpg"
                                cv2.imwrite(f"detected/fall/{timestamp}.jpg", img)
                                event_data ={
                                    'kind':'낙상'
                                }
                                save_to_firebase('fall', event_data)
                                video_path = f"detected/fall/{timestamp}.mp4"
                                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (img.shape[1], img.shape[0]))

                                # 비디오 녹화 시작
                                recording_thread = threading.Thread(target=record_video, args=(video_writer,))
                                recording_thread.start()
                                recording_thread.join()  # 비디오 녹화가 끝날 때까지 기다림

                                # 비디오 녹화가 완료된 후 Firebase Storage에 업로드
                                upload_to_storage(video_path, f"Users/{uid}/{timestamp}.mp4")
                                # fall 감지가 되었으니, fcm 푸시
                                on_fall_detected(uid,image_path)
                        else:
                            if save_frames_fall:
                                save_frames_fall = False
                                fall_detection_count = 0
            except Exception as e:
                print(f"Model 1(Fall Detect) Error {str(e)}\n")

        # 영상 표시
        cv2.imshow('Living Room Stream', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    cap.release()

# 부엌 웹캠 처리 코드
def process_camera_stream1(rtsp_url, uid):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error occurred while accessing stream for user {uid}")
            break

        # 여기서 영상 분석 (예: 낙상 감지, 화재 감지 등)
        results = detect_model(frame, stream=True, save=False, device="cpu", imgsz=640)

        # 감지된 이벤트 처리 코드 추가
        # 여기에 화재나, 연기 감지시 작동할 코드를 넣어주시면 돼요.

    cap.release()

# 모든 사용자(회원)의 카메라를 처리하는 스레드를 생성
def start_processing_for_all_users():
    uids = get_all_user_uids()
    for uid in uids:
        if(uid!=None):
            thread = threading.Thread(target=process_user_cameras, args=(uid,))
            thread.start()

def get_rtsp_address0(uid):
    # Firebase에서 특정 uid 사용자의 RTSP 주소를 가져옴
    rtsp_ref = db.reference(f'Users/{uid}/Cameradata/0/rtspAddress')
    rtsp_address = rtsp_ref.get()

    if rtsp_address:
        return rtsp_address
    else:
        print(f"{uid} -> 해당 uid 유저는 rtspAddress(0)에 rtsp값을 입력하지 않았습니다.")
        return None


# Firebase에서 특정 uid 사용자의 RTSP 주소를 가져옴
def get_rtsp_address1(uid):
    rtsp_ref = db.reference(f'Users/{uid}/Cameradata/1/rtspAddress')
    rtsp_address = rtsp_ref.get()

    if rtsp_address:
        return rtsp_address
    else:
        print(f"{uid} -> 해당 uid 유저는 rtspAddress(1)에 rtsp값을 입력하지 않았습니다.")
        return None

# 프로그램 시작 시 모든 사용자의 웹캠 스트림을 처리
start_processing_for_all_users()
# uid 변경 이벤트 리스너 등록
listen_for_user_changes()
