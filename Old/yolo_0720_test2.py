from ultralytics import YOLO
import datetime
import cv2
import os
import time
import threading
#rtsp값 검사
import re

cv2.setNumThreads(0)
#
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage
#

# Firebase 서비스 계정 키 파일 경로
cred = credentials.Certificate('/Users/taehunkim/Downloads/Eating Person Models/Model/aidetection-d68f6-firebase-adminsdk-hq597-5b05d575dd.json')
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
# 이벤트 리스너 설정
rtsp_ref.listen(rtsp_changed)

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
    hour_minute = timestamp.split('_')[1][:4]  # 시, 분 부분 추출
    time_only = f"{hour_minute[:2]}:{hour_minute[2:]}"  # 'HH:MM'
    # Firebase 경로 설정
    ref = db.reference(f'Users/wR5dbiFCl3OGCffPbsdWXh6Nf9G3/OccurrencesData/{year_month_day}/{time_only}')
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

# 여기서부터 모델 코드
# 모델 가동 상태를 제어하는 플래그
model_running = False

# Load the object detection model
detect_model = YOLO('/Users/taehunkim/VSCode/Python/Practice/yolo/best_5.pt')

# Load the pose estimation model
pose_model = YOLO('yolov8s-pose.pt', task="pose")  # Load pretrained YOLOv8 pose model

#video_path = 'fall_0025.mp4'
#터미널에서 웹캠 스트림을 rtsp로 변경하는 코드
#gst-launch-1.0 autovideosrc ! queue ! videoconvert ! queue ! video/x-raw,format=I420 ! queue ! x264enc ! queue ! rtph264pay ! queue ! udpsink host=127.0.0.1 port=3434
#ffmpeg -i udp://127.0.0.1:3434 -f rtsp rtsp://localhost:8554/mystream

video_path = "rtsp://admin:admin@172.100.1.16:1936"

cap = cv2.VideoCapture(video_path)
frame = 1
fall_detection_count = 0
fire_detection_count = 0
smoke_detection_count = 0
save_frames_fall = False
save_frames_fire = False
save_frames_smoke = False
recording = False

# Create necessary folders if they do not exist
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

fourcc = cv2.VideoWriter_fourcc(*'X264')  # Define the codec for the video file

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

while True:
    # Read frame from webcam
    ret, img = cap.read()
    if not ret:
        break

    # Run object detection model inference
    results = detect_model(img, stream=True, save=False, device="cpu", imgsz=640)

    person_detected = False
    fire_detected = False
    smoke_detected = False

    for result in results:
        try:
            boxes = result.boxes  # Boxes object for bbox outputs

            for box in boxes:
                cls = int(box.cls[0])  # Get class of detected object
                conf = float(box.conf[0])  # Get confidence of detected object

                label = f"{detect_model.names[cls]} {conf:.2f}"  # Create label with class and confidence

                # Check if the detected object is a person
                if cls == 0:  # Assuming '0' is the class ID for 'person'
                    person_detected = True
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Crop person from the frame
                    person_img = img[y1:y2, x1:x2]

                    # Run pose estimation model on the detected person
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
                            # Draw keypoints on img
                            cv2.circle(img, (x1 + kx, y1 + ky), 5, (0, 0, 255), -1)  # Draw a red circle at each keypoint location

                        w = box.xywh[0][2]
                        h = box.xywh[0][3]

                        if w / h > 1.4:
                            fall_detected = True
                            fall_detection_count += 1
                            print(f"Fall detected at frame {frame}")

                            # Print fall on top of person's head
                            cv2.putText(img, "Fallen", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            cv2.putText(img, "Stable", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Check if fall was detected
                    if fall_detected:
                        if fall_detection_count == 3:
                            save_frames_fall = True
                            timestamp = get_timestamp()
                            cv2.imwrite(f"detected/fall/fall_{timestamp}.jpg", img)
                            event_data ={
                                'Kind':'낙상'
                            }
                            save_to_firebase('fall',event_data)
                            video_path = f"detected/fall/fall_{timestamp}.mp4"
                            video_writer = cv2.VideoWriter(f"detected/fall/fall_{timestamp}.mp4", fourcc, 20.0, (img.shape[1], img.shape[0]))
                            # Firebase Storage에 비디오 업로드 (업로드는 비디오 녹화가 끝난 후 수행)
                            video_thread = threading.Thread(target=lambda: upload_to_storage(video_path, f"fall_videos/fall_{timestamp}.mp4"))
                            video_thread.start()

                            recording = True
                            recording_thread = threading.Thread(target=record_video, args=(video_writer,))
                            recording_thread.start()
                    else:
                        if save_frames_fall:
                            # Stop saving frames once fall is no longer detected
                            save_frames_fall = False
                            fall_detection_count = 0

                # Check if the detected object is fire
                elif cls == 1:  # Assuming '1' is the class ID for 'fire'
                    fire_detected = True
                    # Draw bounding box for fire
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Check if fire was detected
                    if fire_detected:
                        fire_detection_count += 1
                        if fire_detection_count == 5:
                            save_frames_fire = True
                            timestamp = get_timestamp()
                            cv2.imwrite(f"detected/fire/fire_{timestamp}.jpg", img)
                    else:
                        if save_frames_fire:
                            # Stop saving frames once fire is no longer detected
                            save_frames_fire = False
                            fire_detection_count = 0

                # Check if the detected object is smoke
                elif cls == 2:  # Assuming '2' is the class ID for 'smoke'
                    smoke_detected = True
                    # Draw bounding box for smoke
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # Check if smoke was detected
                    if smoke_detected:
                        smoke_detection_count += 1
                        if smoke_detection_count >= 3:
                            save_frames_smoke = True
                            timestamp = get_timestamp()
                            cv2.imwrite(f"detected/smoke/smoke_{timestamp}.jpg", img)
                    else:
                        if save_frames_smoke:
                            # Stop saving frames once smoke is no longer detected
                            save_frames_smoke = False
                            smoke_detection_count = 0

                # Check if the detected object is spoon
                elif cls == 3:  # Assuming '3' is the class ID for 'spoon'
                    # Draw bounding box for spoon
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                    # Save spoon detection image to disk
                    timestamp = get_timestamp()
                    cv2.imwrite(f"detected/spoon/spoon_{timestamp}.jpg", img)
                    print(f"Spoon detected at frame {frame}")

                # Check if the detected object is chopstick
                elif cls == 4:  # Assuming '4' is the class ID for 'chopstick'
                    # Draw bounding box for chopstick
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    # Save chopstick detection image to disk
                    timestamp = get_timestamp()
                    cv2.imwrite(f"detected/chopstick/chopstick_{timestamp}.jpg", img)
                    print(f"Chopstick detected at frame {frame}")

        except Exception as e:
            print(f"Error: {e}")

    # Display the resulting frame
    cv2.imshow('Webcam Feed', img)

    frame += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Total falls detected: {fall_detection_count}")
