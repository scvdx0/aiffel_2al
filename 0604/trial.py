import cv2
import dlib
import numpy as np

# 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
# dlib에서 제공하는 pre-trained 모델
predictor = dlib.shape_predictor('./0604/models/shape_predictor_68_face_landmarks.dat')

# 합성할 이미지 로드
overlay_image = cv2.imread('./0604/images/cat-whiskers.png', cv2.IMREAD_UNCHANGED)

def add_overlay(frame, overlay, position, angle):
    x, y, w, h = position
    overlay = cv2.resize(overlay, (w, h))
    
    # 알파 채널 분리
    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # 이미지 회전
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_overlay = cv2.warpAffine(overlay, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    for c in range(0, 3):
        frame[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] +
                                  alpha_l * frame[y:y+h, x:x+w, c])

    return frame

# 얼굴의 각도 계산
def calculate_angle(landmarks):
    left_eye_center = np.mean([(landmarks.part(36).x, landmarks.part(36).y),
                               (landmarks.part(39).x, landmarks.part(39).y)], axis=0)
    right_eye_center = np.mean([(landmarks.part(42).x, landmarks.part(42).y),
                                (landmarks.part(45).x, landmarks.part(45).y)], axis=0)
    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left = (landmarks.part(0).x, landmarks.part(0).y)
        right = (landmarks.part(16).x, landmarks.part(16).y)
        top = (landmarks.part(19).x, landmarks.part(19).y)
        bottom = (landmarks.part(8).x, landmarks.part(8).y)
        
        width = right[0] - left[0]
        height = bottom[1] - top[1]

        # 얼굴 위치에 합성 이미지 추가
        position = (left[0], top[1] - height // 2, width, height)
        
        # 얼굴 각도 계산
        angle = calculate_angle(landmarks)

        frame = add_overlay(frame, overlay_image, position, angle)

    cv2.imshow('Face Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()