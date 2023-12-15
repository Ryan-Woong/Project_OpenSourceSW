import cv2
import numpy as np

# 얼굴과 눈 감지를 위한 Haar 캐스케이드 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 안경 이미지 로드 (투명 배경의 PNG 이미지 권장)
glasses = cv2.imread('./image/glass.png', -1)  # 안경 이미지 경로를 정확히 지정해주세요.

# 대상 사진 로드
img = cv2.imread('./image/face.jpg')  # 사진 이미지 경로를 정확히 지정해주세요.
if img is None:
    print("이미지 파일을 로드할 수 없습니다.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 감지
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # 눈 감지
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # 두 눈의 중앙 지점 찾기
    if len(eyes) == 2:
        eye1, eye2 = eyes[0], eyes[1]
        # 안경의 크기 조정을 위한 눈 사이의 거리 계산
        eye_distance = np.linalg.norm(np.array((eye1[0], eye1[1])) - np.array((eye2[0], eye2[1])))
        glasses_width = int(eye_distance * 2)

        # 중간 지점 계산
        eye1_center = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
        eye2_center = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)
        glasses_center = np.average(np.array([eye1_center, eye2_center]), axis=0).astype(int)

        # Y 좌표는 두 눈 중 더 낮은 눈의 Y 좌표를 기준으로 조정, 안경이 눈 위에 오도록 조정
        y_offset = int(min(eye1_center[1], eye2_center[1]) - glasses_width * 0.125)  # 값을 줄이면 안경이 올라가고, 값을 늘리면 내려갑니다.

        # 안경 이미지의 크기 조정
        glasses_resized = cv2.resize(glasses, (glasses_width, int(glasses.shape[0] * glasses_width / glasses.shape[1])))

        # 안경 이미지의 Y 좌표는 눈의 Y 좌표보다 조금 더 위에 위치
        glasses_h, glasses_w, _ = glasses_resized.shape
        for i in range(glasses_h):
            for j in range(glasses_w):
                # 범위 안에 있는지 확인
                y_index = y_offset + i
                x_index = glasses_center[0] + j - glasses_width // 2
                
                # 합성할 위치가 roi_color 범위 내에 있는지 확인
                if (0 <= x_index < roi_color.shape[1]) and (0 <= y_index < roi_color.shape[0]):
                    if glasses_resized[i, j][3] != 0:  # 알파 채널 검사
                        roi_color[y_index, x_index] = glasses_resized[i, j][:3]

# 결과 이미지 표시
cv2.imshow('Face with Glasses', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
