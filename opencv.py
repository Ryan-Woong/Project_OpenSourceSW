import cv2
import numpy as np
import dlib

# 얼굴 인식기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 루돌프 코 사진
nose = cv2.imread('nose.png') 

# 사진 로드
img = cv2.imread('face.jpg') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 인식
faces = detector(gray)
for face in faces:
    landmarks = predictor(gray, face)

    # 코 중앙
    nose_center_x = (landmarks.part(30).x + landmarks.part(33).x) // 2
    nose_center_y = (landmarks.part(30).y + landmarks.part(33).y) // 2

    # offset
    nose_left_x, nose_left_y = landmarks.part(31).x, landmarks.part(31).y
    nose_right_x, nose_right_y = landmarks.part(35).x, landmarks.part(35).y
    ctx = (landmarks.part(31).x + landmarks.part(35).x) // 2
    cty = (landmarks.part(30).y + landmarks.part(33).y) // 2

    nose = cv2.imread("nose.png", cv2.IMREAD_UNCHANGED)

    nose_distance = np.sqrt((nose_right_x - nose_left_x)**2 + (nose_right_y - nose_left_y)**2)
    
    original_ball_width = 400
    scaling_factor = nose_distance / original_ball_width

    nose = cv2.resize(nose, None, fx=scaling_factor, fy=scaling_factor)

    x_offset = int(ctx - nose.shape[1] // 2)
    y_offset = int(cty - nose.shape[0] // 2)

    for i in range(nose.shape[0]):
        for j in range(nose.shape[1]):
            if nose[i, j, 3] != 0:  # 알파 채널이 0이 아닌 경우에만 추가
                img[y_offset + i, x_offset + j, :3] = nose[i, j, :3]

# 이미지 크기 자동 조정
resized_image = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# 이미지 표시
cv2.imshow('Face with nose', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
