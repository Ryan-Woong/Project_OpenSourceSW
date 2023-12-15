import cv2
import dlib
import numpy as np

# Initialize face recognition
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 얼굴과 눈 감지를 위한 Haar 캐스케이드 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 안경 이미지 로드 (투명 배경의 PNG 이미지 권장)
glasses = cv2.imread('./image/glass.png', -1)  # 안경 이미지 경로를 정확히 지정해주세요.

# 이미지 불러오기
image = cv2.imread("face.jpg")

# 흑백 이미지로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 인식
faces = detector(gray)
faces_gla = face_cascade.detectMultiScale(gray, 1.3, 5)

for face in faces:

    # 모자
    landmarks_hat = predictor(gray, face)

    # Calculate the coordinates of the top-center of the head (for hat placement)
    top_head_x = landmarks_hat.part(29).x
    top_head_y = landmarks_hat.part(24).y

    # Load hat image
    hat = cv2.imread("hat.jpg", cv2.IMREAD_UNCHANGED)

    # Calculate the scaling factor based on the distance between the eyes
    eye_left_x, eye_left_y = landmarks_hat.part(36).x, landmarks_hat.part(36).y
    eye_right_x, eye_right_y = landmarks_hat.part(45).x, landmarks_hat.part(45).y
    eyes_distance = np.sqrt((eye_right_x - eye_left_x)**2 + (eye_right_y - eye_left_y)**2)

    # Calculate the scaling factor based on the original hat size and desired head coverage
    original_hat_width = 400  # Set this to the original width of your hat image
    scaling_factor = eyes_distance / original_hat_width

    # Resize the hat image
    hat = cv2.resize(hat, None, fx=scaling_factor, fy=scaling_factor)

    # Add hat to image
    y_offset = int(top_head_y - hat.shape[0])
    x_offset = int(top_head_x - hat.shape[1] // 2)

    for i in range(hat.shape[0]):
        for j in range(hat.shape[1]):
            if hat[i, j, 3] != 0:  # Add only if alpha channel is non-zero
                image[y_offset + i, x_offset + j, :3] = hat[i, j, :3]



    # 수염
    landmarks_mus = predictor(gray, face)

    # 입 중앙의 좌표 계산
    mouth_center_x_mus = (landmarks_mus.part(48).x + landmarks_mus.part(54).x) // 2
    mouth_center_y_mus = (landmarks_mus.part(48).y + landmarks_mus.part(54).y) // 2

    # 코 부분의 양쪽 끝 특징점 좌표
    nose_left_x_mus, nose_left_y_mus = landmarks_mus.part(31).x, landmarks_mus.part(31).y
    nose_right_x_mus, nose_right_y_mus = landmarks_mus.part(35).x, landmarks_mus.part(35).y

    # 입 중앙과 코 양쪽 끝으로 삼각형을 그리기
    triangle_points = np.array([[nose_left_x_mus, nose_left_y_mus], [nose_right_x_mus, nose_right_y_mus], [mouth_center_x_mus, mouth_center_y_mus]], np.int32)
    triangle = cv2.convexHull(triangle_points)

    # 삼각형의 무게 중심 계산
    M = cv2.moments(triangle)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # 콧수염 이미지 불러오기
    mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)

    # Calculate the scaling factor based on the distance between nose points
    nose_distance = np.sqrt((nose_right_x_mus - nose_left_x_mus)**2 + (nose_right_y_mus - nose_left_y_mus)**2)
    
    # Calculate the scaling factor based on the original mustache size and desired face coverage
    original_mustache_width = 200  # Set this to the original width of your mustache image
    scaling_factor_mus = nose_distance / original_mustache_width

    # Resize the mustache image
    mustache = cv2.resize(mustache, None, fx=scaling_factor_mus, fy=scaling_factor_mus)

    # 이미지에 콧수염 추가
    y_offset_mus = int(cy - mustache.shape[0] // 2)
    x_offset_mus = int(cx - mustache.shape[1] // 2)

    for i in range(mustache.shape[0]):
        for j in range(mustache.shape[1]):
            if mustache[i, j, 3] != 0:  # 알파 채널이 0이 아닌 경우에만 추가
                image[y_offset_mus + i, x_offset_mus + j, :3] = mustache[i, j, :3]



    # 루돌프 코
    landmarks_nose = predictor(gray, face)

    # 코 중앙
    nose_center_x = (landmarks_nose.part(30).x + landmarks_nose.part(33).x) // 2
    nose_center_y = (landmarks_nose.part(30).y + landmarks_nose.part(33).y) // 2

    # offset
    nose_left_x, nose_left_y = landmarks_nose.part(31).x, landmarks_nose.part(31).y
    nose_right_x, nose_right_y = landmarks_nose.part(35).x, landmarks_nose.part(35).y
    ctx = (landmarks_nose.part(31).x + landmarks_nose.part(35).x) // 2
    cty = (landmarks_nose.part(30).y + landmarks_nose.part(33).y) // 2

    nose = cv2.imread("nose.png", cv2.IMREAD_UNCHANGED)

    nose_distance_nose = np.sqrt((nose_right_x - nose_left_x)**2 + (nose_right_y - nose_left_y)**2)
    
    original_ball_width = 400
    scaling_factor_nose = nose_distance_nose / original_ball_width

    nose = cv2.resize(nose, None, fx=scaling_factor_nose, fy=scaling_factor_nose)

    x_offset_nose = int(ctx - nose.shape[1] // 2)
    y_offset_nose = int(cty - nose.shape[0] // 2)

    for i in range(nose.shape[0]):
        for j in range(nose.shape[1]):
            if nose[i, j, 3] != 0:  # 알파 채널이 0이 아닌 경우에만 추가
                image[y_offset_nose + i, x_offset_nose + j, :3] = nose[i, j, :3]



    # 안경
    for (x, y, w, h) in faces_gla:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

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


# 이미지 크기 자동 조정
resized_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

cv2.imshow("Result", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()