import cv2
import dlib
import numpy as np

# 얼굴 인식기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 이미지 불러오기
image = cv2.imread("Hyein.jpg")

# 흑백 이미지로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 인식
faces = detector(gray)
for face in faces:
    landmarks = predictor(gray, face)

    # 입 중앙의 좌표 계산
    mouth_center_x = (landmarks.part(48).x + landmarks.part(54).x) // 2
    mouth_center_y = (landmarks.part(48).y + landmarks.part(54).y) // 2

    # 코 부분의 양쪽 끝 특징점 좌표
    nose_left_x, nose_left_y = landmarks.part(31).x, landmarks.part(31).y
    nose_right_x, nose_right_y = landmarks.part(35).x, landmarks.part(35).y

    # 입 중앙과 코 양쪽 끝으로 삼각형을 그리기
    triangle_points = np.array([[nose_left_x, nose_left_y], [nose_right_x, nose_right_y], [mouth_center_x, mouth_center_y]], np.int32)
    triangle = cv2.convexHull(triangle_points)

    # 삼각형의 무게 중심 계산
    M = cv2.moments(triangle)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # 콧수염 이미지 불러오기
    mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)

    # Calculate the scaling factor based on the distance between nose points
    nose_distance = np.sqrt((nose_right_x - nose_left_x)**2 + (nose_right_y - nose_left_y)**2)
    
    # Calculate the scaling factor based on the original mustache size and desired face coverage
    original_mustache_width = 200  # Set this to the original width of your mustache image
    scaling_factor = nose_distance / original_mustache_width

    # Resize the mustache image
    mustache = cv2.resize(mustache, None, fx=scaling_factor, fy=scaling_factor)

    # 이미지에 콧수염 추가
    y_offset = int(cy - mustache.shape[0] // 2)
    x_offset = int(cx - mustache.shape[1] // 2)

    for i in range(mustache.shape[0]):
        for j in range(mustache.shape[1]):
            if mustache[i, j, 3] != 0:  # 알파 채널이 0이 아닌 경우에만 추가
                image[y_offset + i, x_offset + j, :3] = mustache[i, j, :3]

# 이미지 크기 자동 조정
resized_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

cv2.imshow("Result", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
