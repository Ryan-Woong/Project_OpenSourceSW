import cv2
import dlib
import numpy as np

# Initialize face recognition
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Load image
image = cv2.imread("MinJi.jpg")


# Convert to black and white image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Face recognition
faces = detector(gray)
for face in faces:
    landmarks = predictor(gray, face)


    # Calculate the coordinates of the top-center of the head (for hat placement)
    top_head_x = landmarks.part(29).x
    top_head_y = landmarks.part(24).y


    # Load hat image
    hat = cv2.imread("hat.jpg", cv2.IMREAD_UNCHANGED)


    # Calculate the scaling factor based on the distance between the eyes
    eye_left_x, eye_left_y = landmarks.part(36).x, landmarks.part(36).y
    eye_right_x, eye_right_y = landmarks.part(45).x, landmarks.part(45).y
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


# Auto adjust image size
resized_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

cv2.imshow("Result", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()