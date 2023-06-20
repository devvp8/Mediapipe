import cv2
import mediapipe as mp

def detect_eye_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_eye_landmarks = [face_landmarks[33], face_landmarks[7], face_landmarks[163],
                                   face_landmarks[144], face_landmarks[145], face_landmarks[153], face_landmarks[154],
                                   face_landmarks[155]]
            
            right_eye_landmarks = [face_landmarks[263], face_landmarks[249], face_landmarks[446],
                                    face_landmarks[374], face_landmarks[380], face_landmarks[381], face_landmarks[382],
                                    face_landmarks[383]]

            height, width, _ = image.shape
            left_eye_coords = []
            right_eye_coords = []

            for landmark in left_eye_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                left_eye_coords.append((x, y))

            for landmark in right_eye_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                right_eye_coords.append((x, y))

            return left_eye_coords, right_eye_coords

    return None, None

def draw_eye_landmarks(image, left_eye_coords, right_eye_coords):
    if left_eye_coords:
        for landmark in left_eye_coords:
            cv2.circle(image, landmark, 2, (0, 255, 0), -1)

    if right_eye_coords:
        for landmark in right_eye_coords:
            cv2.circle(image, landmark, 2, (0, 0, 255), -1)

    return image

cap = cv2.VideoCapture(0)

while True:
    reading, frame = cap.read()

    if not reading:
        break

    left_eye_coords, right_eye_coords = detect_eye_landmarks(frame)

    frame = draw_eye_landmarks(frame, left_eye_coords, right_eye_coords)

    print("Left Eye Landmarks:")
    print(left_eye_coords)
    print("Right Eye Landmarks:")
    print(right_eye_coords)

    cv2.imshow('Eye Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
