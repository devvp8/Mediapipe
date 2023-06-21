import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

def detect_eye_landmarks(image):
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_eye_landmarks_3d = [face_landmarks[33], face_landmarks[7], face_landmarks[163],
                                     face_landmarks[144], face_landmarks[145], face_landmarks[153], face_landmarks[154],
                                     face_landmarks[155]]
            
            right_eye_landmarks_3d = [face_landmarks[263], face_landmarks[249], face_landmarks[446],
                                      face_landmarks[374], face_landmarks[380], face_landmarks[381], face_landmarks[382],
                                      face_landmarks[383]]

            return left_eye_landmarks_3d, right_eye_landmarks_3d

    return None, None

def calculate_displacement(current_coords, initial_coords):
    total_displacement = 0

    for current, initial in zip(current_coords, initial_coords):
        # Euclidian Distance
        displacement = math.sqrt((current[0] - initial[0]) ** 2 + (current[1] - initial[1]) ** 2)
        total_displacement += displacement

    return total_displacement

cap = cv2.VideoCapture(0)

initial_left_eye_coords = None
initial_right_eye_coords = None

while True:
    reading, frame = cap.read()

    if not reading:
        break

    left_eye_landmarks_3d, right_eye_landmarks_3d = detect_eye_landmarks(frame)
    left_eye_coords = [] #to store all axes
    right_eye_coords = []
    right_eye_cords=[] #to store x,y axes
    left_eye_cords=[]

    if left_eye_landmarks_3d:
        for landmark in left_eye_landmarks_3d:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            z = landmark.z
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            left_eye_coords.append((x, y, z))
            left_eye_cords.append((x,y))

    if right_eye_landmarks_3d:
        for landmark in right_eye_landmarks_3d:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            z = landmark.z
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            right_eye_coords.append((x,y,z))
            right_eye_cords.append((x,y))

    if left_eye_coords and right_eye_coords:

        if initial_left_eye_coords is None:
            initial_left_eye_coords = left_eye_cords
        if initial_right_eye_coords is None:
            initial_right_eye_coords = right_eye_cords

        left_eye_displacement = calculate_displacement(left_eye_cords, initial_left_eye_coords)
        right_eye_displacement = calculate_displacement(right_eye_cords, initial_right_eye_coords)

        print("Left Eye Displacement:", left_eye_displacement)
        print("Right Eye Displacement:", right_eye_displacement)
        print("Left Eye Landmarks:", left_eye_coords)
        print("Right Eye Landmarks:", right_eye_coords)

    cv2.imshow('Eye Landmarks 3D', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()