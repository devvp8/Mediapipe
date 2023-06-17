import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

def print_result(results):
    for detection in results.detections:
        print('Detection score:', detection.score)
        print('Location data:', detection.location_data.relative_bounding_box)


with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as detector:
    # Open a video stream (you can replace 0 with the video file path)
    video_stream = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video stream
        ret, frame = video_stream.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detector.process(frame_rgb)

        print_result(results)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                    int(bbox.width * iw), int(bbox.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    video_stream.release()
    cv2.destroyAllWindows()
