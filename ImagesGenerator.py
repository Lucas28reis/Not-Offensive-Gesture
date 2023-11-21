import cv2
import mediapipe as mp
import time
from PIL import Image, ImageDraw

# mediapipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# webcam init
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    capture_interval = 5  # Intervalo de captura em segundos
    last_capture_time = time.time() - capture_interval

    prev_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Hands coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))

                if prev_landmarks is not None:
                    # Calculate the mean squared error (MSE) between current and previous landmarks
                    mse = sum(((prev[0] - current[0]) ** 2 + (prev[1] - current[1]) ** 2) for prev, current in zip(prev_landmarks, landmarks)) / len(landmarks)
                    if mse < 1000:  # Adjust this threshold as needed
                        # Bounding box
                        x_min = min(landmarks, key=lambda x: x[0])[0]
                        x_max = max(landmarks, key=lambda x: x[0])[0]
                        y_min = min(landmarks, key=lambda x: x[1])[1]
                        y_max = max(landmarks, key=lambda x: x[1])[1]

                        # Resize bounding box 
                        scale = 1.2
                        width = (x_max - x_min) * 3.5
                        height = y_max - y_min
                        x_min = max(0, x_min - int(width * (scale - 1) / 2))
                        x_max = min(frame.shape[1], x_max + int(width * (scale - 1) / 2))
                        y_min = max(0, y_min - int(height * (scale - 1) / 2))
                        y_max = min(frame.shape[0], y_max + int(height * (scale - 1) / 2))
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Capture image if the capture interval has passed
                        current_time = time.time()
                        if current_time - last_capture_time >= capture_interval:
                            last_capture_time = current_time

                            # Crop and save the hand image with a white background
                            hand_image = frame[y_min:y_max, x_min:x_max]
                            white_background = Image.new('RGB', (hand_image.shape[1], hand_image.shape[0]), (255, 255, 255))
                            white_background.paste(Image.fromarray(hand_image), (0, 0))
                            white_background.save(f'imagens/hand_{int(current_time)}.png')
                
                prev_landmarks = landmarks

        # Show resulting image
        cv2.imshow('Hand Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
