import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('hand_gesture_recognition_model.h5')

# Define unique labels (ensure this matches your training labels)
unique_labels = ['palm', 'I', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand landmarks and process for gesture recognition
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box coordinates for the detected hand
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * width), int(max(x_coords) * width)
            y_min, y_max = int(min(y_coords) * height), int(max(y_coords) * height)
            
            # Crop and resize the detected hand region for classification
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue  # Skip empty frames
            
            hand_img = cv2.resize(hand_img, (64, 64))  # Resize to match model input
            hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension
            hand_img = hand_img / 255.0  # Normalize

            # Predict the gesture
            predictions = model.predict(hand_img)
            predicted_class = np.argmax(predictions, axis=1)
            gesture = unique_labels[predicted_class[0]]

            # Display the result
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
