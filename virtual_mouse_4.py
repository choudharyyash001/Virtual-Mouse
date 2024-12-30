import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize PyAutoGUI
pyautogui.PAUSE = 0.01

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Variables to track gestures
clicking = False
dragging = False

# Function to smooth mouse movement
def smooth_mouse_movement(current_x, current_y, previous_x, previous_y, smoothing_factor=0.7):
    new_x = int(previous_x * smoothing_factor + current_x * (1 - smoothing_factor))
    new_y = int(previous_y * smoothing_factor + current_y * (1 - smoothing_factor))
    return new_x, new_y

# Initialize previous positions
prev_x, prev_y = 0, 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger tip and thumb tip positions
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            index_x = int(index_finger_tip.x * img.shape[1])
            index_y = int(index_finger_tip.y * img.shape[0])
            thumb_x = int(thumb_tip.x * img.shape[1])
            thumb_y = int(thumb_tip.y * img.shape[0])

            # Calculate distance between index finger tip and thumb tip
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

            # Smooth the mouse movement
            index_x, index_y = smooth_mouse_movement(index_x, index_y, prev_x, prev_y)
            prev_x, prev_y = index_x, index_y

            # Move mouse cursor
            pyautogui.moveTo(index_x, index_y)

            # Gesture for clicking
            if distance < 20:
                if not clicking:
                    pyautogui.click()
                    clicking = True
            else:
                clicking = False

            # Gesture for dragging
            if distance < 40:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Gesture for scrolling (vertical movement of index finger)
            y_diff = index_finger_tip.y - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            if abs(y_diff) > 0.02:
                pyautogui.scroll(int(y_diff * 1000))

            # Draw hand landmarks on the image
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Virtual Mouse', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
