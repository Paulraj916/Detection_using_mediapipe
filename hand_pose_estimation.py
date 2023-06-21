import cv2
import mediapipe as mp

# Initialize the MediaPipe Hands model
mp_hands = mp.solutions.hands

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the drawing utilities
mp_drawing = mp.solutions.drawing_utils 

with mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5) as hands:

    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to RGB for input to the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform hand tracking on the frame
        results = hands.process(frame_rgb)

        # If hands are detected, draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with landmarks drawn
        cv2.imshow('Hand Pose Estimation', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
