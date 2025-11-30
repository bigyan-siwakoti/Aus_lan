import cv2
import mediapipe as mp
import joblib # To load our saved model
import numpy as np # To reshape data for the model

# --- 1. Load the Trained "Brain" ---
MODEL_PATH = 'auslan_classifier.joblib'
print(f"Loading trained model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"*** ERROR: Cannot find {MODEL_PATH}! Did you run the training script? ***")
    exit()

# --- 2. Setup MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, # Still look for 2 hands
                       min_detection_confidence=0.3, # Use the same settings as data collection
                       min_tracking_confidence=0.5)

# --- 3. Start Webcam ---
cap = cv2.VideoCapture(0)
print("\nStarting live classification...")
print("Make the 'A', 'B', or 'C' sign. Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip image, convert to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image with MediaPipe
    results = hands.process(image_rgb)

    predicted_label = "???" # Default text if no hands or prediction

    # --- 4. Prepare Landmark Data for Prediction ---
    if results.multi_hand_landmarks:
        # Use the SAME padding logic as data collection
        live_landmarks = []

        # Hand 1
        hand1_landmarks = results.multi_hand_landmarks[0].landmark
        for landmark in hand1_landmarks:
            live_landmarks.extend([landmark.x, landmark.y, landmark.z])

        # Hand 2 (or padding)
        if len(results.multi_hand_landmarks) == 2:
            hand2_landmarks = results.multi_hand_landmarks[1].landmark
            for landmark in hand2_landmarks:
                live_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            live_landmarks.extend([0.0] * 63) # Pad with 63 zeros

        # --- Safety Check: Ensure we have 126 features ---
        if len(live_landmarks) == 126:
            # Convert list to numpy array and reshape for the model
            # The model expects a 2D array: [[feature1, feature2, ...]]
            landmark_array = np.array(live_landmarks).reshape(1, -1)

            # --- 5. Make Prediction ---
            try:
                prediction = model.predict(landmark_array)
                predicted_label = prediction[0].upper() # Get the predicted letter (e.g., 'a') and make it uppercase
            except Exception as e:
                print(f"Error during prediction: {e}")
                predicted_label = "ERR"
        #else: # Optional debug
            #print(f"Debug: Landmark count mismatch: {len(live_landmarks)}")


    # --- 6. Draw Landmarks (Optional but helpful) ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # --- 7. Display Prediction on Image ---
    # Choose font, position, size, color, thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (50, 100) # (x, y) - from top-left corner
    font_scale = 3
    font_color = (255, 0, 0) # Blue in BGR
    line_type = 2

    cv2.putText(image, predicted_label,
                position,
                font,
                font_scale,
                font_color,
                line_type)

    # Show the image
    cv2.imshow('Live Auslan Classifier (A, B, C)', image)

    # Quit with 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 8. Cleanup ---
print("\nShutting down...")
hands.close()
cap.release()
cv2.destroyAllWindows()