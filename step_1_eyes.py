import cv2
import mediapipe as mp

# --- 1. Setup ---

# Get the tools from MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hand Landmarker ("the eyes")
# We set it to find a maximum of 2 hands
hands = mp_hands.Hands(max_num_hands=2, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.5)

# Start the webcam
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

# --- 2. The Main Loop (Runs for every video frame) ---

while cap.isOpened():
    # Read one frame from the webcam
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # --- 3. Process the Image ---
    
    # Flip the image horizontally (so it's a "selfie" view)
    image = cv2.flip(image, 1)

    # IMPORTANT: Convert the image from BGR (OpenCV) to RGB (MediaPipe)
    # MediaPipe was trained on RGB images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tell the "eyes" to process the image and find hands
    results = hands.process(image_rgb)

    # --- 4. Draw the Results ---

    # We need to draw on the *original* BGR image, not the RGB one
    # So we'll use the 'image' variable
    
    # Check if any hands were found
    if results.multi_hand_landmarks:
        # Loop through each hand it found (max 2)
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Use the "drawing kit" to draw the 21 dots
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,  # This draws the lines connecting the dots
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # --- 5. Show the Image ---
    
    # Display the final image in a window
    cv2.imshow('MediaPipe Hand Tracking', image)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---

print("Shutting down...")
hands.close()
cap.release()
cv2.destroyAllWindows()