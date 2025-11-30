import cv2
import mediapipe as mp
import os
import csv

# --- 1. Setup ---
print("Starting Data Collection Tool (FINAL VERSION)...")
print("Saving directly to final_auslan_data.csv")
print("Press 'Esc' to quit.")
print("Press 'a'-'z' (excluding j, z) or '0'-'9' keys to save data.")

# --- 2. Setup MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.5)

# --- 3. Setup CSV File ---
OUTPUT_CSV_PATH = 'final_auslan_data.csv' # Saving directly to the final file name
FILE_EXISTS = os.path.isfile(OUTPUT_CSV_PATH)

header = ['label']
for i in range(1, 43):
    header += [f'x{i}', f'y{i}', f'z{i}']

# Open the file in "append" mode ('a').
# *** IMPORTANT: Make sure you deleted the old final_auslan_data.csv first! ***
csv_file = open(OUTPUT_CSV_PATH, 'a', newline='')
writer = csv.writer(csv_file)

if not FILE_EXISTS:
    writer.writerow(header)
    print(f"Created new file: {OUTPUT_CSV_PATH}")
else:
    print(f"Appending to existing file: {OUTPUT_CSV_PATH}")
    print("!!! WARNING: Make sure you deleted the old CSV before starting! !!!")


# --- 4. Start Webcam ---
cap = cv2.VideoCapture(0)
samples_saved_session = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('Data Collection Tool - Press keys to save', image)

    # --- 5. The "Key Press" Logic ---
    key = cv2.waitKey(5) & 0xFF
    label = None # Reset label

    # Quit with Escape key (27)
    if key == 27:
        print("Escape key pressed, quitting...")
        break

    # Check for lowercase letters (a-z), excluding j (106) and z (122)
    if 97 <= key <= 122 and key != 106 and key != 122:
        label = chr(key)

    # Check for numbers (0-9)
    elif 48 <= key <= 57:
         label = chr(key)

    # --- Process if a valid key was pressed ---
    if label is not None:
        #print(f"DEBUG: Valid key pressed. Just before saving, label is: '{label}'") # Keep commented unless debugging

        if results.multi_hand_landmarks:
            landmarks_row = [label]
            # Hand 1
            hand1_landmarks = results.multi_hand_landmarks[0].landmark
            for landmark in hand1_landmarks:
                landmarks_row.extend([landmark.x, landmark.y, landmark.z])
            # Hand 2 (or padding)
            if len(results.multi_hand_landmarks) == 2:
                hand2_landmarks = results.multi_hand_landmarks[1].landmark
                for landmark in hand2_landmarks:
                    landmarks_row.extend([landmark.x, landmark.y, landmark.z])
            else:
                landmarks_row.extend([0.0] * 63)

            if len(landmarks_row) == 127: # Strict check for 127 columns
                writer.writerow(landmarks_row)
                samples_saved_session += 1
                print(f"SAVED! Label: '{label}', Samples this session: {samples_saved_session}")
            else:
                # This error should NOT happen if logic is correct
                print(f"!!! CRITICAL ERROR: Landmark row length generated was {len(landmarks_row)} instead of 127. Skipping save. PLEASE DEBUG !!!")
        else:
            print(f"Pressed '{label}', but no hands detected!")

# --- 6. Cleanup ---
print(f"\nShutting down... Saved {samples_saved_session} samples this session.")
csv_file.close()
hands.close()
cap.release()
cv2.destroyAllWindows()