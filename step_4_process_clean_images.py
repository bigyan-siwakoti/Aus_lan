import cv2
import mediapipe as mp
import os
import csv
import numpy as np # Needed for PIL conversion
from PIL import Image # Our backup image reader

# --- 1. Setup ---
print("Starting Task 5 (Robust Version): Extracting Landmarks")
print("Using the better dataset and smarter image reading!")

# Get the tools from MediaPipe (the "eyes")
mp_hands = mp.solutions.hands

# Use the low confidence setting
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.1)

# --- 2. Define Paths ---
DATASET_PATH = 'processed_combine_asl_dataset' # Correct folder
OUTPUT_CSV_PATH = 'asl_landmark_data_clean.csv' # Correct output file

# --- 3. Create the "Textbook" (CSV File) ---
# Full 127-column header
header = ['label']
for i in range(1, 43): # 42 landmarks (21 per hand)
    header += [f'x{i}', f'y{i}', f'z{i}']

# Open the new CSV file in 'write' mode
with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    print(f"Created {OUTPUT_CSV_PATH}. Starting to process images...")
    total_images_processed = 0
    total_images_skipped = 0

    # --- 4. The Main Loop (Same as before) ---
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(label_path):
            continue

        print(f"--- Processing folder: {label} ---")
        images_processed_in_folder = 0
        images_skipped_in_folder = 0

        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = None # Reset image variable

            # --- *** ROBUST IMAGE READING *** ---
            try:
                # Try OpenCV first
                image = cv2.imread(image_path)

                if image is None:
                    # If OpenCV fails, try PIL as a backup
                    try:
                        pil_image = Image.open(image_path).convert('RGB')
                        # Convert PIL image to OpenCV format (RGB -> BGR)
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        #print(f"DEBUG: Read {image_name} using PIL") # Optional debug print
                    except Exception as e_pil:
                         #print(f"Warning: Could not read image {image_name} with OpenCV or PIL ({e_pil}). Skipping.") # Optional debug print
                         images_skipped_in_folder += 1
                         continue # Skip to next image

                if image is None: # Double check if PIL also failed
                     #print(f"Warning: Still could not read image {image_name} after PIL attempt. Skipping.") # Optional debug print
                     images_skipped_in_folder += 1
                     continue # Skip to next image

            except Exception as e_read:
                #print(f"Error during image reading {image_name}: {e_read}. Skipping.") # Optional debug print
                images_skipped_in_folder += 1
                continue # Skip to next image
            # --- *** END ROBUST READING *** ---


            # --- MediaPipe Processing ---
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
            except Exception as e_process:
                #print(f"Error processing image {image_name} with MediaPipe: {e_process}. Skipping.") # Optional debug print
                images_skipped_in_folder += 1
                continue # Skip to next image


            # --- 5. The "Padding" Logic (Same as before) ---
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

                writer.writerow( landmarks_row)
                images_processed_in_folder += 1
            else:
                 # MediaPipe found no hands, even with low confidence
                 images_skipped_in_folder += 1


        print(f"--- Finished folder: {label}. Successfully processed {images_processed_in_folder} images. Skipped {images_skipped_in_folder} images. ---")
        total_images_processed += images_processed_in_folder
        total_images_skipped += images_skipped_in_folder

    print("\n--- Landmark Extraction Complete! ---")
    print(f"Total images successfully processed: {total_images_processed}")
    print(f"Total images skipped (unreadable or no hands detected): {total_images_skipped}")
    print(f"Your 'textbook' is ready: {OUTPUT_CSV_PATH}")

# Clean up
hands.close()