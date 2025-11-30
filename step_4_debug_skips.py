import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from PIL import Image

# --- 1. Setup ---
print("Starting Task 5 (DEBUG SKIPS): Let's see why!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.1) # Still low confidence

# --- 2. Define Paths ---
DATASET_PATH = 'processed_combine_asl_dataset'
OUTPUT_CSV_PATH = 'debug_output.csv' # We don't care about the CSV now

# --- 3. Open Dummy CSV ---
# We still need this structure, but we won't save much
header = ['label'] + [f'{coord}{i}' for i in range(1, 43) for coord in ['x', 'y', 'z']]
with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    print(f"Created dummy {OUTPUT_CSV_PATH}. Starting to debug images...")

    # --- 4. The Main Loop (Limit to ONE folder for speed) ---
    # Let's just process ONE folder, e.g., 'a', to speed up debugging
    label_to_debug = 'a' # <--- CHANGE THIS if 'a' is empty or problematic
    label_path = os.path.join(DATASET_PATH, label_to_debug)

    if not os.path.isdir(label_path):
        print(f"*** ERROR: Debug folder '{label_to_debug}' not found! Exiting. ***")
        exit() # Stop the script if the folder doesn't exist

    print(f"--- DEBUGGING folder: {label_to_debug} ---")
    images_processed_in_folder = 0
    images_skipped_reading = 0
    images_skipped_detection = 0
    images_tried = 0

    # Limit the number of images we check per folder for speed
    MAX_IMAGES_TO_CHECK = 20

    for image_name in os.listdir(label_path):
        if images_tried >= MAX_IMAGES_TO_CHECK:
            print(f"--- Reached MAX_IMAGES_TO_CHECK ({MAX_IMAGES_TO_CHECK}). Moving on. ---")
            break # Stop checking images in this folder

        images_tried += 1
        image_path = os.path.join(label_path, image_name)
        image = None
        print(f"\nAttempting to process: {image_name}") # DEBUG PRINT

        # --- *** ROBUST IMAGE READING WITH DEBUG *** ---
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"  DEBUG: cv2.imread failed for {image_name}.") # DEBUG PRINT
                try:
                    pil_image = Image.open(image_path).convert('RGB')
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB_BGR)
                    print(f"  DEBUG: PIL successfully read {image_name}.") # DEBUG PRINT
                except Exception as e_pil:
                     print(f"  DEBUG: PIL failed too for {image_name}. Reason: {e_pil}. SKIPPING.") # DEBUG PRINT
                     images_skipped_reading += 1
                     continue

            if image is None: # Double check
                 print(f"  DEBUG: Image is still None after PIL for {image_name}. SKIPPING.") # DEBUG PRINT
                 images_skipped_reading += 1
                 continue
            #else: # Optional: confirm successful reading
            #    print(f"  DEBUG: Successfully read {image_name} (shape: {image.shape})")

        except Exception as e_read:
            print(f"  DEBUG: CRITICAL ERROR during reading {image_name}: {e_read}. SKIPPING.") # DEBUG PRINT
            images_skipped_reading += 1
            continue
        # --- *** END ROBUST READING *** ---


        # --- MediaPipe Processing WITH DEBUG ---
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
        except Exception as e_process:
            print(f"  DEBUG: MediaPipe process() failed for {image_name}: {e_process}. SKIPPING.") # DEBUG PRINT
            images_skipped_reading += 1 # Count as a reading/processing failure
            continue


        # --- 5. Analyze Results ---
        if results.multi_hand_landmarks:
            print(f"  SUCCESS: MediaPipe FOUND {len(results.multi_hand_landmarks)} hand(s) in {image_name}!") # DEBUG PRINT
            # (We won't actually write to CSV in this debug script)
            images_processed_in_folder += 1
            # You could add the writing logic back here if needed for testing
        else:
             print(f"  DEBUG: MediaPipe found NO hands in {image_name}. SKIPPING.") # DEBUG PRINT
             images_skipped_detection += 1


    print(f"\n--- DEBUG Summary for folder '{label_to_debug}' ---")
    print(f"Images attempted: {images_tried}")
    print(f"Images skipped due to READING failure: {images_skipped_reading}")
    print(f"Images skipped due to DETECTION failure: {images_skipped_detection}")
    print(f"Images successfully processed (hands found): {images_processed_in_folder}")
    print("--- End Debug Script ---")

# Clean up
hands.close()