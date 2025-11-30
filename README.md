Real-Time Auslan Sign Language Detection
 OverviewThis project is a computer vision application designed to detect and translate Australian Sign Language (Auslan) fingerspelling in real-time. It leverages Google MediaPipe for robust hand tracking and a Random Forest Classifier to recognize signs based on skeletal landmark coordinates.The system was built to bridge the communication gap, allowing for the instant translation of static Auslan signs (A-Z, 0-9) into text via a standard webcam.
FeaturesReal-time Detection: Tracks single and double-handed signs instantly.Robust Tracking: Uses MediaPipe to extract 21 skeletal landmarks per hand (x, y, z coordinates).Smart Padding: Automatically handles the difference between one-handed and two-handed signs using zero-padding logic.Machine Learning: Trained on a custom dataset using a Random Forest algorithm for high accuracy. 
Installation & Setup:
1. Clone the Repositorygit clone .x
cd Aus_lan
2. Create a Virtual Environment (Recommended)It is best practice to run this project in a virtual environment to avoid conflicts.
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
3. Install Dependencies: pip install opencv-python mediapipe pandas scikit-learn numpy joblib pillow
 How to Run the ProjectThe project is broken down into numbered steps for clarity. You can run any step independently.
Step 1: Sanity Check (Optional)Test if MediaPipe can see your hands correctly before running the full AI.python step_1_eyes.py
Step 2: Data CollectionIf you want to add new signs to the dataset, run this tool. It saves landmarks directly to CSV.Controls: Press a-z or 0-9 to save a frame. Esc to quit.python step_3_collect_data.py
Step 3: Data ProcessingIf you have raw images in folders, use these scripts to extract landmarks and merge datasets.# Extracts landmarks from images (handles errors robustly)
python step_4_process_clean_images.py

# Merges the alphabet and numeric datasets
python step_4_merge_data.py
Step 4: Train the ModelTrains the Random Forest classifier on final_auslan_data.csv and saves the model as auslan_classifier.joblib.python step_5_train_test.py
Step 5: LIVE TRANSLATION (The Main App)Run this to start the real-time translator.python step_6_live_classifier.py
Technical Workflow: 
Feature Extraction (MediaPipe)We utilize MediaPipe Hands to extract (x, y, z) coordinates for 21 hand landmarks.
Two Hands Supported: The system supports tracking up to 2 hands simultaneously.
Total Features: 21 landmarks × 3 coords × 2 hands = 126 features.
 Preprocessing & Zero PaddingSince some signs use one hand and others use two, the model needs a consistent input shape.If 2 hands detected: We flatten both sets of landmarks into the feature vector.
 If 1 hand detected: We flatten the first hand and fill the remaining 63 features with 0.0 (Zero Padding).This logic ensures the Random Forest model always receives a valid input array of size 126, regardless of the gesture type.
 Model ArchitectureAlgorithm: Random Forest Classifier (Ensemble Learning).
 Estimators: 100 decision trees.
 Input: Flattened 1D array of 126 float values representing spatial coordinates.Output: Predicted class label (e.g., 'A', 'B', '5').
File Structure:
step_1_eyes.py - Basic webcam & MediaPipe test script.
step_2: we kinda tried to over complicated the model and tried to use the combionation od yolo and media pie but it was not working that accurately so had to remove iy.
step_3_collect_data.py - Data collection tool for live webcam entry.
step_4_process_clean_images.py - Converts image folders into CSV data (includes robust reading with PIL/CV2).step_4_merge_data.py - Combines alphabet and numeric datasets into one.
step_5_train_test.py - Cleans data (removes NaNs), trains the Random Forest model, and evaluates accuracy.
step_6_live_classifier.py - The final application for real-time inference.final_auslan_data.csv - The processed training dataset.
Authors:
Bigyan - Lead Developer
Rohan - Collaborator