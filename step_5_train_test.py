import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np # Import numpy for checking

# --- 1. Load the "Textbook" ---
CSV_PATH = 'final_auslan_data.csv'
print(f"Loading data from {CSV_PATH}...")

try:
    # Load only the first 127 columns (index 0 to 126)
    df = pd.read_csv(CSV_PATH, usecols=range(127)) # <<<--- ADD THIS ARGUMENT
except FileNotFoundError:
    print(f"*** ERROR: Cannot find {CSV_PATH}! Did you run the merge script? ***")
    exit()

if df.empty:
    print(f"*** ERROR: {CSV_PATH} is empty! Something went wrong. ***")
    exit()

print(f"Data loaded successfully! Found {len(df)} samples.")
print("Initial sample counts per label:")
print(df['label'].value_counts().sort_index()) # Show counts before cleaning

# --- *** NEW: Data Cleaning Step *** ---
print("\nChecking for non-numeric data in feature columns...")
feature_cols = df.columns[1:] # Get all columns except 'label'
bad_rows = []

# Loop through each feature column and try to convert to numeric
# errors='coerce' will turn any non-numeric value into NaN (Not a Number)
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Find rows that now contain NaN values (meaning they had bad data)
nan_rows_mask = df.isnull().any(axis=1)
bad_rows_df = df[nan_rows_mask]

if not bad_rows_df.empty:
    print(f"\n*** WARNING: Found {len(bad_rows_df)} rows with non-numeric data! ***")
    print("These rows likely caused the error. Displaying first 5:")
    print(bad_rows_df.head())
    print("\nDropping these rows before training...")
    # Drop the rows with NaN values
    df = df.dropna()
    print(f"Data shape after dropping bad rows: {df.shape}")
    if df.empty:
         print(f"*** ERROR: All data was bad after cleaning! Cannot train. Check data collection. ***")
         exit()
    print("\nSample counts after cleaning:")
    print(df['label'].value_counts().sort_index())
else:
    print("Data seems clean (all feature values are numeric).")
# --- *** END Data Cleaning Step *** ---


# --- 2. Prepare the Data ---
print("\nPreparing data for training...")
labels = df['label']
features = df.drop('label', axis=1)

if len(labels) < 2:
    print("*** ERROR: Not enough data left after cleaning to train! Need at least 2 samples. ***")
    exit()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")
except ValueError as e:
    print(f"\n*** WARNING: Could not stratify split ({e}). Splitting without stratifying. ***")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )
    print(f"Data split (unstratified): {len(X_train)} training samples, {len(X_test)} testing samples.")


# --- 3. Train the "Brain" ---
print("\nTraining the RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete!")

# --- 4. Test the "Brain" ---
print("\nTesting the model on unseen data...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- RESULTS ---")
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
print("---------------")

if accuracy > 0.95: # Adjusted threshold for full dataset
    print("Excellent! The model learned well from your data. 🎉")
elif accuracy > 0.80:
    print("Good result! Might improve further with more varied data.")
else:
    print("Hmm, accuracy is lower than expected. Review data quality/quantity for tricky signs.")

# --- 5. Save the Trained "Brain" ---
MODEL_SAVE_PATH = 'auslan_classifier.joblib'
joblib.dump(model, MODEL_SAVE_PATH)
print(f"\nTrained model saved to: {MODEL_SAVE_PATH}")
print("We can now use this file in our final application!")