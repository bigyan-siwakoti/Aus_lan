import pandas as pd
import os

# --- Define File Paths ---
alphabet_csv = 'my_auslan_data.csv'
numeric_csv = 'my_auslan_data_FULL.csv'
output_csv = 'final_auslan_data.csv'

print("Starting data merge...")

# --- Check if files exist ---
if not os.path.exists(alphabet_csv):
    print(f"*** ERROR: Cannot find alphabet data file: {alphabet_csv}! ***")
    exit()
if not os.path.exists(numeric_csv):
    print(f"*** ERROR: Cannot find numeric data file: {numeric_csv}! ***")
    exit()

# --- Load the datasets ---
try:
    print(f"Loading alphabet data from {alphabet_csv}...")
    df_alpha = pd.read_csv(alphabet_csv)
    print(f"Found {len(df_alpha)} alphabet samples.")

    print(f"Loading numeric data from {numeric_csv}...")
    df_numeric = pd.read_csv(numeric_csv)
    print(f"Found {len(df_numeric)} numeric samples.")

except Exception as e:
    print(f"*** ERROR loading CSV files: {e} ***")
    exit()

# --- Combine the datasets ---
print("\nCombining datasets...")
# ignore_index=True resets the row numbers in the new combined dataframe
df_final = pd.concat([df_alpha, df_numeric], ignore_index=True)

# --- Save the final dataset ---
try:
    df_final.to_csv(output_csv, index=False) # index=False prevents pandas from adding an extra column
    print(f"\nSuccessfully merged data!")
    print(f"Total samples: {len(df_final)}")
    print(f"Final dataset saved to: {output_csv}")
    print("\nSample counts per label in final dataset:")
    print(df_final['label'].value_counts().sort_index()) # Show counts sorted

except Exception as e:
    print(f"*** ERROR saving final CSV file: {e} ***")