# import os

# folder = "capno"

# for filename in os.listdir(folder):
#     if filename.endswith(".csv"):
#         # split on underscore to separate ID from rest
#         parts = filename.split("_")
#         file_id = parts[0]  # e.g., '0009'
        
#         # add 1000 to the numeric ID
#         new_id = str(int(file_id) + 1000)
        
#         # build new filename
#         new_name = f"bidmc_{new_id}_Signals.csv"
        
#         # full paths
#         old_path = os.path.join(folder, filename)
#         new_path = os.path.join(folder, new_name)
        
#         # rename
#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} -> {new_name}")


import os
import pandas as pd
import numpy as np
from scipy.signal import resample
from scipy import signal



# def downsample_signal(signal_data: np.ndarray, 
#                       original_rate: int, 
#                       target_rate: int) -> np.ndarray:
#     """Downsample the signal to target sampling rate using resample_poly."""
#     if original_rate == target_rate:
#         return signal_data
    
#     # Use rational resampling (better than decimate for non-integer ratios)
#     downsampled = signal.resample_poly(signal_data, target_rate, original_rate)
#     return downsampled


# # Directory with CapnoBase files
# input_dir = 'capno'

# original_fs = 300
# target_fs = 125

# for filename in os.listdir(input_dir):
#     if filename.endswith('.csv'):
#         filepath = os.path.join(input_dir, filename)
#         df = pd.read_csv(filepath)

#         if 'PPG' in df.columns:
#             print("Processing", filename)
#             df = df.rename(columns={'PPG': 'PLETH'})
#             ppg = df['PLETH'].values

#             # Downsample
#             downsampled_ppg = downsample_signal(ppg, original_fs, target_fs)

#             # Create new DataFrame with correct length
#             new_df = pd.DataFrame({'PLETH': downsampled_ppg})

#             # Save back
#             new_df.to_csv(filepath, index=False)



# Example files
# capno_file = "capno/bidmc_1009_Signals.csv"
# bidmc_file = "data/bidmc_data/bidmc_csv/bidmc_01_Signals.csv"

# def check_range(filepath, col_name_candidates=["PLETH"]):
#     df = pd.read_csv(filepath)
#     # Find the correct column name
#     df.columns = df.columns.str.strip()
#     for col in col_name_candidates:
#         if col in df.columns:
#             sig = df[col].values
#             return sig.min(), sig.max()
#     raise ValueError(f"No PPG column found in {filepath}")

# # Check ranges
# capno_min, capno_max = check_range(capno_file)
# bidmc_min, bidmc_max = check_range(bidmc_file)

# print("CapnoBase:", capno_min, "to", capno_max)
# print("BIDMC    :", bidmc_min, "to", bidmc_max)



input_dir = 'capno'

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        print("yes")
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)
        # Drop the second row (index 1)
        new_df = df.iloc[1:].reset_index(drop=True)

        # Overwrite the same file
        new_df.to_csv(filepath, index=False)