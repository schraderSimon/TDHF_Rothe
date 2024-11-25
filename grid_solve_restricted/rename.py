import os
import re

# Define the directory containing the files
directory = "./"  # Change this to the directory where your files are located

# Loop through all files in the directory
pattern = r"Rothe_wavefunctions(\d+\.\d+)_([\d_]+)_(LiH\d?)_(\d+)_(\d\.\d+e[+-]\d+)\.npz"

# Iterate through the files in the directory
for filename in os.listdir(directory):
    match = re.match(pattern, filename)
    if match:
        # Extract parts of the filename
        param1 = match.group(1)
        param2 = match.group(2)
        molecule = match.group(3)
        param3 = match.group(4)
        param4 = match.group(5)

        # Construct the new filename
        new_filename = f"Rothe_wavefunctions_{molecule}_{param1}_{param2}_{param3}_{param4}.npz"

        # Rename the file
        os.rename(filename, new_filename)
        print(f"Renamed: {filename} -> {new_filename}")