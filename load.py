import numpy as np
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use the full path to the file
file_path = os.path.join(current_dir, 'drone_data_20250414_144053.npz')

print(f"Loading file from: {file_path}")
data = np.load(file_path)

print("Available arrays in the file:")
print(data.files)

# Print the shape of the first array to understand its structure
first_key = data.files[0]
print(f"\nFirst array ({first_key}) shape: {data[first_key].shape}")

# If it's the IQ samples, print some basic statistics
if 'iq_samples' in data.files:
    iq_samples = data['iq_samples']
    print(f"\nIQ samples shape: {iq_samples.shape}")
    print(f"Number of samples: {len(iq_samples)}")
    print(f"Each sample length: {iq_samples[0].shape if len(iq_samples) > 0 else 'N/A'}")
    
    # Print some basic statistics
    if len(iq_samples) > 0:
        first_sample = iq_samples[0]
        print(f"\nFirst sample statistics:")
        print(f"Mean magnitude: {np.mean(np.abs(first_sample)):.6f}")
        print(f"Max magnitude: {np.max(np.abs(first_sample)):.6f}")
        print(f"Min magnitude: {np.min(np.abs(first_sample)):.6f}")

