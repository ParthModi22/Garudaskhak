#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime

# Configuration
MODEL_OUTPUT = 'xgboost_drone_detection_model.json'
DEFAULT_DATASET_DIR = 'drone_dataset'
FFT_SIZE = 2048  # Size for feature extraction
FEATURES_TO_USE = FFT_SIZE // 2  # Use half of FFT bins as features

def extract_features(iq_data, n_fft=FFT_SIZE):
    """Extract frequency domain features from IQ samples"""
    # Use absolute value of IQ samples
    data_magnitude = np.abs(iq_data)
    
    # Compute DFT
    dft = np.fft.fft(data_magnitude, n=n_fft)
    
    # Extract one-sided magnitude spectrum
    magnitude_spectrum = np.abs(dft[:n_fft//2])
    
    # Normalize
    if np.max(magnitude_spectrum) > 0:
        magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
    
    return magnitude_spectrum

def load_dataset(dataset_dir, drone_label=1, background_label=0, test_size=0.2):
    """Load and prepare the drone detection dataset"""
    print(f"Loading dataset from {dataset_dir}")
    
    # Find all drone data files
    drone_files = glob.glob(f"{dataset_dir}/drone_data_*.npz")
    
    # Find all background/noise data files
    background_files = glob.glob(f"{dataset_dir}/background_*.npz")
    
    if not drone_files:
        print(f"No drone data files found in {dataset_dir}")
        return None, None, None, None
    
    if not background_files:
        print(f"Warning: No background data files found. Will need to create synthetic background data.")
    
    # Load drone data
    drone_samples = []
    for file in drone_files:
        try:
            data = np.load(file, allow_pickle=True)
            samples = data['iq_samples']
            drone_samples.extend(samples)
            print(f"Loaded {len(samples)} drone samples from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Load background data if available
    background_samples = []
    if background_files:
        for file in background_files:
            try:
                data = np.load(file, allow_pickle=True)
                samples = data['iq_samples']
                background_samples.extend(samples)
                print(f"Loaded {len(samples)} background samples from {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    else:
        # Generate synthetic background/noise data
        print("Generating synthetic background noise data...")
        num_synthetic = len(drone_samples)
        
        for _ in range(num_synthetic):
            # Generate random noise with similar characteristics to the drone samples
            sample_length = len(drone_samples[0])
            noise = np.random.normal(0, 0.1, sample_length) + 1j * np.random.normal(0, 0.1, sample_length)
            background_samples.append(noise)
        
        print(f"Generated {num_synthetic} synthetic background samples")
    
    # Extract features
    print("Extracting features...")
    
    X_drone = np.array([extract_features(sample) for sample in drone_samples])
    X_background = np.array([extract_features(sample) for sample in background_samples])
    
    # Create labels
    y_drone = np.ones(len(X_drone)) * drone_label
    y_background = np.zeros(len(X_background)) * background_label
    
    # Combine datasets
    X = np.vstack((X_drone, X_background))
    y = np.hstack((y_drone, y_background))
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Drone samples: {len(X_drone)}, Background samples: {len(X_background)}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    """Train an XGBoost model for drone detection"""
    print("Training XGBoost model...")
    
    # Define model parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    # Create and train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Feature importance
    feature_importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'feature_importance_{timestamp}.png')
    plt.close()
    
    return model

def collect_background_data():
    """Instructions for collecting background/noise data"""
    print("\nTo improve model accuracy, you should collect background/noise data:")
    print("1. Run the detector with no drone present to collect background noise.")
    print("2. Use the following command:")
    print("   python drone_detection_system.py --collect-data")
    print("3. Rename the collected file by running:")
    print("   mv drone_dataset/drone_data_*.npz drone_dataset/background_<timestamp>.npz")
    print("\nCollect data under different environmental conditions for best results.")

def main():
    parser = argparse.ArgumentParser(description='Train Drone Detection Model')
    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_DIR,
                        help=f'Directory containing drone and background datasets (default: {DEFAULT_DATASET_DIR})')
    parser.add_argument('--output', type=str, default=MODEL_OUTPUT,
                        help=f'Output model file name (default: {MODEL_OUTPUT})')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    # Ensure dataset directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"Dataset directory {args.dataset_dir} not found.")
        print(f"Creating directory {args.dataset_dir}")
        os.makedirs(args.dataset_dir)
        collect_background_data()
        return
    
    # Load and prepare dataset
    X_train, X_test, y_train, y_test = load_dataset(args.dataset_dir, test_size=args.test_size)
    
    if X_train is None:
        print("No valid dataset found. Please collect data first.")
        collect_background_data()
        return
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    model.save_model(args.output)
    print(f"Model saved to {args.output}")
    
    # Display instructions for using the model
    print("\nTo use this model for drone detection, run:")
    print(f"python drone_detection_system.py")

if __name__ == "__main__":
    main() 