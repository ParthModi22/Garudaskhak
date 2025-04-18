# Drone Detection System

This system uses a Software Defined Radio (SDR) with machine learning to detect drones in the vicinity.

## System Requirements

- Python 3.6+
- PlutoSDR or compatible SDR hardware
- [libiio](https://github.com/analogdevicesinc/libiio) library

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/drone-detection-system.git
cd drone-detection-system
```

2. Install required Python packages:
```
pip install -r requirements.txt
```

3. Connect your PlutoSDR device via USB or network.

## Using the System

### 1. Collecting Training Data

To train an accurate model, you'll need to collect both drone data and background/noise data:

```
# To collect drone data (with drone flying nearby):
python drone_detection_system.py --collect-data --num-samples 1000

# To collect background data (with no drone):
python drone_detection_system.py --collect-data --num-samples 1000
```

After collecting background data, rename the output file:
```
mv drone_dataset/drone_data_*.npz drone_dataset/background_*.npz
```

### 2. Training the Model

You have two options for training models:

#### Basic Training
```
python train_drone_model.py
```

#### Advanced Training with Multiple Models
```
python advanced_model_training.py
```

The advanced training script:
- Uses more sophisticated features
- Trains multiple model types (XGBoost, Random Forest, SVM)
- Performs hyperparameter tuning
- Handles class imbalance with SMOTE
- Generates performance comparisons
- Selects the best model automatically

### 3. Running the Detection System

Start the detection system:

```
python drone_detection_system.py
```

This will:
- Connect to your SDR
- Load the trained model
- Display a real-time waterfall display
- Show drone detection probabilities
- Alert when a drone is detected

## Command-line Options

### Data Collection

```
python drone_detection_system.py --collect-data [--num-samples N]
```

- `--collect-data`: Enables data collection mode
- `--num-samples N`: Number of samples to collect (default: 1000)

### Detection System

```
python drone_detection_system.py [--tx-channel N] [--rx-channel N]
```

- `--tx-channel N`: Transmit channel to use (default: 0)
- `--rx-channel N`: Receive channel to use (default: 0)

### Basic Model Training

```
python train_drone_model.py [--dataset-dir DIR] [--output FILE] [--test-size S]
```

- `--dataset-dir DIR`: Directory containing the dataset (default: 'drone_dataset')
- `--output FILE`: Output model filename (default: 'xgboost_drone_detection_model.json')
- `--test-size S`: Fraction of data to use for testing (default: 0.2)

### Advanced Model Training

```
python advanced_model_training.py [--dataset-dir DIR] [--output-dir DIR] [--test-size S] [--no-smote] [--no-advanced-features]
```

- `--dataset-dir DIR`: Directory containing the dataset (default: 'drone_dataset')
- `--output-dir DIR`: Directory to save models and plots (default: 'models')
- `--test-size S`: Fraction of data to use for testing (default: 0.2)
- `--no-smote`: Disable SMOTE class balancing
- `--no-advanced-features`: Use simple feature extraction instead of advanced features

## Feature Engineering

The system extracts advanced features from the radio signals:

1. **Frequency Domain Features**:
   - Spectral statistics (mean, std, skewness, kurtosis)
   - Spectral entropy
   - Peak analysis

2. **Time Domain Features**:
   - Statistical features of I/Q components
   - Zero-crossing rate
   - Envelope analysis

3. **Modulation Features**:
   - AM component detection
   - FM/PM component detection

4. **Drone-Specific Features**:
   - Control band energy ratios
   - Frequency hopping detection

5. **Wavelet Features**:
   - Multi-resolution time-frequency analysis

## Tips for Better Detection

1. **Collect diverse data**: Include drone data from different drones at various distances and angles.

2. **Include environmental variations**: Collect background data in different environments and conditions.

3. **Use advanced training**: The advanced training script will typically yield better results than the basic one.

4. **Try different models**: The system will compare XGBoost, Random Forest, and SVM models to find the best one.

5. **Adjust SDR parameters**: Try different gain settings and sample rates for optimal performance.

6. **Hardware placement**: Place your SDR antenna with clear line of sight for best reception.

## Troubleshooting

### SDR Connection Issues

If you have trouble connecting to your SDR:

1. Check USB connection or network settings
2. Ensure the SDR is powered properly
3. Verify libiio is installed correctly
4. Try unplugging and reconnecting the device

### Performance Issues

If the system is running slowly:

1. Reduce the sample rate
2. Close other applications consuming CPU resources
3. Adjust buffer sizes for better performance
4. Use simpler feature extraction with `--no-advanced-features`

### Detection Accuracy

If detection accuracy is poor:

1. Collect more training data from your specific drone
2. Ensure training data includes both drone and background samples
3. Adjust detection threshold in the code (`PROBABILITY_THRESHOLD` in drone_detection_system.py)
4. Try running the advanced model training

### Training Failures

If model training fails:

1. Make sure you have sufficient drone and background samples
2. Check that PyWavelets is installed if using advanced features
3. Try using simpler features with `--no-advanced-features`
4. Disable SMOTE with `--no-smote` if you encounter memory errors

## License

This project is licensed under the MIT License - see the LICENSE file for details. 