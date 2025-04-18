#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import xgboost as xgb
import joblib
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Import the advanced feature extraction
from feature_engineering import extract_advanced_features, extract_features_for_ml

# Configuration
MODEL_OUTPUT_DIR = 'models'
DEFAULT_DATASET_DIR = 'drone_dataset'

def load_and_preprocess_data(dataset_dir, test_size=0.2, use_advanced_features=True):
    """Load, extract features, and preprocess the drone detection dataset"""
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
    
    # Extract features - use advanced features if requested
    print("Extracting features...")
    
    if use_advanced_features:
        print("Using advanced feature extraction...")
        X_drone = extract_features_for_ml(drone_samples)
        X_background = extract_features_for_ml(background_samples)
    else:
        # Simple feature extraction (using FFT magnitudes)
        print("Using simple FFT-based feature extraction...")
        # Define a function for simple feature extraction
        def simple_feature_extract(iq_data, n_fft=2048):
            data_magnitude = np.abs(iq_data)
            dft = np.fft.fft(data_magnitude, n=n_fft)
            magnitude_spectrum = np.abs(dft[:n_fft//2])
            if np.max(magnitude_spectrum) > 0:
                magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
            return magnitude_spectrum
        
        X_drone = np.array([simple_feature_extract(sample) for sample in drone_samples])
        X_background = np.array([simple_feature_extract(sample) for sample in background_samples])
    
    # Create labels
    y_drone = np.ones(len(X_drone))
    y_background = np.zeros(len(X_background))
    
    # Combine datasets
    X = np.vstack((X_drone, X_background))
    y = np.hstack((y_drone, y_background))
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)
    
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Drone samples: {len(X_drone)}, Background samples: {len(X_background)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    return X_train, X_test, y_train, y_test, scaler

def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {np.bincount(y_train.astype(int))}")
    print(f"Balanced class distribution: {np.bincount(y_train_balanced.astype(int))}")
    
    return X_train_balanced, y_train_balanced

def tune_xgboost(X_train, y_train, cv=5):
    """Tune XGBoost hyperparameters using GridSearchCV"""
    print("Tuning XGBoost hyperparameters...")
    
    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Create a base model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # Set up GridSearchCV
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=stratified_kfold,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Return the best model
    return grid_search.best_estimator_

def train_and_evaluate_models(X_train, X_test, y_train, y_test, use_smote=True):
    """Train and evaluate multiple models for comparison"""
    results = {}
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # 1. XGBoost with hyperparameter tuning
    print("\n[1/3] Training XGBoost model with hyperparameter tuning...")
    xgb_model = tune_xgboost(X_train_balanced, y_train_balanced)
    
    # Evaluate XGBoost
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_results = {
        'model': xgb_model,
        'name': 'XGBoost',
        'predictions': xgb_pred,
        'probabilities': xgb_proba,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred),
        'f1': f1_score(y_test, xgb_pred),
        'auc': roc_auc_score(y_test, xgb_proba),
        'average_precision': average_precision_score(y_test, xgb_proba)
    }
    results['xgboost'] = xgb_results
    
    # 2. Random Forest
    print("\n[2/3] Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate Random Forest
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    rf_results = {
        'model': rf_model,
        'name': 'Random Forest',
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
        'auc': roc_auc_score(y_test, rf_proba),
        'average_precision': average_precision_score(y_test, rf_proba)
    }
    results['random_forest'] = rf_results
    
    # 3. SVM (with reduced training set for speed if needed)
    print("\n[3/3] Training SVM model (may take a while)...")
    max_samples = 5000  # Limit samples for SVM to avoid excessive training time
    
    if len(X_train_balanced) > max_samples:
        print(f"Reducing training set size to {max_samples} for SVM to improve training speed...")
        indices = np.random.choice(len(X_train_balanced), max_samples, replace=False)
        X_train_svm = X_train_balanced[indices]
        y_train_svm = y_train_balanced[indices]
    else:
        X_train_svm = X_train_balanced
        y_train_svm = y_train_balanced
    
    svm_model = SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    svm_model.fit(X_train_svm, y_train_svm)
    
    # Evaluate SVM
    svm_pred = svm_model.predict(X_test)
    svm_proba = svm_model.predict_proba(X_test)[:, 1]
    
    svm_results = {
        'model': svm_model,
        'name': 'SVM',
        'predictions': svm_pred,
        'probabilities': svm_proba,
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred),
        'recall': recall_score(y_test, svm_pred),
        'f1': f1_score(y_test, svm_pred),
        'auc': roc_auc_score(y_test, svm_proba),
        'average_precision': average_precision_score(y_test, svm_proba)
    }
    results['svm'] = svm_results
    
    return results

def save_models(results, scaler, output_dir):
    """Save trained models and preprocessing objects"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save each model
    for model_name, result in results.items():
        model_filename = f"{output_dir}/{model_name}_model_{timestamp}.joblib"
        joblib.dump(result['model'], model_filename)
        print(f"Saved {model_name} model to {model_filename}")
    
    # Save the best model as the default
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_model_name]['model']
    
    # Save the best model in two formats
    joblib.dump(best_model, f"{output_dir}/best_model.joblib")
    
    if best_model_name == 'xgboost':
        # XGBoost models can be saved in their native format
        best_model.save_model(f"{output_dir}/xgboost_drone_detection_model.json")
    
    # Save the scaler
    scaler_filename = f"{output_dir}/scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_filename)
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")  # Save a version with a standard name
    
    print(f"\nBest model ({best_model_name}) saved as default model")
    print(f"Scaler saved to {scaler_filename}")
    
    # Create a model info file
    info_filename = f"{output_dir}/model_info_{timestamp}.txt"
    with open(info_filename, 'w') as f:
        f.write(f"Model training completed at: {timestamp}\n\n")
        
        f.write("Model Performance Comparison:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name:<15} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
                   f"{result['recall']:<10.4f} {result['f1']:<10.4f} {result['auc']:<10.4f}\n")
    
    print(f"Model information saved to {info_filename}")

def plot_model_comparison(results, output_dir, y_test):
    """Generate plots to compare model performance"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. ROC curves
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        auc = result['auc']
        plt.plot(fpr, tpr, label=f"{result['name']} (AUC = {auc:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    roc_filename = f"{output_dir}/roc_comparison_{timestamp}.png"
    plt.savefig(roc_filename)
    plt.close()
    
    # 2. Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
        ap = result['average_precision']
        plt.plot(recall, precision, label=f"{result['name']} (AP = {ap:.4f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Different Models')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    pr_filename = f"{output_dir}/pr_comparison_{timestamp}.png"
    plt.savefig(pr_filename)
    plt.close()
    
    # 3. Bar chart of metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = [result['name'] for result in results.values()]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.2
    shifts = np.linspace(-width, width, len(results))
    
    for i, (model_name, result) in enumerate(results.items()):
        values = [result[metric] for metric in metrics]
        plt.bar(x + shifts[i], values, width, label=result['name'])
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save the plot
    metrics_filename = f"{output_dir}/metrics_comparison_{timestamp}.png"
    plt.savefig(metrics_filename)
    plt.close()
    
    print(f"Performance comparison plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Drone Detection Model Training')
    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_DIR,
                        help=f'Directory containing drone and background datasets (default: {DEFAULT_DATASET_DIR})')
    parser.add_argument('--output-dir', type=str, default=MODEL_OUTPUT_DIR,
                        help=f'Output directory for models and plots (default: {MODEL_OUTPUT_DIR})')
    parser.add_argument('--no-smote', action='store_true',
                        help='Disable SMOTE class balancing')
    parser.add_argument('--no-advanced-features', action='store_true',
                        help='Use simple feature extraction instead of advanced features')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    # Ensure dataset directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"Dataset directory {args.dataset_dir} not found.")
        print(f"Creating directory {args.dataset_dir}")
        os.makedirs(args.dataset_dir)
        print("\nPlease collect data before training the model.")
        print("Run: python drone_detection_system.py --collect-data")
        return
    
    # Load and preprocess the data
    use_advanced_features = not args.no_advanced_features
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        args.dataset_dir, 
        test_size=args.test_size,
        use_advanced_features=use_advanced_features
    )
    
    if X_train is None:
        print("No valid dataset found. Please collect data first.")
        return
    
    # Train and evaluate models
    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, 
        use_smote=not args.no_smote
    )
    
    # Print results
    print("\nModel Evaluation Results:")
    print("-" * 80)
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} {result['f1']:<10.4f} {result['auc']:<10.4f}")
    
    # Save models
    save_models(results, scaler, args.output_dir)
    
    # Plot comparisons
    plot_model_comparison(results, args.output_dir, y_test)
    
    print("\nTraining complete!")
    print("To use the best model for drone detection, run:")
    print("python drone_detection_system.py")
    
if __name__ == "__main__":
    main() 