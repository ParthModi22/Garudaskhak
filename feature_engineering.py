#!/usr/bin/env python3

import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis
import pywt

def extract_advanced_features(iq_data, sample_rate=40e6):
    """
    Extract a comprehensive set of features from IQ data for drone detection
    
    Parameters:
    -----------
    iq_data : complex numpy array
        IQ samples from SDR
    sample_rate : float
        Sample rate in Hz
        
    Returns:
    --------
    features : dict
        Dictionary containing feature values
    """
    features = {}
    
    # Basic signal metrics
    signal_power = np.mean(np.abs(iq_data)**2)
    features['power_dbm'] = 10 * np.log10(signal_power) + 30  # Convert to dBm
    
    # ------ Frequency domain features ------
    # Apply windowing to reduce spectral leakage
    window = np.blackman(len(iq_data))
    windowed_iq = iq_data * window
    
    # Compute FFT
    nfft = 2048
    fft_data = np.fft.fft(windowed_iq, n=nfft)
    fft_mag = np.abs(fft_data[:nfft//2])
    fft_db = 20 * np.log10(fft_mag + 1e-12)
    
    # Frequency bins
    freq_bins = np.fft.fftfreq(nfft, 1/sample_rate)[:nfft//2]
    
    # Spectral statistics
    features['spectral_mean'] = np.mean(fft_db)
    features['spectral_std'] = np.std(fft_db)
    features['spectral_skew'] = skew(fft_db)
    features['spectral_kurtosis'] = kurtosis(fft_db)
    
    # Spectral entropy
    psd = fft_mag**2
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    features['spectral_entropy'] = spectral_entropy
    
    # Peak analysis
    peak_threshold = np.mean(fft_db) + 10  # 10dB above mean
    peaks, _ = signal.find_peaks(fft_db, height=peak_threshold)
    features['num_peaks'] = len(peaks)
    
    if len(peaks) > 0:
        features['mean_peak_height'] = np.mean(fft_db[peaks])
        features['max_peak_height'] = np.max(fft_db[peaks])
        features['peak_frequency_spread'] = np.std(freq_bins[peaks])
    else:
        features['mean_peak_height'] = 0
        features['max_peak_height'] = 0
        features['peak_frequency_spread'] = 0
    
    # ------ Time domain features ------
    i_component = np.real(iq_data)
    q_component = np.imag(iq_data)
    
    # Statistical features
    features['i_mean'] = np.mean(i_component)
    features['i_std'] = np.std(i_component)
    features['i_skew'] = skew(i_component)
    features['i_kurtosis'] = kurtosis(i_component)
    
    features['q_mean'] = np.mean(q_component)
    features['q_std'] = np.std(q_component)
    features['q_skew'] = skew(q_component)
    features['q_kurtosis'] = kurtosis(q_component)
    
    # Zero-crossing rate
    zero_crossings_i = np.sum(np.diff(np.signbit(i_component)))
    zero_crossings_q = np.sum(np.diff(np.signbit(q_component)))
    features['zero_crossing_rate_i'] = zero_crossings_i / len(i_component)
    features['zero_crossing_rate_q'] = zero_crossings_q / len(q_component)
    
    # ------ Modulation-specific features ------
    # Envelope detection (AM detection)
    envelope = np.abs(iq_data)
    
    # Analyze envelope for modulation properties
    features['env_mean'] = np.mean(envelope)
    features['env_std'] = np.std(envelope)
    features['env_max'] = np.max(envelope)
    features['env_min'] = np.min(envelope)
    
    # Envelope spectrum (looks for AM modulation components)
    env_spec = np.abs(np.fft.fft(envelope * window, n=nfft))[:nfft//2]
    env_spec_db = 20 * np.log10(env_spec + 1e-12)
    features['env_spec_mean'] = np.mean(env_spec_db)
    features['env_spec_std'] = np.std(env_spec_db)
    
    # Phase features (useful for FM/PM modulation detection)
    phase = np.angle(iq_data)
    unwrapped_phase = np.unwrap(phase)
    phase_diff = np.diff(unwrapped_phase)
    
    features['phase_diff_std'] = np.std(phase_diff)
    features['phase_diff_max'] = np.max(np.abs(phase_diff))
    
    # ------ Wavelet-based features ------
    # Multi-resolution analysis with wavelets
    try:
        coeffs = pywt.wavedec(np.abs(iq_data), 'db4', level=4)
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2) / len(coeff)
    except:
        # Fallback if PyWavelets is not available or fails
        for i in range(5):
            features[f'wavelet_energy_level_{i}'] = 0
    
    # ------ Drone-specific features ------
    # Many drones have specific control frequencies around 50-150 Hz
    # Look for energy in control bands
    control_bands = [(50, 150), (150, 250), (250, 400)]
    for i, (low, high) in enumerate(control_bands):
        low_idx = int(low * nfft / sample_rate)
        high_idx = int(high * nfft / sample_rate)
        band_energy = np.sum(psd[low_idx:high_idx])
        total_energy = np.sum(psd)
        features[f'control_band_{i}_energy_ratio'] = band_energy / total_energy if total_energy > 0 else 0
    
    # Hopping pattern detection
    # Drone control signals often hop frequencies
    segments = 10
    seg_len = len(iq_data) // segments
    segment_energies = []
    
    for i in range(segments):
        seg_data = iq_data[i*seg_len:(i+1)*seg_len]
        seg_fft = np.abs(np.fft.fft(seg_data * window[:seg_len], n=nfft))[:nfft//2]
        segment_energies.append(seg_fft)
    
    # Calculate energy variation across time segments (frequency hopping indicator)
    segment_energies = np.array(segment_energies)
    freq_variation = np.std(segment_energies, axis=0)
    features['freq_hopping_indicator'] = np.mean(freq_variation)
    
    # Convert all features to a flat numpy array for ML model
    feature_array = np.array(list(features.values()))
    
    return features, feature_array

def get_feature_names():
    """Return the names of features in the order they appear in the feature array"""
    sample_data = np.random.normal(0, 1, 2000) + 1j * np.random.normal(0, 1, 2000)
    features, _ = extract_advanced_features(sample_data)
    return list(features.keys())

def extract_features_for_ml(iq_data_list, sample_rate=40e6):
    """
    Extract features from a list of IQ samples for machine learning
    
    Parameters:
    -----------
    iq_data_list : list of complex numpy arrays
        List of IQ sample arrays
    sample_rate : float
        Sample rate in Hz
        
    Returns:
    --------
    X : numpy array
        Feature matrix for machine learning (n_samples, n_features)
    """
    if len(iq_data_list) == 0:
        return np.array([])
    
    # Extract features from first sample to get dimension
    _, first_features = extract_advanced_features(iq_data_list[0], sample_rate)
    
    # Pre-allocate feature matrix
    X = np.zeros((len(iq_data_list), len(first_features)))
    
    # Extract features for all samples
    for i, iq_data in enumerate(iq_data_list):
        _, X[i, :] = extract_advanced_features(iq_data, sample_rate)
    
    return X

# If run directly, test with random data
if __name__ == "__main__":
    # Generate random IQ data
    n_samples = 10000
    test_data = np.random.normal(0, 1, n_samples) + 1j * np.random.normal(0, 1, n_samples)
    
    # Extract features
    features_dict, features_array = extract_advanced_features(test_data)
    
    # Print results
    print(f"Extracted {len(features_dict)} features:")
    for name, value in features_dict.items():
        print(f"{name}: {value}")
    
    print(f"\nFeature array shape: {features_array.shape}") 