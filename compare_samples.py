#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy import signal

def load_data(filename):
    """Load and validate data file"""
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return None
    
    print(f"Loading file: {full_path}")
    data = np.load(full_path)
    
    return data

def compute_fft(sample, sample_rate=40e6):
    """Compute FFT with windowing"""
    n_fft = min(len(sample), 65536) 
    win = np.blackman(n_fft)
    fft_data = np.fft.fftshift(np.fft.fft(sample[:n_fft] * win))
    freq = np.fft.fftshift(np.fft.fftfreq(n_fft, 1/sample_rate))
    fft_db = 20 * np.log10(np.abs(fft_data) / n_fft + 1e-12)
    return freq, fft_db

def compare_samples(drone_data, background_data, sample_idx=0, sample_rate=40e6, save_path=None):
    """Compare drone and background data samples"""
    if 'iq_samples' not in drone_data.files or 'iq_samples' not in background_data.files:
        print("Error: iq_samples not found in one or both files")
        return
    
    drone_samples = drone_data['iq_samples']
    background_samples = background_data['iq_samples']
    
    if sample_idx >= len(drone_samples) or sample_idx >= len(background_samples):
        print(f"Error: Sample index {sample_idx} out of range")
        return
    
    drone_sample = drone_samples[sample_idx]
    background_sample = background_samples[sample_idx]
    
    # Create comparison figure
    fig, axs = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle(f"Drone vs Background Comparison - Sample {sample_idx}", fontsize=16)
    
    # Time domain plots - first 10,000 samples
    samples_to_plot = 10000
    t = np.arange(samples_to_plot) / sample_rate * 1000  # ms
    
    # Drone time domain
    axs[0, 0].plot(t, np.real(drone_sample[:samples_to_plot]), 'b-', alpha=0.7, label='I')
    axs[0, 0].plot(t, np.imag(drone_sample[:samples_to_plot]), 'r-', alpha=0.7, label='Q')
    axs[0, 0].set_title('Drone Signal - Time Domain')
    axs[0, 0].set_xlabel('Time (ms)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Background time domain
    axs[0, 1].plot(t, np.real(background_sample[:samples_to_plot]), 'b-', alpha=0.7, label='I')
    axs[0, 1].plot(t, np.imag(background_sample[:samples_to_plot]), 'r-', alpha=0.7, label='Q')
    axs[0, 1].set_title('Background Signal - Time Domain')
    axs[0, 1].set_xlabel('Time (ms)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Compute FFTs
    drone_freq, drone_fft = compute_fft(drone_sample, sample_rate)
    bg_freq, bg_fft = compute_fft(background_sample, sample_rate)
    
    # Plot FFTs
    axs[1, 0].plot(drone_freq/1e6, drone_fft)
    axs[1, 0].set_title('Drone Signal - Frequency Spectrum')
    axs[1, 0].set_xlabel('Frequency (MHz)')
    axs[1, 0].set_ylabel('Magnitude (dB)')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlim(-sample_rate/4e6, sample_rate/4e6)  # Zoom in to show detail
    
    axs[1, 1].plot(bg_freq/1e6, bg_fft)
    axs[1, 1].set_title('Background Signal - Frequency Spectrum')
    axs[1, 1].set_xlabel('Frequency (MHz)')
    axs[1, 1].set_ylabel('Magnitude (dB)')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlim(-sample_rate/4e6, sample_rate/4e6)  # Zoom in to show detail
    
    # Calculate and plot the difference
    # Find the minimum length to compare
    min_len = min(len(drone_fft), len(bg_fft))
    diff_fft = drone_fft[:min_len] - bg_fft[:min_len]
    
    axs[2, 0].plot(drone_freq[:min_len]/1e6, diff_fft)
    axs[2, 0].set_title('Difference (Drone - Background)')
    axs[2, 0].set_xlabel('Frequency (MHz)')
    axs[2, 0].set_ylabel('Magnitude Difference (dB)')
    axs[2, 0].grid(True)
    axs[2, 0].set_xlim(-sample_rate/4e6, sample_rate/4e6)  # Zoom in to show detail
    
    # Plot energy distribution in frequency bands
    # Identify peaks in the difference spectrum
    peak_threshold = np.mean(diff_fft) + 2*np.std(diff_fft)
    peaks, _ = signal.find_peaks(diff_fft, height=peak_threshold)
    
    axs[2, 1].plot(drone_freq[:min_len]/1e6, diff_fft)
    
    # Highlight peaks that might be drone signatures
    if len(peaks) > 0:
        axs[2, 1].plot(drone_freq[peaks]/1e6, diff_fft[peaks], 'ro')
        peak_freqs = drone_freq[peaks]/1e6
        
        # Add annotation for top 3 peaks
        if len(peaks) > 3:
            top_peaks = np.argsort(diff_fft[peaks])[-3:]
            for i in top_peaks:
                axs[2, 1].annotate(f"{peak_freqs[i]:.1f} MHz", 
                                (peak_freqs[i], diff_fft[peaks[i]]), 
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center')
    
    axs[2, 1].set_title('Drone Signal Peaks')
    axs[2, 1].set_xlabel('Frequency (MHz)')
    axs[2, 1].set_ylabel('Magnitude Difference (dB)')
    axs[2, 1].grid(True)
    axs[2, 1].set_xlim(-sample_rate/4e6, sample_rate/4e6)  # Zoom in to show detail
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison to {save_path}")
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare drone and background data')
    parser.add_argument('--drone-file', type=str, help='Drone data NPZ file')
    parser.add_argument('--background-file', type=str, help='Background data NPZ file')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to compare')
    parser.add_argument('--save-dir', type=str, help='Directory to save comparison')
    
    args = parser.parse_args()
    
    # Find default files if not specified
    if not args.drone_file or not args.background_file:
        files = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) 
                if f.endswith('.npz')]
        
        drone_files = [f for f in files if f.startswith('drone_data')]
        bg_files = [f for f in files if f.startswith('background')]
        
        if not args.drone_file and drone_files:
            args.drone_file = drone_files[0]
            print(f"Using drone file: {args.drone_file}")
            
        if not args.background_file and bg_files:
            args.background_file = bg_files[0]
            print(f"Using background file: {args.background_file}")
    
    # Check that we have both files
    if not args.drone_file or not args.background_file:
        print("Error: Need both drone and background data files")
        return
    
    # Load data
    drone_data = load_data(args.drone_file)
    background_data = load_data(args.background_file)
    
    if drone_data is None or background_data is None:
        return
    
    # Setup save path if needed
    save_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"comparison_sample_{args.sample}.png")
    
    # Compare samples
    compare_samples(drone_data, background_data, args.sample, save_path=save_path)

if __name__ == "__main__":
    main() 