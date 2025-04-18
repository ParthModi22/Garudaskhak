#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

def load_data(filename):
    """Load and validate drone data file"""
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        available_files = [f for f in os.listdir(os.path.dirname(full_path)) 
                          if f.endswith('.npz')]
        print(f"Available .npz files: {available_files}")
        return None
    
    print(f"Loading file: {full_path}")
    data = np.load(full_path)
    
    return data

def analyze_sample(sample, sample_rate=40e6, title="Signal Analysis"):
    """Analyze a single IQ sample"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{title}", fontsize=16)
    
    # Plot time domain
    t = np.arange(len(sample)) / sample_rate * 1000  # Convert to ms
    axs[0, 0].plot(t, np.real(sample), 'b-', alpha=0.7, label='I')
    axs[0, 0].plot(t, np.imag(sample), 'r-', alpha=0.7, label='Q')
    axs[0, 0].set_title('Time Domain')
    axs[0, 0].set_xlabel('Time (ms)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Only plot a small segment if the sample is large
    if len(sample) > 10000:
        t_zoom = t[:10000]
        axs[0, 1].plot(t_zoom, np.real(sample[:10000]), 'b-', label='I')
        axs[0, 1].plot(t_zoom, np.imag(sample[:10000]), 'r-', label='Q')
        axs[0, 1].set_title('Time Domain (Zoomed)')
        axs[0, 1].set_xlabel('Time (ms)')
        axs[0, 1].set_ylabel('Amplitude')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
    else:
        # Plot magnitude instead
        axs[0, 1].plot(t, np.abs(sample), 'g-')
        axs[0, 1].set_title('Signal Magnitude')
        axs[0, 1].set_xlabel('Time (ms)')
        axs[0, 1].set_ylabel('Magnitude')
        axs[0, 1].grid(True)
    
    # Plot frequency domain (FFT)
    n_fft = min(len(sample), 65536)  # Use smaller FFT for speed if needed
    win = np.blackman(n_fft)
    fft_data = np.fft.fftshift(np.fft.fft(sample[:n_fft] * win))
    freq = np.fft.fftshift(np.fft.fftfreq(n_fft, 1/sample_rate))
    
    # Convert to dB
    fft_db = 20 * np.log10(np.abs(fft_data) / n_fft + 1e-12)
    
    axs[1, 0].plot(freq/1e6, fft_db)
    axs[1, 0].set_title('Frequency Spectrum')
    axs[1, 0].set_xlabel('Frequency (MHz)')
    axs[1, 0].set_ylabel('Magnitude (dB)')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlim(-sample_rate/2e6, sample_rate/2e6)
    
    # Plot spectrogram - fix for matplotlib's specgram
    # Take only a portion of the sample if it's very large
    max_specgram_samples = 100000
    if len(sample) > max_specgram_samples:
        specgram_sample = sample[:max_specgram_samples]
    else:
        specgram_sample = sample
        
    # Use scipy's spectrogram instead which is more reliable
    from scipy import signal as sg
    f, t, Sxx = sg.spectrogram(np.abs(specgram_sample), fs=sample_rate, 
                              nperseg=1024, noverlap=512)
    axs[1, 1].pcolormesh(t*1000, f/1e6, 10*np.log10(Sxx+1e-12), shading='gouraud')
    axs[1, 1].set_title('Spectrogram')
    axs[1, 1].set_xlabel('Time (ms)')
    axs[1, 1].set_ylabel('Frequency (MHz)')
    
    plt.tight_layout()
    
    return fig

def visualize_data(data, sample_indices=None, save_dir=None):
    """Visualize IQ samples from data file"""
    if 'iq_samples' not in data.files:
        print("No iq_samples found in file")
        return
    
    iq_samples = data['iq_samples']
    num_samples = len(iq_samples)
    
    print(f"Data contains {num_samples} samples, each with {iq_samples[0].shape[0]} points")
    
    # If no sample indices specified, use the first few
    if sample_indices is None:
        if num_samples > 5:
            sample_indices = [0, 1, num_samples//2, num_samples-2, num_samples-1]
        else:
            sample_indices = list(range(num_samples))
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Analyze each requested sample
    for idx in sample_indices:
        if idx < 0 or idx >= num_samples:
            print(f"Sample index {idx} out of range (0-{num_samples-1})")
            continue
        
        print(f"Analyzing sample {idx}...")
        fig = analyze_sample(iq_samples[idx], title=f"Sample {idx} Analysis")
        
        if save_dir:
            # Save the figure
            save_path = os.path.join(save_dir, f"sample_{idx}_analysis.png")
            fig.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()
            input("Press Enter to continue...")
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Visualize drone data')
    parser.add_argument('--file', type=str, help='NPZ file to analyze')
    parser.add_argument('--samples', type=str, default='0,1,2',
                       help='Comma-separated list of sample indices to visualize')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Find default file if none specified
    if not args.file:
        files = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) 
                if f.endswith('.npz')]
        if not files:
            print("No .npz files found in the directory")
            return
        
        # Prefer drone data files over background
        drone_files = [f for f in files if f.startswith('drone_data')]
        if drone_files:
            args.file = drone_files[0]
        else:
            args.file = files[0]
        
        print(f"Using file: {args.file}")
    
    # Load data
    data = load_data(args.file)
    if data is None:
        return
    
    # Parse sample indices
    try:
        sample_indices = [int(idx) for idx in args.samples.split(',')]
    except ValueError:
        print("Invalid sample indices. Using defaults.")
        sample_indices = [0, 1, 2]
    
    # Visualize data
    visualize_data(data, sample_indices, args.save_dir)

if __name__ == "__main__":
    main() 