#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Mac compatibility
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import adi
import time
import scipy.signal as signal
import subprocess
import argparse
import xgboost as xgb
from datetime import datetime
import os
from collections import deque

# Configuration parameters
MODEL_PATH = 'xgboost_drone_detection_model.json'
CENTER_FREQ = 2.4e9  # 2.4 GHz center frequency for drone signals
SAMPLE_RATE = 40e6   # Match training data: 40 MHz
RX_GAIN = 50         # Receiver gain
TX_GAIN = -10        # Transmitter gain
BUFFER_SIZE = int(40e6 * 0.02)  # 20ms of samples at 40MHz
FFT_SIZE = 1024 * 4  # FFT size for spectral analysis

# Waterfall settings
NUM_SLICES = 100     # Number of time slices in waterfall display
PLOT_FREQ = 500e3    # Frequency range to display (Hz)

# Drone detection parameters
PROBABILITY_THRESHOLD = 0.5  # Threshold for drone detection
HISTORY_LENGTH = 50          # Number of samples to keep in history

def get_pluto_ip():
    """Try to discover the PlutoSDR IP address"""
    try:
        result = subprocess.run(["df", "-h"], capture_output=True, text=True)
        output = result.stdout
        
        if "PlutoSDR" in output:
            print("PlutoSDR USB drive detected.")
            return "192.168.2.1"
    except:
        pass
    
    common_ips = ["192.168.2.1", "192.168.3.1", "10.48.0.1", "192.168.1.10"]
    return common_ips[0]

def configure_sdr(args):
    """Configure PlutoSDR for both transmission and reception"""
    try:
        # Try USB connection first
        print("Attempting to connect to SDR via USB...")
        try:
            sdr = adi.Pluto(uri="usb:")
            print("USB connection established")
        except Exception as usb_error:
            print(f"USB connection failed: {usb_error}")
            
            # Try IP connection
            pluto_ip = get_pluto_ip()
            print(f"Trying IP connection at {pluto_ip}...")
            try:
                sdr = adi.Pluto(uri=f"ip:{pluto_ip}")
                print(f"IP connection established at {pluto_ip}")
            except Exception as ip_error:
                print(f"IP connection failed: {ip_error}")
                raise Exception("Could not connect to SDR via USB or IP")

        # Get maximum supported sample rate
        try:
            max_sample_rate = int(SAMPLE_RATE)
            print(f"Attempting to set sample rate to {max_sample_rate/1e6} MHz...")
            sdr.sample_rate = max_sample_rate
        except Exception as e:
            print(f"Warning: Could not set {SAMPLE_RATE/1e6}MHz sample rate: {e}")
            print("Falling back to 20MHz sample rate...")
            try:
                sdr.sample_rate = int(20e6)
            except Exception as e:
                print(f"Warning: Could not set 20MHz sample rate: {e}")
                print("Falling back to 10MHz sample rate...")
                sdr.sample_rate = int(10e6)

        # Configure common settings
        sdr.rx_rf_bandwidth = int(sdr.sample_rate)
        sdr.tx_rf_bandwidth = int(sdr.sample_rate)
        
        # Configure transmitter
        sdr.tx_lo = int(CENTER_FREQ)
        sdr.tx_enabled_channels = [args.tx_channel]
        sdr.tx_hardwaregain_chan0 = TX_GAIN
        
        # Adjust buffer size based on the SDR's capabilities
        # Some SDRs have fixed buffer sizes, so we need to adapt
        try:
            # Try to set our preferred buffer size
            sdr.rx_buffer_size = int(BUFFER_SIZE)
        except Exception as e:
            print(f"Warning: Could not set buffer size to {BUFFER_SIZE}: {e}")
            print("Using default buffer size provided by the SDR")
            # The SDR will use its default buffer size
        
        # Configure receiver
        sdr.rx_lo = int(CENTER_FREQ)  # Set to specified center frequency
        sdr.rx_enabled_channels = [args.rx_channel]
        sdr.gain_control_mode_chan0 = "manual"  
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        
        print("\nSDR Configuration:")
        print(f"Sample Rate: {sdr.sample_rate/1e6} MHz")
        print(f"Center Frequency: {sdr.rx_lo/1e9} GHz")
        print(f"TX Channel: {args.tx_channel}")
        print(f"RX Channel: {args.rx_channel}")
        print(f"TX Gain: {sdr.tx_hardwaregain_chan0} dB")
        print(f"RX Gain: {sdr.rx_hardwaregain_chan0} dB")
        print(f"Buffer Size: {sdr.rx_buffer_size} samples")
        print(f"NOTE: Processing will be done on blocks of {FFT_SIZE} samples")
        
        return sdr
    except Exception as e:
        print(f"Error configuring SDR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure PlutoSDR is properly connected via USB")
        print("2. Check if device shows up in 'lsusb' (Linux) or Device Manager (Windows)")
        print("3. Try unplugging and replugging the device")
        print("4. Verify no other software is currently using the SDR")
        return None

def generate_drone_signal():
    """Generate a realistic drone-like signal for testing purposes"""
    fs = SAMPLE_RATE
    N = BUFFER_SIZE
    t = np.arange(N) / fs
    
    # Create a complex signal matching drone characteristics
    # Control signal frequencies (based on common drone protocols)
    control_freqs = [100e3, 150e3, 200e3, 250e3]  # Common control frequencies
    telemetry_freqs = [50e3, 75e3]  # Telemetry frequencies
    hopping_freqs = np.linspace(-500e3, 500e3, 10)  # Frequency hopping range
    
    # Base signal with control frequencies
    tx_iq_signal = np.zeros(N, dtype=complex)
    for freq in control_freqs:
        amplitude = np.random.uniform(0.2, 0.4)
        tx_iq_signal += amplitude * np.exp(2j * np.pi * freq * t)
    
    # Add telemetry signals
    for freq in telemetry_freqs:
        amplitude = np.random.uniform(0.1, 0.2)
        tx_iq_signal += amplitude * np.exp(2j * np.pi * freq * t)
    
    # Add frequency hopping
    hop_duration = N // 10  # 10 hops per buffer
    for i in range(10):
        hop_freq = np.random.choice(hopping_freqs)
        start_idx = i * hop_duration
        end_idx = (i + 1) * hop_duration
        t_hop = t[start_idx:end_idx]
        tx_iq_signal[start_idx:end_idx] *= np.exp(2j * np.pi * hop_freq * t_hop)
    
    # Add amplitude modulation to simulate control patterns
    am_freq = 50  # 50 Hz AM
    am = 0.3 * np.sin(2 * np.pi * am_freq * t)
    tx_iq_signal = tx_iq_signal * (1 + am)
    
    # Add noise
    noise = np.random.normal(0, 0.05, N) + 1j * np.random.normal(0, 0.05, N)
    tx_iq_signal += noise
    
    # Normalize
    tx_iq_signal = tx_iq_signal / np.max(np.abs(tx_iq_signal)) * 0.8
    return tx_iq_signal

def extract_features(signal_data, n_fft=2048):
    """Extract features for drone detection"""
    # Ensure signal_data is the right size for feature extraction
    if len(signal_data) > n_fft:
        # If signal is too large, take the first n_fft samples
        signal_data = signal_data[:n_fft]
    elif len(signal_data) < n_fft:
        # If signal is too small, pad with zeros
        padding = np.zeros(n_fft - len(signal_data), dtype=complex)
        signal_data = np.concatenate([signal_data, padding])
    
    # Use absolute value of IQ samples
    data_magnitude = np.abs(signal_data)
    
    # Compute DFT
    dft = np.fft.fft(data_magnitude, n=n_fft)
    
    # Extract one-sided magnitude spectrum
    magnitude_spectrum = np.abs(dft[:n_fft//2])
    
    # Normalize
    if np.max(magnitude_spectrum) > 0:
        magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
    
    return magnitude_spectrum

def load_model(model_path):
    """Load the trained XGBoost model"""
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def collect_drone_data(sdr, output_dir='drone_dataset', num_samples=1000):
    """Collect data from a real drone for dataset creation"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Starting drone data collection. Saving to: {output_dir}")
    print("Please ensure your drone is powered on and operating normally.")
    print(f"Will collect {num_samples} samples...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/drone_data_{timestamp}.npz"
    
    # Storage for data
    iq_samples = []
    
    try:
        for i in range(num_samples):
            # Get IQ data
            rx_data = sdr.rx()
            iq_samples.append(rx_data)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Collected {i+1}/{num_samples} samples")
                
            time.sleep(0.1)  # Short delay between samples
            
        # Save collected data
        np.savez(filename, iq_samples=np.array(iq_samples))
        print(f"Data collection complete. Saved to {filename}")
        return filename
    
    except KeyboardInterrupt:
        # Handle user interrupt
        print("\nData collection interrupted by user")
        if len(iq_samples) > 10:  # Only save if we have a reasonable amount of data
            np.savez(filename, iq_samples=np.array(iq_samples))
            print(f"Partial data saved to {filename}")
            return filename
        return None
    except Exception as e:
        print(f"Error during data collection: {e}")
        return None

class DroneDetector:
    def __init__(self, sdr, model):
        self.sdr = sdr
        self.model = model
        
        # Signal processing parameters
        self.sample_rate = sdr.sample_rate
        self.fft_size = FFT_SIZE
        
        # Initialize history trackers
        self.probability_history = deque(maxlen=HISTORY_LENGTH)
        self.time_history = deque(maxlen=HISTORY_LENGTH)
        self.detection_history = deque(maxlen=HISTORY_LENGTH)
        
        # Initialize waterfall display data
        self.img_array = np.ones((NUM_SLICES, self.fft_size)) * (-100)
        
        # Setup plot
        self.setup_plot()
    
    def setup_plot(self):
        """Set up the visualization plots"""
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(3, 2)
        
        # Calculate frequency display range based on sample rate
        center_freq = CENTER_FREQ / 1e9  # Center frequency in GHz
        freq_range = self.sample_rate / 2 / 1e6  # Half bandwidth in MHz
        
        # Frequency axis labels for display
        freq_min = center_freq - (freq_range / 1000)  # Lower frequency in GHz
        freq_max = center_freq + (freq_range / 1000)  # Upper frequency in GHz
        
        # Spectrum plot
        self.ax_spectrum = self.fig.add_subplot(self.gs[0, 0])
        self.ax_spectrum.set_title(f'RF Spectrum around {center_freq:.1f} GHz')
        self.ax_spectrum.set_xlabel('Frequency (GHz)')
        self.ax_spectrum.set_ylabel('Magnitude (dB)')
        self.ax_spectrum.grid(True)
        
        # Waterfall plot - set up with correct orientation
        self.ax_waterfall = self.fig.add_subplot(self.gs[1, 0])
        
        # Create a time axis with meaningful labels (newest at top)
        time_axis = np.linspace(0, NUM_SLICES/10, NUM_SLICES)  # Assuming ~10 updates per second
        
        # Initialize a time-indexed waterfall plot
        # X-axis: frequency (GHz), Y-axis: time (seconds ago)
        self.waterfall_img = self.ax_waterfall.imshow(
            self.img_array, 
            aspect='auto', 
            origin='upper',  # 'upper' to have newest data at the bottom (like scrolling text)
            cmap='viridis',
            extent=[freq_min, freq_max, time_axis[-1], time_axis[0]]  # freq_min to freq_max, newest time to oldest
        )
        self.fig.colorbar(self.waterfall_img, ax=self.ax_waterfall, label='Magnitude (dB)')
        self.ax_waterfall.set_title(f'Waterfall Display ({center_freq:.1f} GHz)')
        self.ax_waterfall.set_xlabel('Frequency (GHz)')
        self.ax_waterfall.set_ylabel('Time (seconds ago)')
        
        # Add timestamp to make time axis more meaningful
        self.waterfall_timestamp = datetime.now()
        
        # Detection probability plot
        self.ax_prob = self.fig.add_subplot(self.gs[0, 1])
        self.ax_prob.set_title('Drone Detection Probability')
        self.ax_prob.set_xlabel('Time')
        self.ax_prob.set_ylabel('Probability')
        self.ax_prob.set_ylim(0, 1)
        self.ax_prob.axhline(y=PROBABILITY_THRESHOLD, color='r', linestyle='--')
        self.ax_prob.grid(True)
        
        # IQ constellation plot
        self.ax_constellation = self.fig.add_subplot(self.gs[1, 1])
        self.ax_constellation.set_title('IQ Constellation')
        self.ax_constellation.set_xlabel('I')
        self.ax_constellation.set_ylabel('Q')
        self.ax_constellation.grid(True)
        self.ax_constellation.set_aspect('equal')
        
        # Status display
        self.ax_status = self.fig.add_subplot(self.gs[2, :])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.05, 0.5, "Initializing...", fontsize=12)
        
        # Add center frequency information
        self.ax_status.text(0.05, 0.8, 
                         f"Center Frequency: {center_freq:.3f} GHz\n"
                         f"Bandwidth: {freq_range*2:.1f} MHz", 
                         fontsize=12)
        
        # Detection indicator
        self.detection_patch = plt.Rectangle((0.8, 0.4), 0.15, 0.2, 
                                         facecolor='gray', transform=self.ax_status.transAxes)
        self.ax_status.add_patch(self.detection_patch)
        self.detection_text = self.ax_status.text(0.8, 0.65, "NO DRONE", 
                                               fontsize=12, transform=self.ax_status.transAxes)
        
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)
    
    def process_signal(self, rx_signal):
        """Process received signal and update visualization"""
        # Handle different input array sizes by resizing or selecting a portion
        if len(rx_signal) != self.fft_size:
            # Only print this message once to avoid console spam
            if not hasattr(self, '_resize_message_shown'):
                print(f"Resizing signal from {len(rx_signal)} to {self.fft_size} samples")
                self._resize_message_shown = True
                
            # Option 1: Take the first fft_size samples
            if len(rx_signal) > self.fft_size:
                rx_signal = rx_signal[:self.fft_size]
            # Option 2: If signal is too small, pad with zeros
            else:
                padding = np.zeros(self.fft_size - len(rx_signal), dtype=complex)
                rx_signal = np.concatenate([rx_signal, padding])
        
        # Apply window function to reduce spectral leakage
        win = np.blackman(len(rx_signal))
        y = rx_signal * win
        
        # Compute FFT
        sp = np.absolute(np.fft.fft(y))
        sp = np.fft.fftshift(sp)
        s_mag = np.abs(sp) / np.sum(win)
        s_mag = np.maximum(s_mag, 10 ** (-15))
        s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
        
        # Create frequency axis (centered at 0 for baseband)
        freq = np.linspace(-self.sample_rate/2, self.sample_rate/2, len(s_dbfs))
        
        # Update waterfall display - we roll the array upwards since origin is 'upper'
        self.img_array = np.roll(self.img_array, 1, axis=0)
        self.img_array[0] = s_dbfs  # Add new data at the bottom (most recent)
        
        # Update timestamp
        elapsed_time = (datetime.now() - self.waterfall_timestamp).total_seconds()
        if elapsed_time > 10:  # Update timestamp every 10 seconds
            self.waterfall_timestamp = datetime.now()
            # We'll update the waterfall extent in update_plot
        
        # Extract features and make prediction
        features = extract_features(rx_signal)
        
        if self.model is not None:
            probability = self.model.predict_proba(features.reshape(1, -1))[0][1]
            prediction = 1 if probability > PROBABILITY_THRESHOLD else 0
        else:
            # If no model, use simple energy detection as fallback
            signal_energy = np.mean(np.abs(rx_signal)**2)
            probability = min(signal_energy * 10, 1.0)  # Simple scaling
            prediction = 1 if probability > PROBABILITY_THRESHOLD else 0
        
        # Update history
        self.probability_history.append(probability)
        self.time_history.append(len(self.time_history))
        self.detection_history.append(prediction)
        
        return freq, s_dbfs, probability, prediction
    
    def update_plot(self, freq, s_dbfs, probability, prediction, rx_signal):
        """Update all plots with new data"""
        # Convert frequency axis from Hz to GHz for display
        freq_ghz = CENTER_FREQ / 1e9 + freq / 1e9
        
        # Current time for status reporting
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Update spectrum plot
        self.ax_spectrum.clear()
        self.ax_spectrum.plot(freq_ghz, s_dbfs)
        center_freq = CENTER_FREQ / 1e9  # Center frequency in GHz
        freq_range = self.sample_rate / 2 / 1e6 / 1000  # Half bandwidth in GHz
        
        self.ax_spectrum.set_title(f'RF Spectrum ({current_time}) - {center_freq:.1f} GHz')
        self.ax_spectrum.set_xlabel('Frequency (GHz)')
        self.ax_spectrum.set_ylabel('Magnitude (dB)')
        self.ax_spectrum.grid(True)
        
        # Set x-axis to show the actual frequency range (not baseband)
        self.ax_spectrum.set_xlim(center_freq - freq_range, center_freq + freq_range)
        self.ax_spectrum.set_ylim(-80, 0)
        
        # Update waterfall display
        self.waterfall_img.set_array(self.img_array)
        self.waterfall_img.set_clim([-60, -20])  # Set color scale
        
        # Update time labels on waterfall (make it more intuitive)
        elapsed_time = (datetime.now() - self.waterfall_timestamp).total_seconds()
        if elapsed_time > 10:  # Only update time labels every 10 seconds to avoid flicker
            # Create a time axis with meaningful labels
            time_axis = np.linspace(0, NUM_SLICES/10, NUM_SLICES)  # ~10 updates per second
            
            # Update extent to show current time at bottom
            freq_min = center_freq - freq_range
            freq_max = center_freq + freq_range
            self.waterfall_img.set_extent([freq_min, freq_max, time_axis[-1], time_axis[0]])
            
            # Update the timestamp for future calculations
            self.waterfall_timestamp = datetime.now()
            
            # Update waterfall title with current time
            self.ax_waterfall.set_title(f'Waterfall Display - History ({NUM_SLICES/10:.0f} seconds)')
            
            # Clear and redraw the axis labels (to avoid overlap)
            self.ax_waterfall.set_xlabel('Frequency (GHz)')
            self.ax_waterfall.set_ylabel('Time (seconds ago)')
        
        # Update probability plot
        self.ax_prob.clear()
        self.ax_prob.plot(list(self.time_history), list(self.probability_history), 'b-')
        self.ax_prob.axhline(y=PROBABILITY_THRESHOLD, color='r', linestyle='--')
        self.ax_prob.set_ylim(0, 1)
        self.ax_prob.set_title('Drone Detection Probability')
        self.ax_prob.set_xlabel('Sample')
        self.ax_prob.set_ylabel('Probability')
        self.ax_prob.grid(True)
        
        # Update constellation plot (downsample for better visualization)
        downsample_factor = max(1, len(rx_signal) // 1000)  
        self.ax_constellation.clear()
        self.ax_constellation.scatter(
            np.real(rx_signal[::downsample_factor]), 
            np.imag(rx_signal[::downsample_factor]),
            s=2, alpha=0.5
        )
        self.ax_constellation.set_title('IQ Constellation')
        self.ax_constellation.set_xlabel('I')
        self.ax_constellation.set_ylabel('Q')
        self.ax_constellation.grid(True)
        self.ax_constellation.set_aspect('equal')
        
        # Update status text
        detection_rate = sum(self.detection_history) / max(1, len(self.detection_history))
        status_msg = (
            f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"Signal Power: {np.mean(np.abs(rx_signal)**2):.2f}\n"
            f"Current Probability: {probability:.4f}\n"
            f"Detection Rate: {detection_rate:.2%}"
        )
        self.status_text.set_text(status_msg)
        
        # Update detection indicator
        if prediction == 1:
            self.detection_patch.set_facecolor('red')
            self.detection_text.set_text("DRONE DETECTED")
        else:
            self.detection_patch.set_facecolor('green')
            self.detection_text.set_text("NO DRONE")
        
        # Refresh plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def run_detection(self):
        """Main detection loop"""
        try:
            print("Starting drone detection. Press Ctrl+C to stop.")
            print(f"Display settings: Center Freq = {CENTER_FREQ/1e9:.3f} GHz, FFT Size = {FFT_SIZE}")
            print(f"Waterfall shows ~{NUM_SLICES/10:.0f} seconds of history")
            
            # For consistent timing
            update_rate = 0.1  # seconds between updates (10 Hz)
            last_update_time = time.time()
            
            while True:
                # Calculate time to next update
                current_time = time.time()
                elapsed = current_time - last_update_time
                
                if elapsed >= update_rate:
                    # Receive IQ samples
                    rx_signal = self.sdr.rx()
                    
                    # Process signal and make prediction
                    freq, s_dbfs, probability, prediction = self.process_signal(rx_signal)
                    
                    # Update visualization
                    self.update_plot(freq, s_dbfs, probability, prediction, rx_signal)
                    
                    # Print status to console (less frequently to avoid spam)
                    if hasattr(self, 'print_counter'):
                        self.print_counter += 1
                    else:
                        self.print_counter = 0
                        
                    if self.print_counter % 10 == 0:  # Print every ~1 second
                        status = "DETECTED" if prediction == 1 else "not detected"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Drone {status} (prob: {probability:.4f})")
                    
                    # Update timing
                    last_update_time = current_time
                else:
                    # Short sleep to prevent CPU hogging in tight loop
                    time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nDetection stopped by user.")
        except Exception as e:
            print(f"Error in detection loop: {e}")
            import traceback
            traceback.print_exc()

def main():
    global FFT_SIZE, CENTER_FREQ  # Move global declarations to the top of the function
    
    parser = argparse.ArgumentParser(description='Drone Detection System')
    parser.add_argument('--collect-data', action='store_true', help='Collect drone data for dataset creation')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to collect for dataset')
    parser.add_argument('--tx-channel', type=int, default=0, help='Transmit channel (0 or 1)')
    parser.add_argument('--rx-channel', type=int, default=0, help='Receive channel (0 or 1)')
    parser.add_argument('--fft-size', type=int, default=FFT_SIZE, help='FFT size for spectral analysis')
    parser.add_argument('--center-freq', type=float, default=CENTER_FREQ/1e9, 
                        help=f'Center frequency in GHz (default: {CENTER_FREQ/1e9:.1f})')
    args = parser.parse_args()
    
    # Update FFT_SIZE if specified
    if args.fft_size != FFT_SIZE:
        print(f"Using custom FFT size: {args.fft_size}")
        FFT_SIZE = args.fft_size
    
    # Update CENTER_FREQ if specified
    center_freq_ghz = args.center_freq
    if center_freq_ghz != CENTER_FREQ/1e9:
        print(f"Using custom center frequency: {center_freq_ghz} GHz")
        CENTER_FREQ = center_freq_ghz * 1e9
    
    # Configure SDR
    sdr = configure_sdr(args)
    if sdr is None:
        return
    
    # Load the model if available
    model = None
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        if model is None:
            print("Warning: Could not load model. Will use energy detection as fallback.")
    else:
        print("Warning: Model file not found. Will use energy detection as fallback.")
    
    # Data collection mode
    if args.collect_data:
        output_file = collect_drone_data(sdr, num_samples=args.num_samples)
        if output_file:
            print(f"Data collection completed. Data saved to {output_file}")
            print("You can use this data to train or improve your drone detection model.")
        return
    
    # Detection mode
    detector = DroneDetector(sdr, model)
    detector.run_detection()

if __name__ == "__main__":
    main() 