import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Mac
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
CENTER_FREQ = 2.4e9  # 2.4 GHz center frequency
SAMPLE_RATE = 40e6   # Match training data: 40 MHz
RX_GAIN = 50         # Receiver gain
TX_GAIN = -10        # Transmitter gain
BUFFER_SIZE = int(40e6 * 0.02)  # 20ms of samples at 40MHz

# Channel configuration
TX_CHANNEL = 0       # Using Tx Channel 0
RX_CHANNEL = 0       # Using Rx Channel 0

# Transmission parameters
TX_PULSE_INTERVAL = 5  # Time in seconds between transmission pulses
TX_PULSE_DURATION = 3  # Duration of each transmission pulse in seconds

# Drone detection parameters
PROBABILITY_THRESHOLD = 0.5  # Threshold for drone detection
HISTORY_LENGTH = 50          # Number of samples to keep in history

# Check if the model exists, otherwise warn about it
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model file '{MODEL_PATH}' not found.")
    print("You need to have a trained XGBoost model file in your directory.")
    print("The script will attempt to run, but prediction functionality will be limited.")

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

def configure_sdr():
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
            max_sample_rate = int(40e6)  # Default max for PlutoSDR
            print(f"Attempting to set sample rate to {max_sample_rate/1e6} MHz...")
            sdr.sample_rate = max_sample_rate
        except Exception as e:
            print(f"Warning: Could not set 40MHz sample rate: {e}")
            print("Falling back to 20MHz sample rate...")
            try:
                sdr.sample_rate = int(20e6)
                max_sample_rate = int(20e6)
            except Exception as e:
                print(f"Warning: Could not set 20MHz sample rate: {e}")
                print("Falling back to 10MHz sample rate...")
                sdr.sample_rate = int(10e6)
                max_sample_rate = int(10e6)

        # Update global sample rate to match what we could actually set
        global SAMPLE_RATE
        SAMPLE_RATE = sdr.sample_rate
        print(f"Successfully set sample rate to {SAMPLE_RATE/1e6} MHz")
        
        # Configure common settings
        sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
        sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
        
        # Configure transmitter
        sdr.tx_lo = int(CENTER_FREQ)
        sdr.tx_hardwaregain_chan0 = TX_GAIN
        
        # Configure receiver
        sdr.rx_lo = int(CENTER_FREQ)
        sdr.rx_buffer_size = int(SAMPLE_RATE * 0.02)  # 20ms of samples
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        
        print("\nSDR Configuration:")
        print(f"Sample Rate: {sdr.sample_rate/1e6} MHz")
        print(f"Center Frequency: {sdr.rx_lo/1e9} GHz")
        print(f"TX Channel: {TX_CHANNEL}")
        print(f"RX Channel: {RX_CHANNEL}")
        print(f"TX Gain: {sdr.tx_hardwaregain_chan0} dB")
        print(f"RX Gain: {sdr.rx_hardwaregain_chan0} dB")
        print(f"Buffer Size: {sdr.rx_buffer_size} samples")
        
        return sdr
    except Exception as e:
        print(f"Error configuring SDR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure PlutoSDR is properly connected via USB")
        print("2. Check if device shows up in 'lsusb' (Linux) or Device Manager (Windows)")
        print("3. Try unplugging and replugging the device")
        print("4. Verify no other software is currently using the SDR")
        return None

def calculate_signal_strength(signal_data):
    """Calculate signal strength in dBm"""
    # Calculate RMS power
    power = np.mean(np.abs(signal_data)**2)
    # Convert to dBm
    power_dbm = 10 * np.log10(power) + 30
    return power_dbm

def generate_drone_signal():
    """Generate a more realistic drone-like signal based on training data characteristics"""
    fs = SAMPLE_RATE
    N = BUFFER_SIZE
    t = np.arange(N) / fs
    
    # Create a more complex signal matching drone characteristics
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
    
    # Add some noise
    noise = np.random.normal(0, 0.05, N) + 1j * np.random.normal(0, 0.05, N)
    tx_iq_signal += noise
    
    # Normalize
    tx_iq_signal = tx_iq_signal / np.max(np.abs(tx_iq_signal)) * 0.8
    return tx_iq_signal

def extract_dft_features(signal_data, n_fft=2048):
    """Extract features exactly as done in training"""
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

def calculate_fft(iq_signal, fs):
    """Calculate FFT with proper frequency axis and windowing"""
    N = len(iq_signal)
    win = np.hamming(N)
    iq_windowed = iq_signal * win
    
    fft_data = np.fft.fftshift(np.fft.fft(iq_windowed))
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    power_db = 20 * np.log10(np.abs(fft_data) / N + 1e-10)
    
    return freq, power_db

def create_spectrogram(iq_signal, fs, nperseg=256):
    """Create a spectrogram from IQ data"""
    f, t, Sxx = signal.spectrogram(iq_signal, fs=fs, nperseg=nperseg, 
                                  return_onesided=False, scaling='spectrum')
    # Shift frequencies to center
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db

def validate_signal(rx_signal, fs=SAMPLE_RATE):
    """Validate received signal characteristics"""
    # Calculate signal metrics
    power_db = 10 * np.log10(np.mean(np.abs(rx_signal)**2))
    freq, psd = signal.welch(rx_signal, fs, nperseg=1024)
    peak_freq = freq[np.argmax(psd)]
    snr = 10 * np.log10(np.max(psd) / np.mean(psd))
    
    print("\nSignal Validation:")
    print(f"Power: {power_db:.2f} dB")
    print(f"Peak Frequency: {peak_freq/1e3:.2f} kHz")
    print(f"SNR: {snr:.2f} dB")
    
    return {
        'power_db': power_db,
        'peak_freq': peak_freq,
        'snr': snr
    }

def update_visualization(ax_spectrum, ax_waterfall, ax_constellation, ax_probability, 
                        rx_signal, freq, power_db, probability_history, time_history):
    """Update all visualization plots with new data"""
    # Update spectrum plot
    ax_spectrum.clear()
    ax_spectrum.plot(freq/1e3, power_db)
    ax_spectrum.set_title('RF Spectrum (Baseband)')
    ax_spectrum.set_xlabel('Frequency (kHz)')
    ax_spectrum.set_ylabel('Power (dB)')
    ax_spectrum.grid(True)
    
    # Update waterfall plot
    f, t, Sxx = signal.spectrogram(rx_signal, fs=SAMPLE_RATE, nperseg=1024)
    ax_waterfall.clear()
    ax_waterfall.pcolormesh(t, f/1e3, 10 * np.log10(Sxx), shading='gouraud')
    ax_waterfall.set_title('Signal Spectrogram')
    ax_waterfall.set_xlabel('Time')
    ax_waterfall.set_ylabel('Frequency (kHz)')
    
    # Update constellation plot
    ax_constellation.clear()
    ax_constellation.scatter(np.real(rx_signal), np.imag(rx_signal), s=1, alpha=0.1)
    ax_constellation.set_title('IQ Constellation')
    ax_constellation.set_xlabel('I')
    ax_constellation.set_ylabel('Q')
    ax_constellation.grid(True)
    ax_constellation.set_aspect('equal')
    
    # Update probability plot
    ax_probability.clear()
    ax_probability.plot(time_history, probability_history, 'b-')
    ax_probability.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax_probability.set_title('Drone Detection Probability')
    ax_probability.set_xlabel('Sample')
    ax_probability.set_ylabel('Probability')
    ax_probability.set_ylim(0, 1)
    ax_probability.grid(True)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Drone Detection Testing')
    parser.add_argument('--skip-tx', action='store_true', help='Skip transmission and only visualize')
    parser.add_argument('--tx-channel', type=int, default=0, help='Transmit channel (0 or 1)')
    parser.add_argument('--rx-channel', type=int, default=0, help='Receive channel (0 or 1)')
    parser.add_argument('--continuous-tx', action='store_true', help='Enable continuous transmission instead of pulsed')
    parser.add_argument('--pulse-interval', type=int, default=TX_PULSE_INTERVAL, 
                        help=f'Seconds between transmission pulses (default: {TX_PULSE_INTERVAL})')
    parser.add_argument('--pulse-duration', type=int, default=TX_PULSE_DURATION,
                        help=f'Duration of each transmission pulse in seconds (default: {TX_PULSE_DURATION})')
    args = parser.parse_args()
    
    # Update transmission parameters based on command line arguments
    pulse_interval = args.pulse_interval
    pulse_duration = args.pulse_duration
    
    # Update channel configuration based on command line arguments
    global TX_CHANNEL, RX_CHANNEL
    TX_CHANNEL = args.tx_channel
    RX_CHANNEL = args.rx_channel

    # Load the trained model if available
    model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    # Configure SDR
    sdr = configure_sdr()
    if sdr is None:
        return

    # Generate drone signal
    tx_signal = generate_drone_signal()

    # Initialize history
    time_history = deque(maxlen=HISTORY_LENGTH)
    probability_history = deque(maxlen=HISTORY_LENGTH)
    signal_strength_history = deque(maxlen=HISTORY_LENGTH)
    
    # Set up visualization
    plt.ion()
    fig = plt.figure(figsize=(15, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3)
    
    # Create subplots
    ax_spectrum = fig.add_subplot(gs[0, 0:2])
    ax_waterfall = fig.add_subplot(gs[1, 0:2]) 
    ax_constellation = fig.add_subplot(gs[0, 2])
    ax_probability = fig.add_subplot(gs[1, 2])
    ax_signal_strength = fig.add_subplot(gs[2, 0])
    ax_status = fig.add_subplot(gs[2, 1:])
    
    # Set up plots
    spectrum_line, = ax_spectrum.plot([], [])
    ax_spectrum.set_title('RF Spectrum at 2.4 GHz')
    ax_spectrum.set_xlabel('Frequency Offset (kHz)')
    ax_spectrum.set_ylabel('Power (dB)')
    ax_spectrum.grid(True)
    
    # Waterfall plot (spectrogram)
    waterfall_img = ax_waterfall.imshow(np.zeros((100, 100)), aspect='auto', 
                                      origin='lower', cmap='viridis')
    ax_waterfall.set_title('Signal Spectrogram')
    ax_waterfall.set_xlabel('Time')
    ax_waterfall.set_ylabel('Frequency (kHz)')
    fig.colorbar(waterfall_img, ax=ax_waterfall, label='Power (dB)')
    
    # Constellation plot
    constellation_plot = ax_constellation.scatter([], [], s=2)
    ax_constellation.set_title('IQ Constellation')
    ax_constellation.set_xlabel('I')
    ax_constellation.set_ylabel('Q')
    ax_constellation.grid(True)
    ax_constellation.set_aspect('equal')
    
    # Probability plot
    probability_line, = ax_probability.plot([], [])
    ax_probability.axhline(y=PROBABILITY_THRESHOLD, color='r', linestyle='--')
    ax_probability.set_title('Drone Detection Probability')
    ax_probability.set_xlabel('Sample')
    ax_probability.set_ylabel('Probability')
    ax_probability.set_ylim(0, 1)
    ax_probability.grid(True)
    
    # Signal strength plot
    signal_line, = ax_signal_strength.plot([], [])
    ax_signal_strength.set_title('Signal Strength')
    ax_signal_strength.set_xlabel('Sample')
    ax_signal_strength.set_ylabel('Power (dBm)')
    ax_signal_strength.grid(True)
    
    # Status display (text box)
    ax_status.axis('off')
    status_text = ax_status.text(0.1, 0.5, "Initializing...", 
                               fontsize=16, transform=ax_status.transAxes)
    
    # Add a detection indicator
    detection_indicator = plt.Rectangle((0.7, 0.4), 0.25, 0.25, 
                                      facecolor='gray', transform=ax_status.transAxes)
    ax_status.add_patch(detection_indicator)
    
    # Add a transmission indicator
    transmission_indicator = plt.Rectangle((0.7, 0.7), 0.25, 0.25, 
                                         facecolor='gray', transform=ax_status.transAxes)
    ax_status.add_patch(transmission_indicator)
    tx_indicator_text = ax_status.text(0.7, 0.98, "TX OFF", 
                                    fontsize=12, transform=ax_status.transAxes)
    
    plt.tight_layout()
    plt.show(block=False)

    # Start transmission if not skipped
    transmitting = False
    last_tx_toggle_time = time.time()
    
    try:
        print("Starting drone detection testing. Press Ctrl+C to stop...")
        if not args.skip_tx:
            if args.continuous_tx:
                print(f"Starting CONTINUOUS transmission on channel {TX_CHANNEL}")
                try:
                    # Configure for continuous transmission
                    sdr.tx_cyclic_buffer = True
                    sdr.tx(tx_signal)
                    transmitting = True
                    print("Transmission started successfully")
                except Exception as e:
                    print(f"Error starting transmission: {e}")
                    return
            else:
                print(f"Starting PULSED transmission on channel {TX_CHANNEL}")
                print(f"Pulse interval: {pulse_interval}s, Duration: {pulse_duration}s")

        sample_count = 0
        detection_count = 0
        total_samples = 0
        
        while True:
            current_time = time.time()
            
            # Handle pulsed transmission if not in continuous mode and not skipped
            if not args.skip_tx and not args.continuous_tx:
                time_since_toggle = current_time - last_tx_toggle_time
                
                # Toggle transmission state based on timing
                if transmitting and time_since_toggle >= pulse_duration:
                    # Turn off transmission
                    try:
                        sdr.tx_destroy_buffer()
                        transmitting = False
                        last_tx_toggle_time = current_time
                        print(f"TX OFF at {datetime.now().strftime('%H:%M:%S')}")
                    except Exception as e:
                        print(f"Error stopping transmission: {e}")
                
                elif not transmitting and time_since_toggle >= pulse_interval:
                    # Turn on transmission
                    try:
                        sdr.tx_cyclic_buffer = True
                        sdr.tx(tx_signal)
                        transmitting = True
                        last_tx_toggle_time = current_time
                        print(f"TX ON at {datetime.now().strftime('%H:%M:%S')}")
                    except Exception as e:
                        print(f"Error starting transmission: {e}")
            
            # Receive IQ samples
            try:
                rx_signal = sdr.rx()
            except Exception as e:
                print(f"Error receiving data: {e}")
                time.sleep(1)
                continue

            # Validate signal quality
            signal_metrics = validate_signal(rx_signal)
            
            # Only process if signal quality is good enough
            if signal_metrics['snr'] > 10:  # Minimum 10dB SNR
                sample_count += 1
                total_samples += 1
                
                # Extract features and make prediction
                features = extract_dft_features(rx_signal)
                prediction = model.predict(features.reshape(1, -1))
                probability = model.predict_proba(features.reshape(1, -1))[0][1]
                
                # Update histories
                probability_history.append(probability)
                time_history.append(sample_count)
                signal_strength_history.append(signal_metrics['power_db'])
                
                if prediction[0] == 1:
                    detection_count += 1
                
                # Calculate detection rate
                detection_rate = detection_count / total_samples
                
                # Update visualization
                freq, power_db = calculate_fft(rx_signal, SAMPLE_RATE)
                update_visualization(ax_spectrum, ax_waterfall, ax_constellation, 
                                  ax_probability, rx_signal, freq, power_db,
                                  probability_history, time_history)
                
                # Update status display
                tx_status = "TRANSMITTING" if transmitting else "TX OFF"
                status_msg = (
                    f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                    f"TX Status: {tx_status}\n"
                    f"Signal Power: {signal_metrics['power_db']:.2f} dB\n"
                    f"SNR: {signal_metrics['snr']:.2f} dB\n"
                    f"Detection Rate: {detection_rate:.2%}\n"
                    f"Current Probability: {probability:.4f}"
                )
                status_text.set_text(status_msg)
                
                # Update detection indicator
                detection_color = 'red' if prediction[0] == 1 else 'green'
                detection_indicator.set_facecolor(detection_color)
                
                # Update transmission indicator
                transmission_color = 'orange' if transmitting else 'blue'
                transmission_indicator.set_facecolor(transmission_color)
                tx_indicator_text.set_text("TX ON" if transmitting else "TX OFF")
                
                # Print status to console
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"TX: {tx_status} | "
                      f"Power: {signal_metrics['power_db']:.2f} dB | "
                      f"SNR: {signal_metrics['snr']:.2f} dB | "
                      f"Prob: {probability:.4f}")
                
                # Refresh the figure
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nTesting stopped by user.")
        print(f"\nFinal Statistics:")
        print(f"Total Samples: {total_samples}")
        print(f"Detection Rate: {detection_count/total_samples:.2%}")
        print(f"Average Probability: {np.mean(probability_history):.4f}")
        print(f"Average SNR: {np.mean([m['snr'] for m in signal_metrics_history]):.2f} dB")
    finally:
        # Clean up
        try:
            if transmitting:
                sdr.tx_destroy_buffer()
                print("Transmission buffer released.")
        except:
            pass
        print("SDR resources released.")
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    main() 