#!/usr/bin/env python3
"""
Generate test audio files for testing the MPC Speech service
"""

import numpy as np
import soundfile as sf
from scipy import signal
import math

def generate_sine_wave(frequency=440, duration=3.0, sample_rate=16000, amplitude=0.3):
    """Generate a sine wave audio signal"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32)

def generate_chirp(duration=3.0, sample_rate=16000, f0=200, f1=2000, amplitude=0.3):
    """Generate a frequency sweep (chirp) signal"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * signal.chirp(t, f0, duration, f1)
    return wave.astype(np.float32)

def generate_white_noise(duration=3.0, sample_rate=16000, amplitude=0.1):
    """Generate white noise"""
    samples = int(sample_rate * duration)
    wave = amplitude * np.random.normal(0, 1, samples)
    return wave.astype(np.float32)

def generate_speech_like_signal(duration=5.0, sample_rate=16000):
    """Generate a speech-like signal with multiple frequency components"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Fundamental frequency (varies like speech)
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)  # Varying pitch
    
    # Multiple harmonics
    signal_wave = np.zeros_like(t)
    for harmonic in range(1, 6):
        amplitude = 0.3 / harmonic  # Decreasing amplitude for higher harmonics
        signal_wave += amplitude * np.sin(2 * np.pi * harmonic * f0 * t)
    
    # Add some formant-like filtering
    # Simple bandpass filtering to simulate formants
    b, a = signal.butter(4, [300, 3000], btype='band', fs=sample_rate)
    signal_wave = signal.filtfilt(b, a, signal_wave)
    
    # Add envelope to make it more speech-like
    envelope = np.exp(-0.5 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    signal_wave *= envelope
    
    return signal_wave.astype(np.float32)

def create_test_audio_files():
    """Create various test audio files"""
    sample_rate = 16000  # Common sample rate for speech recognition
    
    # 1. Simple sine wave (440 Hz - A note)
    sine_wave = generate_sine_wave(440, 3.0, sample_rate)
    sf.write('test_sine_440hz.wav', sine_wave, sample_rate)
    print("Created: test_sine_440hz.wav (3 seconds, 440 Hz sine wave)")
    
    # 2. Frequency sweep
    chirp = generate_chirp(3.0, sample_rate)
    sf.write('test_chirp.wav', chirp, sample_rate)
    print("Created: test_chirp.wav (3 seconds, frequency sweep 200-2000 Hz)")
    
    # 3. White noise (for testing noise handling)
    noise = generate_white_noise(2.0, sample_rate)
    sf.write('test_noise.wav', noise, sample_rate)
    print("Created: test_noise.wav (2 seconds, white noise)")
    
    # 4. Speech-like signal
    speech_like = generate_speech_like_signal(5.0, sample_rate)
    sf.write('test_speech_like.wav', speech_like, sample_rate)
    print("Created: test_speech_like.wav (5 seconds, speech-like signal)")
    
    # 5. Mixed signal (sine + noise)
    mixed_duration = 4.0
    sine_component = generate_sine_wave(300, mixed_duration, sample_rate, 0.4)
    noise_component = generate_white_noise(mixed_duration, sample_rate, 0.1)
    mixed_signal = sine_component + noise_component
    sf.write('test_mixed.wav', mixed_signal, sample_rate)
    print("Created: test_mixed.wav (4 seconds, 300 Hz sine + noise)")
    
    # 6. Multi-tone signal
    multi_tone = (
        0.3 * generate_sine_wave(261.63, 3.0, sample_rate, 1.0) +  # C4
        0.2 * generate_sine_wave(329.63, 3.0, sample_rate, 1.0) +  # E4
        0.2 * generate_sine_wave(392.00, 3.0, sample_rate, 1.0)    # G4
    ) / 3
    sf.write('test_chord.wav', multi_tone, sample_rate)
    print("Created: test_chord.wav (3 seconds, C major chord)")

def create_simple_test_audio():
    """Create a simple test audio file named 'audio.wav' for the examples"""
    sample_rate = 16000
    duration = 3.0
    
    # Create a simple melody-like signal
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00]  # C, D, E, F, G
    note_duration = duration / len(frequencies)
    
    full_signal = np.array([])
    
    for freq in frequencies:
        note = generate_sine_wave(freq, note_duration, sample_rate, 0.3)
        full_signal = np.concatenate([full_signal, note])
    
    sf.write('audio.wav', full_signal, sample_rate)
    print("Created: audio.wav (3 seconds, simple melody for testing)")

if __name__ == '__main__':
    print("Generating test audio files...")
    print(f"Sample rate: 16000 Hz (common for speech recognition)")
    print("-" * 50)
    
    # Install required package if not present
    try:
        from scipy import signal
    except ImportError:
        print("Installing scipy for signal processing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy'])
        from scipy import signal
    
    # Create the simple test file first
    create_simple_test_audio()
    print("-" * 50)
    
    # Create various test files
    create_test_audio_files()
    print("-" * 50)
    print("All test audio files created successfully!")
    print("\nYou can now use these files to test your MPC Speech service:")
    print("- audio.wav (for basic examples)")
    print("- test_*.wav files (for various test scenarios)")
