#!/usr/bin/env python3
"""
Audio Noise Maker - Adds natural-sounding interference to speech recordings

This script takes a .wav speech recording and adds various types of natural
audio interference to damage speech legibility while maintaining realistic
audio artifacts.

Features:
- 3 interferences per 60 seconds of audio
- Bell curve distribution for interference placement
- Multiple natural interference types
- Preserves original audio quality where not affected
"""

import numpy as np
import librosa
import soundfile as sf
import argparse
import os
from scipy import signal
from scipy.stats import norm
import random


class AudioNoiseMaker:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path):
        """Load audio file and return audio data and sample rate"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def save_audio(self, audio, file_path, sample_rate):
        """Save audio data to file"""
        sf.write(file_path, audio, sample_rate)
    
    def generate_static_noise(self, duration_samples, intensity=0.3):
        """Generate realistic static/white noise"""
        # Create pink noise (more natural than white noise)
        noise = np.random.randn(duration_samples)
        
        # Apply pink noise filter (1/f characteristic)
        freqs = np.fft.fftfreq(duration_samples, 1/self.sample_rate)
        freqs[0] = 1  # Avoid division by zero
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        pink_filter[0] = 0
        
        noise_fft = np.fft.fft(noise)
        pink_noise_fft = noise_fft * pink_filter
        pink_noise = np.real(np.fft.ifft(pink_noise_fft))
        
        # Normalize and apply intensity
        pink_noise = pink_noise / np.max(np.abs(pink_noise)) * intensity
        return pink_noise
    
    def generate_click_pop(self, duration_samples, click_type='click'):
        """Generate click or pop sounds"""
        if click_type == 'click':
            # Sharp click sound
            t = np.linspace(0, 0.01, int(0.01 * self.sample_rate))
            click = np.exp(-t * 200) * np.sin(2 * np.pi * 2000 * t)
            # Add some randomness
            click *= np.random.uniform(0.5, 1.5)
        else:  # pop
            # Pop sound (lower frequency, longer duration)
            t = np.linspace(0, 0.02, int(0.02 * self.sample_rate))
            pop = np.exp(-t * 100) * np.sin(2 * np.pi * 800 * t)
            pop *= np.random.uniform(0.3, 1.0)
            click = pop
        
        # Pad to desired duration
        if len(click) < duration_samples:
            click = np.pad(click, (0, duration_samples - len(click)), 'constant')
        else:
            click = click[:duration_samples]
        
        return click * 0.4  # Reduce intensity
    
    def generate_dropout(self, duration_samples):
        """Generate audio dropout (silence with fade in/out)"""
        dropout = np.zeros(duration_samples)
        
        # Add fade in/out to make it more natural
        fade_samples = int(0.005 * self.sample_rate)  # 5ms fade
        if duration_samples > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            # Add very low level noise during dropout
            noise_level = 0.01
            dropout[:fade_samples] = fade_in * np.random.normal(0, noise_level, fade_samples)
            dropout[-fade_samples:] = fade_out * np.random.normal(0, noise_level, fade_samples)
            dropout[fade_samples:-fade_samples] = np.random.normal(0, noise_level, 
                                                                  duration_samples - 2 * fade_samples)
        
        return dropout
    
    def generate_hum(self, duration_samples, frequency=60):
        """Generate electrical hum (60Hz or 50Hz)"""
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples)
        # Add harmonics for more realistic hum
        hum = (np.sin(2 * np.pi * frequency * t) + 
               0.3 * np.sin(2 * np.pi * frequency * 2 * t) +
               0.1 * np.sin(2 * np.pi * frequency * 3 * t))
        
        # Add some amplitude modulation
        hum *= (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))
        return hum * 0.2
    
    def generate_crackle(self, duration_samples):
        """Generate crackling sound (like old vinyl)"""
        crackle = np.random.randn(duration_samples)
        # Apply envelope to make it more realistic
        envelope = np.exp(-np.linspace(0, 3, duration_samples))
        crackle *= envelope
        
        # Add some filtering to make it sound more like vinyl crackle
        b, a = signal.butter(4, [1000, 8000], btype='band', fs=self.sample_rate)
        crackle = signal.filtfilt(b, a, crackle)
        
        return crackle * 0.15
    
    def generate_dramatic_interference(self, duration_samples):
        """Generate dramatic long and loud interference (rare event)"""
        # Create a complex interference pattern
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples)
        
        # Multiple frequency components for realistic interference
        interference = np.zeros(duration_samples)
        
        # Low frequency rumble (like thunder or machinery)
        rumble_freq = np.random.uniform(20, 80)
        rumble = np.sin(2 * np.pi * rumble_freq * t)
        rumble *= np.random.uniform(0.3, 0.7)
        
        # Mid frequency static burst
        static_burst = np.random.randn(duration_samples)
        # Apply bandpass filter for more realistic static
        b, a = signal.butter(4, [200, 2000], btype='band', fs=self.sample_rate)
        static_burst = signal.filtfilt(b, a, static_burst)
        static_burst *= np.random.uniform(0.4, 0.8)
        
        # High frequency screech (like feedback or radio interference)
        screech_freq = np.random.uniform(3000, 8000)
        screech = np.sin(2 * np.pi * screech_freq * t)
        # Add frequency modulation for more realistic effect
        fm = np.sin(2 * np.pi * 5 * t) * 1000  # 5Hz modulation
        screech = np.sin(2 * np.pi * (screech_freq + fm) * t)
        screech *= np.random.uniform(0.2, 0.5)
        
        # Combine all components
        interference = rumble + static_burst + screech
        
        # Apply dramatic envelope - starts loud, fades out
        envelope = np.exp(-np.linspace(0, 2, duration_samples))  # Slower decay
        # Add some amplitude modulation for realism
        amp_mod = 1 + 0.3 * np.sin(2 * np.pi * 2 * t)  # 2Hz amplitude modulation
        envelope *= amp_mod
        
        interference *= envelope
        
        # Add some random spikes for extra drama
        spike_positions = np.random.choice(duration_samples, size=int(duration_samples * 0.01), replace=False)
        for pos in spike_positions:
            if pos < duration_samples - 100:
                spike = np.random.uniform(-1, 1, 100)
                spike *= np.exp(-np.linspace(0, 5, 100))  # Quick decay
                interference[pos:pos+100] += spike * 0.5
        
        # Normalize and apply high intensity
        interference = interference / np.max(np.abs(interference)) * 0.8
        
        return interference
    
    def get_interference_type(self):
        """Randomly select interference type with weighted probabilities"""
        interference_types = [
            ('static', 0.28),
            ('click', 0.2),
            ('pop', 0.15),
            ('dropout', 0.15),
            ('hum', 0.1),
            ('crackle', 0.1),
            ('dramatic', 0.02)  # Very rare dramatic interference
        ]
        
        types, weights = zip(*interference_types)
        return np.random.choice(types, p=weights)
    
    def calculate_interference_positions(self, audio_length_samples, num_interferences):
        """Calculate interference positions using bell curve distribution"""
        # Convert to seconds for easier calculation
        audio_length_sec = audio_length_samples / self.sample_rate
        
        # Create bell curve centered in the middle of the audio
        center = audio_length_sec / 2
        std_dev = audio_length_sec / 6  # 3-sigma rule covers most of the audio
        
        # Generate positions using normal distribution
        positions = np.random.normal(center, std_dev, num_interferences)
        
        # Ensure positions are within audio bounds
        positions = np.clip(positions, 0.1, audio_length_sec - 0.1)
        
        # Convert back to sample indices
        return (positions * self.sample_rate).astype(int)
    
    def add_interference(self, audio, position_samples, interference_type, duration_sec=0.5):
        """Add specific interference to audio at given position"""
        duration_samples = int(duration_sec * self.sample_rate)
        
        # Ensure we don't go beyond audio bounds
        start_idx = max(0, position_samples - duration_samples // 2)
        end_idx = min(len(audio), start_idx + duration_samples)
        actual_duration = end_idx - start_idx
        
        if actual_duration <= 0:
            return audio
        
        # Generate interference
        if interference_type == 'static':
            interference = self.generate_static_noise(actual_duration)
        elif interference_type == 'click':
            interference = self.generate_click_pop(actual_duration, 'click')
        elif interference_type == 'pop':
            interference = self.generate_click_pop(actual_duration, 'pop')
        elif interference_type == 'dropout':
            interference = self.generate_dropout(actual_duration)
        elif interference_type == 'hum':
            interference = self.generate_hum(actual_duration)
        elif interference_type == 'crackle':
            interference = self.generate_crackle(actual_duration)
        elif interference_type == 'dramatic':
            interference = self.generate_dramatic_interference(actual_duration)
        else:
            interference = np.zeros(actual_duration)
        
        # Add interference to audio
        audio[start_idx:end_idx] += interference
        
        return audio
    
    def process_audio(self, input_path, output_path, interferences_per_60sec=3):
        """Main processing function"""
        print(f"Loading audio from: {input_path}")
        audio, sr = self.load_audio(input_path)
        
        # Calculate number of interferences (configurable per 60 seconds)
        audio_length_sec = len(audio) / sr
        num_interferences = max(1, int(audio_length_sec / 60 * interferences_per_60sec))
        
        print(f"Audio length: {audio_length_sec:.2f} seconds")
        print(f"Adding {num_interferences} interferences ({interferences_per_60sec} per 60 seconds)")
        
        # Calculate interference positions using bell curve
        positions = self.calculate_interference_positions(len(audio), num_interferences)
        
        # Add interferences
        for i, pos in enumerate(positions):
            interference_type = self.get_interference_type()
            
            # Dramatic interference gets longer duration and special handling
            if interference_type == 'dramatic':
                duration = np.random.uniform(1.0, 2.5)  # Much longer: 1.0-2.5 seconds
                print(f"🎆 DRAMATIC INTERFERENCE! Adding {interference_type} at {pos/sr:.2f}s (duration: {duration:.2f}s)")
            else:
                duration = np.random.uniform(0.2, 1.0)  # Normal duration: 0.2-1.0 seconds
                print(f"Adding {interference_type} at {pos/sr:.2f}s (duration: {duration:.2f}s)")
            
            audio = self.add_interference(audio, pos, interference_type, duration)
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        
        print(f"Saving processed audio to: {output_path}")
        self.save_audio(audio, output_path, sr)
        
        return audio, sr


def main():
    parser = argparse.ArgumentParser(description='Add natural audio interference to speech recordings')
    parser.add_argument('input', help='Input .wav file path')
    parser.add_argument('-o', '--output', help='Output .wav file path (default: input_noisy.wav)')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Sample rate (default: 22050)')
    parser.add_argument('--interferences-per-60sec', type=int, default=3, 
                       help='Number of interferences per 60 seconds of audio (default: 3)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    if not args.input.lower().endswith('.wav'):
        print("Error: Input file must be a .wav file")
        return 1
    
    # Set output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_noisy.wav"
    
    # Process audio
    noise_maker = AudioNoiseMaker(sample_rate=args.sample_rate)
    
    try:
        noise_maker.process_audio(args.input, args.output, args.interferences_per_60sec)
        print("Processing completed successfully!")
        return 0
    except Exception as e:
        print(f"Error processing audio: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
