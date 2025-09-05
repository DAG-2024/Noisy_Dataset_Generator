# Audio Noise Maker

A Python script that adds natural-sounding audio interference to speech recordings to damage speech legibility while maintaining realistic audio artifacts.

## Features

- **Natural Interference Types**: Static noise, clicks, pops, dropouts, electrical hum, vinyl crackle, and rare dramatic interference
- **Bell Curve Distribution**: Interferences are placed using a normal distribution centered in the middle of the audio
- **Automatic Timing**: Adds approximately 3 interferences per 60 seconds of audio (configurable)
- **Realistic Audio Processing**: Uses pink noise, proper filtering, and natural envelopes
- **Configurable**: Adjustable sample rate, interference rate, and output parameters
- **Rare Events**: Very rare dramatic interference for intense audio disruption

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python noise_maker.py input_speech.wav
```

This will create `input_speech_noisy.wav` with added interference.

### Specify Output File
```bash
python noise_maker.py input_speech.wav -o output_noisy.wav
```

### Custom Sample Rate
```bash
python noise_maker.py input_speech.wav --sample-rate 44100
```

### Custom Interference Rate
```bash
# Light interference (1 per minute)
python noise_maker.py input_speech.wav --interferences-per-60sec 1

# Moderate interference (5 per minute)
python noise_maker.py input_speech.wav --interferences-per-60sec 5

# Heavy interference (10 per minute)
python noise_maker.py input_speech.wav --interferences-per-60sec 10
```

### Batch Processing
```bash
for file in Introductory_Calculus_lecture_segments/audio/*.wav; do
    python noise_maker.py "$file" -o "${file%.wav}_noisy.wav" --interferences-per-60sec 5
done
```

## Interference Types

The script randomly selects from these natural-sounding interference types:

1. **Static Noise** (28% probability) - Pink noise filtered to sound like radio static
2. **Click** (20% probability) - Sharp, brief click sounds
3. **Pop** (15% probability) - Lower frequency pop sounds
4. **Dropout** (15% probability) - Brief silence with natural fade in/out
5. **Electrical Hum** (10% probability) - 60Hz hum with harmonics
6. **Crackle** (10% probability) - Vinyl-like crackling sounds
7. **🎆 Dramatic Interference** (2% probability) - Rare, long-lasting, and intense interference combining:
   - Low frequency rumble (thunder/machinery)
   - Mid frequency static burst
   - High frequency screech with modulation
   - Random audio spikes
   - Lasts 2-5 seconds (vs 0.2-1.0s for others)
   - 80% intensity (vs 15-40% for others)

## Technical Details

- **Distribution**: Interferences are placed using a bell curve (normal distribution) centered in the middle of the audio
- **Duration**: 
  - Normal interferences: 0.2-1.0 seconds (randomly determined)
  - Dramatic interference: 2-5 seconds (much longer and more intense)
- **Intensity**: Carefully calibrated to damage speech legibility while maintaining natural sound
  - Normal interferences: 15-40% intensity
  - Dramatic interference: 80% intensity
- **Audio Quality**: Preserves original audio quality in unaffected regions
- **Rare Events**: Dramatic interference occurs only 2% of the time, making it a special event

## Requirements

- Python 3.7+
- numpy
- scipy
- librosa
- soundfile
- matplotlib (for potential visualization)

## Example

```python
from noise_maker import AudioNoiseMaker

# Create noise maker instance
noise_maker = AudioNoiseMaker(sample_rate=22050)

# Process audio file
noise_maker.process_audio('speech.wav', 'speech_noisy.wav')
```

## License

This project is open source and available under the MIT License.
