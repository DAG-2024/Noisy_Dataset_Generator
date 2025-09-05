# CS50 Lecture Audio Segmenter

This script downloads the CS50 lecture from YouTube and segments it into logical, sentence-based audio chunks perfect for speech processing and noise addition experiments.

## Features

- **YouTube Audio Download**: Downloads high-quality audio from the CS50 lecture
- **Intelligent Segmentation**: Uses OpenAI Whisper for transcription and sentence boundary detection
- **Logical Cuts**: Segments are cut at complete sentence boundaries, not mid-word
- **Duration Control**: Creates segments of 30-120 seconds each
- **Target Quantity**: Generates approximately 100 segments from the full lecture
- **Rich Metadata**: Includes timestamps, transcripts, and segment information

## Quick Start

### Process the CS50 Lecture
```bash
# Install dependencies
pip install -r requirements.txt

# Process the specific CS50 lecture
python process_cs50_lecture.py
```

This will:
1. Download the CS50 lecture audio from [YouTube](https://www.youtube.com/watch?v=8mAITcNt710)
2. Transcribe it using Whisper
3. Create ~100 segments of 30-120 seconds each
4. Save everything to `cs50_lecture_segments/`

### Custom Processing
```bash
# Process any YouTube lecture
python lecture_segmenter.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Custom settings
python lecture_segmenter.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output my_lecture_segments \
    --segments 50 \
    --min-duration 45 \
    --max-duration 90
```

## Output Structure

```
cs50_lecture_segments/
├── audio/                          # Individual audio segments
│   ├── segment_000.wav
│   ├── segment_001.wav
│   └── ...
├── transcripts/                    # Transcription files
│   ├── full_transcript.txt         # Complete lecture transcript
│   └── detailed_transcript.json    # Timestamped word-level transcript
├── segments_metadata.json          # Segment information and timestamps
└── processing_summary.json         # Processing statistics
```

## Segment Metadata

Each segment includes:
- **ID**: Sequential segment number
- **Filename**: Audio file name
- **Start/End Time**: Timestamps in seconds
- **Duration**: Segment length in seconds
- **Text**: Complete transcript of the segment
- **Sentence Count**: Number of sentences in the segment

## Usage with Noise Maker

After creating segments, you can apply noise to them:

```bash
# Apply noise to individual segments
python noise_maker.py cs50_lecture_segments/audio/segment_000.wav
python noise_maker.py cs50_lecture_segments/audio/segment_001.wav

# Batch process all segments (example script)
for file in cs50_lecture_segments/audio/*.wav; do
    python noise_maker.py "$file" -o "${file%.wav}_noisy.wav"
done
```

## Technical Details

### Audio Processing
- **Format**: Downloads as WAV for best quality
- **Sample Rate**: Preserves original YouTube audio quality
- **Segmentation**: Uses sentence boundaries for natural cuts

### Transcription
- **Model**: OpenAI Whisper (configurable size)
- **Language**: Automatic detection (optimized for English lectures)
- **Timestamps**: Word-level timing for precise cuts

### Segmentation Algorithm
1. Transcribes entire lecture with timestamps
2. Identifies sentence boundaries
3. Groups sentences into segments of target duration
4. Ensures segments contain complete sentences
5. Balances segment count and duration constraints

## Requirements

- Python 3.7+
- yt-dlp (YouTube downloader)
- openai-whisper (speech recognition)
- pydub (audio processing)
- nltk (natural language processing)
- numpy, scipy, librosa, soundfile

## Performance Notes

- **Processing Time**: ~2-3x real-time (e.g., 1-hour lecture takes 2-3 hours)
- **Storage**: ~100MB for audio + ~10MB for transcripts
- **Memory**: ~2-4GB RAM during processing
- **Model Size**: "base" model (~140MB) provides good balance of speed/accuracy

## Troubleshooting

### Common Issues
1. **Download Fails**: Check internet connection and YouTube URL
2. **Transcription Slow**: Use smaller Whisper model (`--model tiny`)
3. **Memory Issues**: Process shorter videos or use smaller model
4. **Audio Quality**: Ensure good internet connection for download

### Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# If you encounter issues with specific packages:
pip install --upgrade yt-dlp openai-whisper pydub nltk
```

## License

This project is open source and available under the MIT License.
