# Noisy Dataset Generator for Speech Processing

A comprehensive system for creating speech processing datasets from YouTube lectures. This tool automatically downloads, segments, and processes audio to create clean/noisy pairs perfect for training speech enhancement models.

## 🎯 Features

- **YouTube Processing**: Downloads and segments YouTube lectures using `lecture_segmenter.py`
- **Clean Segments**: Creates high-quality audio segments in `data/clean/`
- **Noisy Segments**: Generates noisy versions using `noise_maker.py` in `data/noisy/`
- **Transcriptions**: Stores individual segment transcriptions in `data/transcriptions/`
- **Database Management**: SQLite database for metadata and easy querying
- **Batch Processing**: Process multiple videos simultaneously
- **Flexible Configuration**: Customizable segment count and noise levels

## 📁 Directory Structure

```
dataset/
├── clean/                    # Clean audio segments
│   ├── clean_1_000.wav
│   ├── clean_1_001.wav
│   └── ...
├── noisy/                    # Noisy audio segments
│   ├── noisy_1_000.wav
│   ├── noisy_1_001.wav
│   └── ...
├── transcriptions/           # Individual segment transcriptions
│   ├── transcript_1_segment_000.txt
│   ├── transcript_1_segment_001.txt
│   └── ...
├── metadata/                 # Additional metadata files
└── dataset.db               # SQLite database with all metadata
```

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Process a Single Video
```bash
python noisy_dataset_generator.py "https://www.youtube.com/watch?v=I3GWzXRectE"
```

### Process Multiple Videos
```bash
python noisy_dataset_generator.py \
    "https://www.youtube.com/watch?v=I3GWzXRectE" \
    "https://www.youtube.com/watch?v=VIDEO_ID_2" \
    --segments 100 \
    --interferences 5
```

### View Dataset Statistics
```bash
python noisy_dataset_generator.py --stats
```

### Query Segments
```bash
python noisy_dataset_generator.py --query clean
python noisy_dataset_generator.py --query noisy
```

## 📊 Database Schema

### Videos Table
- `id`: Primary key
- `youtube_id`: YouTube video ID
- `title`: Video title
- `url`: Original YouTube URL
- `duration_seconds`: Total video duration
- `processing_status`: pending/completed/failed
- `download_date`: When video was processed

### Segments Table
- `id`: Primary key
- `video_id`: Foreign key to videos table
- `segment_index`: Segment number within video
- `filename`: Audio file name
- `start_time`/`end_time`: Timestamps in original video
- `duration`: Segment duration in seconds
- `transcription`: Text content of segment
- `segment_type`: 'clean' or 'noisy'
- `noise_type`: Type of noise applied (for noisy segments)
- `interferences_per_60sec`: Noise intensity setting
- `file_path`: Full path to audio file
- `file_hash`: SHA256 hash for integrity

## 🔧 Advanced Usage

### Custom Dataset Configuration
```python
from dataset_generator import DatasetGenerator

# Initialize with custom settings
generator = DatasetGenerator(
    dataset_root="my_dataset",
    db_path="my_dataset.db"
)

# Process with specific parameters
result = generator.process_youtube_video(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    target_segments=150,  # More segments
    interferences_per_60sec=8  # Heavy interference
)
```

### Batch Processing
```python
urls = [
    "https://www.youtube.com/watch?v=VIDEO_1",
    "https://www.youtube.com/watch?v=VIDEO_2",
    "https://www.youtube.com/watch?v=VIDEO_3"
]

results = generator.batch_process_videos(
    urls=urls,
    target_segments=100,
    interferences_per_60sec=3
)
```

### Querying the Dataset
```python
# Get all clean segments
clean_segments = generator.query_segments(segment_type="clean")

# Get segments longer than 60 seconds
long_segments = generator.query_segments(min_duration=60.0)

# Get segments from specific video
video_segments = generator.query_segments(video_id=1)

# Get dataset statistics
stats = generator.get_dataset_stats()
```

## 📈 Dataset Statistics

The system provides comprehensive statistics:

```python
stats = generator.get_dataset_stats()
print(f"Videos: {stats['videos']['total']} total")
print(f"Segments: {stats['segments']['clean']} clean, {stats['segments']['noisy']} noisy")
print(f"Total duration: {stats['duration']['total_minutes']:.1f} minutes")
print(f"Average segment: {stats['duration']['average_seconds']:.1f} seconds")
```

## 🎵 Audio Processing Pipeline

1. **Download**: YouTube video downloaded using `yt-dlp`
2. **Transcribe**: Audio transcribed using OpenAI Whisper
3. **Segment**: Audio split at sentence boundaries (30-120 seconds)
4. **Clean**: Segments saved to `data/clean/`
5. **Noisy**: Noise applied using `noise_maker.py`, saved to `data/noisy/`
6. **Database**: All metadata stored in SQLite database

## 🔍 Use Cases

### Speech Enhancement Training
- Clean/noisy pairs for supervised learning
- Various noise types and intensities
- Consistent segment lengths and quality

### Speech Recognition
- High-quality transcriptions with timestamps
- Segmented audio for training/testing
- Multiple speakers and topics

### Audio Analysis
- Large dataset of speech segments
- Metadata for filtering and analysis
- Consistent format across all samples

## ⚙️ Configuration Options

### Command Line Arguments
- `--dataset-root`: Root directory for dataset (default: "dataset")
- `--db-path`: Database file path (default: "dataset.db")
- `--segments`: Target segments per video (default: 100)
- `--interferences`: Interferences per 60 seconds (default: 3)
- `--stats`: Show dataset statistics
- `--query`: Query segments with filters

### Processing Parameters
- **Target Segments**: Number of segments to create per video
- **Interferences per 60sec**: Noise intensity (1=light, 10=heavy)
- **Segment Duration**: 30-120 seconds (automatically determined)
- **Noise Types**: Static, clicks, pops, dropouts, hum, crackle, dramatic

## 🛠️ Troubleshooting

### Common Issues
1. **YouTube Download Fails**: Check internet connection and URL validity
2. **Transcription Slow**: Use smaller Whisper model (`--model tiny`)
3. **Memory Issues**: Process shorter videos or reduce segment count
4. **Database Locked**: Ensure no other processes are using the database

### Performance Tips
- Use SSD storage for faster I/O
- Process videos in smaller batches
- Monitor disk space (each video ~1-2GB)
- Use appropriate Whisper model size for your hardware

## 📝 Example Workflow

```bash
# 1. Create dataset from multiple videos
python noisy_dataset_generator.py \
    "https://www.youtube.com/watch?v=LECTURE_1" \
    "https://www.youtube.com/watch?v=LECTURE_2" \
    --segments 100 \
    --interferences 5

# 2. Check dataset statistics
python noisy_dataset_generator.py --stats

# 3. Query specific segments
python noisy_dataset_generator.py --query clean

# 4. Use in your ML pipeline
from dataset_generator import DatasetGenerator
generator = DatasetGenerator()
clean_segments = generator.query_segments(segment_type="clean")
noisy_segments = generator.query_segments(segment_type="noisy")
```

## 📄 License

This project is open source and available under the MIT License.
