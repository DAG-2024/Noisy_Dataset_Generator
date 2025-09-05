#!/usr/bin/env python3
"""
CS50 Lecture Audio Segmenter

This script downloads a CS50 lecture from YouTube and segments it into
logical sentence-based audio chunks of 30-120 seconds each.

Features:
- Downloads YouTube audio using yt-dlp
- Transcribes audio using OpenAI Whisper
- Identifies sentence boundaries for logical cuts
- Creates segments of 30-120 seconds
- Generates approximately 100 segments
- Saves segments with metadata
"""

import os
# Set environment variables to prevent semaphore leaks
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import re
import json
import argparse
import subprocess
import whisper
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import nltk
from nltk.tokenize import sent_tokenize
import yt_dlp
from typing import List, Tuple, Dict
import logging
import warnings
import multiprocessing



# Suppress multiprocessing warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LectureSegmenter:
    def __init__(self, output_dir="lecture_segments", model_size="base"):
        self.output_dir = output_dir
        self.model_size = model_size
        self.whisper_model = None
        self.audio_file = None
        self.transcription = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "transcripts"), exist_ok=True)
    
    def download_youtube_audio(self, url: str) -> str:
        """Download audio from YouTube URL"""
        logger.info(f"Downloading audio from: {url}")
        
        # Configure yt-dlp options
        ydl_opts = {
            # Get best available audio stream
            'format': 'bestaudio/best',
            # Use video ID to make the output name deterministic and easy to find
            'outtmpl': os.path.join(self.output_dir, '%(id)s.%(ext)s'),
            'noplaylist': True,
            # Ensure post-processing to WAV using ffmpeg when available
            'prefer_ffmpeg': True,
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }
            ],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get deterministic identifiers
                info = ydl.extract_info(url, download=False)
                video_id = info.get('id')
                title = info.get('title', 'lecture')
                # Safe human-readable title for optional renaming
                safe_title = re.sub(r'[^\w\s-]', '', title).strip()
                safe_title = re.sub(r'[-\s]+', '-', safe_title)

                # Perform download (with postprocessing to WAV)
                ydl.download([url])

                # Primary expected output (postprocessed)
                expected_wav = os.path.join(self.output_dir, f"{video_id}.wav")
                if os.path.exists(expected_wav):
                    # Optionally rename to readable title while keeping .wav
                    target_readable = os.path.join(self.output_dir, f"{safe_title}.wav")
                    try:
                        if not os.path.exists(target_readable):
                            os.rename(expected_wav, target_readable)
                            self.audio_file = target_readable
                        else:
                            self.audio_file = expected_wav
                    except Exception:
                        # Fallback: keep by id if rename fails
                        self.audio_file = expected_wav
                    logger.info(f"Downloaded audio: {self.audio_file}")
                    return self.audio_file

                # Fallback: If postprocessing did not produce WAV, try to locate the raw audio
                # and convert it to WAV ourselves.
                candidate_raw = None
                for file in os.listdir(self.output_dir):
                    if file.startswith(video_id + ".") and not file.endswith('.part'):
                        path = os.path.join(self.output_dir, file)
                        if os.path.isfile(path) and not file.endswith('.wav'):
                            candidate_raw = path
                            break

                if candidate_raw:
                    try:
                        logger.info(f"Converting to WAV via pydub: {candidate_raw}")
                        audio_seg = AudioSegment.from_file(candidate_raw)
                        target_readable = os.path.join(self.output_dir, f"{safe_title}.wav")
                        audio_seg.export(target_readable, format='wav')
                        self.audio_file = target_readable
                        logger.info(f"Downloaded audio: {self.audio_file}")
                        return self.audio_file
                    except Exception as conv_err:
                        logger.error(f"Conversion to WAV failed: {conv_err}")

                raise FileNotFoundError("Downloaded audio file not found after download and conversion attempts")
                
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise
    
    def transcribe_audio(self) -> Dict:
        """Transcribe audio using Whisper"""
        if not self.audio_file:
            raise ValueError("No audio file loaded")
        
        logger.info("Loading Whisper model (CPU)…")
        self.whisper_model = whisper.load_model(self.model_size, device="cpu")

        
        logger.info("Transcribing audio...")
        
        # Transcribe with settings to minimize semaphore leaks
        result = self.whisper_model.transcribe(
            self.audio_file,
            verbose=False,
            fp16=False,
            condition_on_previous_text=False,
            temperature=0.0,
            best_of=1,
            beam_size=1
        )
        
        self.transcription = result
        logger.info("Transcription completed")
        
        # Save full transcription
        transcript_file = os.path.join(self.output_dir, "transcripts", "full_transcript.txt")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        # Save detailed transcription with timestamps
        detailed_file = os.path.join(self.output_dir, "transcripts", "detailed_transcript.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcription saved to: {transcript_file}")
        return result
    
    def find_sentence_boundaries(self) -> List[Tuple[float, float, str]]:
        """Find sentence boundaries with timestamps"""
        if not self.transcription:
            raise ValueError("No transcription available")
        
        logger.info("Finding sentence boundaries...")
        
        # Get word-level timestamps
        words = self.transcription.get('segments', [])
        if not words:
            # Fallback to full text if no segments
            full_text = self.transcription['text']
            sentences = sent_tokenize(full_text)
            return [(0.0, len(sentences) * 2.0, sent) for sent in sentences]
        
        # Reconstruct text with timestamps
        text_with_timestamps = []
        current_sentence = ""
        sentence_start = None
        
        for segment in words:
            segment_text = segment.get('text', '').strip()
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            
            if not current_sentence:
                sentence_start = segment_start
            
            current_sentence += segment_text
            
            # Check if this segment ends a sentence
            if segment_text.endswith(('.', '!', '?')):
                # Clean up the sentence
                clean_sentence = current_sentence.strip()
                if clean_sentence:
                    text_with_timestamps.append((
                        sentence_start,
                        segment_end,
                        clean_sentence
                    ))
                current_sentence = ""
                sentence_start = None
        
        # Handle remaining text
        if current_sentence.strip():
            text_with_timestamps.append((
                sentence_start or 0,
                words[-1].get('end', 0) if words else 0,
                current_sentence.strip()
            ))
        
        logger.info(f"Found {len(text_with_timestamps)} sentences")
        return text_with_timestamps
    

    def _ffmpeg_cut(self, in_path: str, start: float, end: float, out_path: str):
        # cut with re-encode to PCM16 mono 16k (whisper-friendly)
        # NOTE: use re-encode instead of -c copy so boundaries are exact
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", in_path,
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            out_path
        ]
        subprocess.run(cmd, check=True)

    def create_segments(self, target_count: int = 100, min_duration: int = 30, max_duration: int = 120) -> List[Dict]:
        """Create audio segments based on sentence boundaries"""
        if not self.audio_file:
            raise ValueError("No audio file loaded")
        
        logger.info(f"Creating segments (target: {target_count}, duration: {min_duration}-{max_duration}s)")
        
        # Load audio
        total_duration = self.find_sentence_boundaries()[-1][1] if self.transcription else 0.0

        
        # Get sentence boundaries
        sentences = self.find_sentence_boundaries()
        
        # Calculate optimal segment duration
        target_duration = total_duration / target_count
        
        segments = []
        current_segment_sentences = []
        current_start_time = 0
        segment_count = 0
        
        for i, (start_time, end_time, sentence) in enumerate(sentences):
            current_segment_sentences.append((start_time, end_time, sentence))
            
            # Calculate current segment duration
            if current_segment_sentences:
                segment_duration = current_segment_sentences[-1][1] - current_segment_sentences[0][0]
            else:
                segment_duration = 0
            
            # Check if we should create a segment
            should_create_segment = False
            
            # Create segment if:
            # 1. We've reached target duration, OR
            # 2. We've exceeded max duration, OR
            # 3. This is the last sentence
            if (segment_duration >= target_duration and len(current_segment_sentences) > 1) or \
               segment_duration >= max_duration or \
               i == len(sentences) - 1:
                should_create_segment = True
            
            if should_create_segment and current_segment_sentences:
                # Ensure minimum duration
                if segment_duration < min_duration and i < len(sentences) - 1:
                    continue
                
                # Create segment
                segment_start = current_segment_sentences[0][0]
                segment_end = current_segment_sentences[-1][1]
                segment_text = " ".join([sent[2] for sent in current_segment_sentences])
                
                
                # Save audio segment
                segment_filename = f"segment_{segment_count:03d}.wav"
                segment_path = os.path.join(self.output_dir, "audio", segment_filename)
                self._ffmpeg_cut(self.audio_file, segment_start, segment_end, segment_path)
                
                # Save individual transcript for this segment
                transcript_filename = f"segment_{segment_count:03d}.txt"
                transcript_path = os.path.join(self.output_dir, "transcripts", transcript_filename)
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(segment_text)
                
                # Create segment metadata
                segment_info = {
                    "id": segment_count,
                    "filename": segment_filename,
                    "transcript_filename": transcript_filename,
                    "start_time": segment_start,
                    "end_time": segment_end,
                    "duration": segment_end - segment_start,
                    "text": segment_text,
                    "sentence_count": len(current_segment_sentences)
                }
                
                segments.append(segment_info)
                segment_count += 1
                
                logger.info(f"Created segment {segment_count}: {segment_duration:.1f}s - {segment_text[:50]}...")
                
                # Reset for next segment
                current_segment_sentences = []
        
        # Save segments metadata
        metadata_file = os.path.join(self.output_dir, "segments_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(segments)} segments")
        logger.info(f"Metadata saved to: {metadata_file}")
        
        return segments
    
    def process_lecture(self, url: str, target_segments: int = 100) -> List[Dict]:
        """Complete processing pipeline"""
        logger.info("Starting lecture processing...")
        
        # Download audio
        self.download_youtube_audio(url)
        
        # Transcribe
        self.transcribe_audio()
        
        # Create segments
        segments = self.create_segments(target_count=target_segments)
        
        # Generate summary
        self.generate_summary(segments)
        
        logger.info("Processing completed!")
        return segments
    
    def generate_summary(self, segments: List[Dict]):
        """Generate processing summary"""
        total_duration = sum(seg['duration'] for seg in segments)
        avg_duration = total_duration / len(segments) if segments else 0
        
        summary = {
            "total_segments": len(segments),
            "total_duration_minutes": total_duration / 60,
            "average_duration_seconds": avg_duration,
            "duration_range": {
                "min": min(seg['duration'] for seg in segments) if segments else 0,
                "max": max(seg['duration'] for seg in segments) if segments else 0
            },
            "output_directory": self.output_dir
        }
        
        summary_file = os.path.join(self.output_dir, "processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Created {len(segments)} segments, total duration: {total_duration/60:.1f} minutes")
    
    def cleanup(self):
        """Clean up resources to prevent semaphore leaks"""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        
        # Force garbage collection
        import gc
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Segment CS50 lecture into logical audio chunks')
    parser.add_argument('url', help='YouTube URL of the CS50 lecture')
    parser.add_argument('-o', '--output', default='lecture_segments', help='Output directory')
    parser.add_argument('-n', '--segments', type=int, default=100, help='Target number of segments')
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('--min-duration', type=int, default=30, help='Minimum segment duration (seconds)')
    parser.add_argument('--max-duration', type=int, default=120, help='Maximum segment duration (seconds)')
    
    args = parser.parse_args()
    
    # Validate URL
    if 'youtube.com' not in args.url and 'youtu.be' not in args.url:
        logger.error("Please provide a valid YouTube URL")
        return 1
    
    try:
        # Create segmenter
        segmenter = LectureSegmenter(
            output_dir=args.output,
            model_size=args.model
        )
        
        # Process lecture
        segments = segmenter.process_lecture(args.url, args.segments)
        
        print(f"\n✅ Successfully created {len(segments)} segments!")
        print(f"📁 Output directory: {args.output}")
        print(f"🎵 Audio segments: {args.output}/audio/")
        print(f"📝 Transcripts: {args.output}/transcripts/")
        print(f"📊 Metadata: {args.output}/segments_metadata.json")
        
        # Clean up resources
        segmenter.cleanup()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing lecture: {e}")
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # safest on macOS
    exit(main())


