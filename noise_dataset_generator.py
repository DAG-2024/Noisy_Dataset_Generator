#!/usr/bin/env python3
"""
Dataset Generator for Speech Processing

This script creates a comprehensive dataset by:
1. Downloading and segmenting YouTube lectures into clean audio segments
2. Generating noisy versions of the clean segments
3. Storing all data in a structured database for easy access

Features:
- Processes YouTube videos using lecture_segmenter.py
- Creates clean segments in data/clean/
- Generates noisy segments using noise_maker.py in data/noisy/
- Stores transcriptions in data/transcriptions/
- Maintains a SQLite database for metadata and easy querying
- Supports batch processing of multiple videos
"""

import os
import sqlite3
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

# Import our custom modules
from lecture_segmenter import LectureSegmenter
from noise_maker import AudioNoiseMaker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, dataset_root: str = "dataset", db_path: str = "dataset.db"):
        self.dataset_root = Path(dataset_root)
        self.db_path = db_path
        
        # Create directory structure
        self.clean_dir = self.dataset_root / "clean"
        self.noisy_dir = self.dataset_root / "noisy"
        self.transcriptions_dir = self.dataset_root / "transcriptions"
        self.metadata_dir = self.dataset_root / "metadata"
        
        # Create directories
        for dir_path in [self.clean_dir, self.noisy_dir, self.transcriptions_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Initialize processors
        self.segmenter = None
        self.noise_maker = AudioNoiseMaker()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Videos table - stores information about source videos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                youtube_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                duration_seconds REAL,
                download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_status TEXT DEFAULT 'pending',
                metadata_json TEXT
            )
        ''')
        
        # Segments table - stores information about audio segments
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                segment_index INTEGER,
                filename TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                transcription TEXT,
                sentence_count INTEGER,
                segment_type TEXT DEFAULT 'clean',
                noise_type TEXT,
                noise_intensity REAL,
                interferences_per_60sec INTEGER,
                file_path TEXT NOT NULL,
                file_hash TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_videos_youtube_id ON videos(youtube_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_type ON segments(segment_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_duration ON segments(duration)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def extract_youtube_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        import re
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract YouTube ID from URL: {url}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def process_youtube_video(self, url: str, target_segments: int = 100, 
                            interferences_per_60sec: int = 3) -> Dict:
        """Process a single YouTube video and add to dataset"""
        logger.info(f"Processing YouTube video: {url}")
        
        try:
            # Extract YouTube ID
            youtube_id = self.extract_youtube_id(url)
            
            # Check if video already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM videos WHERE youtube_id = ?', (youtube_id,))
            existing_video = cursor.fetchone()
            
            if existing_video:
                logger.info(f"Video {youtube_id} already exists in database")
                conn.close()
                return {"status": "exists", "video_id": existing_video[0]}
            
            # Create temporary directory for processing
            temp_dir = self.dataset_root / f"temp_{youtube_id}"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Step 1: Download and segment using lecture_segmenter
                logger.info("Step 1: Downloading and segmenting video...")
                segmenter = LectureSegmenter(output_dir=str(temp_dir))
                segments = segmenter.process_lecture(url, target_segments)
                
                # Get video metadata
                video_title = segmenter.transcription.get('text', '')[:100] if segmenter.transcription else 'Unknown'
                total_duration = segments[-1]['end_time'] if segments else 0
                
                # Insert video record
                cursor.execute('''
                    INSERT INTO videos (youtube_id, title, url, duration_seconds, processing_status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (youtube_id, video_title, url, total_duration, 'processing'))
                
                video_id = cursor.lastrowid
                conn.commit()
                
                # Step 2: Copy clean segments to dataset
                logger.info("Step 2: Copying clean segments...")
                clean_segments = self._copy_clean_segments(segments, temp_dir, video_id)
                
                # Step 3: Generate noisy segments
                logger.info("Step 3: Generating noisy segments...")
                noisy_segments = self._generate_noisy_segments(clean_segments, interferences_per_60sec, video_id)
                
                # Step 4: Copy transcriptions
                logger.info("Step 4: Copying transcriptions...")
                self._copy_transcriptions(temp_dir, video_id)
                
                # Update video status
                cursor.execute('UPDATE videos SET processing_status = ? WHERE id = ?', ('completed', video_id))
                conn.commit()
                
                logger.info(f"Successfully processed video {youtube_id}: {len(clean_segments)} clean + {len(noisy_segments)} noisy segments")
                
                return {
                    "status": "success",
                    "video_id": video_id,
                    "clean_segments": len(clean_segments),
                    "noisy_segments": len(noisy_segments),
                    "total_duration": total_duration
                }
                
            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                conn.close()
                
        except Exception as e:
            logger.error(f"Error processing video {url}: {e}")
            # Update video status to failed
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE videos SET processing_status = ? WHERE youtube_id = ?', ('failed', youtube_id))
                conn.commit()
                conn.close()
            except:
                pass
            raise
    
    def _copy_clean_segments(self, segments: List[Dict], temp_dir: Path, video_id: int) -> List[Dict]:
        """Copy clean segments to dataset and insert into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        clean_segments = []
        for segment in segments:
            # Copy audio file
            src_path = temp_dir / "audio" / segment['filename']
            dst_filename = f"clean_{video_id}_{segment['id']:03d}.wav"
            dst_path = self.clean_dir / dst_filename
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                
                # Calculate file hash
                file_hash = self.calculate_file_hash(dst_path)
                
                # Insert into database
                cursor.execute('''
                    INSERT INTO segments (
                        video_id, segment_index, filename, start_time, end_time, duration,
                        transcription, sentence_count, segment_type, file_path, file_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_id, segment['id'], dst_filename,
                    segment['start_time'], segment['end_time'], segment['duration'],
                    segment['text'], segment['sentence_count'], 'clean',
                    str(dst_path), file_hash
                ))
                
                clean_segments.append({
                    'id': segment['id'],
                    'filename': dst_filename,
                    'path': dst_path,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['duration'],
                    'transcription': segment['text']
                })
        
        conn.commit()
        conn.close()
        return clean_segments
    
    def _generate_noisy_segments(self, clean_segments: List[Dict], 
                               interferences_per_60sec: int, video_id: int) -> List[Dict]:
        """Generate noisy versions of clean segments"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        noisy_segments = []
        for clean_segment in clean_segments:
            # Generate noisy version
            noisy_filename = f"noisy_{video_id}_{clean_segment['id']:03d}.wav"
            noisy_path = self.noisy_dir / noisy_filename
            
            try:
                # Process with noise maker
                self.noise_maker.process_audio(
                    str(clean_segment['path']),
                    str(noisy_path),
                    interferences_per_60sec
                )
                
                # Calculate file hash
                file_hash = self.calculate_file_hash(noisy_path)
                
                # Insert into database
                cursor.execute('''
                    INSERT INTO segments (
                        video_id, segment_index, filename, start_time, end_time, duration,
                        transcription, sentence_count, segment_type, interferences_per_60sec,
                        file_path, file_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_id, clean_segment['id'], noisy_filename,
                    clean_segment['start_time'], clean_segment['end_time'], clean_segment['duration'],
                    clean_segment['transcription'], 0, 'noisy', interferences_per_60sec,
                    str(noisy_path), file_hash
                ))
                
                noisy_segments.append({
                    'id': clean_segment['id'],
                    'filename': noisy_filename,
                    'path': noisy_path,
                    'clean_segment_id': clean_segment['id']
                })
                
            except Exception as e:
                logger.error(f"Error generating noisy segment {clean_segment['id']}: {e}")
        
        conn.commit()
        conn.close()
        return noisy_segments
    
    def _copy_transcriptions(self, temp_dir: Path, video_id: int):
        """Copy transcription files to dataset"""
        transcripts_src = temp_dir / "transcripts"
        if transcripts_src.exists():
            for transcript_file in transcripts_src.glob("*.txt"):
                dst_filename = f"transcript_{video_id}_{transcript_file.stem}.txt"
                dst_path = self.transcriptions_dir / dst_filename
                shutil.copy2(transcript_file, dst_path)
    
    def batch_process_videos(self, urls: List[str], target_segments: int = 100,
                           interferences_per_60sec: int = 3) -> Dict:
        """Process multiple YouTube videos in batch"""
        logger.info(f"Starting batch processing of {len(urls)} videos")
        
        results = {
            "total": len(urls),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing video {i}/{len(urls)}: {url}")
            try:
                result = self.process_youtube_video(url, target_segments, interferences_per_60sec)
                results["details"].append({"url": url, "result": result})
                
                if result["status"] == "success":
                    results["successful"] += 1
                elif result["status"] == "exists":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                results["failed"] += 1
                results["details"].append({"url": url, "error": str(e)})
        
        logger.info(f"Batch processing completed: {results['successful']} successful, {results['failed']} failed, {results['skipped']} skipped")
        return results
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Video statistics
        cursor.execute('SELECT COUNT(*) FROM videos')
        total_videos = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM videos WHERE processing_status = "completed"')
        completed_videos = cursor.fetchone()[0]
        
        # Segment statistics
        cursor.execute('SELECT COUNT(*) FROM segments WHERE segment_type = "clean"')
        clean_segments = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM segments WHERE segment_type = "noisy"')
        noisy_segments = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(duration) FROM segments WHERE segment_type = "clean"')
        avg_duration = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT SUM(duration) FROM segments WHERE segment_type = "clean"')
        total_duration = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "videos": {
                "total": total_videos,
                "completed": completed_videos,
                "failed": total_videos - completed_videos
            },
            "segments": {
                "clean": clean_segments,
                "noisy": noisy_segments,
                "total": clean_segments + noisy_segments
            },
            "duration": {
                "average_seconds": avg_duration,
                "total_minutes": total_duration / 60
            }
        }
    
    def query_segments(self, segment_type: str = None, min_duration: float = None,
                      max_duration: float = None, video_id: int = None) -> List[Dict]:
        """Query segments from database with filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM segments WHERE 1=1"
        params = []
        
        if segment_type:
            query += " AND segment_type = ?"
            params.append(segment_type)
        
        if min_duration is not None:
            query += " AND duration >= ?"
            params.append(min_duration)
        
        if max_duration is not None:
            query += " AND duration <= ?"
            params.append(max_duration)
        
        if video_id is not None:
            query += " AND video_id = ?"
            params.append(video_id)
        
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results


def main():
    parser = argparse.ArgumentParser(description='Noisy Dataset Generator - Create speech processing datasets from YouTube videos')
    parser.add_argument('urls', nargs='+', help='YouTube URLs to process')
    parser.add_argument('--dataset-root', default='dataset', help='Root directory for dataset')
    parser.add_argument('--db-path', default='dataset.db', help='Database file path')
    parser.add_argument('--segments', type=int, default=100, help='Target number of segments per video')
    parser.add_argument('--interferences', type=int, default=3, help='Interferences per 60 seconds')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    parser.add_argument('--query', help='Query segments (e.g., "clean", "noisy", "duration>60")')
    
    args = parser.parse_args()
    
    # Initialize dataset generator
    generator = DatasetGenerator(args.dataset_root, args.db_path)
    
    if args.stats:
        # Show statistics
        stats = generator.get_dataset_stats()
        print("\n📊 Dataset Statistics:")
        print(f"Videos: {stats['videos']['total']} total, {stats['videos']['completed']} completed")
        print(f"Segments: {stats['segments']['clean']} clean, {stats['segments']['noisy']} noisy")
        print(f"Total duration: {stats['duration']['total_minutes']:.1f} minutes")
        print(f"Average segment: {stats['duration']['average_seconds']:.1f} seconds")
        return 0
    
    if args.query:
        # Query segments
        if args.query == "clean":
            segments = generator.query_segments(segment_type="clean")
        elif args.query == "noisy":
            segments = generator.query_segments(segment_type="noisy")
        else:
            segments = generator.query_segments()
        
        print(f"\nFound {len(segments)} segments:")
        for seg in segments[:10]:  # Show first 10
            print(f"  {seg['filename']} - {seg['duration']:.1f}s - {seg['segment_type']}")
        if len(segments) > 10:
            print(f"  ... and {len(segments) - 10} more")
        return 0
    
    # Process videos
    if args.urls:
        results = generator.batch_process_videos(
            args.urls, 
            args.segments, 
            args.interferences
        )
        
        print(f"\n✅ Batch processing completed!")
        print(f"📊 Results: {results['successful']} successful, {results['failed']} failed, {results['skipped']} skipped")
        
        return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())
