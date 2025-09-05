#!/usr/bin/env python3
"""
Example usage of the Dataset Generator

This script demonstrates how to use the dataset_generator.py to create
a comprehensive speech processing dataset from YouTube videos.
"""

from dataset_generator import DatasetGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_single_video():
    """Example: Process a single YouTube video"""
    print("🎥 Example 1: Processing a single YouTube video")
    print("=" * 60)
    
    # Initialize dataset generator
    generator = DatasetGenerator(dataset_root="example_dataset", db_path="example_dataset.db")
    
    # Process a single video
    url = "https://www.youtube.com/watch?v=I3GWzXRectE"  # Oxford Calculus lecture
    
    try:
        result = generator.process_youtube_video(
            url=url,
            target_segments=50,  # Create 50 segments
            interferences_per_60sec=5  # 5 interferences per minute
        )
        
        print(f"✅ Successfully processed video!")
        print(f"📊 Results: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_batch_processing():
    """Example: Process multiple YouTube videos in batch"""
    print("\n🎥 Example 2: Batch processing multiple videos")
    print("=" * 60)
    
    # Initialize dataset generator
    generator = DatasetGenerator(dataset_root="example_dataset", db_path="example_dataset.db")
    
    # List of YouTube URLs to process
    urls = [
        "https://www.youtube.com/watch?v=I3GWzXRectE",  # Oxford Calculus
        # Add more URLs here as needed
    ]
    
    try:
        results = generator.batch_process_videos(
            urls=urls,
            target_segments=30,  # 30 segments per video
            interferences_per_60sec=3  # 3 interferences per minute
        )
        
        print(f"✅ Batch processing completed!")
        print(f"📊 Results: {results}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_dataset_queries():
    """Example: Query and analyze the dataset"""
    print("\n🔍 Example 3: Querying the dataset")
    print("=" * 60)
    
    # Initialize dataset generator
    generator = DatasetGenerator(dataset_root="example_dataset", db_path="example_dataset.db")
    
    # Get dataset statistics
    stats = generator.get_dataset_stats()
    print("📊 Dataset Statistics:")
    print(f"  Videos: {stats['videos']['total']} total, {stats['videos']['completed']} completed")
    print(f"  Segments: {stats['segments']['clean']} clean, {stats['segments']['noisy']} noisy")
    print(f"  Total duration: {stats['duration']['total_minutes']:.1f} minutes")
    print(f"  Average segment: {stats['duration']['average_seconds']:.1f} seconds")
    
    # Query clean segments
    clean_segments = generator.query_segments(segment_type="clean")
    print(f"\n🎵 Clean segments: {len(clean_segments)}")
    
    # Query noisy segments
    noisy_segments = generator.query_segments(segment_type="noisy")
    print(f"🔊 Noisy segments: {len(noisy_segments)}")
    
    # Query segments by duration
    long_segments = generator.query_segments(min_duration=60.0)
    print(f"⏱️  Long segments (>60s): {len(long_segments)}")
    
    # Show some examples
    if clean_segments:
        print(f"\n📝 Example clean segments:")
        for i, seg in enumerate(clean_segments[:3]):
            print(f"  {i+1}. {seg['filename']} - {seg['duration']:.1f}s")
            print(f"     Transcription: {seg['transcription'][:100]}...")


def example_custom_dataset():
    """Example: Create a custom dataset with specific parameters"""
    print("\n⚙️  Example 4: Custom dataset configuration")
    print("=" * 60)
    
    # Initialize with custom settings
    generator = DatasetGenerator(
        dataset_root="custom_dataset",
        db_path="custom_dataset.db"
    )
    
    # Process with custom parameters
    urls = [
        "https://www.youtube.com/watch?v=I3GWzXRectE",
    ]
    
    try:
        results = generator.batch_process_videos(
            urls=urls,
            target_segments=100,  # More segments
            interferences_per_60sec=10  # Heavy interference
        )
        
        print(f"✅ Custom dataset created!")
        print(f"📊 Results: {results}")
        
        # Show statistics
        stats = generator.get_dataset_stats()
        print(f"\n📈 Custom dataset stats:")
        print(f"  Total segments: {stats['segments']['total']}")
        print(f"  Average duration: {stats['duration']['average_seconds']:.1f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples"""
    print("🚀 Dataset Generator Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_single_video()
        example_batch_processing()
        example_dataset_queries()
        example_custom_dataset()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("\n📚 Next steps:")
        print("1. Use the generated dataset for speech processing experiments")
        print("2. Query segments by type, duration, or other criteria")
        print("3. Add more YouTube videos to expand the dataset")
        print("4. Use the clean/noisy pairs for training speech enhancement models")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
