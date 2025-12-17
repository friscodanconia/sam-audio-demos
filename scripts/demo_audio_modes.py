"""
Demo script to showcase different audio modes for sports broadcasts
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stadium_mode.pipeline import StadiumModePipeline
from src.stadium_mode.audio_modes import AudioModeConfig
from src.stadium_mode.video_utils import download_video, extract_audio, remux_video
from src.stadium_mode.audio_utils import load_audio, save_audio
from src.stadium_mode.sam_audio_processor import SAMAudioProcessor

def create_demo(video_url, modes=None, use_sam_audio=True):
    """
    Create demo videos showing different audio modes
    
    Args:
        video_url: URL of video to process
        modes: List of modes to process (None = all modes)
        use_sam_audio: Whether to use SAM Audio processing
    """
    print("=" * 70)
    print("AUDIO MODES DEMO")
    print("=" * 70)
    
    # Get modes to process
    if modes is None:
        modes = AudioModeConfig.list_modes()
        # Remove "original" from list (we'll handle it separately)
        if "original" in modes:
            modes.remove("original")
    
    print(f"\nProcessing video: {video_url}")
    print(f"Modes to process: {', '.join(modes)}")
    print(f"SAM Audio enabled: {use_sam_audio}")
    
    # Step 1: Download video once
    print("\n" + "=" * 70)
    print("STEP 1: Downloading video...")
    print("=" * 70)
    video_path = download_video(video_url, output_dir="data/input")
    video_filename = Path(video_path).stem
    
    # Step 2: Extract audio once
    print("\n" + "=" * 70)
    print("STEP 2: Extracting audio...")
    print("=" * 70)
    audio_path = f"data/output/{video_filename}_original_audio.wav"
    extract_audio(str(video_path), audio_path)
    
    # Step 3: Load audio once
    print("\n" + "=" * 70)
    print("STEP 3: Loading audio...")
    print("=" * 70)
    audio_waveform, sample_rate = load_audio(audio_path)
    
    # Step 4: Initialize SAM Audio processor (once)
    processor = None
    if use_sam_audio:
        print("\n" + "=" * 70)
        print("STEP 4: Initializing SAM Audio processor...")
        print("=" * 70)
        try:
            processor = SAMAudioProcessor()
            processor.load_model()
        except Exception as e:
            print(f"⚠ SAM Audio initialization failed: {e}")
            print("Falling back to original audio for all modes")
            use_sam_audio = False
    
    # Step 5: Process each mode
    print("\n" + "=" * 70)
    print("STEP 5: Processing audio in different modes...")
    print("=" * 70)
    
    results = {}
    
    # Process original first (no processing needed)
    print(f"\n[Original] Processing original audio...")
    original_output = f"data/output/{video_filename}_original.mp4"
    remux_video(str(video_path), audio_path, original_output)
    results["original"] = {
        "mode": "original",
        "video": original_output,
        "audio": audio_path
    }
    print(f"✓ Original video saved: {original_output}")
    
    # Process each mode
    for i, mode in enumerate(modes, 1):
        print(f"\n[{i}/{len(modes)}] Processing mode: {mode}")
        print("-" * 70)
        
        mode_config = AudioModeConfig.get_mode_config(mode)
        print(f"Mode: {mode_config['name']}")
        print(f"Description: {mode_config['description']}")
        
        # Process audio
        if use_sam_audio and processor:
            try:
                processed_audio = processor.process_audio(
                    audio_waveform.copy(),
                    sample_rate,
                    mode=mode
                )
            except Exception as e:
                print(f"⚠ Processing failed: {e}")
                print("Using original audio as fallback")
                processed_audio = audio_waveform
        else:
            print("⚠ SAM Audio not available, using original audio")
            processed_audio = audio_waveform
        
        # Save processed audio
        processed_audio_path = f"data/output/{video_filename}_{mode}_audio.wav"
        save_audio(processed_audio, sample_rate, processed_audio_path)
        
        # Re-mux video
        output_video = f"data/output/{video_filename}_{mode}.mp4"
        remux_video(str(video_path), processed_audio_path, output_video)
        
        results[mode] = {
            "mode": mode,
            "name": mode_config['name'],
            "video": output_video,
            "audio": processed_audio_path
        }
        
        print(f"✓ {mode_config['name']} video saved: {output_video}")
    
    # Step 6: Create summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    
    print("\nGenerated videos:")
    print("-" * 70)
    for mode_key, result in results.items():
        print(f"\n{result.get('name', mode_key).upper()}")
        print(f"  Video: {result['video']}")
        print(f"  Audio: {result['audio']}")
    
    print("\n" + "=" * 70)
    print("All demo videos are ready!")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo different audio modes")
    parser.add_argument("video_url", help="URL of video to process")
    parser.add_argument(
        "--modes",
        nargs="+",
        help="Specific modes to process (default: all modes)",
        choices=AudioModeConfig.list_modes()
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Skip SAM Audio processing (test pipeline only)"
    )
    
    args = parser.parse_args()
    
    create_demo(
        args.video_url,
        modes=args.modes,
        use_sam_audio=not args.no_sam
    )
