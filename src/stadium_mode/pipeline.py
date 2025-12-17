"""Main pipeline for Stadium Mode"""
from .video_utils import download_video, extract_audio, remux_video
from .audio_utils import load_audio, save_audio
from .sam_audio_processor import SAMAudioProcessor
from .audio_modes import AudioModeConfig
from pathlib import Path
import os

class StadiumModePipeline:
    """
    Complete pipeline for removing commentary from sports videos
    """
    
    def __init__(self, output_dir="data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_processor = None
    
    def process_video(self, video_url, mode="stadium_atmosphere", use_sam_audio=True):
        """
        Main processing function with mode support
        
        Args:
            video_url: URL of video to process
            mode: Audio processing mode (see AudioModeConfig.MODES)
                  Options: 'stadium_atmosphere', 'commentary_only', 'crowd_only', 
                          'game_sounds_only', 'referee_only', 'pure_game', 'original'
            use_sam_audio: Whether to use SAM Audio (False for testing pipeline)
        
        Returns:
            Path to processed video
        """
        print("=" * 60)
        print("STADIUM MODE PIPELINE")
        print("=" * 60)
        
        # Get mode info
        try:
            mode_config = AudioModeConfig.get_mode_config(mode)
            print(f"Mode: {mode_config['name']}")
            print(f"Description: {mode_config['description']}")
        except ValueError as e:
            print(f"⚠ Warning: {e}")
            print("Using default mode: stadium_atmosphere")
            mode = "stadium_atmosphere"
            mode_config = AudioModeConfig.get_mode_config(mode)
        
        # Step 1: Download video
        print("\n[1/5] Downloading video...")
        video_path = download_video(video_url, output_dir="data/input")
        video_filename = Path(video_path).stem
        
        # Step 2: Extract audio
        print("\n[2/5] Extracting audio...")
        audio_path = self.output_dir / f"{video_filename}_original_audio.wav"
        extract_audio(str(video_path), str(audio_path))
        
        # Step 3: Load audio
        print("\n[3/5] Loading audio...")
        audio_waveform, sample_rate = load_audio(str(audio_path))
        
        # Step 4: Process audio (based on mode)
        print("\n[4/5] Processing audio...")
        processed_audio_path = self.output_dir / f"{video_filename}_{mode}_audio.wav"
        
        # Check if mode is "original" (no processing)
        if mode == "original":
            print("Mode is 'original' - skipping audio processing")
            processed_audio = audio_waveform
        elif use_sam_audio:
            try:
                # Initialize processor if not already done
                if self.audio_processor is None:
                    self.audio_processor = SAMAudioProcessor()
                    self.audio_processor.load_model()
                
                # Process audio with specified mode
                print(f"Processing audio in '{mode_config['name']}' mode...")
                processed_audio = self.audio_processor.process_audio(
                    audio_waveform, sample_rate, mode=mode
                )
                
            except Exception as e:
                print(f"⚠ SAM Audio processing failed: {e}")
                print("Falling back to original audio (no processing)")
                # Fallback: use original audio
                processed_audio = audio_waveform
        else:
            # Skip SAM Audio processing (for testing pipeline)
            print("⚠ Skipping SAM Audio processing (testing mode)")
            processed_audio = audio_waveform
        
        # Save processed audio
        save_audio(processed_audio, sample_rate, str(processed_audio_path))
        
        # Step 5: Re-mux video
        print("\n[5/5] Re-muxing video...")
        output_video_path = self.output_dir / f"{video_filename}_{mode}.mp4"
        remux_video(str(video_path), str(processed_audio_path), str(output_video_path))
        
        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETE!")
        print(f"Mode: {mode_config['name']}")
        print(f"Output video: {output_video_path}")
        print("=" * 60)
        
        return str(output_video_path)
    
    def list_modes(self):
        """List all available audio modes"""
        print("Available audio modes:")
        print("-" * 60)
        for mode in AudioModeConfig.list_modes():
            info = AudioModeConfig.get_mode_info(mode)
            print(f"  • {mode}: {info}")
        print("-" * 60)

# Test the pipeline
if __name__ == "__main__":
    pipeline = StadiumModePipeline()
    
    # Show available modes
    pipeline.list_modes()
    
    print("\n" + "=" * 60)
    print("Testing pipeline (without SAM Audio processing)...")
    print("=" * 60)
    
    # Test with a sample video URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Test with stadium_atmosphere mode (but without SAM Audio)
    result = pipeline.process_video(test_url, mode="stadium_atmosphere", use_sam_audio=False)
    print(f"\n✓ Pipeline test complete!")
    print(f"Output: {result}")
