"""Video download and processing utilities"""
import yt_dlp
import subprocess
import os
from pathlib import Path

def download_video(url, output_dir="data/input"):
    """
    Download video from YouTube/Twitch URL
    
    Args:
        url: Video URL (YouTube, Twitch, etc.)
        output_dir: Directory to save video
    
    Returns:
        Path to downloaded video file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'quiet': False,  # Set to True to suppress output
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract info first to get filename
        info = ydl.extract_info(url, download=False)
        filename = ydl.prepare_filename(info)
        
        # Download the video
        ydl.download([url])
        
        return filename

def extract_audio(video_path, audio_output_path=None):
    """
    Extract audio from video file using FFmpeg
    
    Args:
        video_path: Path to input video
        audio_output_path: Path for output audio (default: same name with .wav)
    
    Returns:
        Path to extracted audio file
    """
    if audio_output_path is None:
        audio_output_path = str(Path(video_path).with_suffix('.wav'))
    
    # Ensure output directory exists
    Path(audio_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '44100',  # Sample rate
        '-ac', '2',  # Stereo
        '-y',  # Overwrite output file
        audio_output_path
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"✓ Audio extracted to: {audio_output_path}")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        print(f"✗ Error extracting audio: {e}")
        raise

def remux_video(video_path, audio_path, output_path):
    """
    Combine processed audio with original video
    
    Args:
        video_path: Path to original video (video track)
        audio_path: Path to processed audio (audio track)
        output_path: Path for final output video
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    command = [
        'ffmpeg',
        '-i', video_path,  # Original video
        '-i', audio_path,  # Processed audio
        '-c:v', 'copy',  # Copy video codec (no re-encoding = faster)
        '-c:a', 'aac',  # Audio codec
        '-map', '0:v:0',  # Use video from first input
        '-map', '1:a:0',  # Use audio from second input
        '-shortest',  # Match shortest stream length
        '-y',  # Overwrite output
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"✓ Video re-muxed to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"✗ Error re-muxing video: {e}")
        raise

# Test function
if __name__ == "__main__":
    # Test with a short video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with test video
    print("Testing video download...")
    video_path = download_video(test_url)
    print(f"✓ Downloaded to: {video_path}")
