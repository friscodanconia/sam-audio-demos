"""Audio processing utilities for SAM Audio"""
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

def load_audio(audio_path, target_sr=44100):
    """
    Load audio file and convert to format SAM Audio expects
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 44100)
    
    Returns:
        audio_waveform: numpy array of audio samples
        sample_rate: sample rate of audio
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=False)
    
    # librosa.load returns shape (channels, samples) for stereo
    # Convert to (samples, channels) if needed
    if len(audio.shape) == 2:
        audio = audio.T  # Transpose to (samples, channels)
    
    # If mono, convert to stereo (SAM Audio may expect stereo)
    if len(audio.shape) == 1:
        audio = np.array([audio, audio])  # Duplicate to stereo
        audio = audio.T  # Transpose to (samples, channels)
    
    print(f"✓ Loaded audio: shape={audio.shape}, sample_rate={sr} Hz")
    return audio, sr

def save_audio(audio_waveform, sample_rate, output_path):
    """
    Save audio waveform to file
    
    Args:
        audio_waveform: numpy array of audio samples
        sample_rate: sample rate
        output_path: path to save audio file
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to correct format for soundfile
    # soundfile expects (samples, channels) format
    if len(audio_waveform.shape) == 1:
        # Mono audio
        sf.write(output_path, audio_waveform, sample_rate)
    else:
        # Stereo or multi-channel
        sf.write(output_path, audio_waveform, sample_rate)
    
    print(f"✓ Saved audio to: {output_path}")

# Test function
if __name__ == "__main__":
    # Test loading the extracted audio
    audio_path = "data/input/test_audio.wav"
    print(f"Testing audio loading from: {audio_path}")
    
    audio, sr = load_audio(audio_path)
    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f} seconds")
    
    # Test saving
    test_output = "data/output/test_saved_audio.wav"
    save_audio(audio, sr, test_output)
    print(f"✓ Test complete! Audio loaded and saved successfully.")
