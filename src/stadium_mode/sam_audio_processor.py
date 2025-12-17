"""SAM Audio processing for commentary removal"""
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
import os
from dotenv import load_dotenv
from .audio_modes import AudioModeConfig

load_dotenv()

class SAMAudioProcessor:
    """
    Handles SAM Audio model for audio processing in different modes
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.loaded = False
    
    def load_model(self):
        """Load SAM Audio model (run this once)"""
        if self.loaded:
            print("Model already loaded")
            return
        
        print("Loading SAM Audio model...")
        token = os.getenv("HF_TOKEN")
        
        if not token:
            raise ValueError(
                "HF_TOKEN not found in environment. "
                "Set it in .env file or export HF_TOKEN=your_token"
            )
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                "facebook/sam-audio-large",
                token=token
            )
            self.model = AutoModel.from_pretrained(
                "facebook/sam-audio-large",
                token=token
            ).to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Hugging Face access is approved")
            print("2. Verify HF_TOKEN is set correctly")
            print("3. Check internet connection")
            print("4. Try: huggingface-cli login")
            raise
    
    def process_audio(self, audio_waveform, sample_rate, mode="stadium_atmosphere"):
        """
        Process audio based on specified mode
        
        Args:
            audio_waveform: numpy array of audio samples (shape: samples, channels)
            sample_rate: sample rate of audio
            mode: Processing mode (see AudioModeConfig.MODES)
        
        Returns:
            processed_audio: numpy array with processed audio
        """
        if not self.loaded:
            self.load_model()
        
        # Get mode configuration
        mode_config = AudioModeConfig.get_mode_config(mode)
        operation = mode_config.get("operation", "remove")
        
        # Handle "none" operation (original audio)
        if operation == "none":
            print(f"Mode '{mode_config['name']}': Keeping original audio")
            return audio_waveform
        
        print(f"Processing audio in '{mode_config['name']}' mode...")
        print(f"  Description: {mode_config['description']}")
        
        # Convert numpy to torch tensor
        if isinstance(audio_waveform, np.ndarray):
            if len(audio_waveform.shape) == 1:
                audio_waveform = np.column_stack([audio_waveform, audio_waveform])
            audio_tensor = torch.from_numpy(audio_waveform).float()
            if len(audio_tensor.shape) == 2:
                audio_tensor = audio_tensor.T  # (samples, channels) -> (channels, samples)
        else:
            audio_tensor = audio_waveform
        
        # Process based on operation type
        if operation == "remove":
            # Remove specified prompts (subtract them)
            prompts = mode_config.get("remove_prompts", [])
            processed_audio = self._remove_audio(audio_tensor, sample_rate, prompts)
            
        elif operation == "isolate":
            # Isolate specified prompts (keep only them)
            prompts = mode_config.get("isolate_prompts", [])
            processed_audio = self._isolate_audio(audio_tensor, sample_rate, prompts)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Convert back to numpy
        if len(processed_audio.shape) == 2:
            processed_audio = processed_audio.T  # (channels, samples) -> (samples, channels)
        
        processed_audio = processed_audio.cpu().numpy()
        
        print(f"✓ Processing complete for '{mode_config['name']}' mode")
        return processed_audio
    
    def _remove_audio(self, audio_tensor, sample_rate, prompts):
        """Remove specified audio components"""
        # Process with SAM Audio to get mask
        inputs = self.processor(
            audio=audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor,
            sampling_rate=sample_rate,
            text_prompts=prompts
        )
        
        with torch.no_grad():
            outputs = self.model(inputs)
            mask = outputs['masks'][0]
        
        # Ensure shapes match
        if audio_tensor.shape != mask.shape:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0),
                size=audio_tensor.shape[-1],
                mode='linear',
                align_corners=False
            ).squeeze(0)
        
        # Subtract mask to remove specified audio
        processed = audio_tensor - mask
        
        # Normalize
        max_val = torch.max(torch.abs(processed))
        if max_val > 0:
            processed = processed / max_val
        
        return processed
    
    def _isolate_audio(self, audio_tensor, sample_rate, prompts):
        """Isolate specified audio components (keep only them)"""
        # Process with SAM Audio to get mask
        inputs = self.processor(
            audio=audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor,
            sampling_rate=sample_rate,
            text_prompts=prompts
        )
        
        with torch.no_grad():
            outputs = self.model(inputs)
            mask = outputs['masks'][0]
        
        # Ensure shapes match
        if audio_tensor.shape != mask.shape:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0),
                size=audio_tensor.shape[-1],
                mode='linear',
                align_corners=False
            ).squeeze(0)
        
        # Use mask to isolate specified audio
        # Multiply original audio by mask to keep only masked portions
        processed = audio_tensor * mask
        
        # Normalize
        max_val = torch.max(torch.abs(processed))
        if max_val > 0:
            processed = processed / max_val
        
        return processed

# Test function
if __name__ == "__main__":
    print("SAM Audio Processor Test")
    print("=" * 50)
    
    processor = SAMAudioProcessor()
    
    try:
        processor.load_model()
        print("✓ SAM Audio processor ready!")
        
        # List available modes
        print("\nAvailable modes:")
        for mode in AudioModeConfig.list_modes():
            print(f"  - {mode}: {AudioModeConfig.get_mode_info(mode)}")
    except Exception as e:
        print(f"⚠ Model loading failed: {e}")
        print("This is expected if Hugging Face access is not yet approved.")
