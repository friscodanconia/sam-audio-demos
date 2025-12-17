"""Test script to verify SAM Audio model can be loaded"""
from transformers import AutoProcessor, AutoModel
import torch

print("Loading SAM Audio model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    print("Downloading processor...")
    processor = AutoProcessor.from_pretrained("facebook/sam-audio-large")
    print("✓ Processor loaded!")
    
    print("Downloading model (this may take a while, ~2-5GB)...")
    model = AutoModel.from_pretrained("facebook/sam-audio-large")
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
