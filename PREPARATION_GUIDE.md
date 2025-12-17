# Preparation Guide: SAM Audio Demo Projects

**Version:** 1.0  
**Date:** December 2024  
**Purpose:** Complete setup and preparation guide for solo developer building SAM Audio demos

---

## Table of Contents

1. [Git Repository Setup](#git-repository-setup)
2. [GPU Requirements & Cloud Resources](#gpu-requirements--cloud-resources)
3. [Pre-Development Checklist](#pre-development-checklist)
4. [Technical Prerequisites Deep Dive](#technical-prerequisites-deep-dive)
5. [MVP vs Full Feature Approach](#mvp-vs-full-feature-approach)
6. [Cost Considerations](#cost-considerations)
7. [Quick Start Checklist](#quick-start-checklist)

---

## Git Repository Setup

### Why Initialize Git?

Even as a solo developer, Git provides:
- **Version Control:** Track changes and revert if needed
- **Backup:** Code history and remote backup
- **Future Collaboration:** Easy to share or collaborate later
- **Professional Practice:** Industry standard workflow
- **Experiment Safely:** Try features without fear of breaking code

### Initial Setup Commands

```bash
# Navigate to project directory
cd "/Users/soumyosinha/Documents/Project SAM"

# Initialize git repository
git init

# Create initial commit
git add .
git commit -m "Initial commit: PRDs and project structure"

# (Optional) Create GitHub repository and connect
# git remote add origin https://github.com/yourusername/sam-audio-demos.git
# git push -u origin main
```

### Recommended .gitignore

Create a `.gitignore` file in your project root:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv
pip-log.txt
pip-delete-this-directory.txt

# Jupyter Notebooks
.ipynb_checkpoints
*.ipynb

# ML Models & Data
*.pth
*.pt
*.h5
*.ckpt
models/
checkpoints/
data/
*.wav
*.mp4
*.mp3
*.avi
*.mkv

# API Keys & Secrets
.env
.env.local
*.key
secrets/
config.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Cloud GPU
*.ipynb
colab/
kaggle/

# Logs
*.log
logs/
```

### Branching Strategy for Solo Development

**Simple Strategy:**
- `main` - Stable, working code
- `develop` - Active development
- `feature/*` - New features (optional)

**Workflow:**
```bash
# Create feature branch
git checkout -b feature/stadium-mode

# Work on feature
# ... make changes ...

# Commit frequently
git add .
git commit -m "Add audio separation logic"

# Merge back to main when done
git checkout main
git merge feature/stadium-mode
```

---

## GPU Requirements & Cloud Resources

### GPU Requirements Analysis

#### SAM Audio Model Specifications

**Model:** `facebook/sam-audio-large`

**Estimated Requirements:**
- **VRAM:** 8-16GB GPU memory recommended
- **RAM:** 8GB+ system RAM
- **Storage:** ~2-5GB for model weights
- **Inference Speed:** 
  - GPU: ~1-5 seconds per minute of audio
  - CPU: ~5-30 minutes per minute of audio (not practical)

**Why GPU is Needed:**
- SAM Audio is a large transformer model
- CPU inference is extremely slow (hours for short videos)
- GPU acceleration is essential for practical use

#### CPU vs GPU Inference Trade-offs

**GPU Advantages:**
- 10-100x faster inference
- Practical for real-world use
- Better for batch processing

**CPU Disadvantages:**
- Extremely slow (impractical)
- May cause system freezing
- Not suitable for production

**Recommendation:** Use GPU for all SAM Audio inference. CPU is only acceptable for pre/post-processing.

### Free/Public GPU Resources

#### 1. Google Colab (Recommended for MVP)

**Free Tier:**
- **GPU:** T4 GPU (16GB VRAM)
- **RAM:** 12GB system RAM
- **Storage:** 100GB (temporary)
- **Session Limit:** 12 hours max
- **Availability:** Usually available, may have wait times

**Setup Steps:**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with Google account
3. Create new notebook
4. Runtime → Change runtime type → GPU
5. Install dependencies in notebook

**Limitations:**
- Sessions disconnect after inactivity
- Files deleted when session ends (use Google Drive)
- May need to wait for GPU availability
- 12-hour session limit

**Best For:** Development, testing, MVP building

#### 2. Kaggle Notebooks

**Free Tier:**
- **GPU:** P100 GPU (16GB VRAM) or T4 GPU
- **RAM:** 13GB system RAM
- **Storage:** 20GB persistent storage
- **Session Limit:** 9 hours max
- **Availability:** Usually available

**Setup Steps:**
1. Go to [kaggle.com](https://kaggle.com)
2. Sign up/login
3. Create new notebook
4. Settings → Accelerator → GPU
5. Install dependencies

**Advantages:**
- Persistent storage (20GB)
- Good for datasets
- Active ML community

**Limitations:**
- 9-hour session limit
- Internet access restrictions
- May need verification

**Best For:** Data-heavy projects, longer sessions

#### 3. RunPod (Pay-as-you-go)

**Pricing:**
- **T4 GPU:** ~$0.20-0.30/hour
- **RTX 3090:** ~$0.40-0.50/hour
- **A100:** ~$1.00-2.00/hour

**Advantages:**
- No session limits
- Persistent storage
- More reliable availability
- Better performance options

**Best For:** Production use, longer processing, when free tiers insufficient

**Setup:**
1. Sign up at [runpod.io](https://runpod.io)
2. Create pod with GPU
3. SSH into pod
4. Install dependencies

#### 4. Hugging Face Spaces (Inference API)

**Free Tier:**
- Inference API for some models
- Limited requests
- May not support SAM Audio directly

**Best For:** Quick API testing, not full development

#### 5. Modal Labs

**Free Tier:**
- Limited free credits
- Pay-as-you-go after

**Best For:** Serverless GPU functions

### Local Development Strategy

#### CPU Fallback Options

**When GPU Unavailable:**
1. Use smaller models if available (`sam-audio-base` instead of `large`)
2. Process very short audio clips (< 30 seconds)
3. Use for testing code logic only
4. Switch to cloud GPU for actual processing

**Model Quantization:**
- Use quantized models (INT8) for CPU
- Still slow but faster than FP32
- May reduce quality slightly

#### Chunked Processing for Memory Efficiency

**Strategy:**
- Process audio in 30-60 second chunks
- Clear GPU memory between chunks
- Reassemble processed chunks

**Code Pattern:**
```python
def process_long_audio(audio, chunk_size=30):
    chunks = split_audio(audio, chunk_size)
    processed_chunks = []
    
    for chunk in chunks:
        # Process chunk
        processed = process_chunk(chunk)
        processed_chunks.append(processed)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # Reassemble
    return combine_chunks(processed_chunks)
```

---

## Pre-Development Checklist

### Environment Setup

#### Python Installation

**Requirement:** Python 3.9 or higher

**Check Version:**
```bash
python3 --version
```

**Install Python (if needed):**
- **macOS:** `brew install python3` or download from python.org
- **Linux:** `sudo apt-get install python3`
- **Windows:** Download from python.org

#### Virtual Environment Setup

**Create Virtual Environment:**
```bash
# Using venv (recommended)
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**Using Conda (Alternative):**
```bash
# Install Miniconda/Anaconda
conda create -n sam-audio python=3.9
conda activate sam-audio
```

#### Package Manager

**Using pip:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Using Poetry (Optional):**
```bash
pip install poetry
poetry init
poetry install
```

### API Keys & Accounts

#### Required Accounts

**1. Hugging Face Account**
- **Purpose:** Access SAM Audio model
- **Sign up:** [huggingface.co](https://huggingface.co)
- **Setup:**
  ```bash
  pip install huggingface_hub
  huggingface-cli login
  ```
- **Cost:** Free

**2. Google Colab Account**
- **Purpose:** Free GPU access
- **Sign up:** [colab.research.google.com](https://colab.research.google.com)
- **Cost:** Free

**3. Kaggle Account (Optional)**
- **Purpose:** Alternative GPU access
- **Sign up:** [kaggle.com](https://kaggle.com)
- **Cost:** Free

#### Optional APIs (For Universal Broadcaster)

**4. Google Cloud Translation API**
- **Purpose:** Text translation
- **Sign up:** [cloud.google.com](https://cloud.google.com)
- **Setup:** Create project, enable Translation API, create API key
- **Free Tier:** $20/month credit
- **Cost:** ~$20 per 1M characters

**5. DeepL API (Alternative Translation)**
- **Purpose:** High-quality translation
- **Sign up:** [deepl.com](https://deepl.com)
- **Free Tier:** 500,000 characters/month
- **Cost:** ~$25 per 1M characters after free tier

**6. ElevenLabs API (TTS)**
- **Purpose:** Text-to-speech generation
- **Sign up:** [elevenlabs.io](https://elevenlabs.io)
- **Free Tier:** 10,000 characters/month
- **Cost:** ~$5 per 1,000 characters after free tier

**7. OpenAI API (Optional - Whisper API)**
- **Purpose:** Speech-to-text (if not using local Whisper)
- **Sign up:** [openai.com](https://openai.com)
- **Free Tier:** $5 credit
- **Cost:** ~$0.006 per minute of audio

**8. YouTube API (Optional)**
- **Purpose:** Enhanced video metadata
- **Sign up:** [console.cloud.google.com](https://console.cloud.google.com)
- **Free Tier:** 10,000 units/day
- **Cost:** Free for most use cases

### Development Tools

#### Code Editor

**VS Code (Recommended):**
- **Download:** [code.visualstudio.com](https://code.visualstudio.com)
- **Extensions:**
  - Python
  - Jupyter
  - GitLens
  - Python Docstring Generator

**PyCharm (Alternative):**
- **Download:** [jetbrains.com/pycharm](https://jetbrains.com/pycharm)
- **Community Edition:** Free

#### Git Client

**Command Line (Built-in):**
- macOS/Linux: Usually pre-installed
- Windows: Install Git for Windows

**GUI Clients (Optional):**
- GitHub Desktop
- SourceTree
- GitKraken

#### Docker (Optional)

**Purpose:** Local testing, containerization

**Install:**
- **macOS:** `brew install docker` or Docker Desktop
- **Linux:** `sudo apt-get install docker.io`
- **Windows:** Docker Desktop

**Note:** Not required for MVP, useful for production

#### FFmpeg

**Purpose:** Audio/video processing

**Install:**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows
# Download from ffmpeg.org
```

**Verify:**
```bash
ffmpeg -version
```

### Dependencies to Install

#### Core Dependencies

**Create `requirements.txt`:**
```txt
# Core ML/AI
torch>=2.0.0
transformers>=4.35.0
numpy>=1.24.0

# SAM Audio
# (Model loaded via transformers)

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.10.0

# Video Processing
yt-dlp>=2023.11.16
opencv-python>=4.8.0
ffmpeg-python>=0.2.0

# Computer Vision (for Visual-Audio Isolation)
ultralytics>=8.0.0  # YOLOv8

# Speech Processing (for Universal Broadcaster)
openai-whisper>=20231117

# API Clients (for Universal Broadcaster)
google-cloud-translate>=3.11.0
elevenlabs>=0.2.0

# Utilities
tqdm>=4.65.0
python-dotenv>=1.0.0
```

**Install:**
```bash
pip install -r requirements.txt
```

#### PyTorch Installation (GPU vs CPU)

**For GPU (Cloud):**
```bash
# CUDA 11.8 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU Only (Not Recommended):**
```bash
pip install torch torchvision torchaudio
```

**Verify Installation:**
```python
import torch
print(torch.cuda.is_available())  # Should be True on GPU
print(torch.__version__)
```

### Project Structure

#### Recommended Folder Organization

```
Project SAM/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── prds/                    # PRD documents
│   ├── PRD_Stadium_Mode.md
│   ├── PRD_Visual_Audio_Isolation.md
│   └── PRD_Universal_Broadcaster.md
│
├── src/                     # Source code
│   ├── stadium_mode/
│   │   ├── __init__.py
│   │   ├── separation.py
│   │   └── processing.py
│   ├── visual_audio/
│   │   ├── __init__.py
│   │   ├── detection.py
│   │   └── isolation.py
│   └── universal_broadcaster/
│       ├── __init__.py
│       ├── transcription.py
│       ├── translation.py
│       └── mixing.py
│
├── models/                  # Model weights (gitignored)
│   └── .gitkeep
│
├── data/                    # Test data (gitignored)
│   ├── input/
│   └── output/
│
├── notebooks/               # Jupyter notebooks for experimentation
│   └── .gitkeep
│
├── tests/                   # Unit tests
│   └── .gitkeep
│
├── scripts/                 # Utility scripts
│   └── setup.sh
│
└── docs/                    # Additional documentation
    └── .gitkeep
```

#### Configuration Management

**Create `config.yaml`:**
```yaml
models:
  sam_audio:
    model_name: "facebook/sam-audio-large"
    cache_dir: "./models"
  
  whisper:
    model_size: "base"  # base, medium, large
  
processing:
  audio:
    sample_rate: 44100
    chunk_size: 30  # seconds
  
  video:
    frame_rate: 1  # frames per second for detection
```

**Create `.env.example`:**
```env
# Hugging Face
HF_TOKEN=your_huggingface_token_here

# Google Cloud (for translation)
GOOGLE_CLOUD_API_KEY=your_google_api_key_here

# DeepL (alternative translation)
DEEPL_API_KEY=your_deepl_api_key_here

# ElevenLabs (for TTS)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# OpenAI (optional, for Whisper API)
OPENAI_API_KEY=your_openai_api_key_here
```

**Load Environment Variables:**
```python
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
```

#### Logging Setup

**Create `src/utils/logger.py`:**
```python
import logging
import sys

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
```

### Testing & Validation

#### Sample Test Videos/Audio

**Prepare Test Data:**
- Download 3-5 sample sports videos (different sports)
- Short clips (1-5 minutes) for faster testing
- Various qualities (HD, SD)
- Different commentary styles

**Sources:**
- YouTube (public domain or your own)
- Create test videos yourself
- Use free stock footage

**Store in:** `data/input/test_videos/`

#### Unit Test Framework Setup

**Create `tests/test_separation.py`:**
```python
import unittest
from src.stadium_mode.separation import separate_audio

class TestSeparation(unittest.TestCase):
    def test_basic_separation(self):
        # Test audio separation
        pass

if __name__ == '__main__':
    unittest.main()
```

**Run Tests:**
```bash
python -m pytest tests/
```

#### Performance Benchmarking

**Create `scripts/benchmark.py`:**
```python
import time
from src.stadium_mode.separation import separate_audio

def benchmark_processing(video_path):
    start = time.time()
    result = separate_audio(video_path)
    end = time.time()
    
    duration = end - start
    print(f"Processing time: {duration:.2f} seconds")
    return duration
```

---

## Technical Prerequisites Deep Dive

### SAM Audio Model

#### Model Download and Caching

**Automatic Download (Recommended):**
```python
from transformers import AutoProcessor, SamAudioModel

# Model downloads automatically on first use
processor = AutoProcessor.from_pretrained("facebook/sam-audio-large")
model = SamAudioModel.from_pretrained("facebook/sam-audio-large")
```

**Manual Download:**
```bash
huggingface-cli download facebook/sam-audio-large
```

**Cache Location:**
- Default: `~/.cache/huggingface/hub/`
- Custom: Set `HF_HOME` environment variable

#### Processor Initialization

**Basic Setup:**
```python
from transformers import AutoProcessor, SamAudioModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("facebook/sam-audio-large")
model = SamAudioModel.from_pretrained("facebook/sam-audio-large").to(device)
model.eval()
```

#### Memory Optimization Techniques

**1. Model Quantization:**
```python
# Quantize model to INT8 (reduces memory by ~50%)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)
model = SamAudioModel.from_pretrained(
    "facebook/sam-audio-large",
    quantization_config=quantization_config
)
```

**2. Gradient Checkpointing:**
```python
model.gradient_checkpointing_enable()
```

**3. Clear Cache:**
```python
import torch
torch.cuda.empty_cache()  # Clear GPU memory
```

#### Batch Processing Considerations

**Process in Chunks:**
```python
def process_audio_chunks(audio, chunk_size=30, sample_rate=44100):
    chunk_samples = chunk_size * sample_rate
    chunks = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i+chunk_samples]
        processed = process_chunk(chunk)
        chunks.append(processed)
        torch.cuda.empty_cache()
    
    return combine_chunks(chunks)
```

### Video Processing Pipeline

#### yt-dlp Configuration

**Basic Usage:**
```python
import yt_dlp

def download_video(url, output_path="data/input/"):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': f'{output_path}%(title)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
```

**Audio Extraction:**
```python
def extract_audio(url, output_path="data/input/audio.wav"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
```

#### Audio Extraction Workflow

**Using FFmpeg:**
```python
import subprocess

def extract_audio_ffmpeg(video_path, audio_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '44100',  # Sample rate
        '-ac', '2',  # Stereo
        audio_path
    ]
    subprocess.run(command, check=True)
```

#### Video Re-muxing Process

**Re-mux Audio with Video:**
```python
def remux_video(video_path, audio_path, output_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # Copy video codec
        '-c:a', 'aac',  # Audio codec
        '-map', '0:v:0',  # Use video from first input
        '-map', '1:a:0',  # Use audio from second input
        '-shortest',  # Match shortest stream
        output_path
    ]
    subprocess.run(command, check=True)
```

#### Format Compatibility

**Supported Input Formats:**
- MP4, MKV, WebM, AVI, MOV
- YouTube, Twitch, m3u8 streams

**Supported Output Formats:**
- MP4 (recommended, most compatible)
- MKV (better quality, larger files)
- WebM (web-friendly)

### Visual Detection (for Visual-Audio Isolation)

#### YOLOv8 Setup

**Install:**
```bash
pip install ultralytics
```

**Model Selection:**
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small (balanced)
- `yolov8m.pt` - Medium (better accuracy)
- `yolov8l.pt` - Large (best accuracy, slower)

**Basic Usage:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Downloads automatically
results = model(frame)
persons = [box for box in results.boxes if box.cls == 0]  # Class 0 = person
```

#### Face Detection Alternatives

**Using face_recognition:**
```bash
pip install face-recognition
```

```python
import face_recognition

# Load known face
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Detect in frame
face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations)

# Match
matches = face_recognition.compare_faces([known_encoding], face_encodings[0])
```

#### Coordinate Extraction and Mapping

**Extract Center Coordinates:**
```python
def get_person_center(bounding_box):
    x1, y1, x2, y2 = bounding_box.xyxy[0].cpu().numpy()
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y
```

### Translation & TTS (for Universal Broadcaster)

#### Whisper Model Selection

**Model Sizes:**
- `base` - Fastest, least accurate (~74M params)
- `small` - Balanced (~244M params)
- `medium` - Better accuracy (~769M params)
- `large` - Best accuracy (~1550M params)

**Usage:**
```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.wav")
text = result["text"]
```

#### Translation Service Comparison

**Google Translate API:**
- Pros: Many languages, reliable
- Cons: Less natural, more expensive
- Cost: ~$20 per 1M characters

**DeepL API:**
- Pros: More natural translations, better quality
- Cons: Fewer languages, more expensive
- Cost: ~$25 per 1M characters

**Recommendation:** Start with DeepL free tier, switch to Google if needed

#### TTS Voice Selection

**ElevenLabs:**
- Many voice options
- Good quality
- Expensive (~$5 per 1K characters)

**Meta Voicebox:**
- Open source alternative
- Good quality
- Free (self-hosted)

**Recommendation:** Start with ElevenLabs free tier, consider Voicebox for scaling

---

## MVP vs Full Feature Approach

### Stadium Mode MVP

**MVP Scope:**
- Process single video URL
- Remove commentary using SAM Audio
- Output processed video
- Basic CLI interface
- Support YouTube URLs only

**Defer:**
- Real-time processing
- Multiple input sources
- Adjustable removal strength
- Batch processing
- Web interface

### Visual-Audio Isolation MVP

**MVP Scope:**
- Process single video file
- Detect people using YOLOv8
- User selects target person (CLI)
- Isolate target person's audio (Fan Mode)
- Output processed video

**Defer:**
- Hater Mode (muting)
- Visual person selection UI
- Multi-person tracking
- Real-time processing
- Face recognition

### Universal Broadcaster MVP

**MVP Scope:**
- Process single video URL
- Support 2-3 target languages (Spanish, Hindi, Japanese)
- Use free/low-cost APIs
- Basic audio mixing
- CLI interface

**Defer:**
- Real-time processing
- Many languages (start with 2-3)
- Voice customization
- Web interface
- Batch processing
- Advanced mixing options

---

## Cost Considerations

### Development Costs

**Free Resources:**
- Google Colab GPU: Free
- Kaggle GPU: Free
- Hugging Face models: Free
- OpenAI Whisper: Free (local)
- Git/GitHub: Free

**Estimated MVP Cost:** $0-50
- API testing: $20-50
- Optional paid GPU: $0-30

### Production Costs (Future)

**Per Minute of Video:**
- GPU compute: $0.05-0.20 (if using paid)
- Translation API: $0.01-0.05
- TTS API: $0.05-0.20
- **Total:** ~$0.10-0.50 per minute

**Monthly Estimates (100 hours/month):**
- GPU: $50-200
- APIs: $100-500
- Storage: $10-50
- **Total:** ~$160-750/month

### Cost Optimization Strategies

1. **Use Free Tiers:** Maximize free API credits
2. **Cache Results:** Cache translations for repeated content
3. **Optimize API Calls:** Batch requests, reduce redundancy
4. **Consider Self-Hosted:** Use open-source alternatives (Voicebox, local Whisper)
5. **Monitor Usage:** Track costs closely, set alerts

---

## Quick Start Checklist

### Week 1: Foundation

- [ ] Initialize git repository
- [ ] Set up Python virtual environment
- [ ] Install core dependencies
- [ ] Create project structure
- [ ] Set up Google Colab account
- [ ] Set up Hugging Face account
- [ ] Test SAM Audio model loading
- [ ] Install FFmpeg
- [ ] Create `.gitignore`
- [ ] Set up logging

### Week 2: First Project (Stadium Mode)

- [ ] Test yt-dlp video extraction
- [ ] Implement basic audio separation
- [ ] Test SAM Audio commentary detection
- [ ] Implement audio subtraction
- [ ] Test video re-muxing
- [ ] Create basic CLI
- [ ] Test on sample videos

### Week 3: Refinement

- [ ] Improve prompt engineering
- [ ] Optimize processing speed
- [ ] Add error handling
- [ ] Add progress indicators
- [ ] Test on diverse videos
- [ ] Document usage

### Week 4: Next Projects

- [ ] Choose next project (Visual-Audio or Universal Broadcaster)
- [ ] Set up required APIs
- [ ] Begin implementation
- [ ] Follow similar workflow

---

## Additional Resources

### Documentation Links

- [SAM Audio Paper](https://arxiv.org/abs/2312.06663)
- [SAM Audio Hugging Face](https://huggingface.co/facebook/sam-audio-large)
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [Whisper GitHub](https://github.com/openai/whisper)
- [yt-dlp GitHub](https://github.com/yt-dlp/yt-dlp)
- [FFmpeg Docs](https://ffmpeg.org/documentation.html)

### Community Resources

- Hugging Face Forums
- Reddit: r/MachineLearning, r/Python
- Stack Overflow
- Discord: AI/ML communities

### Learning Resources

- PyTorch Tutorials
- Audio Processing with librosa
- Computer Vision with OpenCV
- Video Processing with FFmpeg

---

## Troubleshooting

### Common Issues

**1. GPU Not Available:**
- Check Colab runtime type
- Verify CUDA installation
- Try different cloud provider

**2. Model Download Fails:**
- Check internet connection
- Verify Hugging Face token
- Try manual download

**3. Out of Memory:**
- Use smaller model
- Process in smaller chunks
- Clear GPU cache frequently

**4. Audio Sync Issues:**
- Check sample rates match
- Verify timestamps preserved
- Test re-muxing process

**5. API Rate Limits:**
- Implement retry logic
- Use multiple API keys
- Cache results

---

**Document Status:** Complete  
**Next Steps:** Begin Week 1 foundation setup, then proceed with Stadium Mode MVP

