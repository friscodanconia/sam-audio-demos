# Product Requirements Document: Stadium Mode (The Atmosphere Filter)

**Version:** 1.0  
**Date:** December 2024  
**Author:** Solo Developer  
**Status:** Planning Phase

---

## Executive Summary

### Problem Statement
Sports fans often find broadcast commentary distracting, biased, or overly verbose. Many viewers prefer the raw stadium atmosphere—the roar of the crowd, referee whistles, and game sounds—without the constant commentary overlay. Current solutions don't exist for removing commentary while preserving ambient stadium audio.

### Solution Overview
Stadium Mode is an AI-powered audio processing tool that removes sports commentary from live or recorded broadcasts while preserving all stadium atmosphere sounds. Using Meta's SAM Audio model, the tool performs subtractive audio separation to isolate and remove commentator voices, leaving users with an immersive stadium experience.

### Target Users
- **Primary:** Sports fans who prefer raw game audio without commentary
- **Secondary:** Content creators wanting to create highlight reels with ambient audio
- **Tertiary:** Broadcasters exploring alternative audio tracks

### Key Value Proposition
Transform any sports broadcast into an immersive stadium experience by removing commentary while preserving crowd noise, game sounds, and referee signals.

---

## Product Vision

### Long-Term Goals
Become the go-to tool for sports fans seeking authentic stadium audio experiences. Expand beyond sports to concerts, conferences, and other live events where users want to filter out specific audio sources.

### Market Opportunity
- Millions of sports fans watch broadcasts daily
- Growing demand for customizable viewing experiences
- No direct competitors offering this specific functionality
- Potential integration with streaming platforms and broadcasters

### Value Proposition
- **For Users:** Experience games as if you're in the stadium, without commentary distractions
- **For Content Creators:** Create unique highlight reels with authentic atmosphere
- **For Platforms:** Offer differentiated viewing experiences to attract subscribers

---

## Key Features

### Core Features

#### 1. URL-Based Video Processing
**Description:** Accept video URLs from YouTube, Twitch, or m3u8 streams as input.

**User Benefit:** No need to download videos manually; process directly from streaming platforms.

**Technical Details:**
- Support for YouTube, Twitch, and m3u8 stream URLs
- Automatic video/audio extraction using yt-dlp
- High-fidelity audio extraction (minimum 44.1kHz, stereo)

**Implementation Priority:** P0 (Must Have)

#### 2. AI-Powered Commentary Removal
**Description:** Use SAM Audio to identify and remove commentator voices using text prompts.

**User Benefit:** Accurate separation that preserves background sounds that exist underneath commentary.

**Technical Details:**
- Model: `facebook/sam-audio-large`
- Positive prompts: ["sports commentator", "play-by-play announcer", "human speech"]
- Negative prompts: ["crowd cheering", "whistle", "ball impact", "sneaker squeak"]
- Subtractive separation (waveform subtraction, not muting)

**Implementation Priority:** P0 (Must Have)

#### 3. Stadium Atmosphere Preservation
**Description:** Maintain crowd noise, referee whistles, ball impacts, and other game sounds.

**User Benefit:** Authentic stadium experience with all ambient audio intact.

**Technical Details:**
- Waveform subtraction preserves background noise layers
- Normalization to prevent audio clipping
- Dynamic range preservation

**Implementation Priority:** P0 (Must Have)

#### 4. Video Re-muxing
**Description:** Combine processed audio with original video to create final output.

**User Benefit:** Seamless viewing experience with synchronized audio/video.

**Technical Details:**
- FFmpeg-based re-muxing
- Maintain original video quality
- Support for common formats (MP4, MKV, WebM)

**Implementation Priority:** P0 (Must Have)

### Enhanced Features (Future Iterations)

#### 5. Real-Time Processing Mode
**Description:** Process live streams with minimal latency.

**User Benefit:** Use Stadium Mode during live broadcasts.

**Technical Details:**
- Chunked processing (5-10 second windows)
- Streaming audio pipeline
- Latency optimization

**Implementation Priority:** P2 (Future)

#### 6. Adjustable Commentary Removal Strength
**Description:** Slider to control how aggressively commentary is removed.

**User Benefit:** Fine-tune the experience—partial removal for some commentary, full removal for others.

**Implementation Priority:** P1 (Nice to Have)

#### 7. Batch Processing
**Description:** Process multiple videos or entire playlists.

**User Benefit:** Efficiently process entire seasons or highlight collections.

**Implementation Priority:** P1 (Nice to Have)

---

## Technical Architecture

### System Design

```
┌─────────────────┐
│  User Input     │
│  (Video URL)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  yt-dlp         │
│  Video/Audio    │
│  Extraction     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio          │
│  Preprocessing  │
│  (Normalize,    │
│   Resample)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SAM Audio      │
│  Model          │
│  (Commentary    │
│   Detection)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio          │
│  Subtraction    │
│  (Remove        │
│   Commentary)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio          │
│  Postprocessing │
│  (Normalize,    │
│   Enhance)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FFmpeg         │
│  Re-muxing      │
│  (Video + Audio)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Video   │
└─────────────────┘
```

### Data Flow

1. **Input Stage:** User provides video URL
2. **Extraction Stage:** yt-dlp downloads and extracts audio/video streams
3. **Processing Stage:** SAM Audio identifies commentary mask
4. **Separation Stage:** Subtractive audio processing removes commentary
5. **Output Stage:** Processed audio re-muxed with original video

### Technology Stack

**Core Technologies:**
- **Python 3.9+** - Primary development language
- **PyTorch** - Deep learning framework for SAM Audio
- **Transformers (Hugging Face)** - Model loading and inference
- **yt-dlp** - Video/audio extraction from URLs
- **FFmpeg** - Audio/video processing and re-muxing
- **librosa/soundfile** - Audio manipulation

**AI/ML Components:**
- **SAM Audio Large Model** (`facebook/sam-audio-large`) - Commentary detection
- **Text Prompts** - Guide model to identify commentary vs. atmosphere

**Infrastructure:**
- **Cloud GPU** (Google Colab/Kaggle) - Model inference
- **Local CPU** - Pre/post-processing and file management

---

## Technical Requirements

### Dependencies

**Python Packages:**
```python
torch>=2.0.0
transformers>=4.35.0
yt-dlp>=2023.11.16
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
ffmpeg-python>=0.2.0
```

**System Requirements:**
- Python 3.9 or higher
- FFmpeg installed system-wide
- 8GB+ RAM (for local processing)
- GPU access recommended (via cloud services)

### API Requirements

**Hugging Face:**
- Account for model access
- Model: `facebook/sam-audio-large`
- Token for private models (if needed)

**Optional APIs:**
- YouTube Data API (for enhanced metadata)
- Cloud storage API (for output storage)

### Infrastructure Needs

**Development:**
- Local development environment
- Cloud GPU access (Google Colab/Kaggle free tier)
- Storage for test videos and outputs

**Production (Future):**
- Cloud compute with GPU (AWS EC2, GCP, RunPod)
- Object storage (S3, GCS) for video files
- CDN for video delivery

### Performance Requirements

**Processing Speed:**
- Target: Process 1 minute of video in < 2 minutes
- Acceptable: Process 1 minute of video in < 5 minutes

**Audio Quality:**
- Maintain original audio sample rate
- Preserve stereo channels
- Dynamic range: -60dB to 0dB

**Output Quality:**
- Video: Same resolution and codec as input
- Audio: 44.1kHz or higher, stereo

---

## User Stories & Use Cases

### User Story 1: The Purist Fan
**As a** sports fan who dislikes commentary  
**I want to** remove commentator voices from game broadcasts  
**So that** I can experience the game with pure stadium atmosphere

**Acceptance Criteria:**
- Can input YouTube URL of a game
- Commentary is completely removed
- Crowd noise and game sounds remain intact
- Output video plays smoothly

### User Story 2: The Content Creator
**As a** content creator making highlight reels  
**I want to** extract stadium atmosphere from broadcasts  
**So that** I can create unique content with authentic audio

**Acceptance Criteria:**
- Can process multiple videos
- Audio quality is high enough for professional use
- Processing completes in reasonable time
- Output format is compatible with editing software

### User Story 3: The Casual Viewer
**As a** casual sports viewer  
**I want to** try Stadium Mode on a game  
**So that** I can see if I prefer it over regular commentary

**Acceptance Criteria:**
- Simple interface to paste URL
- Clear processing status
- Quick preview of result
- Easy download of processed video

### Use Case Scenarios

**Scenario 1: NBA Game Processing**
1. User finds YouTube highlight of NBA game
2. Copies URL and pastes into Stadium Mode
3. Tool processes 10-minute highlight
4. User downloads video with crowd noise only
5. User shares on social media

**Scenario 2: Soccer Match Live Stream**
1. User watches live Premier League match on Twitch
2. User wants to experience stadium atmosphere
3. Tool processes stream in real-time (future feature)
4. User watches with commentary removed
5. User enjoys authentic crowd reactions

---

## Success Metrics

### Product Metrics

**Adoption Metrics:**
- Number of videos processed per week
- User retention rate (users who process multiple videos)
- Average video length processed

**Quality Metrics:**
- User satisfaction score (1-5 scale)
- Percentage of successful processing (no errors)
- Audio quality rating from users

**Performance Metrics:**
- Average processing time per minute of video
- Success rate of commentary removal (user feedback)
- System uptime/availability

### Technical Metrics

**Model Performance:**
- Commentary detection accuracy (manual evaluation)
- False positive rate (removing non-commentary audio)
- Processing speed (seconds per minute of audio)

**System Performance:**
- Memory usage during processing
- GPU utilization efficiency
- Error rate and types

### Validation Criteria

**MVP Success:**
- Successfully process 10 test videos from different sports
- Commentary removal accuracy > 85% (manual evaluation)
- Processing time < 5 minutes per minute of video
- User feedback score > 3.5/5.0

**Full Product Success:**
- Process videos from 5+ different sports
- Commentary removal accuracy > 90%
- Processing time < 2 minutes per minute of video
- User feedback score > 4.0/5.0

---

## Development Roadmap

### Phase 1: MVP Development (Weeks 1-3)

**Week 1: Foundation**
- Set up development environment
- Initialize git repository
- Set up cloud GPU access (Google Colab/Kaggle)
- Test SAM Audio model loading and basic inference
- Create basic project structure

**Deliverables:**
- Working SAM Audio inference pipeline
- Basic audio processing utilities
- Test script with sample audio

**Week 2: Core Processing**
- Implement yt-dlp integration for URL processing
- Build audio extraction pipeline
- Implement SAM Audio commentary detection
- Create subtractive audio separation logic
- Add audio normalization and post-processing

**Deliverables:**
- End-to-end audio processing pipeline
- Test on 3-5 sample videos
- Basic error handling

**Week 3: Video Integration**
- Implement FFmpeg re-muxing
- Create video output pipeline
- Build simple CLI interface
- Test full pipeline end-to-end
- Document usage instructions

**Deliverables:**
- Complete MVP working on test videos
- CLI tool for processing
- Basic documentation

### Phase 2: Enhancement (Weeks 4-5)

**Week 4: Quality & Robustness**
- Improve prompt engineering for better detection
- Add audio quality enhancements
- Implement better error handling
- Add progress indicators
- Optimize processing speed

**Deliverables:**
- Improved commentary removal accuracy
- Better user experience
- Performance optimizations

**Week 5: Testing & Refinement**
- Comprehensive testing on diverse videos
- User feedback collection
- Bug fixes and improvements
- Documentation updates
- Prepare for demo

**Deliverables:**
- Stable, tested MVP
- User documentation
- Demo-ready product

### Phase 3: Future Enhancements (Post-MVP)

**Potential Features:**
- Real-time processing mode
- Adjustable removal strength
- Batch processing
- Web interface
- API for integration
- Support for more video sources

**Timeline:** TBD based on MVP success and user feedback

---

## Risks & Mitigations

### Technical Risks

**Risk 1: SAM Audio Model Performance**
- **Description:** Model may not accurately identify commentary in all scenarios
- **Impact:** High - Core functionality depends on accurate detection
- **Mitigation:**
  - Test extensively on diverse sports and commentary styles
  - Refine prompt engineering based on results
  - Consider fine-tuning model if needed
  - Have fallback to simpler audio processing methods

**Risk 2: Processing Speed**
- **Description:** SAM Audio inference may be too slow for practical use
- **Impact:** Medium - Users expect reasonable processing times
- **Mitigation:**
  - Use GPU acceleration (cloud resources)
  - Optimize batch processing
  - Consider model quantization
  - Implement chunked processing for long videos

**Risk 3: Audio Quality Degradation**
- **Description:** Subtractive separation may introduce artifacts
- **Impact:** Medium - Quality is key user value
- **Mitigation:**
  - Careful normalization and post-processing
  - Test on various audio qualities
  - Implement quality enhancement filters
  - User feedback loop for quality assessment

**Risk 4: GPU Resource Availability**
- **Description:** Free GPU resources may have limits or downtime
- **Impact:** Medium - Blocks development and usage
- **Mitigation:**
  - Use multiple cloud providers (Colab, Kaggle)
  - Implement CPU fallback (slower but functional)
  - Consider paid options for critical usage
  - Cache models locally when possible

### Product Risks

**Risk 5: Limited Use Cases**
- **Description:** May only work well for specific sports or commentary styles
- **Impact:** Medium - Limits market appeal
- **Mitigation:**
  - Test on diverse content early
  - Be transparent about limitations
  - Iterate based on user feedback
  - Expand to other use cases (concerts, etc.)

**Risk 6: Copyright Concerns**
- **Description:** Processing copyrighted content may raise legal issues
- **Impact:** Low-Medium - Could limit distribution
- **Mitigation:**
  - Focus on personal use cases
  - Add disclaimer about copyright
  - Consider partnerships with content creators
  - Research fair use implications

### Operational Risks

**Risk 7: Solo Developer Burnout**
- **Description:** Building alone may lead to project abandonment
- **Impact:** High - Project may not complete
- **Mitigation:**
  - Set realistic milestones
  - Focus on MVP first
  - Take breaks between phases
  - Consider open-sourcing for community support

---

## Solo Developer Considerations

### Resource Constraints

**Time:**
- Limited to evenings/weekends or dedicated blocks
- Focus on MVP first, iterate later
- Set realistic deadlines with buffer time

**Budget:**
- Rely on free cloud GPU resources (Colab, Kaggle)
- Minimize paid API usage
- Use open-source tools exclusively

**Skills:**
- May need to learn SAM Audio API
- Video/audio processing knowledge required
- Python and ML fundamentals assumed

### Prioritization Strategy

**Must Have (P0):**
- URL input processing
- Commentary removal
- Video output
- Basic CLI interface

**Nice to Have (P1):**
- Progress indicators
- Error messages
- Quality improvements
- Batch processing

**Future (P2):**
- Web interface
- Real-time processing
- Advanced features

### MVP Approach

**MVP Scope:**
- Process single video URL
- Remove commentary using SAM Audio
- Output processed video file
- Basic error handling
- Command-line interface

**What to Defer:**
- Web interface
- Real-time processing
- Batch processing
- Advanced quality controls
- Multiple output formats

### Development Workflow

1. **Start Small:** Get basic SAM Audio working on sample audio
2. **Iterate:** Add features one at a time
3. **Test Frequently:** Test on real videos early and often
4. **Document:** Keep notes on what works and what doesn't
5. **Get Feedback:** Share early demos for feedback

### Cost Management

**Free Resources:**
- Google Colab (free GPU tier)
- Kaggle Notebooks (free GPU)
- Hugging Face (free model access)
- GitHub (free code hosting)

**Potential Costs:**
- Cloud storage for videos (minimal with free tiers)
- Paid GPU if free tier insufficient (~$0.20/hour)
- Domain/hosting if building web interface (future)

**Budget Estimate:**
- MVP Development: $0 (using free resources)
- Testing/Demo: $0-20 (optional paid GPU)
- Future scaling: TBD based on usage

---

## Open Questions

1. **Model Selection:** Is `sam-audio-large` the best choice, or should we test smaller models for speed?
2. **Prompt Engineering:** What combination of prompts gives best results across different sports?
3. **Processing Strategy:** Should we process entire audio at once or in chunks?
4. **Output Format:** What video formats should we prioritize?
5. **User Interface:** CLI for MVP, but what should the future interface be?

---

## References & Resources

- [Meta SAM Audio Paper](https://arxiv.org/abs/2312.06663)
- [SAM Audio Hugging Face Model](https://huggingface.co/facebook/sam-audio-large)
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [PyTorch Audio Processing](https://pytorch.org/audio/stable/index.html)

---

## Appendix: Code Snippet Reference

```python
# Core Commentary Removal Logic (from blueprint)
import torch
from transformers import AutoProcessor, SamAudioModel

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/sam-audio-large")
model = SamAudioModel.from_pretrained("facebook/sam-audio-large")

# Process audio
prompts = ["sports commentator", "announcer voice"]
inputs = processor(audio=waveform, sampling_rate=sr, text_prompts=prompts)

# Get commentary mask
with torch.no_grad():
    outputs = model(inputs)
    commentary_mask = outputs['masks'][0]

# Subtract commentary to leave stadium atmosphere
stadium_atmosphere = original_waveform - commentary_mask

# Normalize to prevent clipping
stadium_atmosphere = stadium_atmosphere / torch.max(torch.abs(stadium_atmosphere))
```

---

**Document Status:** Ready for Review  
**Next Steps:** Begin Phase 1 development after preparation guide review

