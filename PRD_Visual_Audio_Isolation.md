# Product Requirements Document: Visual-Audio Isolation Filter

**Version:** 1.0  
**Date:** December 2024  
**Author:** Solo Developer  
**Status:** Planning Phase

---

## Executive Summary

### Problem Statement
Current audio processing tools can isolate or mute audio sources, but they're "blind"—they can't identify WHO is speaking. Users watching videos with multiple people want to hear only a specific person (e.g., a celebrity, speaker, or performer) or mute a specific person while hearing everyone else. Existing solutions require manual audio editing or don't support person-specific isolation.

### Solution Overview
Visual-Audio Isolation Filter combines computer vision (YOLOv8) with audio AI (SAM Audio) to create a "Smart Mute" system. The tool detects a specific person in video frames, extracts their location, and uses visual prompting with SAM Audio to isolate or mute that person's audio. This creates a unique "Focus Mode" that works on specific individuals, not just generic voice detection.

### Target Users
- **Primary:** Content creators editing videos with multiple speakers
- **Secondary:** Fans wanting to isolate specific performers (singers, speakers, celebrities)
- **Tertiary:** Educational users wanting to focus on specific instructors in multi-person videos

### Key Value Proposition
The first tool that combines vision and audio AI to isolate or mute specific people in videos, creating unprecedented control over audio sources based on visual identification.

---

## Product Vision

### Long-Term Goals
Become the standard tool for person-specific audio isolation in video content. Expand to support real-time processing, multiple person tracking, and integration with video editing software.

### Market Opportunity
- Millions of videos uploaded daily with multiple speakers
- Growing creator economy needs better editing tools
- No existing solution combines visual detection with audio isolation
- Potential for viral "wow factor" demos

### Value Proposition
- **For Content Creators:** Isolate specific speakers without manual audio editing
- **For Fans:** Focus on favorite performers in crowded videos
- **For Educators:** Create focused learning experiences from multi-person content

---

## Key Features

### Core Features

#### 1. Person Detection in Video
**Description:** Use YOLOv8 or face detection to identify and track specific people in video frames.

**User Benefit:** Automatically find the target person without manual frame-by-frame selection.

**Technical Details:**
- Frame extraction (1 frame per second, configurable)
- YOLOv8 for person detection
- Face recognition for specific person identification (optional)
- Bounding box coordinate extraction

**Implementation Priority:** P0 (Must Have)

#### 2. Visual Prompting with SAM Audio
**Description:** Pass detected person coordinates to SAM Audio as visual prompts for audio isolation.

**User Benefit:** Precise audio isolation based on visual location, not just voice characteristics.

**Technical Details:**
- Convert bounding box center to [x, y] coordinates
- Pass frame image and coordinates to SAM Audio
- Use `input_points=[[[x, y]]]` for visual prompting
- Process audio segments corresponding to detected frames

**Implementation Priority:** P0 (Must Have)

#### 3. Fan Mode (Isolation)
**Description:** Keep only the audio from the detected person, removing all other audio sources.

**User Benefit:** Hear only what the target person is saying/singing, even in noisy environments.

**Technical Details:**
- SAM Audio returns mask for target person's audio
- Keep only masked audio segments
- Blend isolated segments smoothly
- Maintain audio sync with video

**Implementation Priority:** P0 (Must Have)

#### 4. Hater Mode (Muting)
**Description:** Remove the detected person's audio while keeping everything else.

**User Benefit:** Mute specific people (e.g., annoying commentators, background speakers) while preserving other audio.

**Technical Details:**
- SAM Audio returns mask for target person's audio
- Subtract masked audio from original waveform
- Preserve background audio and other speakers
- Smooth transitions between segments

**Implementation Priority:** P0 (Must Have)

#### 5. Video-Audio Synchronization
**Description:** Ensure processed audio remains synchronized with video frames.

**User Benefit:** Seamless viewing experience without audio/video drift.

**Technical Details:**
- Track frame timestamps
- Map audio segments to corresponding frames
- Maintain original video timing
- Re-mux processed audio with video

**Implementation Priority:** P0 (Must Have)

### Enhanced Features (Future Iterations)

#### 6. Multi-Person Tracking
**Description:** Track and isolate/mute multiple people simultaneously.

**User Benefit:** Complex audio editing scenarios with multiple targets.

**Implementation Priority:** P2 (Future)

#### 7. Real-Time Processing
**Description:** Process live video streams with person-specific audio isolation.

**User Benefit:** Use during live broadcasts or video calls.

**Implementation Priority:** P2 (Future)

#### 8. Person Selection UI
**Description:** Visual interface to select target person from video preview.

**User Benefit:** Easy person selection without technical knowledge.

**Implementation Priority:** P1 (Nice to Have)

#### 9. Confidence Thresholds
**Description:** Adjustable sensitivity for person detection and audio isolation.

**User Benefit:** Fine-tune results for different video qualities and scenarios.

**Implementation Priority:** P1 (Nice to Have)

---

## Technical Architecture

### System Design

```
┌─────────────────┐
│  Video Input    │
│  (File/Stream)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Frame          │
│  Extraction     │
│  (1 fps)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv8         │
│  Person         │
│  Detection      │
└────────┬────────┘
         │
         │ [x, y] coordinates
         ▼
┌─────────────────┐
│  Person         │
│  Identification │
│  (Face Rec/     │
│   User Select)  │
└────────┬────────┘
         │
         │ Target person confirmed
         ▼
┌─────────────────┐
│  Audio          │
│  Extraction     │
│  (Per Segment)  │
└────────┬────────┘
         │
         │ Frame + Coordinates
         ▼
┌─────────────────┐
│  SAM Audio      │
│  Visual         │
│  Prompting      │
└────────┬────────┘
         │
         │ Audio mask
         ▼
┌─────────────────┐
│  Audio          │
│  Processing     │
│  (Isolate/      │
│   Mute)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio          │
│  Blending       │
│  (Smooth        │
│   Transitions)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Video-Audio    │
│  Re-muxing      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Video   │
└─────────────────┘
```

### Data Flow

1. **Input Stage:** Video file or stream loaded
2. **Frame Extraction:** Extract frames at regular intervals (1 fps)
3. **Person Detection:** YOLOv8 detects all people in each frame
4. **Person Identification:** User selects or system identifies target person
5. **Coordinate Extraction:** Extract [x, y] coordinates of target person
6. **Audio Segmentation:** Extract audio segments corresponding to frames
7. **Visual Prompting:** Pass frame image and coordinates to SAM Audio
8. **Audio Isolation/Muting:** Process audio based on mode (Fan/Hater)
9. **Audio Blending:** Smooth transitions between processed segments
10. **Output Stage:** Re-mux processed audio with original video

### Technology Stack

**Core Technologies:**
- **Python 3.9+** - Primary development language
- **PyTorch** - Deep learning framework
- **OpenCV** - Video processing and frame extraction
- **YOLOv8 (Ultralytics)** - Person detection
- **SAM Audio** - Audio isolation via visual prompting
- **FFmpeg** - Video/audio processing and re-muxing

**AI/ML Components:**
- **YOLOv8** - Person detection in video frames
- **Face Recognition (Optional)** - Specific person identification
- **SAM Audio** - Visual prompting for audio isolation
- **Visual Prompting** - Using image coordinates to guide audio processing

**Infrastructure:**
- **Cloud GPU** - For YOLOv8 and SAM Audio inference
- **Local CPU** - Video processing and file management

---

## Technical Requirements

### Dependencies

**Python Packages:**
```python
torch>=2.0.0
transformers>=4.35.0
ultralytics>=8.0.0  # YOLOv8
opencv-python>=4.8.0
numpy>=1.24.0
librosa>=0.10.0
soundfile>=0.12.0
ffmpeg-python>=0.2.0
face-recognition>=1.3.0  # Optional, for face recognition
```

**System Requirements:**
- Python 3.9 or higher
- FFmpeg installed system-wide
- 8GB+ RAM (for video processing)
- GPU access recommended (via cloud services)

### API Requirements

**Hugging Face:**
- Account for SAM Audio model access
- Model: `facebook/sam-audio-large`

**Optional:**
- Face recognition models (local)
- Pre-trained YOLOv8 weights (downloaded automatically)

### Infrastructure Needs

**Development:**
- Local development environment
- Cloud GPU access (Google Colab/Kaggle)
- Storage for test videos

**Production (Future):**
- Cloud compute with GPU
- Object storage for video files
- CDN for video delivery

### Performance Requirements

**Processing Speed:**
- Target: Process 1 minute of video in < 10 minutes
- Acceptable: Process 1 minute of video in < 20 minutes

**Detection Accuracy:**
- Person detection: > 90% accuracy
- Target person identification: > 85% accuracy
- Audio isolation quality: Subjectively good (user feedback)

**Output Quality:**
- Video: Same resolution and codec as input
- Audio: Maintain original quality, smooth transitions

---

## User Stories & Use Cases

### User Story 1: The Content Creator
**As a** video content creator  
**I want to** isolate a specific speaker's audio from a multi-person video  
**So that** I can create focused content without manual audio editing

**Acceptance Criteria:**
- Can select target person from video
- Audio isolation works accurately
- Output maintains video quality
- Processing completes in reasonable time

### User Story 2: The Music Fan
**As a** music fan watching concert videos  
**I want to** isolate the lead singer's voice  
**So that** I can hear their performance clearly over the band and crowd

**Acceptance Criteria:**
- Can identify singer in video
- Singer's voice is isolated clearly
- Background music is removed or minimized
- Audio quality is good enough for listening

### User Story 3: The Student
**As a** student watching educational videos  
**I want to** focus on a specific instructor's audio  
**So that** I can learn from their explanations without distractions

**Acceptance Criteria:**
- Can select instructor from video
- Their audio is isolated clearly
- Other speakers are muted
- Video remains synchronized

### Use Case Scenarios

**Scenario 1: Concert Video Processing**
1. User uploads concert video with multiple performers
2. User selects lead singer using visual interface
3. Tool processes video to isolate singer's audio
4. User downloads video with only singer's voice
5. User creates highlight reel with isolated audio

**Scenario 2: Interview Muting**
1. User has interview video with annoying background speaker
2. User selects background speaker to mute
3. Tool processes video to remove their audio
4. Main interviewer's audio remains intact
5. User shares cleaned interview video

**Scenario 3: Conference Talk Isolation**
1. User records conference with multiple speakers
2. User wants to focus on one speaker's presentation
3. Tool isolates that speaker's audio
4. User creates focused learning content
5. User shares isolated presentation

---

## Success Metrics

### Product Metrics

**Adoption Metrics:**
- Number of videos processed per week
- User retention rate
- Average video length processed
- Fan Mode vs Hater Mode usage ratio

**Quality Metrics:**
- User satisfaction score (1-5 scale)
- Audio isolation accuracy (user feedback)
- Person detection accuracy
- Video processing success rate

**Performance Metrics:**
- Average processing time per minute of video
- Person detection speed (frames per second)
- System uptime/availability

### Technical Metrics

**Model Performance:**
- Person detection accuracy (> 90% target)
- Target person identification accuracy (> 85% target)
- Audio isolation quality (subjective user feedback)
- False positive rate (isolating wrong person)

**System Performance:**
- Memory usage during processing
- GPU utilization efficiency
- Error rate and types
- Frame processing throughput

### Validation Criteria

**MVP Success:**
- Successfully process 10 test videos with multiple people
- Person detection accuracy > 85%
- Audio isolation works in Fan Mode
- Audio muting works in Hater Mode
- Processing time < 20 minutes per minute of video
- User feedback score > 3.5/5.0

**Full Product Success:**
- Process videos with 3+ people accurately
- Person detection accuracy > 90%
- Audio isolation quality rated > 4.0/5.0
- Processing time < 10 minutes per minute of video
- User feedback score > 4.0/5.0

---

## Development Roadmap

### Phase 1: MVP Development (Weeks 1-4)

**Week 1: Foundation & Person Detection**
- Set up development environment
- Initialize git repository
- Set up cloud GPU access
- Implement YOLOv8 person detection
- Test on sample video frames
- Create basic video frame extraction

**Deliverables:**
- Working YOLOv8 person detection
- Frame extraction pipeline
- Test script with sample video

**Week 2: Person Identification & Visual Prompting**
- Implement person selection/identification
- Extract coordinates from bounding boxes
- Integrate SAM Audio model
- Implement visual prompting with coordinates
- Test visual prompting on sample frames
- Create audio extraction pipeline

**Deliverables:**
- Person identification system
- Visual prompting with SAM Audio
- Audio extraction per segment

**Week 3: Audio Processing**
- Implement Fan Mode (isolation)
- Implement Hater Mode (muting)
- Create audio blending logic
- Test audio processing on sample videos
- Implement video-audio synchronization

**Deliverables:**
- Working Fan Mode
- Working Hater Mode
- Audio blending and sync

**Week 4: Integration & Testing**
- Integrate all components
- Build CLI interface
- Test end-to-end pipeline
- Fix bugs and optimize
- Document usage instructions

**Deliverables:**
- Complete MVP working on test videos
- CLI tool for processing
- Basic documentation

### Phase 2: Enhancement (Weeks 5-6)

**Week 5: Quality & Robustness**
- Improve person detection accuracy
- Optimize visual prompting
- Enhance audio blending
- Add progress indicators
- Improve error handling

**Deliverables:**
- Improved accuracy and quality
- Better user experience
- Performance optimizations

**Week 6: Testing & Refinement**
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
- Multi-person tracking
- Real-time processing
- Web interface with visual person selection
- Confidence thresholds
- Batch processing
- Integration with video editing software

**Timeline:** TBD based on MVP success and user feedback

---

## Risks & Mitigations

### Technical Risks

**Risk 1: Person Detection Accuracy**
- **Description:** YOLOv8 may miss people or misidentify targets
- **Impact:** High - Core functionality depends on accurate detection
- **Mitigation:**
  - Test on diverse videos early
  - Consider face recognition for specific person ID
  - Allow manual person selection as fallback
  - Fine-tune detection thresholds

**Risk 2: Visual Prompting Effectiveness**
- **Description:** SAM Audio visual prompting may not work as expected
- **Impact:** High - Core audio isolation depends on this
- **Mitigation:**
  - Test visual prompting extensively
  - Experiment with coordinate formats
  - Consider text prompts as supplement
  - Have fallback to audio-only methods

**Risk 3: Audio-Video Synchronization**
- **Description:** Processed audio may drift from video timing
- **Impact:** Medium - Affects user experience
- **Mitigation:**
  - Careful timestamp tracking
  - Test synchronization on various videos
  - Implement sync correction algorithms
  - User feedback for sync issues

**Risk 4: Processing Speed**
- **Description:** Combining vision and audio AI may be very slow
- **Impact:** Medium - Users expect reasonable processing times
- **Mitigation:**
  - Optimize frame extraction rate
  - Use GPU acceleration
  - Process in parallel where possible
  - Consider caching detected persons

**Risk 5: Complex Scenes**
- **Description:** Videos with many people or rapid movement may fail
- **Impact:** Medium - Limits use cases
- **Mitigation:**
  - Test on complex videos early
  - Implement robust tracking
  - Allow manual intervention
  - Set realistic expectations

### Product Risks

**Risk 6: Limited Use Cases**
- **Description:** May only work well for specific video types
- **Impact:** Medium - Limits market appeal
- **Mitigation:**
  - Test on diverse content early
  - Be transparent about limitations
  - Iterate based on user feedback
  - Focus on high-value use cases first

**Risk 7: User Experience Complexity**
- **Description:** Selecting target person may be confusing
- **Impact:** Medium - Affects adoption
- **Mitigation:**
  - Create simple CLI for MVP
  - Build visual UI in future iterations
  - Provide clear documentation
  - Include example videos

### Operational Risks

**Risk 8: Resource Intensity**
- **Description:** Requires both vision and audio models (high compute)
- **Impact:** Medium - May limit free tier usage
- **Mitigation:**
  - Use efficient models (YOLOv8 nano)
  - Optimize processing pipeline
  - Consider model quantization
  - Use cloud GPU resources efficiently

---

## Solo Developer Considerations

### Resource Constraints

**Time:**
- More complex than Stadium Mode (vision + audio)
- Focus on MVP with one mode first (Fan Mode)
- Set realistic 4-week timeline for MVP
- Iterate based on results

**Budget:**
- Requires more GPU time (two models)
- Use free cloud resources efficiently
- Consider smaller models for speed
- Batch process when possible

**Skills:**
- Need computer vision knowledge
- Video processing experience helpful
- Audio processing knowledge required
- Python and ML fundamentals assumed

### Prioritization Strategy

**Must Have (P0):**
- Person detection in video
- Visual prompting with SAM Audio
- Fan Mode (isolation)
- Video output

**Nice to Have (P1):**
- Hater Mode (muting)
- Person selection UI
- Progress indicators
- Error handling

**Future (P2):**
- Multi-person tracking
- Real-time processing
- Advanced features

### MVP Approach

**MVP Scope:**
- Process single video file
- Detect people using YOLOv8
- User selects target person (CLI input)
- Isolate target person's audio (Fan Mode)
- Output processed video
- Basic error handling

**What to Defer:**
- Hater Mode (add in Phase 2)
- Visual person selection UI
- Multi-person tracking
- Real-time processing
- Batch processing

### Development Workflow

1. **Start with Vision:** Get YOLOv8 working on sample videos
2. **Add Audio:** Integrate SAM Audio visual prompting
3. **Test Integration:** Ensure vision + audio work together
4. **Build Pipeline:** Create end-to-end processing
5. **Iterate:** Improve accuracy and speed

### Cost Management

**Free Resources:**
- Google Colab (free GPU tier)
- Kaggle Notebooks (free GPU)
- Hugging Face (free model access)
- YOLOv8 (open source)

**Potential Costs:**
- Cloud storage for videos
- Paid GPU if free tier insufficient (~$0.20-0.50/hour)
- More GPU time needed (two models)

**Budget Estimate:**
- MVP Development: $0-30 (using free resources, some paid GPU)
- Testing/Demo: $0-50
- Future scaling: TBD

---

## Open Questions

1. **Person Identification:** Should we use face recognition or simple bounding box selection?
2. **Frame Rate:** What frame extraction rate balances accuracy vs speed?
3. **Visual Prompting:** What coordinate format works best with SAM Audio?
4. **Audio Blending:** How to smooth transitions between processed segments?
5. **Multi-Person:** Should MVP support multiple targets or single person only?

---

## References & Resources

- [Meta SAM Audio Paper](https://arxiv.org/abs/2312.06663)
- [SAM Audio Hugging Face Model](https://huggingface.co/facebook/sam-audio-large)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Video Processing](https://docs.opencv.org/)
- [Visual Prompting Research](https://arxiv.org/abs/2312.06663)

---

## Appendix: Technical Implementation Notes

### Person Detection Code Snippet
```python
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # nano for speed

# Detect people in frame
results = model(frame)
persons = [box for box in results.boxes if box.cls == 0]  # class 0 = person

# Extract coordinates
for person in persons:
    x1, y1, x2, y2 = person.xyxy[0]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
```

### Visual Prompting Code Snippet
```python
# Pass frame image and coordinates to SAM Audio
inputs = processor(
    audio=audio_segment,
    images=frame_image,
    input_points=[[[center_x, center_y]]],
    sampling_rate=sr
)

# Get audio mask for target person
with torch.no_grad():
    outputs = model(inputs)
    person_audio_mask = outputs['masks'][0]
```

---

**Document Status:** Ready for Review  
**Next Steps:** Begin Phase 1 development after preparation guide review

