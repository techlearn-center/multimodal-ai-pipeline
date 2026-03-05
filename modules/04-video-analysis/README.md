# Module 04: Video Analysis

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 03 completed |

---

## Learning Objectives
- Extract frames from videos at configurable intervals
- Analyze video content using vision models
- Build a video summarization pipeline
- Detect scenes and key moments

---

## 1. Frame Extraction

```python
import cv2

def extract_frames(video_path: str, interval_seconds: int = 5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append({"frame": frame, "timestamp": count / fps, "frame_number": count})
        count += 1
    cap.release()
    return frames
```

---

## 2. Video Summarization

```python
from src.processors.image_processor import ImageProcessor
import cv2, tempfile

processor = ImageProcessor()
frames = extract_frames("data/videos/demo.mp4", interval_seconds=10)

captions = []
for f in frames:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, f["frame"])
        caption = processor.caption(tmp.name)
        captions.append({"timestamp": f["timestamp"], "caption": caption})

# Combine into summary
from openai import OpenAI
client = OpenAI()
timeline = "\n".join([f"[{c['timestamp']:.0f}s] {c['caption']}" for c in captions])
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Summarize this video based on frame descriptions."},
        {"role": "user", "content": timeline},
    ],
)
print(response.choices[0].message.content)
```

---

## 3. Scene Detection

```python
import numpy as np

def detect_scene_changes(video_path: str, threshold: float = 30.0):
    cap = cv2.VideoCapture(video_path)
    prev_hist = None
    scenes = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > threshold:
                scenes.append({"timestamp": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, "diff_score": diff})
        prev_hist = hist
    cap.release()
    return scenes
```

---

## 4. Hands-On Lab

Build a video analysis pipeline: frame extraction, scene detection, frame captioning, timestamped summary.

## Validation
```bash
bash modules/04-video-analysis/validation/validate.sh
```

**Next: [Module 05 →](../05-audio-transcription/)**
