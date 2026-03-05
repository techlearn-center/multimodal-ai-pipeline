# Module 05: Audio Transcription

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 04 completed |

---

## Learning Objectives
- Transcribe audio using OpenAI Whisper API
- Handle multiple languages and translation
- Build a meeting notes pipeline with timestamps

---

## 1. Whisper Transcription

```python
from openai import OpenAI
client = OpenAI()

with open("data/audio/meeting.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

for segment in transcript.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
```

---

## 2. Translation

```python
with open("data/audio/spanish.mp3", "rb") as f:
    translation = client.audio.translations.create(model="whisper-1", file=f)
print(translation.text)  # Always returns English
```

---

## 3. Meeting Notes Pipeline

```python
def process_meeting(audio_path: str) -> dict:
    client = OpenAI()
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract: Summary, Key Decisions, Action Items (who/what/deadline), Open Questions."},
            {"role": "user", "content": transcript.text},
        ],
    )
    return {"transcript": transcript.text, "notes": response.choices[0].message.content}
```

---

## 4. Handling Large Files

```python
from pydub import AudioSegment

def split_audio(audio_path: str, chunk_minutes: int = 10):
    audio = AudioSegment.from_file(audio_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i in range(0, len(audio), chunk_ms):
        chunk_path = f"/tmp/chunk_{i // chunk_ms}.mp3"
        audio[i:i + chunk_ms].export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    return chunks
```

---

## 5. Hands-On Lab

Build a meeting transcription service: audio upload, auto-split, transcribe with timestamps, generate structured notes.

## Validation
```bash
bash modules/05-audio-transcription/validation/validate.sh
```

**Next: [Module 06 →](../06-multimodal-rag/)**
