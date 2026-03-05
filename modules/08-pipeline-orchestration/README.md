# Module 08: Pipeline Orchestration

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Modules 01-07 completed |

---

## Learning Objectives
- Build multi-step processing pipelines
- Handle errors, retries, and fallbacks
- Orchestrate parallel processing

---

## 1. Pipeline Architecture

```
Input --> Classifier --> [Image] --> ImageProcessor --> Embedder --> VectorStore
                     --> [PDF]   --> DocProcessor   --> Chunker  --> VectorStore
                     --> [Audio] --> Whisper         --> Chunker  --> VectorStore
```

## 2. Implementation

```python
from pathlib import Path
from typing import Dict, Any

class MultimodalPipeline:
    def __init__(self):
        self.processors = {
            "image": self._process_image,
            "document": self._process_document,
            "audio": self._process_audio,
        }

    def classify_input(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            return "image"
        elif ext in (".pdf", ".docx", ".txt"):
            return "document"
        elif ext in (".mp3", ".wav", ".m4a"):
            return "audio"
        raise ValueError(f"Unsupported: {ext}")

    def process(self, file_path: str) -> Dict[str, Any]:
        input_type = self.classify_input(file_path)
        return self.processors[input_type](file_path)

    def _process_image(self, path):
        from src.processors.image_processor import ImageProcessor
        proc = ImageProcessor()
        return {"type": "image", "caption": proc.caption(path), "text": proc.extract_text(path)}

    def _process_document(self, path):
        from src.processors.document_processor import DocumentProcessor
        proc = DocumentProcessor()
        result = proc.process_pdf(path)
        text = "\n".join([p["text"] for p in result["pages"]])
        chunks = proc.chunk_document(text)
        return {"type": "document", "pages": result["num_pages"], "chunks": chunks}

    def _process_audio(self, path):
        from openai import OpenAI
        with open(path, "rb") as f:
            transcript = OpenAI().audio.transcriptions.create(model="whisper-1", file=f)
        return {"type": "audio", "transcript": transcript.text}
```

---

## 3. Error Handling and Retries

```python
import time
from functools import wraps

def retry(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait = delay * (2 ** attempt)
                    print(f"Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                    time.sleep(wait)
        return wrapper
    return decorator

@retry(max_retries=3, delay=2)
def process_with_retry(pipeline, file_path):
    return pipeline.process(file_path)
```

---

## 4. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_batch(file_paths: list, max_workers: int = 4):
    pipeline = MultimodalPipeline()
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(pipeline.process, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                results[fp] = future.result()
            except Exception as e:
                results[fp] = {"error": str(e)}
    return results
```

---

## 5. Hands-On Lab

Build a unified pipeline that auto-classifies inputs, processes them in parallel, and stores results.

## Validation
```bash
bash modules/08-pipeline-orchestration/validation/validate.sh
```

**Next: [Module 09 →](../09-evaluation-and-testing/)**
