# Module 09: Evaluation and Testing

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 08 completed |

---

## Learning Objectives
- Evaluate multimodal pipeline quality with LLM-as-judge
- Measure latency and throughput
- Build automated evaluation suites

---

## 1. LLM-as-Judge Evaluation

```python
from openai import OpenAI
import json

client = OpenAI()

def evaluate_caption(generated: str, reference: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Rate the caption on accuracy (1-5), completeness (1-5), fluency (1-5). Return JSON with scores and explanation."},
            {"role": "user", "content": f"Generated: {generated}\nReference: {reference}"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

scores = evaluate_caption(
    "A tall iron tower in Paris at sunset",
    "The Eiffel Tower illuminated at sunset with golden light reflecting off the Seine River"
)
print(f"Accuracy: {scores['accuracy']}/5, Completeness: {scores['completeness']}/5")
```

---

## 2. Retrieval Evaluation

```python
def evaluate_retrieval(questions, expected_sources, rag):
    metrics = {"precision": 0, "recall": 0}
    for q, expected in zip(questions, expected_sources):
        results = rag.query(q)
        retrieved = [r["metadata"].get("source", "") for r in results]
        hits = sum(1 for e in expected if e in retrieved)
        metrics["precision"] += hits / len(retrieved) if retrieved else 0
        metrics["recall"] += hits / len(expected) if expected else 0
    n = len(questions)
    return {k: v / n for k, v in metrics.items()}
```

---

## 3. Latency Benchmarking

```python
import time

def benchmark(func, *args, n_runs=10):
    times = []
    for _ in range(n_runs):
        start = time.time()
        func(*args)
        times.append(time.time() - start)
    times.sort()
    return {
        "avg_ms": sum(times) / len(times) * 1000,
        "p50_ms": times[len(times) // 2] * 1000,
        "p95_ms": times[int(len(times) * 0.95)] * 1000,
        "min_ms": times[0] * 1000,
        "max_ms": times[-1] * 1000,
    }

# Example
from src.processors.image_processor import ImageProcessor
proc = ImageProcessor()
results = benchmark(proc.caption, "data/images/test.jpg", n_runs=5)
print(f"Caption latency: avg={results['avg_ms']:.0f}ms, p95={results['p95_ms']:.0f}ms")
```

---

## 4. Hands-On Lab

Build an evaluation suite: LLM-as-judge for captions, retrieval metrics for RAG, latency benchmarks.

## Validation
```bash
bash modules/09-evaluation-and-testing/validation/validate.sh
```

**Next: [Module 10 →](../10-production-deployment/)**
