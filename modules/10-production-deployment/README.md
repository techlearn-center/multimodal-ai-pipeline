# Module 10: Production Deployment

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 09 completed |

---

## Learning Objectives
- Deploy multimodal pipeline with Docker Compose
- Add health checks, rate limiting, and monitoring
- Scale with async processing and queues

---

## 1. Production Docker Compose

```yaml
services:
  app:
    build: .
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 2

  chromadb:
    image: chromadb/chroma:latest
    restart: always
    volumes:
      - chroma-data:/chroma/chroma

  redis:
    image: redis:7-alpine
    restart: always
```

---

## 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/image/caption")
@limiter.limit("10/minute")
async def caption_image(request: Request, file: UploadFile = File(...)):
    ...
```

---

## 3. Async Processing with Redis Queues

```python
import redis
import json

r = redis.Redis(host="redis", port=6379)

def enqueue_processing(file_path: str, task_type: str):
    task_id = r.incr("task_counter")
    task = {"file_path": file_path, "type": task_type, "status": "pending"}
    r.set(f"task:{task_id}", json.dumps(task))
    r.lpush("processing_queue", task_id)
    return task_id

def worker():
    """Background worker that processes queued tasks."""
    while True:
        _, task_id = r.brpop("processing_queue")
        task = json.loads(r.get(f"task:{task_id}"))
        task["status"] = "processing"
        r.set(f"task:{task_id}", json.dumps(task))
        # Process based on type...
        task["status"] = "completed"
        r.set(f"task:{task_id}", json.dumps(task))
```

---

## 4. Monitoring with Prometheus

```python
from prometheus_client import Counter, Histogram
import time

REQUEST_COUNT = Counter("requests_total", "Total requests", ["endpoint"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    LATENCY.labels(request.url.path).observe(time.time() - start)
    REQUEST_COUNT.labels(request.url.path).inc()
    return response
```

---

## 5. Hands-On Lab

Deploy the full pipeline with health checks, rate limiting, Redis queues, and Prometheus metrics.

## Validation
```bash
bash modules/10-production-deployment/validation/validate.sh
```

**Next: [Capstone Project →](../../capstone/)**
