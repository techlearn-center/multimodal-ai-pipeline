# Capstone: Production Multimodal Document Intelligence Platform

## Overview
Build a complete document intelligence system that processes PDFs, images, and text documents using multimodal AI, stores them in a vector database, and provides a natural language query interface.

## Architecture
```
Client --> FastAPI --> [Upload Handler] --> File Classifier
                                              |
                              +---------------+---------------+
                              |               |               |
                         ImageProcessor  DocProcessor    Whisper
                              |               |               |
                              v               v               v
                           Embedder -----> ChromaDB <----- Embedder
                              |
                              v
                      RAG Query Engine --> LLM --> Response with Citations
```

## Requirements
1. **Document Ingestion**: Upload PDFs, images, and text files via REST API
2. **Multimodal Processing**: Extract text via OCR, generate image captions, chunk and embed all content
3. **Vector Storage**: Store in ChromaDB with metadata (source, type, page number, chunk ID)
4. **RAG Query API**: Answer questions using retrieved multimodal context with source citations
5. **Evaluation Dashboard**: Track pipeline quality metrics (caption quality, retrieval precision, latency)
6. **Dockerized Deployment**: All services run via `docker compose up`

## Acceptance Criteria
- [ ] API accepts PDF, image, and text uploads at `/api/upload`
- [ ] Documents are automatically classified and processed
- [ ] Processed content stored in ChromaDB with proper metadata
- [ ] RAG queries return relevant answers with source citations
- [ ] Evaluation endpoint returns quality metrics
- [ ] Health check at `/health` returns service status
- [ ] All services start with `docker compose up`
- [ ] API documentation available at `/docs`
- [ ] Handles errors gracefully (invalid files, API failures)
- [ ] Processing latency < 30s for standard documents

## Getting Started
```bash
cd capstone/starter
cp ../../.env.example .env
# Edit .env with your API keys
docker compose up -d
```

## Validation
```bash
bash capstone/validation/validate.sh
```
