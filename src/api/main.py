"""
Multimodal AI Pipeline API
FastAPI endpoints for image analysis, document processing, and multimodal RAG.
"""
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Multimodal AI Pipeline", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    filter_type: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "healthy", "service": "multimodal-pipeline"}


@app.post("/api/image/caption")
async def caption_image(file: UploadFile = File(...)):
    from src.processors.image_processor import ImageProcessor
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    processor = ImageProcessor()
    caption = processor.caption(str(file_path))
    return {"filename": file.filename, "caption": caption}


@app.post("/api/image/analyze")
async def analyze_image(file: UploadFile = File(...), question: str = Form(...)):
    from src.processors.image_processor import ImageProcessor
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    processor = ImageProcessor()
    answer = processor.analyze(str(file_path), question)
    return {"filename": file.filename, "question": question, "answer": answer}


@app.post("/api/document/process")
async def process_document(file: UploadFile = File(...)):
    from src.processors.document_processor import DocumentProcessor
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    processor = DocumentProcessor()
    return processor.process_pdf(str(file_path))


@app.post("/api/rag/query")
async def query_rag(req: QueryRequest):
    from src.pipeline.multimodal_rag import MultimodalRAG
    rag = MultimodalRAG(chroma_host=os.getenv("CHROMA_HOST", "localhost"), chroma_port=int(os.getenv("CHROMA_PORT", "8001")))
    answer = rag.answer(req.question, req.n_results)
    return {"question": req.question, "answer": answer}
