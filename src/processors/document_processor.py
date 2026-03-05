"""
Document Processing Pipeline
Extracts text, tables, and structure from PDFs and documents.
"""
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader


class DocumentProcessor:
    """Extract and process content from documents."""

    def __init__(self):
        self.supported_formats = [".pdf", ".docx", ".txt", ".md"]

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        reader = PdfReader(pdf_path)
        result = {
            "filename": Path(pdf_path).name,
            "num_pages": len(reader.pages),
            "metadata": dict(reader.metadata) if reader.metadata else {},
            "pages": [],
        }
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            result["pages"].append({
                "page_number": i + 1,
                "text": text,
                "char_count": len(text),
            })
        return result

    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                last_period = text.rfind(".", start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"chunk_id": chunk_id, "text": chunk_text, "start_char": start, "end_char": end})
                chunk_id += 1
            start = end - overlap
        return chunks

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        results = []
        for ext in self.supported_formats:
            for file_path in Path(dir_path).glob(f"**/*{ext}"):
                if ext == ".pdf":
                    results.append(self.process_pdf(str(file_path)))
                elif ext in (".txt", ".md"):
                    text = file_path.read_text(encoding="utf-8")
                    results.append({"filename": file_path.name, "num_pages": 1, "pages": [{"page_number": 1, "text": text}]})
        return results


if __name__ == "__main__":
    import sys
    processor = DocumentProcessor()
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <pdf_path>")
        sys.exit(1)
    result = processor.process_pdf(sys.argv[1])
    print(f"Pages: {result['num_pages']}")
    for page in result["pages"]:
        print(f"\n--- Page {page['page_number']} ({page['char_count']} chars) ---")
        print(page["text"][:500])
