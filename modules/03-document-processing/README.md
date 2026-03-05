# Module 03: Document Processing

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 02 completed |

---

## Learning Objectives
- Extract text and structure from PDFs
- Chunk documents for RAG with different strategies
- Handle multi-format documents (PDF, DOCX, images)

---

## 1. PDF Processing

```python
from src.processors.document_processor import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_pdf("data/documents/sample.pdf")

print(f"Pages: {result['num_pages']}")
for page in result['pages']:
    print(f"Page {page['page_number']}: {page['char_count']} chars")
    print(page['text'][:200])
```

### Advanced: Unstructured Library
```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="data/documents/report.pdf",
    strategy="hi_res",
    infer_table_structure=True,
)
for element in elements:
    print(f"Type: {type(element).__name__}, Text: {str(element)[:100]}")
```

---

## 2. Chunking Strategies

| Strategy | Best For | Chunk Quality | Speed |
|----------|---------|---------------|-------|
| Fixed-size | Simple text | Low | Fast |
| Header-based | Structured docs | High | Fast |
| Recursive | General-purpose | Medium-High | Fast |
| Semantic | Mixed content | Highest | Slow |

```python
# Recursive (best general-purpose)
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(full_text)
```

---

## 3. Hands-On Lab

Build a pipeline: PDF upload -> text extraction -> chunking -> ChromaDB storage.

## Validation
```bash
bash modules/03-document-processing/validation/validate.sh
```

**Next: [Module 04 →](../04-video-analysis/)**
