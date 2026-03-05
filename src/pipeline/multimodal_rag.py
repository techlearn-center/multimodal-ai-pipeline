"""
Multimodal RAG Pipeline
Combines text, image, and document understanding for retrieval-augmented generation.
"""
from typing import List, Dict, Any, Optional

import chromadb
from openai import OpenAI


class MultimodalRAG:
    """RAG pipeline that handles text, images, and documents."""

    def __init__(self, collection_name: str = "multimodal_docs", chroma_host: str = "localhost", chroma_port: int = 8001):
        self.client = OpenAI()
        self.chroma = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.collection = self.chroma.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{"type": "text"} for _ in texts]
        embeddings = [self.embed_text(t) for t in texts]
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def add_image_descriptions(self, image_paths: List[str], descriptions: List[str]):
        metadatas = [{"type": "image", "image_path": p} for p in image_paths]
        ids = [f"img_{i}" for i in range(len(image_paths))]
        self.add_documents(descriptions, metadatas, ids)

    def query(self, question: str, n_results: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embed_text(question)
        where_filter = {"type": filter_type} if filter_type else None
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where_filter, include=["documents", "metadatas", "distances"])
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({"text": results["documents"][0][i], "metadata": results["metadatas"][0][i], "distance": results["distances"][0][i]})
        return formatted

    def answer(self, question: str, n_results: int = 5) -> str:
        results = self.query(question, n_results)
        context_parts = []
        for r in results:
            if r["metadata"].get("type") == "image":
                context_parts.append(f"[Image: {r['metadata'].get('image_path', 'unknown')}]\n{r['text']}")
            else:
                context_parts.append(r["text"])
        context = "\n\n---\n\n".join(context_parts)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer the question based on the provided context. The context may include text documents and image descriptions. Cite sources when possible."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
