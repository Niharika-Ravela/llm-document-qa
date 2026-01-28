# qa_engine.py
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
import ollama

from load_docs import read_pdf, chunk_text
from embed_store import embed_texts, build_faiss_index, search_similar_chunks


def build_index_for_pdf(pdf_path: str):
    """
    Reads a PDF, chunks it, creates embeddings, and builds a FAISS index.
    Returns (chunks, index).
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_file}")

    print(f"Reading PDF from: {pdf_file}")
    full_text = read_pdf(str(pdf_file))
    print(f"Total characters in document: {len(full_text)}")

    chunks = chunk_text(full_text, chunk_size=500, chunk_overlap=100)
    print(f"Number of chunks created: {len(chunks)}")

    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts, model_name="nomic-embed-text")
    index = build_faiss_index(embeddings)

    return chunks, index


def build_context_from_results(results: List[Tuple[int, float, str]]) -> str:
    """
    Takes search results and combines their text into a single context string.
    """
    context_parts = []
    for chunk_id, distance, text in results:
        context_parts.append(f"[Chunk {chunk_id}, distance={distance:.4f}]\n{text}")

    return "\n\n---\n\n".join(context_parts)


def ask_llm(question: str, context: str, model_name: str = "llama3") -> str:
    """
    Sends the context + question to LLaMA 3 using Ollama and returns the answer.
    """
    prompt = f"""
You are an assistant that answers questions about invoices or financial documents.

Use ONLY the information in the CONTEXT below. If the answer is not present in the context, say you cannot find it.

CONTEXT:
{context}

QUESTION: {question}

Answer in one or two clear sentences.
"""

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Depending on Ollama's Python client, the answer is usually here:
    return response["message"]["content"]


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "data" / "sample.pdf"

    # 1) Build index for this PDF
    chunks, index = build_index_for_pdf(str(pdf_path))

    # 2) Ask a question (you can change this)
    question = "What is the total due amount on this invoice?"

    # 3) Retrieve top chunks relevant to the question
    results = search_similar_chunks(
        query=question,
        chunks=chunks,
        index=index,
        model_name="nomic-embed-text",
        top_k=2
    )

    print("\n=== Retrieved Chunks ===")
    for cid, dist, text in results:
        print(f"\nChunk ID: {cid}, Distance: {dist:.4f}")
        print(text[:300])

    # 4) Build a context string from those chunks
    context = build_context_from_results(results)

    # 5) Ask LLaMA 3 for the final answer
    print("\n=== LLaMA 3 Answer ===")
    answer = ask_llm(question, context, model_name="llama3")
    print(answer)
