# embed_store.py
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
import ollama

from load_docs import read_pdf, chunk_text


def embed_texts(texts: List[str], model_name: str = "nomic-embed-text") -> np.ndarray:
    """
    Takes a list of texts and returns a 2D numpy array of embeddings.
    Shape: (num_texts, embedding_dim)
    """
    embeddings = []

    for i, t in enumerate(texts):
        t = t.strip()
        if not t:
            # Empty text, use a zero vector later
            print(f"Skipping empty chunk at index {i}")
            embeddings.append(None)
            continue

        print(f"Embedding chunk {i+1}/{len(texts)}...")
        try:
            response = ollama.embeddings(
                model=model_name,
                prompt=t
            )
            emb = response["embedding"]
            embeddings.append(emb)
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")
            embeddings.append(None)

    # Replace any None with zeros (same dimension)
    # First, find the first non-None embedding to get dimension
    valid_emb = next((e for e in embeddings if e is not None), None)
    if valid_emb is None:
        raise ValueError("No valid embeddings were created.")

    dim = len(valid_emb)
    arr = np.zeros((len(embeddings), dim), dtype="float32")

    for i, emb in enumerate(embeddings):
        if emb is not None:
            arr[i] = np.array(emb, dtype="float32")
        # else: stays as zeros

    return arr


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index from a 2D numpy array of embeddings.
    """
    num_rows, dim = embeddings.shape
    print(f"Building FAISS index with {num_rows} vectors of dimension {dim}...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search_similar_chunks(
    query: str,
    chunks: List[Dict],
    index: faiss.IndexFlatL2,
    model_name: str = "nomic-embed-text",
    top_k: int = 2
) -> List[Tuple[int, float, str]]:
    """
    Given a query string, chunk list, and FAISS index:
    - Embed the query
    - Search for top_k closest chunks
    - Return list of (chunk_id, distance, text)
    """
    # Embed the query
    print(f"\nEmbedding query: {query}")
    resp = ollama.embeddings(model=model_name, prompt=query)
    query_emb = np.array(resp["embedding"], dtype="float32").reshape(1, -1)

    # Search in FAISS
    distances, indices = index.search(query_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[int(idx)]
        results.append((int(idx), float(dist), chunk["text"]))

    return results


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "data" / "sample.pdf"

    print(f"Reading PDF from: {pdf_path}")
    full_text = read_pdf(str(pdf_path))
    print(f"Total characters in document: {len(full_text)}")

    chunks = chunk_text(full_text, chunk_size=500, chunk_overlap=100)
    print(f"Number of chunks created: {len(chunks)}")

    # Get just the text from chunks
    chunk_texts = [c["text"] for c in chunks]

    # 1) Create embeddings for all chunks
    embeddings = embed_texts(chunk_texts, model_name="nomic-embed-text")

    # 2) Build a FAISS index
    index = build_faiss_index(embeddings)

    # 3) Ask a test question
    test_query = "What is the total due amount on this invoice?"
    results = search_similar_chunks(
        test_query,
        chunks,
        index,
        model_name="nomic-embed-text",
        top_k=2
    )

    print("\n=== Top matching chunks for query ===")
    for chunk_id, distance, text in results:
        print(f"\nChunk ID: {chunk_id}, Distance: {distance:.4f}")
        print("Text preview:")
        print(text[:400])  # print only first 400 chars for readability
