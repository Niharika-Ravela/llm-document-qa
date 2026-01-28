# load_docs.py
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

def read_pdf(file_path: str) -> str:
    """
    Reads a PDF file and returns all text as a single string.
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    reader = PdfReader(pdf_path)
    all_text = []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"Error reading page {page_num}: {e}")
            text = ""
        all_text.append(text)

    return "\n".join(all_text)

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Splits text into overlapping chunks.

    Example:
      chunk_size = 500, overlap = 100
      Chunk 1: characters 0-500
      Chunk 2: characters 400-900
      Chunk 3: characters 800-1300
      ...
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        chunk = {
            "id": len(chunks),
            "text": chunk_text,
            "start": start,
            "end": end,
        }
        chunks.append(chunk)

        # move start forward by (chunk_size - overlap)
        start += chunk_size - chunk_overlap

        if start >= text_length:
            break

    return chunks

if __name__ == "__main__":
    pdf_path = "C:\\Users\\nihar\\OneDrive\\Desktop\\Projects\\llm-dcument-qa\\data\\sample.pdf"

    print(f"Reading PDF from: {pdf_path}")
    full_text = read_pdf(pdf_path)

    print(f"Total characters in document: {len(full_text)}")

    chunks = chunk_text(full_text, chunk_size=500, chunk_overlap=100)
    print(f"Number of chunks created: {len(chunks)}")

    # Show a preview of the first chunk
    if chunks:
        first_chunk = chunks[0]
        print("\n=== First chunk (id=0) ===")
        print(first_chunk["text"])
        print(f"\nChunk range: {first_chunk['start']} - {first_chunk['end']}")
