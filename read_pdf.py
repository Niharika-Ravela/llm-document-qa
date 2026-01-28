# read_pdf.py
from pathlib import Path
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

if __name__ == "__main__":
    pdf_file = "C:\\Users\\nihar\\OneDrive\\Desktop\\Projects\\llm-dcument-qa\\data\\sample.pdf"  # make sure this file exists
    text = read_pdf(pdf_file)

    print("=== First 500 characters of extracted text ===")
    print(text[:500])
    print("\n=== Total characters extracted:", len(text))
