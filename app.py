# app.py
import streamlit as st
from pathlib import Path

from qa_engine import build_index_for_pdf, ask_llm
from embed_store import search_similar_chunks


def main():
    st.set_page_config(page_title="Document QA with LLaMA 3", page_icon="📄")
    st.title("📄 LLM Document Q&A (Local RAG)")
    st.write(
        "Upload a PDF (invoice, contract, etc.), then ask a question. "
        "The app will search the document and answer using a local LLaMA 3 model via Ollama."
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Question input
    question = st.text_input("Ask a question about the document:")

    if uploaded_file is not None:
        # Show file details
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size} bytes")

        # Save uploaded file to a temporary path
        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data"
        data_dir.mkdir(exist_ok=True)

        temp_pdf_path = data_dir / "uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"File saved to: {temp_pdf_path}")

        if question:
            if st.button("🔍 Ask LLaMA 3"):
                with st.spinner("Building index and querying the document..."):
                    # 1) Build index for this PDF
                    chunks, index = build_index_for_pdf(str(temp_pdf_path))

                    # 2) Retrieve similar chunks
                    results = search_similar_chunks(
                        query=question,
                        chunks=chunks,
                        index=index,
                        model_name="nomic-embed-text",
                        top_k=3,
                    )

                    # 3) Show retrieved chunks (optional debug/inspection)
                    st.subheader("🔎 Retrieved Chunks (Context Used)")
                    for cid, dist, text in results:
                        with st.expander(f"Chunk {cid} (distance={dist:.4f})"):
                            st.write(text)

                    # 4) Build context and ask LLaMA 3
                    context = ""
                    for cid, dist, text in results:
                        context += f"[Chunk {cid}, distance={dist:.4f}]\n{text}\n\n"

                    st.subheader("🤖 Answer from LLaMA 3")
                    answer = ask_llm(question, context, model_name="llama3")
                    st.write(answer)
    else:
        st.info("Please upload a PDF to get started.")


if __name__ == "__main__":
    main()
