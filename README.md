# LLM Document Q&A (Local RAG with LLaMA 3)

This project implements a local Retrieval-Augmented Generation (RAG) system that allows users to upload PDFs
(invoices, contracts, financial documents) and ask natural language questions.

The system extracts text from documents, splits it into chunks, generates embeddings for semantic search, and retrieves
the most relevant content to provide grounded answers using a locally hosted LLaMA 3 model via Ollama.

This application runs fully offline, requires no cloud services or paid APIs, and is designed for privacy-safe
document intelligence use cases.
