# Agentic RAG Developer Assistant
# Full end to end RAG pipeline

An AI-powered document assistant for querying, summarizing, comparing, and extracting action items from long-form documents using Retrieval-Augmented Generation (RAG).

## Features
- Upload and index PDF documents
- Ask grounded questions from document content
- Summarize a selected document
- Extract action items from a selected document
- Compare sections within a selected document
- Filter by document source
- Local LLM inference with Ollama
- Local vector storage with ChromaDB

## Tech Stack
- Python
- FastAPI
- Ollama
- ChromaDB
- Sentence Transformers
- LangChain text splitting
- Pydantic

## Project Structure
```text
app/
  agents/
  chunking/
  embeddings/
  llm/
  loaders/
  retrieval/
  schemas/
  vectorstore/
  main.py
data/
tests/
README.md
requirements.txt
