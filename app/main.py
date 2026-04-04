from app.agents.router import detect_mode
from app.schemas.response_schema import QueryResponse
from fastapi import FastAPI, UploadFile, File
from app.llm.generator import generate_answer
from app.agents.router import detect_mode
from app.schemas.response_schema import QueryResponse
import os

from app.loaders.pdf_loader import load_pdf
from app.chunking.text_splitter import chunk_documents
from app.embeddings.embedder import embed_texts
from app.vectorstore.chroma_store import store_chunks
from app.retrieval.retriever import retrieve_context

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Agentic RAG Developer Assistant is running",
        "docs": "Open /docs to test the API"
    }
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    pages = load_pdf(file_path)
    chunks = chunk_documents(pages)
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    store_chunks(chunks, embeddings)

    return {"message": f"{file.filename} uploaded and indexed successfully"}

@app.get("/query", response_model=QueryResponse)
def query_docs(query: str):
    mode = detect_mode(query)
    retrieved = retrieve_context(query)
    answer = generate_answer(query, retrieved)

    docs = retrieved.get("documents", [[]])[0]
    metas = retrieved.get("metadatas", [[]])[0]

    results = [
        {
            "source": meta["source"],
            "page_number": meta["page_number"],
            "text": doc
        }
        for doc, meta in zip(docs, metas)
    ]

    return {
        "query": query,
        "mode": mode,
        "answer": answer,
        "matched_chunks": results
    }

@app.get("/summarize", response_model=QueryResponse)
def summarize_doc(query: str = "Summarize this document"):
    mode = "summarize"
    retrieved = retrieve_context(query)
    answer = generate_answer(query, retrieved)

    docs = retrieved.get("documents", [[]])[0]
    metas = retrieved.get("metadatas", [[]])[0]

    results = [
        {
            "source": meta["source"],
            "page_number": meta["page_number"],
            "text": doc
        }
        for doc, meta in zip(docs, metas)
    ]

    return {
        "query": query,
        "mode": mode,
        "answer": answer,
        "matched_chunks": results
    }