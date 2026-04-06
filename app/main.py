from fastapi import FastAPI, UploadFile, File, HTTPException
import os

from app.agents.router import detect_mode
from app.schemas.response_schema import QueryResponse
from app.llm.generator import (
    generate_answer,
    generate_summary,
    generate_action_items,
    generate_comparison,
)
from app.loaders.pdf_loader import load_pdf
from app.chunking.text_splitter import chunk_documents
from app.embeddings.embedder import embed_texts
from app.vectorstore.chroma_store import store_chunks
from app.retrieval.retriever import retrieve_context, get_document_chunks

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Agentic RAG Developer Assistant is running",
        "docs": "Open /docs to test the API",
    }


UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def format_results(retrieved):
    docs = retrieved.get("documents", [[]])[0]
    metas = retrieved.get("metadatas", [[]])[0]

    return [
        {
            "source": meta["source"],
            "page_number": meta["page_number"],
            "text": doc,
        }
        for doc, meta in zip(docs, metas)
    ]


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        pages = load_pdf(file_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.")

    if not pages:
        raise HTTPException(status_code=400, detail="No readable content found in the PDF.")

    chunks = chunk_documents(pages)

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not create chunks from the uploaded PDF.")

    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    store_chunks(chunks, embeddings)

    return {"message": f"{file.filename} uploaded and indexed successfully"}


@app.get("/summarize", response_model=QueryResponse)
def summarize_doc(source: str):
    mode = "summarize"
    retrieved = get_document_chunks(source)
    answer = generate_summary(retrieved)

    paired = retrieved.get("ordered_pairs", [])
    results = [
        {
            "source": meta["source"],
            "page_number": meta["page_number"],
            "text": doc,
        }
        for doc, meta in paired
    ]

    return {
        "query": f"Summarize document: {source}",
        "mode": mode,
        "answer": answer,
        "matched_chunks": results,
    }

@app.get("/summarize", response_model=QueryResponse)
def summarize_doc(source: str):
    mode = "summarize"
    retrieved = get_document_chunks(source)
    answer = generate_summary(retrieved)

    paired = retrieved.get("ordered_pairs", [])
    results = [
        {
            "source": meta["source"],
            "page_number": meta["page_number"],
            "text": doc,
        }
        for doc, meta in paired
    ]

    return {
        "query": f"Summarize document: {source}",
        "mode": mode,
        "answer": answer,
        "matched_chunks": results,
    }


@app.get("/action-items", response_model=QueryResponse)
def extract_action_items(source: str):
    mode = "extract_actions"
    retrieved = get_document_chunks(source)
    answer = generate_action_items(retrieved)

    paired = retrieved.get("ordered_pairs", [])
    results = [
        {
            "source": meta["source"],
            "page_number": meta["page_number"],
            "text": doc,
        }
        for doc, meta in paired
    ]

    return {
        "query": f"Extract action items from document: {source}",
        "mode": mode,
        "answer": answer,
        "matched_chunks": results,
    }


@app.get("/compare", response_model=QueryResponse)
def compare_docs(query: str, source: str):
    mode = "compare"
    retrieved = get_document_chunks(source)
    answer = generate_comparison(query, retrieved)

    paired = retrieved.get("ordered_pairs", [])
    results = [
        {
            "source": meta["source"],
            "page_number": meta["page_number"],
            "text": doc,
        }
        for doc, meta in paired
    ]

    return {
        "query": query,
        "mode": mode,
        "answer": answer,
        "matched_chunks": results,
    }
@app.get("/sources")
def list_sources():
    all_results = retrieve_context("document", top_k=100)
    metas = all_results.get("metadatas", [[]])[0]

    sources = []
    seen = set()

    for meta in metas:
        source = meta.get("source")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)

    return {"sources": sorted(sources)}