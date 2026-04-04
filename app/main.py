from fastapi import FastAPI, UploadFile, File
import os

from app.loaders.pdf_loader import load_pdf
from app.chunking.text_splitter import chunk_documents
from app.embeddings.embedder import embed_texts
from app.vectorstore.chroma_store import store_chunks
from app.retrieval.retriever import retrieve_context
from app.llm.generator import generate_answer

app = FastAPI()

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

@app.get("/query")
def query_docs(query: str):
    retrieved = retrieve_context(query)
    answer = generate_answer(query, retrieved)

    citations = [
        {
            "source": meta["source"],
            "page_number": meta["page_number"]
        }
        for meta in retrieved["metadatas"][0]
    ]

    return {
        "answer": answer,
        "citations": citations
    }