from app.vectorstore.chroma_store import collection
from app.embeddings.embedder import embed_query

def retrieve_context(query: str, top_k: int = 4):
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results