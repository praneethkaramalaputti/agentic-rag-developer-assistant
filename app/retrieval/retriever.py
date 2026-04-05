from app.vectorstore.chroma_store import collection
from app.embeddings.embedder import embed_query


def retrieve_context(query: str, top_k: int = 4, source: str = None):
    query_embedding = embed_query(query)

    if source:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"source": source}
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    return results


def get_document_chunks(source: str):
    results = collection.get(
        where={"source": source},
        include=["documents", "metadatas"]
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    paired = list(zip(documents, metadatas))

    paired.sort(key=lambda x: x[1].get("page_number", 0))

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "ordered_pairs": paired
    }