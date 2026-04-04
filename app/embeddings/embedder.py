from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts).tolist()

def embed_query(query: str):
    return model.encode([query])[0].tolist()