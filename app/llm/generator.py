import ollama

def generate_answer(query: str, retrieved_docs):
    docs = retrieved_docs.get("documents", [[]])[0]
    metas = retrieved_docs.get("metadatas", [[]])[0]

    if not docs:
        return "I could not find enough relevant context in the uploaded document."

    context = "\n\n".join(
        [
            f"Source: {meta['source']} | Page: {meta['page_number']}\n{doc}"
            for doc, meta in zip(docs, metas)
        ]
    )

    prompt = f"""
You are a grounded AI assistant.
Answer only from the provided context.
If the answer is not in the context, say you could not find enough evidence.

Question:
{query}

Context:
{context}
"""

    response = ollama.chat(
        model="phi3",
        messages=[
            {"role": "system", "content": "You answer using only retrieved context."},
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]