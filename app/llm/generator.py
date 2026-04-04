import ollama


def generate_answer(query: str, retrieved_docs):
    docs = retrieved_docs.get("documents", [[]])[0]
    metas = retrieved_docs.get("metadatas", [[]])[0]

    if not docs:
        return "I could not find the answer in the document."

    context = "\n\n".join(
        [
            f"Source: {meta['source']} | Page: {meta['page_number']}\n{doc}"
            for doc, meta in zip(docs, metas)
        ]
    )

    prompt = f"""
You are a precise document question-answering assistant.

Rules:
1. Use only the provided context.
2. Answer only the user's current question.
3. Be concise and direct.
4. Do not invent details.
5. If the answer is not present, reply exactly:
I could not find the answer in the document.

Question:
{query}

Context:
{context}
"""

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "system",
                "content": "You answer questions only from the provided document context."
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()