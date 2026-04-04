import ollama


def build_context(retrieved_docs):
    docs = retrieved_docs.get("documents", [[]])[0]
    metas = retrieved_docs.get("metadatas", [[]])[0]

    if not docs:
        return None, [], []

    context = "\n\n".join(
        [
            f"Source: {meta['source']} | Page: {meta['page_number']}\n{doc}"
            for doc, meta in zip(docs, metas)
        ]
    )

    return context, docs, metas


def generate_answer(query: str, retrieved_docs):
    context, docs, metas = build_context(retrieved_docs)

    if not docs:
        return "I could not find the answer in the document."

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


def generate_summary(retrieved_docs):
    context, docs, metas = build_context(retrieved_docs)

    if not docs:
        return "I could not generate a summary because no relevant document content was found."

    prompt = f"""
You are a document summarization assistant.

Rules:
1. Summarize only from the provided context.
2. Write a clear and concise summary.
3. Focus on the main purpose, important points, and notable details.
4. Do not invent anything not present in the context.

Context:
{context}
"""

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "system",
                "content": "You summarize documents only from the provided context."
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()

def generate_action_items(retrieved_docs):
    context, docs, metas = build_context(retrieved_docs)

    if not docs:
        return "I could not find any relevant content to extract action items."

    prompt = f"""
You are an action-item extraction assistant.

Rules:
1. Use only the provided context.
2. Extract clear action items, tasks, or next steps if they exist.
3. Return them as short bullet points.
4. If no action items are present, reply exactly:
No clear action items were found in the document.

Context:
{context}
"""

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "system",
                "content": "You extract action items only from the provided document context."
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()