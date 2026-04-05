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
1. Summarize only from the provided document content.
2. Write a concise and accurate summary of the document.
3. Focus on:
   - what the document is about
   - main sections or themes
   - important technical or business points
4. Do not invent details.
5. Do not merge unrelated ideas.
6. If the content looks like a resume, summarize it as a professional profile, experience, skills, and projects summary.

Document Content:
{context}
"""

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "system",
                "content": "You summarize documents accurately using only the provided document content."
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
def generate_comparison(query: str, retrieved_docs):
    context, docs, metas = build_context(retrieved_docs)

    if not docs:
        return "I could not find enough relevant content to perform a comparison."

    prompt = f"""
You are a document comparison assistant.

Rules:
1. Use only the provided context.
2. Compare the relevant items, sections, or documents based on the user's query.
3. Highlight key similarities, differences, and notable points.
4. Be concise and structured.
5. If there is not enough information to compare, reply exactly:
I could not find enough information in the document to make a comparison.

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
                "content": "You compare information only from the provided document context."
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()