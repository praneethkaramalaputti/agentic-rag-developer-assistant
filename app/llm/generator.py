from openai import OpenAI
from app.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(query: str, retrieved_docs):
    docs = retrieved_docs["documents"][0]
    metas = retrieved_docs["metadatas"][0]

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

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You answer using only retrieved context."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content