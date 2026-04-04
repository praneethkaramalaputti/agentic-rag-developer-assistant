from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []

    for page in pages:
        split_texts = splitter.split_text(page["text"])
        for idx, chunk in enumerate(split_texts):
            chunks.append({
                "chunk_id": f'{page["source"]}_p{page["page_number"]}_c{idx}',
                "text": chunk,
                "page_number": page["page_number"],
                "source": page["source"]
            })

    return chunks