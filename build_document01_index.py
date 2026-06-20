import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DOCUMENT_FILE = r"D:\llama\data\arabic_document01.txt"
OUTPUT_DIR = r"D:\llama\rag_index"

CHUNKS_FILE = os.path.join(OUTPUT_DIR, "chunks01.json")
FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss01.index")

# Good multilingual embedding model for Arabic + English
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def clean_arabic_text(text):
    # Remove repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove Tatweel
    text = text.replace("ـ", "")

    return text.strip()


def split_into_paragraphs(text):
    paragraphs = []

    for part in text.split("\n"):
        part = part.strip()
        if part:
            paragraphs.append(part)

    return paragraphs


def build_chunks(paragraphs, max_chars=1200, overlap_chars=200):
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= max_chars:
            current += "\n" + paragraph
        else:
            if current.strip():
                chunks.append(current.strip())

            # Add small overlap from previous chunk
            overlap = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = overlap + "\n" + paragraph

    if current.strip():
        chunks.append(current.strip())

    return chunks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DOCUMENT_FILE):
        raise FileNotFoundError(f"Document file not found: {DOCUMENT_FILE}")

    print("Reading Arabic document...")
    with open(DOCUMENT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_arabic_text(text)
    paragraphs = split_into_paragraphs(text)
    chunks = build_chunks(paragraphs)

    print(f"Paragraphs: {len(paragraphs)}")
    print(f"Chunks: {len(chunks)}")

    chunk_records = []
    for i, chunk in enumerate(chunks):
        chunk_records.append({
            "id": i,
            "text": chunk
        })

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Encoding chunks...")
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    print("Building FAISS index...")
    index.add(embeddings)

    print("Saving files...")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False, indent=2)

    faiss.write_index(index, FAISS_INDEX_FILE)

    print("Done.")
    print("Chunks saved to:", CHUNKS_FILE)
    print("Index saved to:", FAISS_INDEX_FILE)


if __name__ == "__main__":
    main()

