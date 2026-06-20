# Technical Report: Local Arabic Novel RAG with Qwen3-8B

## 1. Objective

The objective of this experiment is to make a local LLM answer detailed Arabic questions about a specific Arabic novel, **"حكايات التاروت"**, using Retrieval-Augmented Generation. The experiment does not fine-tune the model. Instead, it builds an external searchable knowledge base from the novel text and connects it to a local Qwen3-8B model through a Flask interface.

The main value of the experiment is that it demonstrates how a local model can answer questions that a normal local LLM or an online LLM may not answer correctly because the model does not have access to the exact document.

## 2. Problem Statement

A general-purpose LLM has broad knowledge but cannot reliably answer detailed questions about a specific document unless that document is available in the prompt or in an external retrieval system. Arabic novels are especially challenging because questions may depend on exact character names, events, dialogue, or scenes.

A direct question to a normal LLM may lead to hallucinated answers, vague summaries, incorrect character relationships, English answers although the desired answer is Arabic, or inability to answer because the model does not know the source text.

## 3. Proposed Solution

The proposed solution is a local RAG system. The system stores the Arabic novel as a text file, splits it into smaller chunks, converts each chunk into a vector embedding, and indexes all vectors using FAISS. At question time, the user question is also converted to an embedding. FAISS retrieves the most similar chunks, and those chunks are inserted into the prompt sent to the local LLM.

This means the LLM does not answer from memory only. It answers using the text retrieved from the novel.

## 4. System Architecture

```text
User Browser / Flutter Client
          ↓
Flask RAG API: /ask
          ↓
SentenceTransformer Query Embedding
          ↓
FAISS Similarity Search
          ↓
Top-K Arabic Novel Chunks
          ↓
Prompt Construction
          ↓
llama-server OpenAI-Compatible Endpoint
          ↓
Local Qwen3-8B GGUF Model
          ↓
Arabic Answer returned as HTML or JSON
```

## 5. Indexing Stage

The indexing stage is implemented in `build_document01_index.py`.

### 5.1 Input

The input document is `data/arabic_document01.txt`. In the original Windows configuration, the path is `D:\llama\data\arabic_document01.txt`.

### 5.2 Text Cleaning

The cleaning step removes repeated spaces, normalizes excessive newlines, removes Arabic Tatweel `ـ`, and strips leading and trailing spaces.

### 5.3 Chunking

The document is split into paragraphs, then grouped into chunks. The default chunking strategy is:

```text
max_chars = 1200
overlap_chars = 200
```

The overlap preserves context across chunk boundaries. Without overlap, the answer to a question may be split between two chunks.

### 5.4 Embedding

The system uses:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This multilingual embedding model can represent Arabic and English text in vector form.

### 5.5 FAISS Index

The script uses normalized embeddings and FAISS `IndexFlatIP`. Because embeddings are normalized, inner product search behaves like cosine similarity. The output files are `rag_index/chunks01.json` and `rag_index/faiss01.index`.

## 6. Question Answering Stage

The answering stage is implemented in `pyrun_rag01.py`.

| Route | Purpose |
|---|---|
| `/` | Home page and question form. |
| `/status` | JSON status of the RAG system. |
| `/ask` | Main question-answering endpoint. |

For each question, the Flask app reads `q`, `format`, `top_k`, and `mode`, embeds the question, searches the FAISS index, constructs a prompt using the retrieved passages, sends the prompt to `llama-server`, and returns the answer in Arabic.

## 7. Prompt Design

The prompt instructs the model to answer in Arabic only, use only retrieved passages, avoid inventing events or characters, say that the answer cannot be determined when evidence is missing, keep the answer concise, and avoid showing reasoning.

## 8. Why the System Can Answer Novel-Specific Questions

A normal local LLM may not know the novel details. The RAG system can answer because it retrieves exact passages from the supplied Arabic text file. The model is guided by retrieved evidence rather than pure memory.

## 9. Probabilistic Nature of LLMs

The demo highlights an important behavior: the same question may produce two different correct answers. This happens because LLMs generate text probabilistically. Even with the same context, the model may choose different wording, focus on different details, or summarize the same facts differently.

This does not necessarily mean the model is wrong. Two answers can both be correct if they are supported by the retrieved passages.

To reduce variation, use a low temperature, reduce `top_k` if too many passages confuse the model, make the question more specific, ask the model to cite passage IDs, and use deterministic decoding if supported by the server.

## 10. Evaluation Approach

Suggested evaluation steps:

1. Prepare a list of factual questions from the novel.
2. Run each question using `top_k=3`, `top_k=5`, and `top_k=10`.
3. Check whether the retrieved passages contain the answer.
4. Check whether the generated answer is faithful to the passages.
5. Compare direct local LLM answers with RAG answers.
6. Mark each answer as correct and grounded, partially correct, wrong retrieval, hallucinated, or cannot be determined.

## 11. Strengths and Limitations

Strengths: fully local workflow, Arabic document support, HTML and JSON output, Flutter-ready API, no fine-tuning required, and easy document replacement.

Limitations: it does not permanently train the model, requires good OCR/text quality, retrieval can fail if wording is very different, and the model may still hallucinate if prompt or retrieval quality is weak.

## 12. Reproducibility Checklist

```text
[ ] Put the model in D:\llama\models\
[ ] Put arabic_document01.txt in D:\llama\data\
[ ] Install requirements
[ ] Run build_document01_index.py
[ ] Start llama-server on port 8080
[ ] Start pyrun_rag01.py on port 5000
[ ] Open http://127.0.0.1:5000
[ ] Ask questions from examples/questions.txt
```

## 13. Conclusion

This project is a proof-of-concept showing that a local LLM can be connected to private Arabic data through RAG. The result is a custom Arabic question-answering assistant that can answer detailed questions about a specific novel without modifying the model weights.
