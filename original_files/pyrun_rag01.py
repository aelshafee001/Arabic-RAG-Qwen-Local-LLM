from flask import Flask, request, jsonify
import requests
import html
import os
import json
import re
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ============================================================
# Configuration
# ============================================================

LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

PROFILE_FILE = r"D:\llama\profile01.txt"
CHUNKS_FILE = r"D:\llama\rag_index\chunks01.json"
FAISS_INDEX_FILE = r"D:\llama\rag_index\faiss01.index"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_TOP_K = 5
MAX_TOP_K = 10

FAST_MAX_TOKENS = 250
NORMAL_MAX_TOKENS = 400
DETAILED_MAX_TOKENS = 700

LLAMA_TIMEOUT = 600

# ============================================================
# Global RAG resources
# ============================================================

embedding_model = None
faiss_index = None
chunks = None

RAG_READY = False
RAG_LOAD_ERROR = ""


# ============================================================
# Basic routes
# ============================================================

@app.route("/favicon.ico")
def favicon():
    return "", 204


# ============================================================
# Utility functions
# ============================================================

def read_file(path):
    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_profile():
    profile = read_file(PROFILE_FILE).strip()

    if not profile:
        profile = (
            "The user is working on local LLM experiments using Qwen3-8B, "
            "llama.cpp, Flask, Flutter, and Arabic RAG."
        )

    return profile


def load_rag_resources():
    """
    Load the embedding model, document chunks, and FAISS index when the server starts.
    This prevents timeout on the first question.
    """
    global embedding_model, faiss_index, chunks, RAG_READY, RAG_LOAD_ERROR

    try:
        print("=" * 70)
        print("Starting Arabic Document RAG Flask Server")
        print("=" * 70)

        if not os.path.exists(CHUNKS_FILE):
            raise FileNotFoundError(
                f"Chunks file not found: {CHUNKS_FILE}. "
                "Run build_document_index.py first."
            )

        if not os.path.exists(FAISS_INDEX_FILE):
            raise FileNotFoundError(
                f"FAISS index file not found: {FAISS_INDEX_FILE}. "
                "Run build_document_index.py first."
            )

        print("Loading embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")

        print("Loading chunks...")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Chunks loaded: {len(chunks)}")

        print("Loading FAISS index...")
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        print("FAISS index loaded.")

        RAG_READY = True
        RAG_LOAD_ERROR = ""

        print("=" * 70)
        print("RAG system is ready.")
        print("Open: http://127.0.0.1:5000")
        print("=" * 70)

    except Exception as e:
        RAG_READY = False
        RAG_LOAD_ERROR = str(e)

        print("=" * 70)
        print("RAG system failed to load.")
        print(RAG_LOAD_ERROR)
        print("=" * 70)


def get_safe_top_k():
    raw_top_k = request.args.get("top_k", str(DEFAULT_TOP_K)).strip()

    try:
        top_k = int(raw_top_k)
    except ValueError:
        top_k = DEFAULT_TOP_K

    if top_k < 1:
        top_k = 1

    if top_k > MAX_TOP_K:
        top_k = MAX_TOP_K

    return top_k


def get_answer_mode_settings():
    mode = request.args.get("mode", "fast").strip().lower()

    if mode == "detailed":
        return "detailed", DETAILED_MAX_TOKENS

    if mode == "normal":
        return "normal", NORMAL_MAX_TOKENS

    return "fast", FAST_MAX_TOKENS


def contains_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))


def count_english_words(text):
    return len(re.findall(r"[A-Za-z]{3,}", text))


def count_arabic_words(text):
    return len(re.findall(r"[\u0600-\u06FF]+", text))


def answer_looks_english(answer):
    """
    Detect if the model answered mostly in English.
    If yes, a second request rewrites the answer into Arabic.
    """
    english_count = count_english_words(answer)
    arabic_count = count_arabic_words(answer)

    if english_count > arabic_count:
        return True

    if not contains_arabic(answer) and english_count > 5:
        return True

    return False


def force_arabic_answer(answer):
    """
    Second-pass correction:
    If Qwen answers in English, rewrite the same answer in Arabic.
    """
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "أنت مساعد عربي. مهمتك الوحيدة هي إعادة صياغة النص التالي باللغة العربية فقط. "
                    "لا تضف معلومات جديدة. لا تشرح. لا تستخدم الإنجليزية. "
                    "أعد كتابة الإجابة فقط باللغة العربية الفصحى الواضحة."
                )
            },
            {
                "role": "user",
                "content": answer
            }
        ],
        "temperature": 0.1,
        "max_tokens": 400
    }

    response = requests.post(
        LLAMA_SERVER_URL,
        json=payload,
        timeout=LLAMA_TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    corrected = data["choices"][0]["message"].get("content", "").strip()

    if corrected:
        return corrected

    return answer


# ============================================================
# HTML UI functions
# ============================================================

def question_form(
    default_question="",
    default_format="html",
    default_top_k=DEFAULT_TOP_K,
    default_mode="fast"
):
    safe_question = html.escape(default_question)

    html_selected = "selected" if default_format == "html" else ""
    json_selected = "selected" if default_format == "json" else ""

    fast_selected = "selected" if default_mode == "fast" else ""
    normal_selected = "selected" if default_mode == "normal" else ""
    detailed_selected = "selected" if default_mode == "detailed" else ""

    return f"""
    <div class="question-card">
        <h3>Ask the Local LLM</h3>

        <form method="get" action="/ask">
            <label for="q"><b>Question</b></label>

            <textarea
                id="q"
                name="q"
                placeholder="Type your question here... Example: من هو د. رفعت؟"
                dir="auto"
                required>{safe_question}</textarea>

            <div class="form-row">
                <div>
                    <label for="format"><b>Output format</b></label><br>
                    <select id="format" name="format">
                        <option value="html" {html_selected}>HTML for Browser</option>
                        <option value="json" {json_selected}>JSON for Mobile / Flutter</option>
                    </select>
                </div>

                <div>
                    <label for="top_k"><b>Retrieved passages</b></label><br>
                    <input
                        id="top_k"
                        name="top_k"
                        type="number"
                        value="{default_top_k}"
                        min="1"
                        max="{MAX_TOP_K}">
                </div>

                <div>
                    <label for="mode"><b>Answer mode</b></label><br>
                    <select id="mode" name="mode">
                        <option value="fast" {fast_selected}>Fast</option>
                        <option value="normal" {normal_selected}>Normal</option>
                        <option value="detailed" {detailed_selected}>Detailed</option>
                    </select>
                </div>
            </div>

            <button type="submit">Ask Local LLM</button>
        </form>

        <div class="examples">
            <b>Quick examples:</b><br>
            <a href="/ask?q=من هو لوسيفر؟ &format=html&top_k=5&mode=fast">من هو لوسيفر؟  </a><br>
            <a href="/ask?q=من هو د. رفعت؟ &format=html&top_k=5&mode=fast">من هو د. رفعت؟</a><br>
            <a href="/ask?q=أذكر أهم شخصيات القصة؟  &format=html&top_k=5&mode=fast">أذكر أهم شخصيات القصة؟</a><br>
        </div>
    </div>
    """


def page_template(title, body):
    return f"""
    <html>
        <head>
            <title>{html.escape(title)}</title>
            <meta charset="UTF-8">

            <style>
                body {{
                    font-family: Arial, Tahoma, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    background-color: #f8f9fa;
                    direction: ltr;
                }}

                .container {{
                    max-width: 1000px;
                    margin: auto;
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}

                h2 {{
                    color: #1f4e79;
                }}

                textarea {{
                    width: 100%;
                    height: 120px;
                    font-size: 16px;
                    padding: 12px;
                    border-radius: 8px;
                    border: 1px solid #ccc;
                    box-sizing: border-box;
                    direction: auto;
                }}

                input[type="number"] {{
                    width: 120px;
                    font-size: 16px;
                    padding: 8px;
                    border-radius: 6px;
                    border: 1px solid #ccc;
                }}

                select {{
                    font-size: 16px;
                    padding: 8px;
                    border-radius: 6px;
                    border: 1px solid #ccc;
                }}

                button {{
                    background-color: #1f75cb;
                    color: white;
                    border: none;
                    padding: 12px 22px;
                    font-size: 16px;
                    border-radius: 8px;
                    cursor: pointer;
                    margin-top: 12px;
                }}

                button:hover {{
                    background-color: #155a99;
                }}

                pre {{
                    background: #f2f2f2;
                    padding: 12px;
                    border-radius: 8px;
                    overflow-x: auto;
                    direction: ltr;
                    text-align: left;
                }}

                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-top: 15px;
                    direction: ltr;
                }}

                th, td {{
                    border: 1px solid #ccc;
                    padding: 8px;
                    text-align: left;
                    vertical-align: top;
                }}

                th {{
                    background-color: #eaf4ff;
                }}

                .note {{
                    background: #fff8df;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 5px solid #f0c040;
                    margin-bottom: 15px;
                }}

                .question-card {{
                    background: #f4f9ff;
                    padding: 18px;
                    border-radius: 10px;
                    border: 1px solid #d6e9ff;
                    margin-bottom: 25px;
                }}

                .form-row {{
                    display: flex;
                    gap: 20px;
                    margin-top: 12px;
                    flex-wrap: wrap;
                }}

                .answer-box {{
                    background:#eaf4ff;
                    padding:15px;
                    border-radius:8px;
                    white-space:pre-wrap;
                    direction: rtl;
                    text-align: right;
                    font-size: 17px;
                }}

                .question-box {{
                    background:#f2f2f2;
                    padding:15px;
                    border-radius:8px;
                    direction:auto;
                    font-size: 17px;
                }}

                .arabic-preview {{
                    direction: rtl;
                    text-align: right;
                    font-size: 15px;
                }}

                .examples {{
                    margin-top: 15px;
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                }}

                .error {{
                    background: #ffecec;
                    border-left: 5px solid #d9534f;
                    padding: 12px;
                    border-radius: 8px;
                }}

                a {{
                    color: #155a99;
                }}
            </style>
        </head>

        <body>
            <div class="container">
                {body}
            </div>
        </body>
    </html>
    """


# ============================================================
# RAG functions
# ============================================================

def retrieve_chunks(question, top_k=DEFAULT_TOP_K):
    if not RAG_READY:
        raise RuntimeError(
            "RAG resources are not ready. "
            f"Load error: {RAG_LOAD_ERROR}"
        )

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        record = chunks[int(idx)]

        results.append({
            "id": record["id"],
            "score": float(score),
            "text": record["text"]
        })

    return results


def call_llama(question, retrieved_chunks, max_tokens=FAST_MAX_TOKENS, answer_mode="fast"):
    profile_text = load_profile()

    context_blocks = []

    for item in retrieved_chunks:
        context_blocks.append(
            f"[Passage ID: {item['id']} | Score: {item['score']:.4f}]\n{item['text']}"
        )

    document_context = "\n\n---\n\n".join(context_blocks)

    system_prompt = f"""
You are a local Arabic document question-answering assistant.

User profile:
{profile_text}

Mandatory rules:
1. The final answer MUST be in Arabic only, even if the question is written in English.
2. Do not answer in English.
3. Use ONLY the provided Arabic document passages.
4. If the answer is not clearly found in the passages, say exactly:
   "لا أستطيع تحديد الإجابة من النصوص المتاحة."
5. Do not invent events, characters, places, relationships, or explanations.
6. Do not repeat the full retrieved passages.
7. Keep the answer concise and clear.
8. Mention passage IDs only if useful.
9. Give the final answer only. Do not show reasoning.
10. If an English technical term is unavoidable, keep it minimal; the explanation must remain Arabic.

Answer mode:
{answer_mode}

Arabic document passages:
{document_context}
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    "Answer the following question in Arabic only using the provided passages:\n\n"
                    + question
                )
            }
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }

    response = requests.post(
        LLAMA_SERVER_URL,
        json=payload,
        timeout=LLAMA_TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    message = data["choices"][0]["message"]
    answer = message.get("content", "").strip()

    if not answer:
        answer = (
            "لم يرجع النموذج إجابة نهائية. "
            "جرّب تقليل عدد المقاطع أو زيادة المهلة أو تعطيل reasoning."
        )

    if answer_looks_english(answer):
        answer = force_arabic_answer(answer)

    timings = data.get("timings", {})
    usage = data.get("usage", {})

    return {
        "question": question,
        "answer": answer,
        "model": data.get("model", ""),
        "answer_language": "Arabic only",
        "answer_mode": answer_mode,
        "retrieved_passages": [
            {
                "id": item["id"],
                "score": item["score"],
                "preview": item["text"][:300]
            }
            for item in retrieved_chunks
        ],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "generation_tokens_per_second": timings.get("predicted_per_second")
    }


# ============================================================
# Pages
# ============================================================

def help_screen():
    if not RAG_READY:
        body = f"""
        <h2>RAG System Not Ready</h2>

        <div class="error">
            <h3>The index was not loaded</h3>
            <p>The server started, but the embedding model or FAISS index was not loaded.</p>
            <p><b>Error:</b></p>
            <pre>{html.escape(RAG_LOAD_ERROR)}</pre>
        </div>

        <h3>How to fix</h3>
        <p>Make sure these files exist:</p>
        <pre>{html.escape(CHUNKS_FILE)}
{html.escape(FAISS_INDEX_FILE)}</pre>

        <p>If they do not exist, run:</p>
        <pre>python build_document_index.py</pre>

        <p>Then restart the server:</p>
        <pre>python pyrun_rag.py</pre>
        """

        return page_template("RAG System Not Ready", body)

    body = f"""
    <h2>Arabic Document RAG Assistant</h2>

    <div class="note">
        <b>Status:</b> RAG index is loaded and ready.<br>
        <b>Loaded chunks:</b> {len(chunks)}<br>
        <b>Embedding model:</b> {html.escape(EMBEDDING_MODEL_NAME)}<br>
        <b>Interface language:</b> English<br>
        <b>Answer language:</b> Arabic only
    </div>

    <p>
        This server receives a question, retrieves relevant passages from the Arabic document,
        sends them to the local Qwen3-8B model, and returns the answer in Arabic.
    </p>

    {question_form(default_format="html", default_top_k=DEFAULT_TOP_K)}

    <h3>Manual browser example</h3>
    <pre>http://127.0.0.1:5000/ask?q=من هو د. رفعت؟&format=html&top_k=3&mode=fast</pre>

    <h3>Manual mobile / Flutter example</h3>
    <pre>http://127.0.0.1:5000/ask?q=من هو د. رفعت؟&format=json&top_k=3&mode=fast</pre>

    <h3>Required services</h3>

    <p>Run llama-server first:</p>

    <pre>llama-server.exe -m "D:\\llama\\models\\Qwen_Qwen3-8B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 --ctx-size 4096 --threads 8</pre>

    <p>Then run this Flask server:</p>

    <pre>python pyrun_rag.py</pre>
    """

    return page_template("Arabic Document RAG Assistant", body)


@app.get("/")
def home():
    return help_screen()


@app.get("/status")
def status():
    if RAG_READY:
        return jsonify({
            "ready": True,
            "chunks": len(chunks),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "interface_language": "English",
            "answer_language": "Arabic only"
        })

    return jsonify({
        "ready": False,
        "error": RAG_LOAD_ERROR
    }), 503


@app.get("/ask")
def ask():
    question = request.args.get("q", "").strip()
    output_format = request.args.get("format", "html").strip().lower()
    top_k = get_safe_top_k()
    answer_mode, max_tokens = get_answer_mode_settings()

    if output_format not in ["html", "json"]:
        output_format = "html"

    if not question:
        if output_format == "json":
            return jsonify({
                "message": "No question provided.",
                "usage": "/ask?q=Your question here&format=json&top_k=3&mode=fast",
                "ready": RAG_READY
            }), 400

        return help_screen(), 200

    if not RAG_READY:
        error = {
            "error": "RAG system is not ready.",
            "details": RAG_LOAD_ERROR,
            "solution": "Run build_document_index.py, then restart pyrun_rag.py."
        }

        if output_format == "json":
            return jsonify(error), 503

        body = f"""
        <h2>RAG System Not Ready</h2>

        <div class="error">
            <pre>{html.escape(RAG_LOAD_ERROR)}</pre>
        </div>

        <p>Run:</p>
        <pre>python build_document_index.py</pre>

        <p>Then restart:</p>
        <pre>python pyrun_rag.py</pre>

        <br>
        <a href="/">Back to Home</a>
        """

        return page_template("RAG System Not Ready", body), 503

    try:
        retrieved = retrieve_chunks(question, top_k=top_k)

        result = call_llama(
            question,
            retrieved,
            max_tokens=max_tokens,
            answer_mode=answer_mode
        )

        if output_format == "json":
            return jsonify(result)

        safe_question = html.escape(result["question"])
        safe_answer = html.escape(result["answer"])

        passage_rows = ""
        for p in result["retrieved_passages"]:
            passage_rows += f"""
            <tr>
                <td>{p["id"]}</td>
                <td>{p["score"]:.4f}</td>
                <td class="arabic-preview">{html.escape(p["preview"])}</td>
            </tr>
            """

        body = f"""
        <h2>Arabic Document RAG Answer</h2>

        {question_form(
            default_question="",
            default_format="html",
            default_top_k=top_k,
            default_mode=answer_mode
        )}

        <h3>Question</h3>
        <div class="question-box">
            {safe_question}
        </div>

        <h3>Answer</h3>
        <div class="answer-box">
            {safe_answer}
        </div>

        <h3>Retrieved Passages</h3>
        <table>
            <tr>
                <th>Passage ID</th>
                <th>Similarity Score</th>
                <th>Preview</th>
            </tr>
            {passage_rows}
        </table>

        <h3>Model Information</h3>
        <table>
            <tr><td>Model</td><td>{html.escape(str(result["model"]))}</td></tr>
            <tr><td>Answer mode</td><td>{html.escape(str(result["answer_mode"]))}</td></tr>
            <tr><td>Answer language</td><td>Arabic only</td></tr>
            <tr><td>Prompt tokens</td><td>{result["prompt_tokens"]}</td></tr>
            <tr><td>Completion tokens</td><td>{result["completion_tokens"]}</td></tr>
            <tr><td>Total tokens</td><td>{result["total_tokens"]}</td></tr>
            <tr><td>Prompt speed</td><td>{result["prompt_tokens_per_second"]} tokens/sec</td></tr>
            <tr><td>Generation speed</td><td>{result["generation_tokens_per_second"]} tokens/sec</td></tr>
        </table>

        <br>
        <a href="/">Back to Home</a>
        """

        return page_template("Arabic Document RAG Answer", body)

    except requests.exceptions.ConnectionError:
        error = {
            "error": "Cannot connect to llama-server.",
            "details": "Make sure llama-server is running on http://127.0.0.1:8080",
            "start_command": 'llama-server.exe -m "D:\\llama\\models\\Qwen_Qwen3-8B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 --ctx-size 4096 --threads 8'
        }

        if output_format == "json":
            return jsonify(error), 503

        body = f"""
        <h2>Cannot Connect to llama-server</h2>

        <div class="error">
            <p>{html.escape(error["details"])}</p>
        </div>

        <p>Start llama-server using:</p>
        <pre>{html.escape(error["start_command"])}</pre>

        {question_form(
            default_question=question,
            default_format="html",
            default_top_k=top_k,
            default_mode=answer_mode
        )}

        <br>
        <a href="/">Back to Home</a>
        """

        return page_template("Connection Error", body), 503

    except requests.exceptions.Timeout:
        error = {
            "error": "Timeout.",
            "details": "Try reducing retrieved passages, using Fast mode, or shortening the question."
        }

        if output_format == "json":
            return jsonify(error), 504

        body = f"""
        <h2>Timeout</h2>

        <div class="error">
            <p>{html.escape(error["details"])}</p>
        </div>

        {question_form(
            default_question=question,
            default_format="html",
            default_top_k=top_k,
            default_mode="fast"
        )}

        <br>
        <a href="/">Back to Home</a>
        """

        return page_template("Timeout", body), 504

    except Exception as e:
        error = {
            "error": "Unexpected error.",
            "details": str(e)
        }

        if output_format == "json":
            return jsonify(error), 500

        body = f"""
        <h2>Unexpected Error</h2>
        <pre>{html.escape(str(e))}</pre>

        {question_form(
            default_question=question,
            default_format="html",
            default_top_k=top_k,
            default_mode=answer_mode
        )}

        <br>
        <a href="/">Back to Home</a>
        """

        return page_template("Unexpected Error", body), 500


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    load_rag_resources()
    app.run(host="0.0.0.0", port=5000)