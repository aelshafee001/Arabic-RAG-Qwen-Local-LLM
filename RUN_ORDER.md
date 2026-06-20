# Run Order

## 1. Install requirements

```bat
cd D:\llama
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Build the index

```bat
cd D:\llama
python build_document01_index.py
```

## 3. Start llama-server

```bat
cd D:\llama
llama-server.exe -m "D:\llama\models\Qwen_Qwen3-8B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 --ctx-size 4096 --threads 8 --n-gpu-layers all --reasoning off
```

## 4. Start Flask RAG server

Open another CMD:

```bat
cd D:\llama
python pyrun_rag01.py
```

## 5. Open the interface

```text
http://127.0.0.1:5000
```
