@echo off
cd /d D:\llama
start "llama-server Qwen" cmd /k llama-server.exe -m "D:\llama\models\Qwen_Qwen3-8B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 --ctx-size 4096 --threads 8 --n-gpu-layers all --reasoning off
timeout /t 5
start "Flask RAG" cmd /k call .venv\Scripts\activate ^& python pyrun_rag01.py
