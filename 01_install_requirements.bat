@echo off
cd /d D:\llama
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
pause
