@echo off
cd /d "%~dp0"
C:\Users\Admin\anaconda3\Scripts\conda.exe run -n tf python -m uvicorn app:app --host 127.0.0.1 --port 8000
