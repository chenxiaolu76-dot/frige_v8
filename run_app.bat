@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run app\ui\streamlit_app.py --server.fileWatcherType none
