@echo off
setlocal

set "PYTHON_EXE=D:\conda-envs\smart-fridge-vision-py311\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    exit /b 1
)

cd /d "%~dp0"
"%PYTHON_EXE%" -m streamlit run app\ui\streamlit_app.py
