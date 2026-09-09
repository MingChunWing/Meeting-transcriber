@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV_PY=C:\Users\tu144\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [!] Hermes venv not found: %VENV_PY%
  echo     Edit VENV_PY below to point at the python where you want deps installed.
  pause & exit /b 1
)

echo Installing torch (CUDA cu124) ...
"%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

echo Installing其余 deps ...
"%VENV_PY%" -m pip install faster-whisper pyannote.audio==3.3.2 huggingface_hub==0.25.2 pyworld soundfile flask

echo.
echo Done. Remember to put your HF token in hf_token.txt (single line) or set HF_TOKEN env var.
echo Then double-click start_app.bat
pause
