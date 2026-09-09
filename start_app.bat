@echo off
chcp 65001 >nul
setlocal
REM Run from THIS folder (the repo), not Videos — single source of truth
cd /d "%~dp0"

REM Use the Hermes venv python (where soundfile / torch / pyannote are installed)
set "VENV_PY=C:\Users\tu144\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [!] Cannot find Hermes venv python: %VENV_PY%
  echo     Please confirm the Hermes install path or point VENV_PY at the python that has soundfile installed.
  pause
  exit /b 1
)

REM ---- load HF token: 1) env var  2) local hf_token.txt next to app.py (gitignored) ----
if defined HF_TOKEN goto :have_token
if exist "%~dp0hf_token.txt" (
  set /p HF_TOKEN=<"%~dp0hf_token.txt"
  if defined HF_TOKEN goto :have_token
)
echo [!] HF_TOKEN not found.
echo     Option A: paste your token alone on line 1 of hf_token.txt in this folder (%~dp0)
echo     Option B: run once in PowerShell:
echo       [System.Environment]::SetEnvironmentVariable("HF_TOKEN","hf_xxx","User")
echo     then re-run this bat.
pause
exit /b 1

:have_token
echo Starting FYP Transcriber ...
echo   server: http://127.0.0.1:5000  (do not close this window)
start "FYP Transcriber Server" cmd /k "%VENV_PY% app.py"
timeout /t 4 >nul
start "" http://127.0.0.1:5000
echo Server window opened + browser launched. First model load is slow; once ready drag mp4/mp3 onto the page.
endlocal
