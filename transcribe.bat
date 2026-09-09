@echo off
chcp 65001 >nul
REM 將任何 mp4 / mp3 拖去呢個 bat 上面就會自動轉 transcript
REM (喺 repo 自己資料夾度跑 transcribe.py)
if "%~1"=="" (
  echo 請將 mp4 或 mp3 檔案拖落呢個 bat 上面。
  pause
  exit /b
)
set "VENV_PY=C:\Users\tu144\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
"%VENV_PY%" "%~dp0transcribe.py" "%~1"
echo.
echo === 完成! 出咗 *_transcript.txt / .srt / .vtt 喺你條片隔離 ===
pause
