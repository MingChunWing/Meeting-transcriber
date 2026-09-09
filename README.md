# FYP Meeting Transcriber

本地 offline 嘅 meeting 錄音轉寫工具：拖放 mp4/mp3 → 聲紋分辨講者 → 標 `Supervisor (F)` / `Member A–D (M)` → 中/英轉寫 → 粵語口語字版。

全部本地跑（NVIDIA GPU + CUDA），錄音唔使上傳任何雲端。

## Features
- faster-whisper `large-v3` (CUDA) 轉寫中文（廣東話 + 國語）+ 英文
- pyannote `speaker-diarization-3.1` 聲紋分辨（5 個講者 cluster）
- 自動按音高判別唯一女聲 = `Supervisor (F)`，其餘 = `Member A–D (M)`
- 輸出：`.txt`（純文字可 copy）/ `.srt` / `.vtt`（帶 timestamp）/ `_cantonese.txt`（粵語口語字版）
- 本地 web UI：拖放即轉，雙 tab 切換 書面中文 / 粵語口語字

## Requirements
- Windows + NVIDIA GPU（測試：RTX 2080 Ti）
- Python 3.11（包喺 Hermes venv，或者你自己嘅 venv）
- ffmpeg on PATH
- HuggingFace read token（免費，pyannote 需要）：https://huggingface.co/settings/tokens
  - 仲要 accept 兩個 model licence：
    - https://huggingface.co/pyannote/speaker-diarization-3.1
    - https://huggingface.co/pyannote/segmentation-3.0

## Setup
1. clone / copy 呢個 folder
2. 執 `setup_env.bat`（會幫你裝晒 dependencies 去 Hermes venv）
   - 或者手動：`pip install -r requirements.txt`
   - 注意 torch 要 CUDA 版：`pip install torch --index-url https://download.pytorch.org/whl/cu124`
3. 將你嘅 HF token 單獨一行貼入 `hf_token.txt`（唔好加其他字 / 註解）
   - 或者 set User 環境變數 `HF_TOKEN`
4. 雙擊 `start_app.bat` → 瀏覽器開 `http://127.0.0.1:5000` → 拖 mp4/mp3 落去

## Notes
- diarization 用 CPU 跑（torch 2.6+cu124 喺呢部機缺 cuDNN symbol）；whisper ASR 用 GPU
- 47 分 audio 約 25 分鐘處理
- `Member A–D` 係按音高排序嘅假名，要對真名就手動改 output
- `hf_token.txt` 已 gitignore，唔會入 git

## Files
- `app.py` — Flask web app（主體）
- `start_app.bat` — 雙擊起動（自動用 venv python + 讀 token）
- `transcribe.py` / `transcribe.bat` — 輕量 CLI 版（唔做 diarization，純轉寫）
- `requirements.txt` — dependencies
- `setup_env.bat` — 一鍵裝 dependencies（Hermes venv）
