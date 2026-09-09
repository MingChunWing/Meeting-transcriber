#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transcribe any mp4/mp3 -> transcript (.txt/.srt/.vtt) next to the source.

Usage:
  python transcribe.py "C:/path/to/meeting.mp4"
Drag & drop a file onto transcribe.bat to run this automatically.

Needs: faster-whisper, ffmpeg on PATH, NVIDIA GPU (CUDA) for speed.
Set WHISPER_MODEL=large-v3-turbo for ~8x faster (slightly less accurate).
"""
import os, sys, subprocess, datetime, tempfile
from faster_whisper import WhisperModel

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

if len(sys.argv) < 2:
    print("用法: python transcribe.py <file.mp4|file.mp3>")
    print("或將檔案拖去 transcribe.bat 上面。")
    sys.exit(1)

SRC = sys.argv[1]
if not os.path.exists(SRC):
    print(f"搵唔到檔案: {SRC}")
    sys.exit(1)

MODEL = os.environ.get("WHISPER_MODEL", "large-v3")  # large-v3 / large-v3-turbo
BASE = os.path.splitext(SRC)[0]            # same folder, same base name
WAV = BASE + ".tmp_transcribe.wav"
TXT = BASE + "_transcript.txt"
SRT = BASE + "_transcript.srt"
VTT = BASE + "_transcript.vtt"

# 1) extract 16k mono audio (works for both mp4 and mp3)
log(f"抽 audio 自: {os.path.basename(SRC)}")
subprocess.run(
    ["ffmpeg", "-y", "-i", SRC, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", WAV],
    check=True,
)

# 2) transcribe
log(f"Load model {MODEL} (CUDA) ...")
model = WhisperModel(MODEL, device="cuda", compute_type="float16")
log("轉寫中 (zh) ...")
segments, info = model.transcribe(
    WAV,
    language="zh",
    beam_size=5,
    vad_filter=True,
    initial_prompt="呢段係FYP final year project 會議錄音，廣東話同國語夾雜，請盡量準確逐字轉寫。",
)
log(f"語言: {info.language} (prob {info.language_probability:.2f})")

segs = list(segments)
log(f"共 {len(segs)} 段")

# 3) write
def fmt_time(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60); ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

with open(TXT, "w", encoding="utf-8") as ft, \
     open(SRT, "w", encoding="utf-8") as fs, \
     open(VTT, "w", encoding="utf-8") as fv:
    fv.write("WEBVTT\n\n")
    for i, seg in enumerate(segs, 1):
        text = seg.text.strip()
        ft.write(text + "\n")
        fs.write(f"{i}\n{fmt_time(seg.start)} --> {fmt_time(seg.end)}\n{text}\n\n")
        start = fmt_time(seg.start).replace(",", ".")
        end = fmt_time(seg.end).replace(",", ".")
        fv.write(f"{start} --> {end}\n{text}\n\n")

# cleanup temp wav
try: os.remove(WAV)
except OSError: pass

log(f"完成! 出咗: {os.path.basename(TXT)} / .srt / .vtt  (喺 {os.path.dirname(TXT)})")
