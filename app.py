#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local FYP meeting transcriber web app (speaker diarization + gender + cantonese).

Pipeline:
  1. faster-whisper (large-v3, CUDA) ASR  -> zh/en text
  2. pyannote speaker-diarization-3.1     -> speaker clusters (voiceprint, not pitch)
  3. per-cluster mean F0 (WORLD)          -> the single highest-pitch cluster = female Supervisor
  4. labels: Supervisor (F) / Member A..D (M)
  5. optional cantonese colloquial pass   -> *_cantonese.txt

Run:  python app.py   ->  http://127.0.0.1:5000
Needs env var HF_TOKEN (read token, pyannote license accepted).
"""
import os, subprocess, datetime, re, tempfile
from flask import Flask, request, send_file, render_template_string
import numpy as np
import soundfile as sf
import pyworld
from faster_whisper import WhisperModel

# torch 2.6 changed torch.load default weights_only=True; pyannote 3.x checkpoints
# use the old pickle format. Allowlist the known globals instead of disabling
# weights_only entirely (safer). Add new ones here if the error names more.
try:
    import torch
    from torch.serialization import add_safe_globals  # type: ignore
    from torch.torch_version import TorchVersion  # type: ignore
    import pyannote.audio.core.task as _task
    _allow = [TorchVersion]
    for _n in ("Specifications", "Problem", "Task", "Resolution"):
        _c = getattr(_task, _n, None)
        if _c is not None:
            _allow.append(_c)
    # also any dataclasses used inside the checkpoint
    try:
        from pyannote.audio.core.task import Task  # noqa
    except Exception:
        pass
    add_safe_globals(_allow)
except Exception as _e:
    print("warn: safe_globals patch failed:", _e)

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = None          # whisper
DIARIZER = None       # pyannote

APP = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- cantonese colloquial replacements (zh -> yue spoken) ----
CANTONESE_MAP = [
    ("我們", "我哋"), ("咱们", "我哋"), ("你們", "你哋"), ("他們", "佢哋"),
    ("她們", "佢哋"), ("它們", "佢哋"), ("它", "佢"),
    ("的", "嘅"),
    ("這", "呢"), ("那", "嗰"), ("這裡", "呢度"), ("那裡", "嗰度"),
    ("這個", "呢個"), ("那個", "嗰個"), ("這些", "呢啲"), ("那些", "嗰啲"),
    ("嗎", "嘛"), ("什麼", "咩"), ("怎麼", "點解"), ("怎樣", "點樣"),
    ("哪裡", "邊度"), ("哪個", "邊個"), ("誰", "邊個"),
    ("會", "會"),  # keep
    ("是", "係"), ("不是", "唔係"), ("有", "有"), ("沒有", "冇"),
    ("不", "唔"), ("很", "好"), ("都", "都"), ("就", "就"),
    ("才", "先"), ("再", "再"), ("又", "又"), ("也", "都"),
    ("呢", "呢"),
    ("了", "咗"), ("過", "過"), ("給", "畀"), ("和", "同"), ("跟", "同"),
    ("與", "同"), ("看", "睇"), ("說", "講"), ("話", "話"),
    ("吃", "食"), ("喝", "飲"), ("買", "買"), ("賣", "賣"),
    ("做", "做"), ("想", "想"), ("要", "要"), ("可以", "可以"),
    ("知道", "知"), ("不知道", "唔知"), ("現在", "而家"), ("先", "先"),
    ("後", "後"), ("時候", "時候"), ("東西", "嘢"), ("地方", "地方"),
    ("為什麼", "點解"), ("所以", "所以"), ("但是", "但係"), ("不過", "不過"),
    ("如果", "如果"), ("因為", "因為"), ("應該", "應該"), ("需要", "需要"),
    ("可以", "得"),
]
CANTONESE_RE = [(re.compile(re.escape(a)), b) for a, b in CANTONESE_MAP]

def to_cantonese(text):
    # Only apply on zh-heavy lines; keep English words intact.
    for rx, b in CANTONESE_RE:
        text = rx.sub(b, text)
    return text

def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)

def extract_wav(src, wav):
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
                    "-f", "wav", wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_whisper():
    global MODEL
    if MODEL is None:
        log(f"Loading whisper {MODEL_SIZE} (CUDA) ...")
        MODEL = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    return MODEL

def get_diarizer():
    global DIARIZER
    if DIARIZER is None:
        from pyannote.audio import Pipeline
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN 未設定 (set User env var or pass inline).")
        log("Loading pyannote diarization (CPU; CUDA cuDNN symbol missing on this box) ...")
        DIARIZER = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            cache_dir=os.path.join(UPLOAD_DIR, "..", "pyannote_cache"))
        # Keep diarization on CPU: torch 2.6+cu124 here lacks cuDNN symbol.
        # Whisper ASR still runs on CUDA (faster). Diarization on CPU is fine.
    return DIARIZER

def torch_available_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_f0(chunk, sr=16000):
    x = np.ascontiguousarray(chunk, dtype=np.float64)
    f0, _, _ = pyworld.wav2world(x, fs=sr)
    f0 = f0[f0 > 0]
    return float(np.mean(f0)) if len(f0) else None

def fmt_srt_time(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60); ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def transcribe_file(src):
    base = os.path.splitext(os.path.basename(src))[0]
    wav = os.path.join(UPLOAD_DIR, base + ".tmp.wav")
    txt = os.path.join(UPLOAD_DIR, base + "_transcript.txt")
    srt = os.path.join(UPLOAD_DIR, base + "_transcript.srt")
    vtt = os.path.join(UPLOAD_DIR, base + "_transcript.vtt")
    can = os.path.join(UPLOAD_DIR, base + "_cantonese.txt")

    extract_wav(src, wav)
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 1) ASR
    model = get_whisper()
    log("Transcribing ...")
    segs, info = model.transcribe(
        wav, language=None, beam_size=5, vad_filter=True,
        initial_prompt="FYP final year project 會議錄音，廣東話同國語夾雜，請盡量準確逐字轉寫。")
    segs = list(segs)
    log(f"segments={len(segs)} lang={info.language} ({info.language_probability:.2f})")

    # 2) diarization
    dia = get_diarizer()
    log("Diarizing (voiceprint) ...")
    diar = dia(wav)
    # build list of (start,end,speaker)
    turns = [(turn.start, turn.end, spk) for turn, _, spk in diar.itertracks(yield_label=True)]
    log(f"speakers found: {sorted(set(t[2] for t in turns))}")

    def speaker_at(t):
        best = None; best_overlap = -1
        for s, e, spk in turns:
            ov = min(e, t + 0.01) - max(s, t - 0.01)
            if ov > best_overlap:
                best_overlap = ov; best = spk
        return best or "SPEAKER_00"

    # 3) assign speaker per segment + collect per-speaker audio for F0/gender
    spk_chunks = {}
    raw_lines = []
    for seg in segs:
        text = seg.text.strip()
        if not text:
            continue
        spk = speaker_at((seg.start + seg.end) / 2)
        raw_lines.append({"start": seg.start, "end": seg.end, "spk": spk, "text": text})
        a0, a1 = int(seg.start * sr), int(seg.end * sr)
        chunk = audio[a0:a1] if a1 <= len(audio) else audio[max(0, a0):]
        spk_chunks.setdefault(spk, []).append(chunk)

    # 4) per-speaker mean F0 -> highest = female (Supervisor)
    spk_f0 = {}
    for spk, chunks in spk_chunks.items():
        allc = np.concatenate(chunks) if chunks else np.zeros(sr, dtype=np.float32)
        f = mean_f0(allc, sr)
        spk_f0[spk] = f if f is not None else 0.0
    log(f"per-speaker F0: {spk_f0}")
    # rank speakers by F0 desc; top-1 female, rest male; label A..D
    ordered = sorted(spk_f0.items(), key=lambda kv: kv[1], reverse=True)
    female_spk = ordered[0][0] if ordered else None
    male_order = [s for s, _ in ordered if s != female_spk]
    label_map = {female_spk: "Supervisor (F)"}
    for i, s in enumerate(male_order[:4]):
        label_map[s] = f"Member {chr(65+i)} (M)"
    for s in male_order[4:]:
        label_map[s] = "Member (M)"

    # 5) build final lines
    lines = []
    for ln in raw_lines:
        tag = label_map.get(ln["spk"], "Member (M)")
        lines.append({"start": ln["start"], "end": ln["end"], "tag": tag, "text": ln["text"]})

    # 6) write txt/srt/vtt + cantonese
    with open(txt, "w", encoding="utf-8") as ft, \
         open(srt, "w", encoding="utf-8") as fs, \
         open(vtt, "w", encoding="utf-8") as fv, \
         open(can, "w", encoding="utf-8") as fc:
        fv.write("WEBVTT\n\n")
        for i, ln in enumerate(lines, 1):
            ft.write(f"{ln['tag']}: {ln['text']}\n")
            fc.write(f"{ln['tag']}: {to_cantonese(ln['text'])}\n")
            fs.write(f"{i}\n{fmt_srt_time(ln['start'])} --> {fmt_srt_time(ln['end'])}\n{ln['tag']}: {ln['text']}\n\n")
            st = fmt_srt_time(ln['start']).replace(",", ".")
            en = fmt_srt_time(ln['end']).replace(",", ".")
            fv.write(f"{st} --> {en}\n{ln['tag']}: {ln['text']}\n\n")

    try: os.remove(wav)
    except OSError: pass
    log("done")
    return {"txt": txt, "srt": srt, "vtt": vtt, "cantonese": can, "n": len(lines),
            "lang": info.language, "speakers": list(label_map.values())}

# ---------- web ----------
INDEX = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>FYP Meeting Transcriber</title>
<style>
        body{margin:0;font-family:-apple-system,'Segoe UI','Noto Sans HK',sans-serif;
             color:#1c2b3a;background:#f6f7f9;padding:40px 20px 64px;line-height:1.5}
        .wrap{max-width:860px;margin:0 auto}
        h1{font-size:22px;margin:0;letter-spacing:-.01em;font-weight:700}
        p.sub{color:#5c6b7a;margin:6px 0 32px;font-size:13.5px}
        /* upload */
        #drop{border:1.5px dashed #b9c2cc;border-radius:14px;padding:44px 24px;text-align:center;
              color:#5c6b7a;cursor:pointer;transition:border-color .18s,background .18s;
              background:#fff}
        #drop:focus-visible{outline:2px solid #2563eb;outline-offset:3px}
        #drop.hot{border-color:#2563eb;color:#1c2b3a;background:#f0f5ff}
        #drop .icon{display:block;margin:0 auto 12px;width:40px;height:40px;color:#8a97a5}
        #drop strong{display:block;color:#1c2b3a;font-size:15px;font-weight:600;margin-bottom:4px}
        #drop span{font-size:12.5px}
        /* status */
        #status{margin:24px 0 0;font-size:14px;min-height:20px;color:#5c6b7a}
        #status.err{color:#b42318}
        .bar{height:4px;background:#e4e8ec;border-radius:99px;overflow:hidden;margin:10px 0 0}
        .bar>i{display:block;height:100%;width:0;background:#2563eb;border-radius:99px;
               transition:width .3s ease-out}
        /* output */
        #out{display:none;margin-top:32px}
        .resbar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:0 0 14px}
        .resbar .meta{font-size:12.5px;color:#5c6b7a}
        .tabs{display:inline-flex;background:#e9edf1;border-radius:9px;padding:3px}
        .tabs button{font:inherit;font-size:13px;border:none;background:none;color:#5c6b7a;
                     padding:6px 14px;border-radius:7px;cursor:pointer}
        .tabs button.active{background:#fff;color:#1c2b3a;font-weight:600;
                            box-shadow:0 1px 2px rgba(28,43,58,.12)}
        .tabs button:focus-visible{outline:2px solid #2563eb;outline-offset:1px}
        .toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 14px}
        button,a.btn{font:inherit;font-size:13.5px;border:1px solid #d4dae0;background:#fff;
             color:#1c2b3a;padding:8px 14px;border-radius:9px;cursor:pointer;text-decoration:none;
             display:inline-flex;align-items:center;gap:7px}
        button:hover,a.btn:hover{border-color:#9aa6b2;background:#fafbfc}
        button:focus-visible,a.btn:focus-visible{outline:2px solid #2563eb;outline-offset:2px}
        button svg,a.btn svg{width:15px;height:15px;color:#5c6b7a;flex:none}
        button.primary{background:#2563eb;border-color:#2563eb;color:#fff}
        button.primary:hover{background:#1d4fc4}
        button.primary svg{color:#fff}
        /* transcript */
        #transcript{white-space:pre-wrap;line-height:1.85;font-size:15px;color:#2a3a4a;
             border:1px solid #e2e7eb;border-radius:12px;padding:24px;background:#fff;
             max-height:62vh;overflow:auto;box-shadow:0 1px 3px rgba(28,43,58,.06)}
        #transcript::-webkit-scrollbar{width:10px}
        #transcript::-webkit-scrollbar-thumb{background:#d4dae0;border-radius:99px;
             border:2.5px solid #fff}
        .f{color:#b42318;font-weight:700}
        .m{color:#1d4fc4;font-weight:600}
        .hint{color:#8a97a5;font-size:12px;margin-top:14px;line-height:1.6}
        .hint kbd{background:#e9edf1;border-radius:4px;padding:1px 5px;font-size:11px;
                  font-family:inherit}
        @media (prefers-reduced-motion: reduce){
          *{transition-duration:.01ms!important;animation-duration:.01ms!important}
        }
        @media (max-width:560px){
          body{padding:24px 14px 48px}
          #drop{padding:32px 16px}
          #transcript{padding:16px;font-size:14.5px}
        }
    </style></head>
    <body>
    <div class="wrap">
    <h1>FYP Meeting Transcriber</h1>
    <p class="sub">拖放錄音 → 聲紋分辨講者 → 標 <strong>Supervisor (F)</strong> / <strong>Member A–D (M)</strong> → 中/英轉寫 → 粵語口語字版</p>
    <div id="drop" role="button" tabindex="0" aria-label="上載會議錄音檔">
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 16V4m0 0l-4 4m4-4l4 4"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>
      </svg>
      <strong>將 meeting 錄音拖落呢度</strong>
      <span>或者 click 揀檔 — 支援 mp4 / mp3 / wav / m4a,全程本地處理</span>
    </div>
    <input id="file" type="file" accept=".mp4,.mp3,.wav,.m4a" hidden>
    <div id="status" role="status" aria-live="polite"></div>
    <div class="bar" id="bar" style="display:none" aria-hidden="true"><i></i></div>
    <div id="out">
      <div class="resbar">
        <div class="tabs" role="tablist">
          <button id="tab-zh" class="active" role="tab" aria-selected="true" onclick="showTab('zh')">書面中文</button>
          <button id="tab-yue" role="tab" aria-selected="false" onclick="showTab('yue')">粵語口語字</button>
        </div>
        <div class="meta" id="meta"></div>
      </div>
      <div class="toolbar">
        <button onclick="copyAll()" class="primary">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>
          </svg>Copy 全部</button>
        <a class="btn" id="dl-txt" download>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 4v12m0 0l-4-4m4 4l4-4"/><path d="M5 20h14"/>
          </svg>.txt</a>
        <a class="btn" id="dl-srt" download>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 4v12m0 0l-4-4m4 4l4-4"/><path d="M5 20h14"/>
          </svg>.srt</a>
        <a class="btn" id="dl-vtt" download>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 4v12m0 0l-4-4m4 4l4-4"/><path d="M5 20h14"/>
          </svg>.vtt</a>
        <a class="btn" id="dl-can" download>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 4v12m0 0l-4-4m4 4l4-4"/><path d="M5 20h14"/>
          </svg>粵語 .txt</a>
      </div>
      <div id="transcript" tabindex="0"></div>
      <p class="hint">聲紋分辨：最高音 cluster 自動標 <strong>Supervisor</strong>（唯一女聲），其餘標 Member A–D。
        4 位男組員間以聲紋區分，如錯配可手動改 output。</p>
    </div>
    </div>
    <script>
let cur={zh:"",yue:""}, curFiles={};
const drop=document.getElementById('drop'),file=document.getElementById('file');
const status=document.getElementById('status'),bar=document.getElementById('bar'),out=document.getElementById('out');
drop.onclick=()=>file.click();
drop.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();file.click()}};
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hot')};
drop.ondragleave=()=>drop.classList.remove('hot');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hot');if(e.dataTransfer.files[0])upload(e.dataTransfer.files[0])};
file.onchange=()=>{if(file.files[0])upload(file.files[0])};
const STEPS=['抽 audio','聲紋分辨','轉寫','貼標籤'];
function upload(f){
  let step=0;
  status.className='';status.textContent='⏳ 處理中（'+STEPS[0]+'）— 首次模型載入會慢';
  out.style.display='none';bar.style.display='block';bar.firstElementChild.style.width='6%';
  const tick=setInterval(()=>{step=(step+1)%STEPS.length;
    status.textContent='⏳ 處理中（'+STEPS[step]+'）— 首次模型載入會慢';},4000);
  const fd=new FormData();fd.append('file',f);
  fetch('/transcribe',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
    clearInterval(tick);
    if(d.error){status.className='err';status.textContent='❌ '+d.error;bar.style.display='none';return;}
    bar.firstElementChild.style.width='100%';
    status.textContent='✅ 完成：'+d.n+' 句，語言 '+d.lang+'，講者 '+d.speakers.join(' / ');
    document.getElementById('meta').textContent=d.n+' 句 · '+d.lang.toUpperCase();
    cur={zh:d.html, yue:d.html_yue}; curFiles=d.files;
    document.getElementById('transcript').innerHTML=d.html;
    document.getElementById('dl-txt').href='/download?f='+encodeURIComponent(d.files.txt);
    document.getElementById('dl-srt').href='/download?f='+encodeURIComponent(d.files.srt);
    document.getElementById('dl-vtt').href='/download?f='+encodeURIComponent(d.files.vtt);
    document.getElementById('dl-can').href='/download?f='+encodeURIComponent(d.files.cantonese);
    out.style.display='block';
  }).catch(e=>{clearInterval(tick);status.className='err';status.textContent='❌ 失敗：'+e;bar.style.display='none'});
}
function showTab(t){
  document.getElementById('tab-zh').classList.toggle('active',t==='zh');
  document.getElementById('tab-zh').setAttribute('aria-selected',t==='zh');
  document.getElementById('tab-yue').classList.toggle('active',t==='yue');
  document.getElementById('tab-yue').setAttribute('aria-selected',t==='yue');
  document.getElementById('transcript').innerHTML = t==='yue'?cur.yue:cur.zh;
  document.getElementById('transcript')._cur=t;
}
function copyAll(){
  const t=document.getElementById('transcript').innerText;
  navigator.clipboard.writeText(t);status.className='';status.textContent='已 copy 全文到剪貼簿';
}
    </script></body></html>"""


@APP.route("/")
def index():
    return render_template_string(INDEX)

@APP.route("/transcribe", methods=["POST"])
def transcribe():
    f = request.files.get("file")
    if not f:
        return {"error": "無檔案"}
    src = os.path.join(UPLOAD_DIR, f.filename)
    f.save(src)
    try:
        res = transcribe_file(src)
    except Exception as e:
        return {"error": str(e)}
    def htmlify(path):
        out = []
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("Supervisor (F):"):
                body = line[len("Supervisor (F):"):].strip()
                out.append(f'<span class="f">Supervisor (F):</span> {body}')
            elif line.startswith("Member") and ":" in line:
                pre, _, body = line.partition(":")
                out.append(f'<span class="m">{pre}:</span> {body.strip()}')
            else:
                out.append(line)
        return "\n".join(out)
    return {"html": htmlify(res["txt"]), "html_yue": htmlify(res["cantonese"]),
            "n": res["n"], "lang": res["lang"], "speakers": res["speakers"],
            "files": res}

@APP.route("/download")
def download():
    p = request.args.get("f")
    if not p or not os.path.exists(p):
        return "not found", 404
    return send_file(p, as_attachment=True)

if __name__ == "__main__":
    log("Starting on http://127.0.0.1:5000")
    APP.run(host="127.0.0.1", port=5000, debug=False)
