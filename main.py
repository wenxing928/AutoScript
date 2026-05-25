#!/usr/bin/env python3
"""
AI-Powered Media Player with Closed Captions
─────────────────────────────────────────────
Install deps (RTX GPU — CUDA 12.1, ctranslate2 4.x):
    pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "torchaudio==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
    pip install "nvidia-cudnn-cu12>=9.0"              # cuDNN 9 for ctranslate2
    pip install "ctranslate2>=4.5,<5" "faster-whisper>=1.1"   # int8_float16 works on RTX 40-series
    pip install "pyannote-audio<4.0.0" python-vlc ollama deep-translator
    python -c "import torch; print(torch.cuda.get_device_name(0))"
    # Also install VLC media player from https://www.videolan.org/

Usage:
    python 1.py

Pipeline:
    1. Browse a media file (video or audio)
    2. Set optional translation
    3. Click "Prepare & Process" — runs faster-whisper ASR → cleanup → (optional) Google Translate
    4. Press Play — subtitles sync automatically during playback
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import re
import sys
import faulthandler
from dataclasses import dataclass
from typing import List, Optional

# ── Crash log — written before anything else so it survives a hard crash ───────
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")
_log_file = open(_LOG_PATH, "w", buffering=1, encoding="utf-8")
faulthandler.enable(_log_file)   # dumps C-level stack on segfault/fatal signal
sys.stderr = _log_file           # ctranslate2 writes fatal errors to stderr

# ── CUDA env fixes (must be set before any CUDA import) ───────────────────────
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")        # prevents init-time crashes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

# hf_xet (Rust download accelerator) crashes with illegal instruction on this CPU
os.environ["HF_HUB_DISABLE_XET"] = "1"

# ── NVIDIA + torch DLL registration (Windows) ─────────────────────────────────
# ctranslate2 needs cudart64_12.dll (bundled in torch/lib) AND cuDNN/cuBLAS
# (from nvidia-* pip packages). Register both so Windows can find them all.
if sys.platform == "win32":
    # 1. torch/lib — contains cudart64_12.dll and core CUDA runtime DLLs
    try:
        import torch as _torch_pre
        _tlib = os.path.join(os.path.dirname(_torch_pre.__file__), "lib")
        if os.path.isdir(_tlib):
            os.add_dll_directory(_tlib)
            _log_file.write(f"[startup] torch/lib registered: {_tlib}\n")
    except Exception as _e:
        _log_file.write(f"[startup] torch/lib registration failed: {_e}\n")

    # 2. site-packages/nvidia/*/bin — cuDNN 9, cuBLAS, nvRTC …
    import site as _site
    for _sp in _site.getsitepackages():
        _nvidia = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia):
            for _lib in os.listdir(_nvidia):
                _bin = os.path.join(_nvidia, _lib, "bin")
                if os.path.isdir(_bin):
                    os.add_dll_directory(_bin)
                    _log_file.write(f"[startup] DLL registered: {_bin}\n")
            break


# ─── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class Subtitle:
    start: float       # seconds
    end: float
    text: str
    translated: str = ""

    def display(self, use_translation: bool = False) -> str:
        return (self.translated or self.text) if use_translation else self.text


# ─── Helpers ───────────────────────────────────────────────────────────────────

def srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def hms(ms: int) -> str:
    if ms < 0:
        ms = 0
    total = ms // 1000
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


def save_srt(subs: List[Subtitle], path: str, use_translation: bool) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(subs, 1):
            body = (s.translated or s.text) if use_translation else s.text
            f.write(f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{body}\n\n")


def cache_path(media_file: str) -> str:
    return os.path.splitext(media_file)[0] + "_ai.json"


def save_cache(subs: List[Subtitle], media_file: str, translate_lang: str = "") -> None:
    import json
    data = {
        "version": 1,
        "media_file": media_file,
        "translate_lang": translate_lang,
        "subtitles": [
            {"start": s.start, "end": s.end,
             "text": s.text, "translated": s.translated}
            for s in subs
        ],
    }
    with open(cache_path(media_file), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cache(media_file: str) -> Optional[List[Subtitle]]:
    import json
    path = cache_path(media_file)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    subs = []
    for d in data.get("subtitles", []):
        subs.append(Subtitle(
            start=d["start"], end=d["end"],
            text=d["text"], translated=d.get("translated", ""),
        ))
    return subs


# ─── .env loader (no extra deps) ─────────────────────────────────────────────────

def _load_dotenv(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    result = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                result[k.strip()] = v.strip().strip("\"'")
    except FileNotFoundError:
        pass
    return result

_dotenv = _load_dotenv()


# ─── Processing Pipeline ───────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, log_cb, progress_cb,
                 asr_prog_cb=None, tra_prog_cb=None):
        self._log = log_cb
        self._prog = progress_cb
        self._asr_prog = asr_prog_cb or (lambda p, m: None)
        self._tra_prog = tra_prog_cb or (lambda p, m: None)

    def _emit(self, msg: str):
        self._log(msg)
        _log_file.write(msg + "\n")

    # ── Regex cleanup ─────────────────────────────────────────────────────

    def _regex_cleanup(self, text: str) -> str:
        text = text[0].upper() + text[1:] if text else text
        text = re.sub(r'\bi\b', 'I', text)
        if text and text[-1].isalpha():
            text += "."
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])(\S)', r'\1 \2', text)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    # ── LLM cleanup ───────────────────────────────────────────────────────

    def _llm_cleanup_batch(self, batch, model_name: str):
        """Fix ASR errors per-line via LLM. Returns cleaned lines 1:1 with input."""
        import ollama

        lines = [s.text for s in batch]
        block = "\n".join(f"<L{i}>{t}</L{i}>" for i, t in enumerate(lines))
        prompt = (
            "Fix punctuation, capitalization, and ASR errors in each line below.\n"
            "Keep the exact same number of lines. Keep all <L#> tags unchanged.\n"
            "Output only the corrected lines — nothing else.\n\n"
            + block
        )
        try:
            self._emit(f"[LLM] calling Ollama ({model_name}, {len(lines)} lines, {len(block)} chars)")
            resp = ollama.generate(
                model=model_name, prompt=prompt,
                options={"num_predict": 512, "temperature": 0.0},
            )
            cleaned = []
            for l in resp.response.strip().split("\n"):
                l = l.strip()
                if not l:
                    continue
                l = re.sub(r'</?L\d+>', '', l).strip()
                cleaned.append(l)
            self._emit(f"[LLM] Ollama returned {len(cleaned)} lines (expected {len(lines)})")
        except Exception as e:
            self._log(f"Ollama error: {e}")
            self._emit(f"[LLM] Ollama FAILED — falling back to regex")
            cleaned = [self._regex_cleanup(line) for line in lines]

        # Ensure 1:1 alignment — pad or trim to match input count
        while len(cleaned) < len(lines):
            cleaned.append(lines[len(cleaned)])
        cleaned = cleaned[:len(lines)]

        # Build output subtitles, preserving original timestamps
        out = []
        for orig, text in zip(batch, cleaned):
            out.append(Subtitle(
                start=orig.start, end=orig.end,
                text=text or orig.text,
            ))
        return out

    # ── Streaming pipeline ────────────────────────────────────────────────

    def run(self, path: str, device: str, whisper_model: str,
            translate_lang: str = None, do_llm: bool = False,
            model_name: str = "llama3.2:1b",
            on_subtitle=None, on_first=None) -> List[Subtitle]:
        """
        Streaming 3-stage pipeline: ASR → LLM/regex → Translation.
        All stages run concurrently via queues.
        Returns the final subtitle list.
        """
        import queue
        import torch
        from faster_whisper import WhisperModel

        if device == "cuda" and not torch.cuda.is_available():
            self._emit("WARNING: CUDA not available — falling back to CPU.")
            device = "cpu"
        ctype = "int8_float16" if device == "cuda" else "int8"
        self._emit(f"[pipe] device={device}  model={whisper_model}  do_llm={do_llm}")
        if device == "cuda":
            free, total = torch.cuda.mem_get_info(0)
            self._emit(f"       VRAM free={free//1024**2} MB / {total//1024**2} MB")

        self._prog(5, "Loading ASR model…")
        self._asr_prog(5, "loading model…")

        asr = WhisperModel(whisper_model, device=device, compute_type=ctype)
        self._asr_prog(10, "model loaded")
        if device == "cuda":
            import torch as _t
            _x = _t.zeros(4, 4, device="cuda")
            _t.matmul(_x, _x)
            _t.cuda.synchronize()

        seg_gen, info = asr.transcribe(
            path, beam_size=5, word_timestamps=True, vad_filter=False)
        duration = info.duration
        self._asr_prog(12, "transcribing…")

        # Queues: raw → cleaned → final
        raw_q = queue.Queue(maxsize=200)
        clean_q = queue.Queue(maxsize=200)

        final_subs: List[Subtitle] = []
        total_segs = [0]   # mutable counter shared across threads
        cleaned_count = [0]
        tra_count = [0]

        # ── Stage 1: ASR producer ─────────────────────────────────────────
        def asr_stage():
            for seg in seg_gen:
                text = seg.text.strip()
                if not text:
                    continue
                total_segs[0] += 1
                raw_q.put((seg.start, seg.end, text))
                if duration > 0:
                    pct = 12 + int(83 * seg.end / duration)
                    self._asr_prog(min(pct, 99), f"{seg.end:.0f}s / {duration:.0f}s")
            raw_q.put(None)  # sentinel
            self._emit(f"[ASR] done — {total_segs[0]} segments")
            self._asr_prog(100, f"done — {total_segs[0]} segs")

        # ── Stage 2: Cleanup (LLM or regex) ───────────────────────────────
        if do_llm:
            self._emit("[LLM] per-line cleanup mode")
            def cleanup_stage():
                batch = []
                batch_num = [0]
                first = [True]
                while True:
                    item = raw_q.get()
                    if item is None:
                        if batch:
                            self._emit("[LLM] processing final batch…")
                            for sub in self._llm_cleanup_batch(batch, model_name):
                                clean_q.put((sub.start, sub.end, sub.text))
                                cleaned_count[0] += 1
                        clean_q.put(None)
                        self._emit(f"[LLM] done — {cleaned_count[0]} segments")
                        # Unload Ollama model to free GPU memory for next run
                        try:
                            import ollama
                            ollama.generate(model=model_name, prompt="ok", keep_alive=0,
                                            options={"num_predict": 1})
                        except Exception:
                            pass
                        break
                    if first[0]:
                        first[0] = False
                        self._emit("[LLM] first segment arrived — buffering batches")
                    batch.append(Subtitle(start=item[0], end=item[1], text=item[2]))
                    chars = sum(len(s.text) for s in batch)
                    if len(batch) >= 8 or chars >= 400:
                        batch_num[0] += 1
                        self._emit(f"[LLM] batch {batch_num[0]} — {len(batch)} segs, {chars} chars")
                        for sub in self._llm_cleanup_batch(batch, model_name):
                            clean_q.put((sub.start, sub.end, sub.text))
                            cleaned_count[0] += 1
                        self._emit(f"[LLM] batch {batch_num[0]} done — total {cleaned_count[0]}")
                        batch = []
        else:
            self._emit("[cleanup] regex fast path")
            def cleanup_stage():
                while True:
                    item = raw_q.get()
                    if item is None:
                        clean_q.put(None)
                        self._emit(f"[cleanup] done — {cleaned_count[0]} segments")
                        break
                    text = self._regex_cleanup(item[2])
                    clean_q.put((item[0], item[1], text))
                    cleaned_count[0] += 1

        # ── Stage 3: Translation ──────────────────────────────────────────
        if translate_lang:
            self._emit(f"[translate] → {translate_lang}")
            self._tra_prog(0, "waiting…")
            def tra_stage():
                from deep_translator import GoogleTranslator
                tr = GoogleTranslator(source="auto", target=translate_lang)
                first = True
                while True:
                    item = clean_q.get()
                    if item is None:
                        self._emit(f"[translate] done — {tra_count[0]}")
                        self._tra_prog(100, f"done — {tra_count[0]}")
                        break
                    start_t, end_t, text = item
                    translated = tr.translate(text) or text
                    sub = Subtitle(start=start_t, end=end_t,                                    text=text, translated=translated)
                    final_subs.append(sub)
                    if on_subtitle:
                        on_subtitle(sub)
                    if first and on_first:
                        first = False; on_first()
                    tra_count[0] += 1
                    if cleaned_count[0] > 0:
                        pct = int(100 * tra_count[0] / max(1, cleaned_count[0]))
                        self._tra_prog(min(pct, 99), f"{tra_count[0]}/{cleaned_count[0]}")
                    time.sleep(0.05)
        else:
            self._tra_prog(0, "waiting…")
            def tra_stage():
                first = True
                while True:
                    item = clean_q.get()
                    if item is None:
                        self._tra_prog(100, "done")
                        break
                    sub = Subtitle(start=item[0], end=item[1],                                    text=item[2])
                    final_subs.append(sub)
                    if on_subtitle:
                        on_subtitle(sub)
                    if first and on_first:
                        first = False; on_first()
                    tra_count[0] += 1

        # ── Launch all stages ─────────────────────────────────────────────
        t1 = threading.Thread(target=asr_stage, name="asr")
        t2 = threading.Thread(target=cleanup_stage, name="cleanup")
        t3 = threading.Thread(target=tra_stage, name="tra")
        t1.start(); t2.start(); t3.start()
        t1.join(); t2.join(); t3.join()

        self._emit(f"[pipe] complete — {len(final_subs)} subtitles")
        self._prog(100, f"Done — {len(final_subs)} subtitles")
        # Free GPU memory for next run
        del asr, seg_gen
        if device == "cuda":
            import torch, gc
            torch.cuda.empty_cache()
            gc.collect()
        return final_subs


# ─── UI Colours ────────────────────────────────────────────────────────────────

DARK   = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#0f3460"
BTN    = "#1a5276"
BLUE   = "#4a9eff"
GREEN  = "#27ae60"


# ─── Application ───────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("AI Media Player")
        self.geometry("980x760")
        self.minsize(800, 600)
        self.configure(bg=DARK)

        # runtime state
        self.media_file: Optional[str] = None
        self.subtitles: List[Subtitle] = []
        self.srt_path: Optional[str] = None
        self.is_processed = False
        self._seeking = False
        self._sub_running = False
        self._cur_sub = ""
        self.player = None
        self.vlc = None
        self.vlc_inst = None

        # tk variables
        self.v_device       = tk.StringVar(value="cuda")
        self.v_whisper_model = tk.StringVar(value="large-v2")
        self.v_translate    = tk.BooleanVar(value=True)
        self.v_llm_cleanup = tk.BooleanVar(value=True)
        self.v_lang      = tk.StringVar(value="ko")
        self.v_model     = tk.StringVar(value="llama3.2:1b")
        self.v_status    = tk.StringVar(value="Ready — browse a file to begin.")
        self.v_progress  = tk.DoubleVar(value=0)
        self.v_asr_prog  = tk.DoubleVar(value=0)
        self.v_tra_prog  = tk.DoubleVar(value=0)
        self.v_asr_label = tk.StringVar(value="")
        self.v_tra_label = tk.StringVar(value="")
        self.v_subtitle  = tk.StringVar(value="")
        self.v_time      = tk.StringVar(value="0:00:00 / 0:00:00")
        self.v_seek      = tk.DoubleVar(value=0)

        self._build_ui()
        self._on_llm_cleanup_toggle()   # sync UI with default (True)
        self._on_translate_toggle()     # sync UI with default (True)
        self._init_vlc()

        # Keyboard shortcuts: Space / Enter to toggle play/pause
        self.bind("<space>", self._on_space_toggle)
        self.bind("<Return>", self._on_space_toggle)
        self.after(200, self._cuda_check)
        self._ui_loop()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        ttk.Style(self).configure("Bar.Horizontal.TProgressbar",
                                  background=BLUE, troughcolor="#333")

        # ── Top control panel ──────────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL, pady=6)
        top.pack(fill="x", padx=8, pady=(8, 0))

        # File row
        fr = tk.Frame(top, bg=PANEL)
        fr.pack(fill="x", padx=6, pady=2)
        self._lbl(fr, "File:").pack(side="left")
        self.lbl_file = self._lbl(fr, "No file selected", fg="#888")
        self.lbl_file.pack(side="left", fill="x", expand=True, padx=6)
        self._btn(fr, "Browse…", self._browse, BLUE).pack(side="right")

        # Settings row
        sr = tk.Frame(top, bg=PANEL)
        sr.pack(fill="x", padx=6, pady=3)

        self._lbl(sr, "Device:").pack(side="left")
        self.cmb_device = ttk.Combobox(sr, textvariable=self.v_device,
                                        width=6, state="readonly")
        self.cmb_device["values"] = ["cuda", "cpu"]
        self.cmb_device.pack(side="left", padx=(2, 14))

        tk.Checkbutton(sr, text="LLM Cleanup", variable=self.v_llm_cleanup,
                       bg=PANEL, fg="white", selectcolor=ACCENT,
                       activebackground=PANEL, activeforeground="white",
                       command=self._on_llm_cleanup_toggle).pack(side="left")
        self.ent_model = tk.Entry(sr, textvariable=self.v_model, width=13,
                                   bg=ACCENT, fg="white", insertbackground="white",
                                   relief="flat", state="disabled")
        self.ent_model.pack(side="left", padx=(2, 14))

        tk.Checkbutton(sr, text="Translate", variable=self.v_translate,
                       bg=PANEL, fg="white", selectcolor=ACCENT,
                       activebackground=PANEL, activeforeground="white",
                       command=self._on_translate_toggle).pack(side="left")
        self.cmb_lang = ttk.Combobox(sr, textvariable=self.v_lang,
                                      width=8, state="disabled")
        self.cmb_lang["values"] = [
            "ko", "zh-CN", "zh-TW", "ja", "fr", "de", "es", "ru", "ar", "pt"
        ]
        self.cmb_lang.pack(side="left", padx=(2, 14))

        self._lbl(sr, "Whisper:").pack(side="left")
        cmb_wm = ttk.Combobox(sr, textvariable=self.v_whisper_model,
                               width=10, state="readonly")
        cmb_wm["values"] = ["tiny", "base", "small", "medium",
                             "large-v2", "large-v3"]
        cmb_wm.pack(side="left", padx=2)

        # Action row
        ar = tk.Frame(top, bg=PANEL)
        ar.pack(fill="x", padx=6, pady=4)
        self.btn_proc = self._btn(ar, "▶  Prepare & Process",
                                   self._start_process, GREEN)
        self.btn_proc.pack(side="left")
        ttk.Progressbar(ar, variable=self.v_progress, maximum=100,
                         length=340, style="Bar.Horizontal.TProgressbar"
                         ).pack(side="left", padx=10)
        self._lbl(ar, textvariable=self.v_status, fg="#aaa").pack(side="left")

        # ── Parallel progress bars ──────────────────────────────────────────
        pp = tk.Frame(self, bg=PANEL, pady=2)
        pp.pack(fill="x", padx=8, pady=(2, 0))
        for var, lbl_var, color, name in [
            (self.v_asr_prog, self.v_asr_label, "#e74c3c", "ASR"),
            (self.v_tra_prog, self.v_tra_label, "#2ecc71", "TRA"),
        ]:
            row = tk.Frame(pp, bg=PANEL)
            row.pack(fill="x", padx=4, pady=1)
            self._lbl(row, f"{name}:", fg=color, bg=PANEL, width=5,
                      anchor="e").pack(side="left", padx=(0, 4))
            ttk.Progressbar(row, variable=var, maximum=100,
                            length=300, style="Bar.Horizontal.TProgressbar"
                            ).pack(side="left", fill="x", expand=True)
            self._lbl(row, textvariable=lbl_var, fg="#aaa", bg=PANEL,
                      width=18, anchor="w").pack(side="left", padx=4)

        # ── Video area ─────────────────────────────────────────────────────
        self.video_frame = tk.Frame(self, bg="black")
        self.video_frame.pack(fill="both", expand=True, padx=8, pady=4)

        # ── Subtitle bar ───────────────────────────────────────────────────
        sub_bg = tk.Frame(self, bg="black", height=44)
        sub_bg.pack(fill="x", padx=8)
        sub_bg.pack_propagate(False)
        tk.Label(sub_bg, textvariable=self.v_subtitle,
                 bg="black", fg="#FFFF00",
                 font=("Arial", 14, "bold"),
                 wraplength=940, justify="center"
                 ).pack(fill="both", expand=True)

        # ── Player controls ────────────────────────────────────────────────
        pc = tk.Frame(self, bg=ACCENT, pady=5)
        pc.pack(fill="x", padx=8, pady=(0, 4))

        icon_kw = dict(fg="white", relief="flat",
                       padx=10, pady=4, font=("Arial", 11), cursor="hand2")
        self._btn(pc, "⏮", self._seek_start, BTN, **icon_kw).pack(side="left", padx=2)
        self.btn_play = tk.Button(pc, text="▶", command=self._toggle_play,
                                   bg=GREEN, **icon_kw)
        self.btn_play.pack(side="left", padx=2)
        self._btn(pc, "⏭", self._seek_end, BTN, **icon_kw).pack(side="left", padx=2)

        seek = ttk.Scale(pc, variable=self.v_seek,
                          from_=0, to=1000, orient="horizontal")
        seek.pack(side="left", fill="x", expand=True, padx=8)
        seek.bind("<ButtonPress-1>", self._on_seek_press)
        seek.bind("<ButtonRelease-1>", self._on_seek_release)

        tk.Label(pc, textvariable=self.v_time,
                 bg=ACCENT, fg="white",
                 font=("Courier", 10)).pack(side="left", padx=6)

        self._lbl(pc, "Vol:", bg=ACCENT).pack(side="left")
        self.vol_scale = tk.Scale(pc, from_=0, to=100, orient="horizontal",
                                   length=90, bg=ACCENT, fg="white",
                                   troughcolor=BTN, highlightthickness=0,
                                   showvalue=False, command=self._set_volume)
        self.vol_scale.set(80)
        self.vol_scale.pack(side="left", padx=4)

        # ── Log console ────────────────────────────────────────────────────
        log_fr = tk.Frame(self, bg=PANEL)
        log_fr.pack(fill="x", padx=8, pady=(0, 8))
        self.log_box = scrolledtext.ScrolledText(
            log_fr, height=4, bg="#0d1b2a", fg="#7ec8e3",
            font=("Courier", 9), state="disabled", relief="flat")
        self.log_box.pack(fill="x")

    # ── VLC init ──────────────────────────────────────────────────────────────

    def _init_vlc(self):
        try:
            import vlc
            self.vlc = vlc
            self.vlc_inst = vlc.Instance("--no-xlib", "--quiet")
            self.player = self.vlc_inst.media_player_new()
            self.player.audio_set_volume(80)
            self._log("VLC initialised.")
        except Exception as e:
            self._log(f"VLC unavailable: {e}")
            messagebox.showwarning(
                "VLC Required",
                "Install VLC media player, then:\n  pip install python-vlc\n\n"
                "Transcription/translation will still work without VLC.")

    def _embed_player(self):
        if not self.player:
            return
        self.update()   # realise widget before grabbing HWND
        wid = self.video_frame.winfo_id()
        if sys.platform == "win32":
            self.player.set_hwnd(wid)
        elif sys.platform == "darwin":
            self.player.set_nsobject(wid)
        else:
            self.player.set_xwindow(wid)

    def _cuda_check(self):
        """Log CUDA status at startup so GPU availability is always visible."""
        try:
            import torch
            info = f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}"
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
                msg  = f"CUDA OK  →  {name}  ({mem} MB)  |  {info}"
                self._log(msg)
                _log_file.write(f"[startup] {msg}\n")
                self.v_device.set("cuda")
            else:
                msg = f"CUDA NOT available  |  {info}"
                self._log(msg)
                _log_file.write(f"[startup] {msg}\n")
                self.v_device.set("cpu")
            try:
                import ctranslate2
                ct2_ver = ctranslate2.__version__
                self._log(f"ctranslate2={ct2_ver}")
                _log_file.write(f"[startup] ctranslate2={ct2_ver}\n")
            except Exception:
                pass
        except ImportError:
            self._log("torch not installed — pip install torch ...")

    # ── Pipeline thread ───────────────────────────────────────────────────────

    def _start_process(self):
        if not self.media_file:
            messagebox.showwarning("No File", "Browse a media file first.")
            return
        self.btn_proc.config(state="disabled", text="Processing…")
        self._batch_lock = threading.Lock()
        self.is_processed = False
        self.subtitles = []
        self._reset_parallel_bars()
        threading.Thread(target=self._pipeline_thread, daemon=True).start()

    def _pipeline_thread(self):
        try:
            pipe = Pipeline(self._log, self._set_prog,
                            asr_prog_cb=self._set_asr_prog,
                            tra_prog_cb=self._set_tra_prog)
            mdl           = self.v_model.get().strip() or "llama3.2:1b"
            device        = self.v_device.get()
            whisper_model = self.v_whisper_model.get()
            translate_lang = self.v_lang.get() if self.v_translate.get() else None
            do_llm        = self.v_llm_cleanup.get()

            def on_subtitle(sub):
                with self._batch_lock:
                    self.subtitles.append(sub)

            def on_first():
                self.after(200, self._enable_early_play)

            all_subs = pipe.run(
                self.media_file, device, whisper_model,
                translate_lang=translate_lang,
                do_llm=do_llm, model_name=mdl,
                on_subtitle=on_subtitle, on_first=on_first)

            # Set subtitles and save
            with self._batch_lock:
                self.subtitles = all_subs
                self.is_processed = True

            base = os.path.splitext(self.media_file)[0]
            self.srt_path = base + "_ai.srt"
            save_cache(all_subs, self.media_file, translate_lang or "")
            save_srt(all_subs, self.srt_path, self.v_translate.get())
            self._log(f"Saved → {self.srt_path}  (+ cache)")

        except Exception as e:
            import traceback
            self._log(f"ERROR:\n{traceback.format_exc()}")
            self.after(0, lambda: messagebox.showerror("Pipeline Error", str(e)))
            self._set_prog(0, f"Error: {e}")
        finally:
            self.after(0, lambda: self.btn_proc.config(
                state="normal", text="▶  Prepare & Process"))

    def _enable_early_play(self):
        """Start playback as soon as the first batch of subtitles is ready."""
        if self.player and self.media_file and not self.player.is_playing():
            self._log("Early playback — first batch ready, loading video…")
            self._load_and_play()

    # ── Playback ──────────────────────────────────────────────────────────────

    def _load_and_play(self):
        if not self.player or not self.media_file:
            return
        self._embed_player()
        media = self.vlc_inst.media_new(self.media_file)
        media.add_option(":no-sub")            # skip loading subtitle tracks
        self.player.set_media(media)
        self.player.play()
        # Retry after a tick — spu track may not be available until playback starts
        self.after(100, lambda: self.player.video_set_spu(-1))
        self.btn_play.config(text="⏸")
        self._sub_running = True
        threading.Thread(target=self._sub_sync_loop, daemon=True).start()

    def _sub_sync_loop(self):
        """Background thread: push the right subtitle line to the UI."""
        while self._sub_running:
            try:
                if self.player and self.vlc:
                    state = self.player.get_state()
                    if state in (self.vlc.State.Playing, self.vlc.State.Paused):
                        pos_s = self.player.get_time() / 1000.0
                        text = ""
                        with getattr(self, "_batch_lock", threading.Lock()):
                            for s in self.subtitles:
                                if s.start <= pos_s <= s.end:
                                    text = s.display(self.v_translate.get())
                                    break
                        if text != self._cur_sub:
                            self._cur_sub = text
                            self.after(0, lambda t=text: self.v_subtitle.set(t))
            except Exception:
                pass
            time.sleep(0.08)

    def _on_space_toggle(self, event):
        """Space / Enter toggles play/pause, but not when typing in a text field."""
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        self._toggle_play()

    def _toggle_play(self):
        if not self.player:
            return
        if self.player.is_playing():
            self.player.pause()
            self.btn_play.config(text="▶")
        else:
            if self.player.get_media() is None:
                if self.subtitles:
                    self._load_and_play()
            else:
                # If playback ended, seek back to start before replaying
                if self.vlc and self.player.get_state() == self.vlc.State.Ended:
                    self.player.stop()
                self.player.play()
                self.btn_play.config(text="⏸")
                if not self._sub_running:
                    self._sub_running = True
                    threading.Thread(target=self._sub_sync_loop,
                                     daemon=True).start()

    def _seek_start(self):
        if self.player:
            self.player.set_time(0)

    def _seek_end(self):
        if self.player:
            L = self.player.get_length()
            if L > 0:
                self.player.set_time(max(0, L - 5000))

    def _on_seek_press(self, event):
        self._seeking = True
        w = event.widget.winfo_width()
        if w > 0:
            frac = max(0.0, min(1.0, event.x / w))
            self.v_seek.set(frac * 1000)
            if self.player and self.player.get_length() > 0:
                if self.vlc and self.player.get_state() == self.vlc.State.Ended:
                    self.player.stop()
                self.player.set_position(frac)

    def _on_seek_release(self, _event):
        if self.player and self.player.get_length() > 0:
            if self.vlc and self.player.get_state() == self.vlc.State.Ended:
                self.player.stop()
            self.player.set_position(self.v_seek.get() / 1000.0)
        self._seeking = False

    def _set_volume(self, val):
        if self.player:
            self.player.audio_set_volume(int(float(val)))

    def _ui_loop(self):
        """Periodic UI refresh: seek bar + time label + end-of-media detection."""
        try:
            if self.player:
                L = self.player.get_length()
                t = self.player.get_time()
                if L > 0 and not self._seeking:
                    self.v_seek.set((t / L) * 1000)
                    self.v_time.set(f"{hms(t)} / {hms(L)}")
                if self.vlc and self.player.get_state() == self.vlc.State.Ended:
                    self.btn_play.config(text="▶")
                    self._sub_running = False
                    self.v_subtitle.set("")
        except Exception:
            pass
        self.after(300, self._ui_loop)

    # ── Misc callbacks ────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=[
                ("Media", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv "
                           "*.mp3 *.wav *.m4a *.flac *.ogg *.aac"),
                ("All",   "*.*"),
            ],
        )
        if path:
            self.media_file = path
            self.lbl_file.config(text=os.path.basename(path), fg="white")
            self.is_processed = False
            self.subtitles = []
            self.v_subtitle.set("")
            self._log(f"Selected: {path}")

            # Check for cached results
            cached = load_cache(path)
            if cached is not None:
                self.subtitles = cached
                self.is_processed = True
                self.srt_path = cache_path(path).replace(".json", ".srt")
                self._log(f"Loaded cached results — {len(cached)} subtitles ready to play")
                self._set_prog(100, "Ready — cached results loaded.")

    def _on_translate_toggle(self):
        self.cmb_lang.config(
            state="readonly" if self.v_translate.get() else "disabled")

    def _on_llm_cleanup_toggle(self):
        self.ent_model.config(
            state="normal" if self.v_llm_cleanup.get() else "disabled")

    def _log(self, msg: str):
        def _do():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _do)

    def _set_prog(self, pct: float, msg: str):
        self.after(0, lambda: self.v_progress.set(pct))
        self.after(0, lambda: self.v_status.set(msg))

    def _set_asr_prog(self, pct: float, msg: str = ""):
        self.after(0, lambda: self.v_asr_prog.set(pct))
        self.after(0, lambda: self.v_asr_label.set(msg))

    def _set_tra_prog(self, pct: float, msg: str = ""):
        self.after(0, lambda: self.v_tra_prog.set(pct))
        self.after(0, lambda: self.v_tra_label.set(msg))

    def _reset_parallel_bars(self):
        self.after(0, lambda: self.v_asr_prog.set(0))
        self.after(0, lambda: self.v_tra_prog.set(0))
        self.after(0, lambda: self.v_asr_label.set(""))
        self.after(0, lambda: self.v_tra_label.set(""))

    def _on_close(self):
        self._sub_running = False
        if self.player:
            self.player.stop()
        self.destroy()

    # ── Widget helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _lbl(parent, text="", fg="white", bg=None, **kw):
        return tk.Label(parent, text=text, fg=fg,
                        bg=bg or parent.cget("bg"), **kw)

    @staticmethod
    def _btn(parent, text, cmd, color, **kw):
        kw.setdefault("fg", "white")
        kw.setdefault("relief", "flat")
        kw.setdefault("padx", 10)
        kw.setdefault("pady", 4)
        kw.setdefault("cursor", "hand2")
        return tk.Button(parent, text=text, command=cmd, bg=color, **kw)


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()
