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
    2. Set speaker count + optional translation
    3. Click "Prepare & Process" — runs faster-whisper ASR → Ollama LLM cleanup → (optional) Google Translate
    4. Press Play — subtitles sync automatically during playback
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import re
import sys
import subprocess
import tempfile
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
    speaker: str
    text: str
    translated: str = ""

    def display(self, use_translation: bool = False) -> str:
        body = (self.translated or self.text) if use_translation else self.text
        return f"[{self.speaker}]  {body}"


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
            f.write(f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{s.speaker}: {body}\n\n")


def cache_path(media_file: str) -> str:
    return os.path.splitext(media_file)[0] + "_ai.json"


def save_cache(subs: List[Subtitle], media_file: str, translate_lang: str = "") -> None:
    import json
    data = {
        "version": 1,
        "media_file": media_file,
        "translate_lang": translate_lang,
        "subtitles": [
            {"start": s.start, "end": s.end, "speaker": s.speaker,
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
            speaker=d.get("speaker", "SPEAKER_00"),
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
                 asr_prog_cb=None, dia_prog_cb=None, tra_prog_cb=None):
        self._log = log_cb
        self._prog = progress_cb
        self._asr_prog = asr_prog_cb or (lambda p, m: None)
        self._dia_prog = dia_prog_cb or (lambda p, m: None)
        self._tra_prog = tra_prog_cb or (lambda p, m: None)

    def _emit(self, msg: str):
        self._log(msg)
        _log_file.write(msg + "\n")

    def _ensure_wav(self, path: str) -> str:
        """If path is not a .wav, convert to temp .wav via ffmpeg so soundfile can read it."""
        if path.lower().endswith(".wav"):
            return path
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        self._emit(f"[ffmpeg] extracting audio → {tmp.name}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True,
        )
        return tmp.name

    # ── Parallel ASR + Diarization ─────────────────────────────────────────

    def _run_asr(self, device: str, whisper_model: str, ctype: str, path: str):
        """Load faster-whisper + transcribe. Runs in its own thread."""
        from faster_whisper import WhisperModel

        self._emit("[ASR] load — start")
        self._asr_prog(5, "loading model…")
        asr = WhisperModel(whisper_model, device=device, compute_type=ctype)
        self._emit("[ASR] load — done")
        self._asr_prog(10, "model loaded")

        if device == "cuda":
            import torch as _t
            _x = _t.zeros(4, 4, device="cuda")
            _t.matmul(_x, _x)
            _t.cuda.synchronize()
            self._emit("[ASR] CUDA matmul OK")

        self._emit(f"[ASR] transcribe — start")
        self._asr_prog(12, "transcribing…")
        seg_gen, info = asr.transcribe(
            path, beam_size=5, word_timestamps=True, vad_filter=False)
        duration = info.duration
        raw_segs = []
        for seg in seg_gen:
            raw_segs.append(seg)
            if duration > 0:
                pct = 12 + int(83 * seg.end / duration)
                self._asr_prog(min(pct, 95), f"{seg.end:.0f}s / {duration:.0f}s")
        self._emit(f"[ASR] transcribe done — lang={info.language}  segs={len(raw_segs)}")
        self._asr_prog(100, f"done — {len(raw_segs)} segments")
        return raw_segs, info

    def _run_diarization(self, hf_token: str, path: str, n_speakers: int):
        """Load pyannote + diarize. Runs in its own thread (CPU only)."""
        import torch
        from pyannote.audio import Pipeline as PyannoteP

        self._emit("[DIA] load — start")
        self._dia_prog(5, "loading model…")
        dia_pipe = PyannoteP.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        dia_pipe = dia_pipe.to(torch.device("cpu"))
        self._emit("[DIA] load — done")
        self._dia_prog(15, "model loaded")

        audio_path = self._ensure_wav(path)
        dia_kwargs = {}
        if n_speakers > 0:
            dia_kwargs["num_speakers"] = n_speakers
        self._emit("[DIA] diarize — start")
        self._dia_prog(20, "diarizing…")
        diarization = dia_pipe(audio_path, **dia_kwargs)
        turns = sorted(
            (seg.start, seg.end, lbl)
            for seg, _, lbl in diarization.itertracks(yield_label=True)
        )
        self._emit(f"[DIA] diarize done — turns={len(turns)}")
        self._dia_prog(100, f"done — {len(turns)} turns")
        return turns

    def transcribe(self, path: str, n_speakers: int, hf_token: str,
                   device: str = "cuda",
                   whisper_model: str = "medium",
                   translate_lang: str = None) -> List[Subtitle]:
        """
        Runs ASR and diarization in parallel, then merges results.
        If translate_lang is set, translation starts as soon as ASR finishes
        (runs in parallel with diarization on CPU).
        ASR uses GPU (faster-whisper/ctranslate2), diarization uses CPU (pyannote).
        """
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            self._emit("WARNING: CUDA not available — falling back to CPU.")
            device = "cpu"

        ctype = "int8_float16" if device == "cuda" else "int8"
        self._emit(f"[pipe] device={device}  model={whisper_model}  compute_type={ctype}")
        if device == "cuda":
            free, total = torch.cuda.mem_get_info(0)
            self._emit(f"       VRAM free={free//1024**2} MB / {total//1024**2} MB")

        self._prog(5, "Loading ASR + diarization models in parallel…")

        # ── Launch ASR and diarization in parallel threads ────────────────
        asr_result = {}
        dia_result = {}
        has_dia = bool(hf_token)

        def _asr_thread():
            try:
                asr_result["data"] = self._run_asr(device, whisper_model, ctype, path)
            except Exception as e:
                asr_result["error"] = e

        def _dia_thread():
            try:
                dia_result["data"] = self._run_diarization(hf_token, path, n_speakers)
            except Exception as e:
                dia_result["error"] = e

        t_asr = threading.Thread(target=_asr_thread, name="asr")
        t_dia = threading.Thread(target=_dia_thread, name="dia")

        if has_dia:
            t_asr.start()
            t_dia.start()
        else:
            t_asr.start()

        # ── Wait for ASR, then start translation early (parallel with DIA) ─
        t_asr.join()
        if "error" in asr_result:
            self._emit(f"[ASR] FAILED: {asr_result['error']}")
            self._emit("  Model files may not be cached. First run downloads them from HuggingFace.")
            raise asr_result["error"]
        raw_segs, info = asr_result["data"]

        # Build temp subtitles from ASR and start translation while DIA runs
        temp_subs = []
        for seg in raw_segs:
            text = seg.text.strip()
            if not text:
                continue
            temp_subs.append(Subtitle(start=seg.start, end=seg.end,
                                       speaker="", text=text))

        tra_done = [True]  # mutable; True if no translation needed
        if translate_lang and temp_subs:
            self._emit("[translate] starting early — parallel with diarization")
            self._tra_prog(0, "starting…")
            tra_done[0] = False
            def _tra_thread():
                self._translate_batch(temp_subs, translate_lang,
                                       offset=0, total=len(temp_subs))
                tra_done[0] = True
                self._emit("[translate] early pass done")
            threading.Thread(target=_tra_thread, name="tra-early", daemon=True).start()

        # ── Wait for diarization ──────────────────────────────────────────
        if has_dia:
            t_dia.join()

        # ── Check diarization result ──────────────────────────────────────
        turns = None
        if has_dia:
            if "error" in dia_result:
                self._emit(f"[DIA] FAILED: {dia_result['error']}")
            else:
                turns = dia_result["data"]

        # ── Build speaker lookup ──────────────────────────────────────────
        if turns:
            def _speaker_at(t: float) -> str:
                for s, e, lbl in turns:
                    if s <= t <= e:
                        return lbl
                    if s > t:
                        break
                return "SPEAKER_00"
        else:
            self._emit("[DIA] unavailable — using default speaker")
            self._emit("  (HF token may be missing, or accept the license at:")
            self._emit("   https://huggingface.co/pyannote/speaker-diarization-3.1)")
            def _speaker_at(t: float) -> str:
                return "SPEAKER_00"

        # ── Merge ASR segments with speaker labels ────────────────────────
        self._prog(65, "Merging speakers…")
        self._emit("[merge] building subtitles")
        subs = []
        ti = 0  # index into temp_subs (for copying early translations)
        for seg in raw_segs:
            text = seg.text.strip()
            if not text:
                continue
            mid = (seg.start + seg.end) / 2
            # Copy translation from temp_subs if early translation ran
            translated = ""
            if translate_lang and ti < len(temp_subs):
                translated = temp_subs[ti].translated
            subs.append(Subtitle(
                start=seg.start, end=seg.end,
                speaker=_speaker_at(mid), text=text,
                translated=translated,
            ))
            ti += 1
        self._emit(f"[merge] done — subtitles={len(subs)}")
        return subs

    # ── Fast regex cleanup (no LLM, instant) ──────────────────────────────

    def _regex_cleanup(self, text: str) -> str:
        """Basic ASR cleanup with regex — instant, no network/LLM needed."""
        # Capitalize first letter
        text = text[0].upper() + text[1:] if text else text
        # Capitalize " i " → " I "
        text = re.sub(r'\bi\b', 'I', text)
        # Add period at end of sentence-like lines lacking punctuation
        if text and text[-1].isalpha():
            text += "."
        # Fix spacing: no space before punctuation, one space after
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])(\S)', r'\1 \2', text)
        # Remove repeated words (common ASR artifact)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        # Collapse multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def _fast_cleanup(self, subs: List[Subtitle]) -> List[Subtitle]:
        """Apply regex cleanup to all subtitles — instant."""
        for s in subs:
            s.text = self._regex_cleanup(s.text)
        return subs

    # ── Streaming batch processing ────────────────────────────────────────

    def _cleanup_one_batch(self, batch: List[Subtitle], n_speakers: int,
                           model_name: str) -> List[Subtitle]:
        """LLM cleanup for a single batch via Ollama."""
        import ollama

        lines = [f"[{s.speaker}] {s.text}" for s in batch]
        block = "\n".join(lines)
        prompt = (
            f"You are a transcript editor. There are {n_speakers} speaker(s).\n"
            "Fix punctuation, capitalization, and obvious ASR errors.\n"
            "Keep the EXACT same number of lines and [SPEAKER_XX] labels unchanged.\n"
            "Output only the corrected lines — nothing else.\n\n"
            + block
        )

        try:
            resp = ollama.generate(model=model_name, prompt=prompt)
            cleaned = [l for l in resp.response.strip().split("\n") if l.strip()]
        except Exception as e:
            self._log(f"Ollama error: {e}")
            cleaned = lines

        out = []
        for i, orig in enumerate(batch):
            raw = cleaned[i] if i < len(cleaned) else f"[{orig.speaker}] {orig.text}"
            m = re.match(r"\[([^\]]+)\]\s*(.*)", raw)
            out.append(Subtitle(
                start=orig.start,
                end=orig.end,
                speaker=m.group(1) if m else orig.speaker,
                text=(m.group(2).strip() if m else raw.strip()) or orig.text,
            ))
        return out

    def _translate_batch(self, batch: List[Subtitle], target: str,
                          offset: int = 0, total: int = 0) -> List[Subtitle]:
        """Translate a single batch via Google Translate."""
        from deep_translator import GoogleTranslator

        tr = GoogleTranslator(source="auto", target=target)
        n = len(batch)
        for i, s in enumerate(batch):
            try:
                s.translated = tr.translate(s.text) or s.text
            except Exception as e:
                self._log(f"Translate error: {e}")
                s.translated = s.text
            if total > 0:
                pct = int(100 * (offset + i + 1) / total)
                self._tra_prog(pct, f"{offset+i+1}/{total}")
            time.sleep(0.05)
        return batch

    def process_batches(self, subs: List[Subtitle], n_speakers: int,
                        model_name: str, translate_lang: str = None,
                        on_batch=None, llm_cleanup: bool = False) -> List[Subtitle]:
        """
        Process subtitles: regex cleanup (instant) or LLM cleanup (slow), then optional translate.
        Calls on_batch(batch_index, batch_subs) after each batch completes.
        Returns the full processed list.
        """
        # Fast path: regex-only cleanup, no batching needed
        if not llm_cleanup:
            self._emit("[cleanup] regex fast path")
            subs = self._fast_cleanup(subs)
            # Trigger playback immediately — translation runs in background after
            if on_batch:
                on_batch(0, subs)
            if translate_lang:
                self._emit("[translate] background — original text shown until ready")
                subs = self._translate_batch(subs, translate_lang,
                                              offset=0, total=len(subs))
            self._emit("[cleanup] done")
            return subs

        # Slow path: LLM cleanup in batches
        chunk_size = 25
        total_batches = (len(subs) + chunk_size - 1) // chunk_size
        out: List[Subtitle] = []

        for bi in range(total_batches):
            batch = subs[bi * chunk_size : (bi + 1) * chunk_size]
            self._emit(f"[batch {bi+1}/{total_batches}] LLM cleanup…")
            cleaned = self._cleanup_one_batch(batch, n_speakers, model_name)
            out.extend(cleaned)
            # Trigger playback after first batch — translation runs after
            if on_batch:
                on_batch(bi, cleaned)
            if translate_lang:
                cleaned = self._translate_batch(cleaned, translate_lang,
                                                 offset=bi * chunk_size,
                                                 total=len(subs))

        self._emit(f"[batch] all {total_batches} batches done")
        return out


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
        self.v_speakers     = tk.IntVar(value=2)
        self.v_device       = tk.StringVar(value="cuda")
        self.v_whisper_model = tk.StringVar(value="medium")
        self.v_translate    = tk.BooleanVar(value=False)
        self.v_llm_cleanup = tk.BooleanVar(value=False)
        self.v_lang      = tk.StringVar(value="zh-CN")
        self.v_hf        = tk.StringVar(value=_dotenv.get("HF_TOKEN", ""))
        self.v_model     = tk.StringVar(value="llama3.2")
        self.v_status    = tk.StringVar(value="Ready — browse a file to begin.")
        self.v_progress  = tk.DoubleVar(value=0)
        self.v_asr_prog  = tk.DoubleVar(value=0)
        self.v_dia_prog  = tk.DoubleVar(value=0)
        self.v_tra_prog  = tk.DoubleVar(value=0)
        self.v_asr_label = tk.StringVar(value="")
        self.v_dia_label = tk.StringVar(value="")
        self.v_tra_label = tk.StringVar(value="")
        self.v_subtitle  = tk.StringVar(value="")
        self.v_time      = tk.StringVar(value="0:00:00 / 0:00:00")
        self.v_seek      = tk.DoubleVar(value=0)

        self._build_ui()
        self._init_vlc()
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

        self._lbl(sr, "Speakers:").pack(side="left")
        tk.Spinbox(sr, from_=1, to=10, textvariable=self.v_speakers,
                   width=3, bg=ACCENT, fg="white",
                   buttonbackground=ACCENT, insertbackground="white",
                   relief="flat").pack(side="left", padx=(2, 14))

        self._lbl(sr, "Device:").pack(side="left")
        self.cmb_device = ttk.Combobox(sr, textvariable=self.v_device,
                                        width=6, state="readonly")
        self.cmb_device["values"] = ["cuda", "cpu"]
        self.cmb_device.pack(side="left", padx=(2, 14))

        tk.Checkbutton(sr, text="Translate", variable=self.v_translate,
                       bg=PANEL, fg="white", selectcolor=ACCENT,
                       activebackground=PANEL, activeforeground="white",
                       command=self._on_translate_toggle).pack(side="left")
        self.cmb_lang = ttk.Combobox(sr, textvariable=self.v_lang,
                                      width=8, state="disabled")
        self.cmb_lang["values"] = [
            "zh-CN", "zh-TW", "ja", "ko", "fr", "de", "es", "ru", "ar", "pt"
        ]
        self.cmb_lang.pack(side="left", padx=(2, 14))

        tk.Checkbutton(sr, text="LLM Cleanup", variable=self.v_llm_cleanup,
                       bg=PANEL, fg="white", selectcolor=ACCENT,
                       activebackground=PANEL, activeforeground="white",
                       command=self._on_llm_cleanup_toggle).pack(side="left")
        self.ent_model = tk.Entry(sr, textvariable=self.v_model, width=13,
                                   bg=ACCENT, fg="white", insertbackground="white",
                                   relief="flat", state="disabled")
        self.ent_model.pack(side="left", padx=2)

        # Whisper model row
        wr = tk.Frame(top, bg=PANEL)
        wr.pack(fill="x", padx=6, pady=(0, 2))
        self._lbl(wr, "Whisper Model:").pack(side="left")
        cmb_wm = ttk.Combobox(wr, textvariable=self.v_whisper_model,
                               width=10, state="readonly")
        cmb_wm["values"] = ["tiny", "base", "small", "medium",
                             "large-v2", "large-v3"]
        cmb_wm.pack(side="left", padx=(2, 0))
        self._lbl(wr, "  (start with 'medium' if large crashes)",
                  fg="#888").pack(side="left")

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
            (self.v_dia_prog, self.v_dia_label, "#3498db", "DIA"),
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
        seek.bind("<ButtonPress-1>",   lambda _: setattr(self, "_seeking", True))
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
                            dia_prog_cb=self._set_dia_prog,
                            tra_prog_cb=self._set_tra_prog)
            n             = self.v_speakers.get()
            tok           = self.v_hf.get().strip()
            mdl           = self.v_model.get().strip() or "llama3.2"
            device        = self.v_device.get()
            whisper_model = self.v_whisper_model.get()
            translate_lang = self.v_lang.get() if self.v_translate.get() else None
            do_llm        = self.v_llm_cleanup.get()

            # Phase 1+2: Parallel ASR + diarization, then merge.
            # Fast path: start translation early (parallel with diarization)
            early_tra = translate_lang if not do_llm else None
            subs = pipe.transcribe(self.media_file, n, tok, device, whisper_model,
                                   translate_lang=early_tra)

            # Phase 3: Cleanup (regex or LLM) + optional translation
            # Skip translation if already done early (fast path)
            post_tra = translate_lang if do_llm else None
            if do_llm:
                self._set_prog(70, "LLM cleanup…")
            else:
                self._set_prog(70, "Cleanup…")

            first_batch = [True]
            def on_batch(batch_idx: int, batch_subs: list):
                with self._batch_lock:
                    self.subtitles.extend(batch_subs)
                total = max(1, (len(subs) + 24) // 25) if do_llm else 1
                pct = 70 + int(30 * (batch_idx + 1) / total)
                done_msg = ""
                if first_batch[0]:
                    first_batch[0] = False
                    done_msg = " — ✓ ready to play"
                    self.after(200, self._enable_early_play)
                self._set_prog(pct, f"Batch {batch_idx+1}/{total}{done_msg}")

            all_subs = pipe.process_batches(subs, n, mdl, post_tra, on_batch,
                                            llm_cleanup=do_llm)

            # Save cache + SRT with final subtitle list
            base = os.path.splitext(self.media_file)[0]
            self.srt_path = base + "_ai.srt"
            save_cache(all_subs, self.media_file, translate_lang or "")
            save_srt(all_subs, self.srt_path, self.v_translate.get())
            self._log(f"Saved → {self.srt_path}  (+ cache)")

            self._set_prog(100, "Done")
            with self._batch_lock:
                self.is_processed = True

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
        self.player.set_media(media)
        self.player.play()
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

    def _toggle_play(self):
        if not self.player:
            return
        if self.player.is_playing():
            self.player.pause()
            self.btn_play.config(text="▶")
        else:
            if self.player.get_media() is None:
                # Start playback if subtitles are available (even if still processing)
                if self.subtitles:
                    self._load_and_play()
            else:
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

    def _on_seek_release(self, _event):
        if self.player and self.player.get_length() > 0:
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

    def _set_dia_prog(self, pct: float, msg: str = ""):
        self.after(0, lambda: self.v_dia_prog.set(pct))
        self.after(0, lambda: self.v_dia_label.set(msg))

    def _set_tra_prog(self, pct: float, msg: str = ""):
        self.after(0, lambda: self.v_tra_prog.set(pct))
        self.after(0, lambda: self.v_tra_label.set(msg))

    def _reset_parallel_bars(self):
        self.after(0, lambda: self.v_asr_prog.set(0))
        self.after(0, lambda: self.v_dia_prog.set(0))
        self.after(0, lambda: self.v_tra_prog.set(0))
        self.after(0, lambda: self.v_asr_label.set(""))
        self.after(0, lambda: self.v_dia_label.set(""))
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
