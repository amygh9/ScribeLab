from __future__ import annotations

"""
Welcome to the metrics module !

This module contains helper functions and classes for evaluating
ASR output. It computes transcription quality metrics such as WER,
CER, PER and BERTScore, tracks latency values during
live streaming, and stores session results in CSV files.
"""

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List


# --------------------------------------------------
# Optional metric dependencies
# --------------------------------------------------

#  PER support
try:
    from phonemizer import phonemize
    _HAS_PHONEMIZER = True
except Exception:
    phonemize = None
    _HAS_PHONEMIZER = False


#  BERTScore support
try:
    from bert_score import score as bertscore_score
    _HAS_BERTSCORE = True
except Exception:
    bertscore_score = None
    _HAS_BERTSCORE = False


# --------------------------------------------------
# Text normalization patterns
# --------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\säöüÄÖÜß-]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------
# Text normalization configuration
# --------------------------------------------------

@dataclass
class TextNormConfig:
    lowercase: bool = True
    remove_punct: bool = True
    collapse_whitespace: bool = True
    strip: bool = True


# --------------------------------------------------
# Text preprocessing helpers
# --------------------------------------------------

def normalize_text(text: str, cfg: TextNormConfig) -> str:
    # Normalize text  before metric calculation
    t = text or ""

    if cfg.lowercase:
        t = t.lower()

    if cfg.remove_punct:
        t = _PUNCT_RE.sub(" ", t)

    if cfg.collapse_whitespace:
        t = _WS_RE.sub(" ", t)

    if cfg.strip:
        t = t.strip()

    return t


def split_words(text: str) -> List[str]:
    # Split text into word tokens for WER
    t = text.strip()
    return t.split() if t else []


def split_chars(text: str) -> List[str]:
    # Split text into characters, ignoring whitespace for CER
    return [c for c in text if not c.isspace()]


def split_sentences(text: str) -> List[str]:
    # Split text into simple sentence-like units
    t = text.strip()

    if not t:
        return []

    parts = re.split(r"[.!?\n]+", t)

    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------
# Edit-distance  with levenshtein matrix
# --------------------------------------------------

def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    
    n, m = len(ref), len(hyp)

    if n == 0:
        return m

    if m == 0:
        return n

    prev = list(range(m + 1))

    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]

        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1

            cur[j] = min(
                prev[j] + 1,         # deletion
                cur[j - 1] + 1,      # insertion
                prev[j - 1] + cost,  # substitution
            )

        prev = cur

    return prev[m]


def error_rate(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0

    d = levenshtein_distance(ref_tokens, hyp_tokens)

    return d / float(len(ref_tokens))


# --------------------------------------------------
# phoneme conversion
# --------------------------------------------------

def phonemize_tokens(
    text: str,
    language: str = "de",
    backend: str = "espeak",
) -> Optional[List[str]]:
    # Convert text into phoneme tokens
    if not _HAS_PHONEMIZER:
        return None

    ph = phonemize(
        text,
        language=language,
        backend=backend,
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        njobs=1,
    )

    ph = _WS_RE.sub(" ", ph).strip()

    return ph.split() if ph else []


# --------------------------------------------------
# Metric result data structures
# --------------------------------------------------

@dataclass
class MetricsResult:
    wer: float
    cer: float
    ser: float
    per: Optional[float]
    bertscore_precision: Optional[float]
    bertscore_recall: Optional[float]
    bertscore_f1: Optional[float]


@dataclass
class LatencyResult:
    stream_duration_s: Optional[float]
    audio_duration_s: Optional[float]
    model_compute_time_s: Optional[float]
    rtf: Optional[float]

    stream_start_to_first_partial_s: Optional[float]
    stream_start_to_first_final_s: Optional[float]
    speech_start_to_first_partial_s: Optional[float]
    speech_start_to_first_final_s: Optional[float]
    stop_to_last_final_s: Optional[float]


@dataclass
class OutputLatencyResult:
    start_to_first_output_s: Optional[float]
    start_to_first_partial_output_s: Optional[float]
    start_to_first_final_output_s: Optional[float]

    num_outputs: int
    mean_inter_output_s: Optional[float]
    min_inter_output_s: Optional[float]
    max_inter_output_s: Optional[float]
    last_output_time_from_start_s: Optional[float]

    output_times_s: List[float]
    output_types: List[str]


# --------------------------------------------------
# Latency tracking
# --------------------------------------------------

class LatencyTracker:
    def __init__(self, speech_rms_threshold: float = 0.0025):
        self.speech_rms_threshold = speech_rms_threshold
        self.reset()

    def reset(self):
        # Reset all timestamps and counters
        self.stream_start_t: Optional[float] = None
        self.stream_stop_t: Optional[float] = None
        self.speech_start_t: Optional[float] = None
        self.first_partial_t: Optional[float] = None
        self.first_final_t: Optional[float] = None
        self.last_final_t: Optional[float] = None

        self.audio_duration_s: float = 0.0
        self.model_compute_time_s: float = 0.0

    def on_stream_start(self):
        # Mark the start of the live stream
        self.reset()
        self.stream_start_t = time.time()

    def on_stream_stop(self):
        # Mark the end of the live stream
        self.stream_stop_t = time.time()

    def on_audio_chunk(self, rms: float, chunk_duration_s: float):
        # Accumulate processed audio duration
        self.audio_duration_s += float(chunk_duration_s)

        if self.stream_start_t is None:
            return

        # Estimate speech start using the RMS threshold
        if self.speech_start_t is None and rms >= self.speech_rms_threshold:
            self.speech_start_t = time.time()

    def on_partial(self):
        # Store the first partial output timestamp
        if self.first_partial_t is None:
            self.first_partial_t = time.time()

    def on_final(self):
        # Store first and latest final output timestamps
        now = time.time()

        if self.first_final_t is None:
            self.first_final_t = now

        self.last_final_t = now

    def result(self) -> LatencyResult:
        # Convert collected timestamps into latency values
        def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None:
                return None

            return max(0.0, b - a)

        audio_duration = self.audio_duration_s if self.audio_duration_s > 0 else None
        compute_time = self.model_compute_time_s if self.model_compute_time_s > 0 else None

        rtf = None

        if audio_duration is not None and compute_time is not None:
            rtf = compute_time / audio_duration

        return LatencyResult(
            stream_duration_s=delta(self.stream_start_t, self.stream_stop_t),
            audio_duration_s=audio_duration,
            model_compute_time_s=compute_time,
            rtf=rtf,
            stream_start_to_first_partial_s=delta(self.stream_start_t, self.first_partial_t),
            stream_start_to_first_final_s=delta(self.stream_start_t, self.first_final_t),
            speech_start_to_first_partial_s=delta(self.speech_start_t, self.first_partial_t),
            speech_start_to_first_final_s=delta(self.speech_start_t, self.first_final_t),
            stop_to_last_final_s=delta(self.stream_stop_t, self.last_final_t),
        )


# --------------------------------------------------
# Output latency tracking
# --------------------------------------------------

class OutputLatencyTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        # Reset output timing information
        self.start_t: Optional[float] = None
        self.first_output_t: Optional[float] = None
        self.first_partial_output_t: Optional[float] = None
        self.first_final_output_t: Optional[float] = None

        self.output_times_abs: List[float] = []
        self.output_types: List[str] = []

    def on_stream_start(self):
        # Mark the start used for output latency measurement
        self.reset()
        self.start_t = time.time()

    def on_output(self, output_type: str):
        # Register one partial or final output event
        if self.start_t is None:
            return

        now = time.time()

        if self.first_output_t is None:
            self.first_output_t = now

        if output_type == "partial" and self.first_partial_output_t is None:
            self.first_partial_output_t = now

        if output_type == "final" and self.first_final_output_t is None:
            self.first_final_output_t = now

        self.output_times_abs.append(now)
        self.output_types.append(output_type)

    def result(self) -> OutputLatencyResult:
        # Summarize output timings and inter-output intervals
        def delta_from_start(t: Optional[float]) -> Optional[float]:
            if self.start_t is None or t is None:
                return None

            return max(0.0, t - self.start_t)

        if self.start_t is None:
            return OutputLatencyResult(
                start_to_first_output_s=None,
                start_to_first_partial_output_s=None,
                start_to_first_final_output_s=None,
                num_outputs=0,
                mean_inter_output_s=None,
                min_inter_output_s=None,
                max_inter_output_s=None,
                last_output_time_from_start_s=None,
                output_times_s=[],
                output_types=[],
            )

        output_times_s = [
            max(0.0, t - self.start_t)
            for t in self.output_times_abs
        ]

        inter = []

        for i in range(1, len(self.output_times_abs)):
            inter.append(
                max(0.0, self.output_times_abs[i] - self.output_times_abs[i - 1])
            )

        mean_inter = sum(inter) / len(inter) if inter else None
        min_inter = min(inter) if inter else None
        max_inter = max(inter) if inter else None
        last_output = output_times_s[-1] if output_times_s else None

        return OutputLatencyResult(
            start_to_first_output_s=delta_from_start(self.first_output_t),
            start_to_first_partial_output_s=delta_from_start(self.first_partial_output_t),
            start_to_first_final_output_s=delta_from_start(self.first_final_output_t),
            num_outputs=len(self.output_times_abs),
            mean_inter_output_s=mean_inter,
            min_inter_output_s=min_inter,
            max_inter_output_s=max_inter,
            last_output_time_from_start_s=last_output,
            output_times_s=output_times_s,
            output_types=list(self.output_types),
        )


# --------------------------------------------------
# Hypothesis session storage
# --------------------------------------------------

class HypothesisSessionStore:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(
            self.out_dir,
            "hypotheses_sessions.csv",
        )

        self._ensure_header()

    def _ensure_header(self):
        # Create the CSV file with a header if it does not exist yet
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            w.writerow([
                "session_id",
                "model",
                "test_id",
                "stream_start_utc",
                "stream_stop_utc",
                "hyp_text",
                "meta_json",
            ])

    def append_session(
        self,
        session_id: str,
        model: str,
        test_id: str,
        stream_start_utc: str,
        stream_stop_utc: str,
        hyp_text: str,
        meta: Optional[Dict[str, Any]] = None,
    ):
        # Append one complete transcription session to the CSV file
        meta = meta or {}

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            w.writerow([
                session_id,
                model,
                test_id,
                stream_start_utc,
                stream_stop_utc,
                hyp_text,
                json.dumps(meta, ensure_ascii=False),
            ])


# --------------------------------------------------
# Metrics calculation and storage
# --------------------------------------------------

class MetricsStore:
    def __init__(
        self,
        out_dir: str,
        norm_cfg: Optional[TextNormConfig] = None,
        per_language: str = "de",
        per_backend: str = "espeak",
        bertscore_model_type: str = "bert-base-multilingual-cased",
        bertscore_device: Optional[str] = None,
    ):
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(
            self.out_dir,
            "metrics.csv",
        )

        self.norm_cfg = norm_cfg or TextNormConfig()

        self.per_language = per_language
        self.per_backend = per_backend

        self.bertscore_model_type = bertscore_model_type
        self.bertscore_device = bertscore_device

        self._ensure_header()

    def _ensure_header(self):
        # Create the metrics CSV file with a header if needed
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            w.writerow([
                "timestamp",
                "session_id",
                "model",
                "test_id",
                "person_id",
                "gender",
                "interface",
                "gain",
                "denoise",
                "wer",
                "cer",
                "ser",
                "per",
                "bertscore_precision",
                "bertscore_recall",
                "bertscore_f1",
                "stream_duration_s",
                "audio_duration_s",
                "model_compute_time_s",
                "rtf",
                "stream_start_to_first_partial_s",
                "stream_start_to_first_final_s",
                "speech_start_to_first_partial_s",
                "speech_start_to_first_final_s",
                "stop_to_last_final_s",
                "start_to_first_output_s",
                "start_to_first_partial_output_s",
                "start_to_first_final_output_s",
                "num_outputs",
                "mean_inter_output_s",
                "min_inter_output_s",
                "max_inter_output_s",
                "last_output_time_from_start_s",
                "gt_text",
                "hyp_text",
                "meta_json",
            ])

    @property
    def has_per(self) -> bool:
        # Indicates whether PER can be computed
        return _HAS_PHONEMIZER

    @property
    def has_bertscore(self) -> bool:
        # Indicates whether BERTScore can be computed
        return _HAS_BERTSCORE

    def compute_bertscore(
        self,
        gt_text: str,
        hyp_text: str,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        # Compute semantic similarity using BERTScore 
        if not _HAS_BERTSCORE:
            print("[BERTSCORE] bert-score package not installed")
            return None, None, None

        cand = hyp_text or ""
        ref = gt_text or ""

        if not cand.strip() and not ref.strip():
            return 1.0, 1.0, 1.0

        if not cand.strip() or not ref.strip():
            return 0.0, 0.0, 0.0

        try:
            P, R, F1 = bertscore_score(
                [cand],
                [ref],
                model_type=self.bertscore_model_type,
                device=self.bertscore_device,
                lang=None,
                verbose=False,
                rescale_with_baseline=False,
                use_fast_tokenizer=True,
            )

            return (
                float(P[0].item()),
                float(R[0].item()),
                float(F1[0].item()),
            )

        except Exception as e:
            print(f"[BERTSCORE ERROR] {e}")
            return None, None, None

    def compute(self, gt_text: str, hyp_text: str) -> MetricsResult:
        # Compute all available transcription quality metrics
        gt_n = normalize_text(gt_text, self.norm_cfg)
        hyp_n = normalize_text(hyp_text, self.norm_cfg)

        wer = error_rate(
            split_words(gt_n),
            split_words(hyp_n),
        )

        cer = error_rate(
            split_chars(gt_n),
            split_chars(hyp_n),
        )

        gt_s = split_sentences(gt_n)
        hyp_s = split_sentences(hyp_n)

        max_len = max(len(gt_s), len(hyp_s))

        if max_len == 0:
            ser = 0.0
        else:
            errors = 0

            for i in range(max_len):
                g = gt_s[i] if i < len(gt_s) else ""
                h = hyp_s[i] if i < len(hyp_s) else ""

                if g != h:
                    errors += 1

            ser = errors / float(max_len)

        gt_ph = phonemize_tokens(
            gt_n,
            language=self.per_language,
            backend=self.per_backend,
        )

        hyp_ph = phonemize_tokens(
            hyp_n,
            language=self.per_language,
            backend=self.per_backend,
        )

        if gt_ph is None or hyp_ph is None:
            per = None
        else:
            per = error_rate(gt_ph, hyp_ph)

        bert_p, bert_r, bert_f1 = self.compute_bertscore(
            gt_text,
            hyp_text,
        )

        return MetricsResult(
            wer=float(wer),
            cer=float(cer),
            ser=float(ser),
            per=None if per is None else float(per),
            bertscore_precision=bert_p,
            bertscore_recall=bert_r,
            bertscore_f1=bert_f1,
        )

    def append(
        self,
        session_id: str,
        model: str,
        test_id: str,
        gt_text: str,
        hyp_text: str,
        latency: Optional[LatencyResult] = None,
        output_latency: Optional[OutputLatencyResult] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MetricsResult:
        # Compute metrics and append them together with metadata to CSV
        meta = meta or {}

        latency = latency or LatencyResult(
            None, None, None, None,
            None, None, None, None, None,
        )

        output_latency = output_latency or OutputLatencyResult(
            None, None, None,
            0, None, None, None, None,
            [], [],
        )

        res = self.compute(
            gt_text,
            hyp_text,
        )

        meta = dict(meta)
        meta["output_times_s"] = output_latency.output_times_s
        meta["output_types"] = output_latency.output_types

        person_id = meta.get("person_id", "")
        gender = meta.get("gender", "")
        interface = meta.get("interface", "")
        gain = meta.get("gain", "")
        denoise = meta.get("denoise", "")

        def fmt(x: Optional[float]) -> str:
            return "" if x is None else f"{x:.6f}"

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            w.writerow([
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                session_id,
                model,
                test_id,
                person_id,
                gender,
                interface,
                gain,
                denoise,
                f"{res.wer:.6f}",
                f"{res.cer:.6f}",
                f"{res.ser:.6f}",
                "" if res.per is None else f"{res.per:.6f}",
                fmt(res.bertscore_precision),
                fmt(res.bertscore_recall),
                fmt(res.bertscore_f1),
                fmt(latency.stream_duration_s),
                fmt(latency.audio_duration_s),
                fmt(latency.model_compute_time_s),
                fmt(latency.rtf),
                fmt(latency.stream_start_to_first_partial_s),
                fmt(latency.stream_start_to_first_final_s),
                fmt(latency.speech_start_to_first_partial_s),
                fmt(latency.speech_start_to_first_final_s),
                fmt(latency.stop_to_last_final_s),
                fmt(output_latency.start_to_first_output_s),
                fmt(output_latency.start_to_first_partial_output_s),
                fmt(output_latency.start_to_first_final_output_s),
                str(output_latency.num_outputs),
                fmt(output_latency.mean_inter_output_s),
                fmt(output_latency.min_inter_output_s),
                fmt(output_latency.max_inter_output_s),
                fmt(output_latency.last_output_time_from_start_s),
                gt_text,
                hyp_text,
                json.dumps(meta, ensure_ascii=False),
            ])

        return res


# --------------------------------------------------
# Session transcript aggregation
# --------------------------------------------------

class SessionAggregator:
    def __init__(self):
        self.reset()

    def reset(self):
        # Clear timestamps and collected final transcript blocks
        self.stream_start_utc: Optional[str] = None
        self.stream_stop_utc: Optional[str] = None
        self.blocks: List[str] = []

    def start(self):
        # Mark the beginning of a new session
        self.reset()
        self.stream_start_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def stop(self):
        # Mark the end of the current session
        self.stream_stop_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def add_final(self, text: str):
        # Store one finalized ASR output block
        t = (text or "").strip()

        if t:
            self.blocks.append(t)

    def full_hypothesis(self) -> str:
        # Join all final outputs into one hypothesis text
        return " ".join(self.blocks).strip()

    def make_session_id(self, model: str, test_id: str) -> str:
        # Create a unique session ID from model, test ID and timestamp
        stamp = self.stream_start_utc or (
            datetime.utcnow().isoformat(timespec="seconds") + "Z"
        )

        safe_stamp = stamp.replace(":", "").replace("-", "")

        return f"{model}_{test_id}_{safe_stamp}"