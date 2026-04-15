import os
import queue
import threading
import time
from typing import Optional

import numpy as np
import sherpa_onnx

from core.types import AudioChunk
from models.model_base import ASRModel


def pick_existing(*paths: str) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("None exist:\n  " + "\n  ".join(paths))


class ParakeetStreamingModel(ASRModel):
    def __init__(
        self,
        model_dir: str,
        sample_rate: int = 16000,
        num_threads: int = 1,
        decode_interval_s: float = 1.0,
        max_buffer_s: float = 6.0,
        silence_rms_threshold: float = 0.0025,
        silence_finalize_s: float = 0.7,
        min_final_chars: int = 8,
    ):
        self.model_dir = model_dir
        self.sr = sample_rate
        self.num_threads = num_threads
        self.decode_interval_s = decode_interval_s
        self.max_buffer_s = max_buffer_s
        self.silence_rms_threshold = silence_rms_threshold
        self.silence_finalize_s = silence_finalize_s
        self.min_final_chars = min_final_chars

        self._q_audio: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue(maxsize=300)
        self._q_events: "queue.Queue[dict]" = queue.Queue(maxsize=300)
        self._running = False
        self._worker: Optional[threading.Thread] = None

        self.recognizer = None
        self._last_partial = ""
        self.compute_time_s = 0.0

        self._buffer = np.empty((0,), dtype=np.float32)
        self._last_decode_time = 0.0
        self._lock = threading.Lock()

        self._silence_run_s = 0.0

    def start(self):
        if self._running:
            return

        tokens = pick_existing(
            os.path.join(self.model_dir, "tokens.txt")
        )
        encoder = pick_existing(
            os.path.join(self.model_dir, "encoder.int8.onnx"),
            os.path.join(self.model_dir, "encoder.onnx"),
        )
        decoder = pick_existing(
            os.path.join(self.model_dir, "decoder.int8.onnx"),
            os.path.join(self.model_dir, "decoder.onnx"),
        )
        joiner = pick_existing(
            os.path.join(self.model_dir, "joiner.int8.onnx"),
            os.path.join(self.model_dir, "joiner.onnx"),
        )

        print("[PARAKEET] loading")
        print("  tokens :", tokens)
        print("  encoder:", encoder)
        print("  decoder:", decoder)
        print("  joiner :", joiner)
        print("[PARAKEET] creating recognizer")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=self.num_threads,
            sample_rate=self.sr,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

        self._last_partial = ""
        self.compute_time_s = 0.0
        self._buffer = np.empty((0,), dtype=np.float32)
        self._last_decode_time = time.perf_counter()
        self._silence_run_s = 0.0

        self._running = True
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._running = False
        if self._worker:
            self._worker.join(timeout=2.0)

    def accept_chunk(self, chunk: AudioChunk):
        if not self._running:
            return
        try:
            audio = chunk.data.astype(np.float32, copy=False)
            rms = float(getattr(chunk, "rms", 0.0))
            self._q_audio.put_nowait((audio, rms))
        except queue.Full:
            pass

    def poll(self) -> Optional[dict]:
        try:
            return self._q_events.get_nowait()
        except queue.Empty:
            return None

    def _emit(self, ev: dict):
        try:
            self._q_events.put_nowait(ev)
        except queue.Full:
            pass

    def get_compute_time_s(self) -> float:
        return self.compute_time_s

    def _decode_buffer(self) -> str:
        if self._buffer.size == 0:
            return ""

        t0 = time.perf_counter()

        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sr, self._buffer)
        self.recognizer.decode_stream(stream)

        result = stream.result

        self.compute_time_s += (time.perf_counter() - t0)

        if hasattr(result, "text"):
            return (result.text or "").strip()
        return str(result).strip()

    def _finalize_current_segment(self):
        text = self._decode_buffer()
        if text and len(text) >= self.min_final_chars:
            self._emit({"type": "final", "text": text})

        self._last_partial = ""
        self._buffer = np.empty((0,), dtype=np.float32)
        self._silence_run_s = 0.0

    def flush_final(self):
        try:
            with self._lock:
                if self._buffer.size > 0:
                    self._finalize_current_segment()
        except Exception as e:
            self._emit({"type": "error", "message": f"parakeet flush_final failed: {e}"})

    def _loop(self):
        max_samples = int(self.max_buffer_s * self.sr)

        while self._running:
            try:
                chunk, rms = self._q_audio.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._lock:
                self._buffer = np.concatenate([self._buffer, chunk])

                if self._buffer.size > max_samples:
                    self._buffer = self._buffer[-max_samples:]

                chunk_duration_s = len(chunk) / float(self.sr)

                if rms < self.silence_rms_threshold:
                    self._silence_run_s += chunk_duration_s
                else:
                    self._silence_run_s = 0.0

                now = time.perf_counter()
                should_decode_partial = (now - self._last_decode_time) >= self.decode_interval_s
                should_finalize = (
                    self._buffer.size > 0
                    and self._last_partial
                    and self._silence_run_s >= self.silence_finalize_s
                )

                if should_finalize:
                    try:
                        self._finalize_current_segment()
                        self._last_decode_time = now
                    except Exception as e:
                        self._emit({"type": "error", "message": f"parakeet finalize failed: {e}"})
                    continue

                if not should_decode_partial:
                    continue

                self._last_decode_time = now

                try:
                    text = self._decode_buffer()
                    if text and text != self._last_partial:
                        self._emit({"type": "partial", "text": text})
                        self._last_partial = text
                except Exception as e:
                    self._emit({"type": "error", "message": f"parakeet decode failed: {e}"})