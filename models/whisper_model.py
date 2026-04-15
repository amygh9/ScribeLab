import queue
import threading
import time
from typing import Optional

import numpy as np
import whisper

from core.types import AudioChunk
from models.model_base import ASRModel


class WhisperWindowModel(ASRModel):
    def __init__(
        self,
        language: str = "de",
        model_name: str = "small",
        target_seconds: float = 4.0,
        overlap_seconds: float = 0.0,
        sample_rate: int = 16000,
        silence_rms_threshold: float = 0.003,
        no_speech_threshold: float = 0.8,
    ):
        self.language = language
        self.model_name = model_name
        self.target_seconds = target_seconds
        self.overlap_seconds = overlap_seconds
        self.sr = sample_rate

        self.silence_rms_threshold = silence_rms_threshold
        self.no_speech_threshold = no_speech_threshold

        self._q_events = queue.Queue(maxsize=50)
        self._running = False

        self._lock = threading.Lock()
        self._buf = []
        self._processing = False

        self.model = None
        self.compute_time_s = 0.0

    def start(self):
        if self._running:
            return

        print(f"[WHISPER] loading {self.model_name} (CPU)")
        self.model = whisper.load_model(self.model_name)

        self.compute_time_s = 0.0
        self._buf = []
        self._processing = False
        self._running = True

    def stop(self):
        self._running = False

    def accept_chunk(self, chunk: AudioChunk):
        if not self._running:
            return

        should_start = False

        with self._lock:
            # Always keep incoming audio. Do not drop chunks while processing.
            self._buf.append(chunk.data)

            total = sum(b.size for b in self._buf)
            dur = total / self.sr

            if (not self._processing) and (dur >= self.target_seconds):
                self._processing = True
                should_start = True

        if should_start:
            threading.Thread(target=self._process, daemon=True).start()

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

    def _schedule_next_if_ready(self):
        with self._lock:
            total = sum(b.size for b in self._buf)
            dur = total / self.sr

            if self._running and dur >= self.target_seconds:
                threading.Thread(target=self._process, daemon=True).start()
            else:
                self._processing = False

    def _process(self):
        with self._lock:
            audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)

            keep = int(self.overlap_seconds * self.sr)

            if keep <= 0:
                self._buf = []
            elif audio.size > keep:
                self._buf = [audio[-keep:].copy()]
            else:
                self._buf = [audio.copy()] if audio.size > 0 else []

        if audio.size == 0:
            self._processing = False
            return

        audio = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)

        # Silence gate to reduce hallucinations during pauses/background noise
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size > 0 else 0.0
        if rms < self.silence_rms_threshold:
            self._emit({
                "type": "info",
                "message": f"skip whisper decode: low energy rms={rms:.5f}",
            })
            self._schedule_next_if_ready()
            return

        try:
            t0 = time.perf_counter()

            result = self.model.transcribe(
                audio,
                fp16=False,
                language=self.language,
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=True,
                no_speech_threshold=self.no_speech_threshold,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                verbose=False,
            )

            self.compute_time_s += (time.perf_counter() - t0)

            text = (result.get("text") or "").strip()
            if text:
                self._emit({"type": "final", "text": text})
            else:
                self._emit({"type": "info", "message": "empty whisper output"})

        except Exception as e:
            self._emit({"type": "error", "message": str(e)})

        finally:
            self._schedule_next_if_ready()