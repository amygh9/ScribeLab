import json
import queue
import threading
import time
from typing import Optional

import numpy as np
from vosk import Model, KaldiRecognizer

from core.types import AudioChunk
from models.model_base import ASRModel


class VoskStreamingModel(ASRModel):
    def __init__(self, model_path: str, sample_rate: int = 16000, min_chunk_s: float = 0.25):
        self.model_path = model_path
        self.sr = sample_rate
        self.min_bytes = int(min_chunk_s * self.sr * 2)  # int16 bytes

        self._q_audio = queue.Queue(maxsize=300)
        self._q_events = queue.Queue(maxsize=300)
        self._running = False

        self._accumulated = b""
        self._worker = None

        self.model = None
        self.rec = None

        # RTF tracking
        self.compute_time_s = 0.0

    def start(self):
        if self._running:
            return

        print("[VOSK] loading model:", self.model_path)
        self.model = Model(self.model_path)
        self.rec = KaldiRecognizer(self.model, self.sr)
        self.rec.SetWords(True)

        self._accumulated = b""
        self.compute_time_s = 0.0

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

        # float32 [-1,1] -> int16 bytes for Vosk
        audio_i16 = (np.clip(chunk.data, -1, 1) * 32767.0).astype(np.int16)

        try:
            self._q_audio.put_nowait(audio_i16.tobytes())
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

    def _loop(self):
        while self._running:
            try:
                b = self._q_audio.get(timeout=0.1)
            except queue.Empty:
                continue

            self._accumulated += b

            if len(self._accumulated) < self.min_bytes:
                continue

            data = self._accumulated
            self._accumulated = b""

            try:
                t0 = time.perf_counter()

                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = (result.get("text") or "").strip()

                    if text:
                        self._emit({"type": "final", "text": text})

                else:
                    partial = json.loads(self.rec.PartialResult())
                    p = (partial.get("partial") or "").strip()

                    if p:
                        self._emit({"type": "partial", "text": p})

                # accumulate compute time
                self.compute_time_s += (time.perf_counter() - t0)

            except Exception as e:
                self._emit({"type": "error", "message": str(e)})