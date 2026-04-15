import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import queue
import threading
import time
from typing import Optional

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from core.types import AudioChunk
from models.model_base import ASRModel


class Wav2Vec2WindowModel(ASRModel):
    def __init__(self, model_id: str, target_seconds=20.0, overlap_seconds=10.0, sample_rate=16000):
        self.model_id = model_id
        self.target_seconds = target_seconds
        self.overlap_seconds = overlap_seconds
        self.sr = sample_rate

        self._q_events = queue.Queue(maxsize=50)
        self._running = False

        self._lock = threading.Lock()
        self._buf = []
        self._processing = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

        # RTF tracking
        self.compute_time_s = 0.0

    def start(self):
        if self._running:
            return

        print(f"[WAV2VEC2] loading {self.model_id} on {self.device}")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        self.compute_time_s = 0.0
        self._buf = []
        self._processing = False
        self._running = True

    def stop(self):
        self._running = False

    def accept_chunk(self, chunk: AudioChunk):
        if not self._running or self._processing:
            return

        with self._lock:
            self._buf.append(chunk.data)
            total = sum(b.size for b in self._buf)
            dur = total / self.sr

        if dur >= self.target_seconds:
            self._processing = True
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

    def _process(self):
        with self._lock:
            audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
            self._buf.clear()

        if audio.size == 0:
            self._processing = False
            return

        # normalize
        m = float(np.max(np.abs(audio)))
        if m > 0:
            audio = audio / m

        try:
            t0 = time.perf_counter()

            inputs = self.processor(
                audio,
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(pred_ids)[0].strip()

            self.compute_time_s += (time.perf_counter() - t0)

            if text:
                self._emit({"type": "final", "text": text})
            else:
                self._emit({"type": "info", "message": "empty wav2vec2 output"})

        except Exception as e:
            self._emit({"type": "error", "message": str(e)})

        # overlap
        keep = int(self.overlap_seconds * self.sr)
        if audio.size > keep:
            with self._lock:
                self._buf.append(audio[-keep:])

        self._processing = False