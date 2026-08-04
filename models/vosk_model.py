"""
Welcome to Vosk integration !

Incoming audio chunks are converted to the format expected by Vosk,
buffered for efficient processing, and transcribed continuously.
The model produces both partial and final transcription results.
"""

import json
import queue
import threading
import time
from typing import Optional

import numpy as np
from vosk import Model, KaldiRecognizer

from core.types import AudioChunk
from models.model_base import ASRModel


# --------------------------------------------------
# Vosk streaming model
# --------------------------------------------------

class VoskStreamingModel(ASRModel):
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        min_chunk_s: float = 0.25,
    ):
        # Model and audio configuration
        self.model_path = model_path
        self.sr = sample_rate

        # Minimum audio size before sending data to Vosk
        self.min_bytes = int(min_chunk_s * self.sr * 2)

        # Audio and event queues
        self._q_audio = queue.Queue(maxsize=300)
        self._q_events = queue.Queue(maxsize=300)

        self._running = False

        # Temporary audio buffer
        self._accumulated = b""

        self._worker = None

        self.model = None
        self.rec = None

        # Total recognition compute time
        self.compute_time_s = 0.0

    # --------------------------------------------------
    # Model initialization
    # --------------------------------------------------

    def start(self):
        if self._running:
            return

        print("[VOSK] loading model:", self.model_path)

        # Load the Vosk model
        self.model = Model(self.model_path)

        # Create streaming recognizer
        self.rec = KaldiRecognizer(self.model, self.sr)
        self.rec.SetWords(True)

        self._accumulated = b""
        self.compute_time_s = 0.0

        self._running = True

        # Start recognition worker thread
        self._worker = threading.Thread(
            target=self._loop,
            daemon=True,
        )

        self._worker.start()

    # --------------------------------------------------
    # Model shutdown
    # --------------------------------------------------

    def stop(self):
        self._running = False

        if self._worker:
            self._worker.join(timeout=2.0)

    # --------------------------------------------------
    # Audio input interface
    # --------------------------------------------------

    def accept_chunk(self, chunk: AudioChunk):
        if not self._running:
            return

        # Convert float32 audio to int16 format expected by Vosk
        audio_i16 = (
            np.clip(chunk.data, -1, 1) * 32767.0
        ).astype(np.int16)

        try:
            self._q_audio.put_nowait(audio_i16.tobytes())
        except queue.Full:
            pass

    # --------------------------------------------------
    # Event handling
    # --------------------------------------------------

    def poll(self) -> Optional[dict]:
        # Return the next available event
        try:
            return self._q_events.get_nowait()
        except queue.Empty:
            return None

    def _emit(self, ev: dict):
        # Add a new event if space is available
        try:
            self._q_events.put_nowait(ev)
        except queue.Full:
            pass

    # --------------------------------------------------
    # Compute time for RTF
    # --------------------------------------------------

    def get_compute_time_s(self) -> float:
        return self.compute_time_s

    # --------------------------------------------------
    # Recognition worker
    # --------------------------------------------------

    def _loop(self):
        while self._running:

            # Wait for incoming audio
            try:
                b = self._q_audio.get(timeout=0.1)
            except queue.Empty:
                continue

            # Collect audio until a minimum chunk size is reached
            self._accumulated += b

            if len(self._accumulated) < self.min_bytes:
                continue

            data = self._accumulated
            self._accumulated = b""

            try:
                t0 = time.perf_counter()

                # Check whether Vosk detected a complete utterance
                if self.rec.AcceptWaveform(data):

                    result = json.loads(self.rec.Result())
                    text = (result.get("text") or "").strip()

                    if text:
                        self._emit({
                            "type": "final",
                            "text": text
                        })

                # Otherwise return an intermediate result
                else:
                    partial = json.loads(self.rec.PartialResult())
                    p = (partial.get("partial") or "").strip()

                    if p:
                        self._emit({
                            "type": "partial",
                            "text": p
                        })

                # Accumulate total inference time
                self.compute_time_s += (
                    time.perf_counter() - t0
                )

            except Exception as e:
                self._emit({
                    "type": "error",
                    "message": str(e)
                })