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


class NemoStreamingModel(ASRModel):
    def __init__(self, model_dir: str, sample_rate: int = 16000):
        self.model_dir = model_dir
        self.sr = sample_rate

        self._q_audio = queue.Queue(maxsize=300)
        self._q_events = queue.Queue(maxsize=300)
        self._running = False
        self._worker = None

        self.recognizer = None
        self.stream = None
        self._last_partial = ""

        # for RTF / latency metrics
        self.compute_time_s = 0.0

    def start(self):
        if self._running:
            return

        tokens = pick_existing(
            os.path.join(self.model_dir, "tokens.txt")
        )
        model = pick_existing(
            os.path.join(self.model_dir, "model.int8.onnx"),
            os.path.join(self.model_dir, "model.onnx"),
        )

        print("[NEMO] loading")
        print("  tokens:", tokens)
        print("  model :", model)

        # Streaming NeMo CTC recognizer
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
            tokens=tokens,
            model=model,
            num_threads=4,
            sample_rate=self.sr,
            feature_dim=80,
            decoding_method="greedy_search",
        )

        self.stream = self.recognizer.create_stream()

        self._last_partial = ""
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
        try:
            self._q_audio.put_nowait(chunk.data.astype(np.float32, copy=False))
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

    def _result_text(self) -> str:
        r = self.recognizer.get_result(self.stream)
        if hasattr(r, "text"):
            return (r.text or "").strip()
        return str(r).strip()

    def get_compute_time_s(self) -> float:
        return self.compute_time_s

    def flush_final(self):
        """
        Force-emits the current best hypothesis as a FINAL event.
        Useful when stopping the demo and you want the last partial
        to be treated as final.
        """
        if not self.recognizer or not self.stream:
            return

        try:
            # Let the recognizer know no more audio is coming
            if hasattr(self.stream, "input_finished"):
                self.stream.input_finished()

            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

            text = self._result_text()
            if text:
                self._emit({"type": "final", "text": text})
                self._last_partial = ""
        except Exception as e:
            self._emit({"type": "error", "message": f"nemo flush_final failed: {e}"})

    def _loop(self):
        has_is_endpoint = hasattr(self.recognizer, "is_endpoint")

        while self._running:
            try:
                chunk = self._q_audio.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            self.stream.accept_waveform(self.sr, chunk)

            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

            partial = self._result_text()

            self.compute_time_s += (time.perf_counter() - t0)

            if partial and partial != self._last_partial:
                self._emit({"type": "partial", "text": partial})
                self._last_partial = partial

            # Some online recognizers expose endpoint detection.
            # If available, emit final and start a new stream.
            if has_is_endpoint and self.recognizer.is_endpoint(self.stream):
                final_text = self._result_text()
                if final_text:
                    self._emit({"type": "final", "text": final_text})

                self.stream = self.recognizer.create_stream()
                self._last_partial = ""