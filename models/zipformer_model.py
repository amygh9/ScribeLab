"""
Welcome to the Kroko Zipformer RNN-T integration.

We use a Zipformer Transducer model through sherpa-onnx. Incoming audio
chunks are processed continuously and partial hypotheses are
generated during speech. Final results are emitted when an
endpoint is detected.

"""

import os
import queue
import threading
import time
from typing import Optional

import numpy as np
import sherpa_onnx

from core.types import AudioChunk
from models.model_base import ASRModel


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def pick_existing(*paths: str) -> str:
    # Return the first existing file from a list of candidates
    for p in paths:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError("None exist:\n  " + "\n  ".join(paths))


# --------------------------------------------------
# Zipformer streaming model
# --------------------------------------------------

class ZipformerStreamingModel(ASRModel):
    def __init__(self, model_dir: str, sample_rate: int = 16000):

        # Model configuration
        self.model_dir = model_dir
        self.sr = sample_rate

        # Audio and event queues
        self._q_audio = queue.Queue(maxsize=300)
        self._q_events = queue.Queue(maxsize=300)

        self._running = False
        self._worker = None

        self.recognizer = None
        self.stream = None

        # Last emitted partial result
        self._last_partial = ""

        # Total decoding time for RTF calculation
        self.compute_time_s = 0.0

    # --------------------------------------------------
    # Model initialization
    # --------------------------------------------------

    def start(self):
        if self._running:
            return

        # Locate model files
        tokens = pick_existing(
            f"{self.model_dir}/tokens.txt"
        )

        encoder = pick_existing(
            f"{self.model_dir}/encoder.int8.onnx",
            f"{self.model_dir}/encoder.onnx",
        )

        decoder = pick_existing(
            f"{self.model_dir}/decoder.int8.onnx",
            f"{self.model_dir}/decoder.onnx",
        )

        joiner = pick_existing(
            f"{self.model_dir}/joiner.int8.onnx",
            f"{self.model_dir}/joiner.onnx",
        )

        print("[ZIPFORMER] loading")
        print("  tokens:", tokens)
        print("  encoder:", encoder)
        print("  decoder:", decoder)
        print("  joiner:", joiner)

        # Create streaming transducer recognizer
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=4,
            sample_rate=self.sr,
            feature_dim=80,
        )

        # Create streaming state
        self.stream = self.recognizer.create_stream()

        self._last_partial = ""
        self.compute_time_s = 0.0

        self._running = True

        # Start decoding worker
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

        try:
            # Forward audio chunk to the decoding thread
            self._q_audio.put_nowait(
                chunk.data.astype(np.float32, copy=False)
            )
        except queue.Full:
            pass

    # --------------------------------------------------
    # Event handling
    # --------------------------------------------------

    def poll(self) -> Optional[dict]:
        # Return the next event without blocking
        try:
            return self._q_events.get_nowait()
        except queue.Empty:
            return None

    def _emit(self, ev: dict):
        # Push a new event to the output queue
        try:
            self._q_events.put_nowait(ev)
        except queue.Full:
            pass

    # --------------------------------------------------
    # Result handling
    # --------------------------------------------------

    def _result_text(self) -> str:
        # Extract text from the recognizer result object
        r = self.recognizer.get_result(self.stream)

        if hasattr(r, "text"):
            return (r.text or "").strip()

        return (r or "").strip()

    # --------------------------------------------------
    # Performance statistics
    # --------------------------------------------------

    def get_compute_time_s(self) -> float:
        return self.compute_time_s

    # --------------------------------------------------
    # Force final output
    # --------------------------------------------------

    def flush_final(self):
        """
        Force-emits the current best hypothesis as a final result.
        Useful before stopping the application.
        """

        if not self.recognizer or not self.stream:
            return

        try:
            text = self._result_text()

            if text:
                self._emit({
                    "type": "final",
                    "text": text
                })

        except Exception as e:
            self._emit({
                "type": "error",
                "message": f"zipformer flush_final failed: {e}"
            })

    # --------------------------------------------------
    # Streaming recognition loop
    # --------------------------------------------------

    def _loop(self):

        # Endpoint detection is optional depending on sherpa version
        has_is_endpoint = hasattr(
            self.recognizer,
            "is_endpoint"
        )

        while self._running:

            # Wait for incoming audio
            try:
                chunk = self._q_audio.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            # Feed audio into the streaming recognizer
            self.stream.accept_waveform(
                self.sr,
                chunk
            )

            # Decode as long as the recognizer is ready
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

            partial = self._result_text()

            # Accumulate total compute time
            self.compute_time_s += (
                time.perf_counter() - t0
            )

            # Emit updated partial hypotheses
            if partial and partial != self._last_partial:
                self._emit({
                    "type": "partial",
                    "text": partial
                })

                self._last_partial = partial

            # Emit final result when an endpoint is detected
            if (
                has_is_endpoint
                and self.recognizer.is_endpoint(self.stream)
            ):
                final_text = self._result_text()

                if final_text:
                    self._emit({
                        "type": "final",
                        "text": final_text
                    })

                # Start a fresh stream for the next utterance
                self.stream = self.recognizer.create_stream()
                self._last_partial = ""