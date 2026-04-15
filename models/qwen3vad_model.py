import os
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import sherpa_onnx

from core.types import AudioChunk
from models.model_base import ASRModel


def pick_existing(*paths: str) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("None exist:\n " + "\n ".join(paths))


class Qwen3VadModel(ASRModel):
    def __init__(
        self,
        model_dir: str,
        sample_rate: int = 16000,
        num_threads: int = 3,
        vad_threshold: float = 0.5,
        vad_min_silence_s: float = 0.5,
        vad_min_speech_s: float = 0.25,
        vad_window_size: int = 512,
        vad_max_speech_s: float = 20.0,
        pre_speech_ms: int = 200,
    ):
        self.model_dir = model_dir
        self.sr = sample_rate
        self.num_threads = num_threads

        self.vad_threshold = vad_threshold
        self.vad_min_silence_s = vad_min_silence_s
        self.vad_min_speech_s = vad_min_speech_s
        self.vad_window_size = vad_window_size
        self.vad_max_speech_s = vad_max_speech_s
        self.pre_speech_ms = pre_speech_ms

        self._q_events = queue.Queue(maxsize=100)
        self._running = False
        self._lock = threading.Lock()

        self.recognizer = None
        self.vad = None
        self.compute_time_s = 0.0

        # streaming state
        self._remainder = np.zeros(0, dtype=np.float32)
        self._pre_buffer = None
        self._speech_buffer = []
        self._speech_active = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._decode_thread = None

        self._vad_window = None
        self._min_speech_samples = None
        self._min_silence_samples = None
        self._pre_speech_samples = int(self.pre_speech_ms * self.sr / 1000)

    def start(self):
        if self._running:
            return

        conv_frontend = pick_existing(
            f"{self.model_dir}/conv_frontend.int8.onnx",
            f"{self.model_dir}/conv_frontend.onnx",
        )
        encoder = pick_existing(
            f"{self.model_dir}/encoder.int8.onnx",
            f"{self.model_dir}/encoder.onnx",
        )
        decoder = pick_existing(
            f"{self.model_dir}/decoder.int8.onnx",
            f"{self.model_dir}/decoder.onnx",
        )

        tokenizer = os.path.join(self.model_dir, "tokenizer")
        if not os.path.isdir(tokenizer):
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer}")

        silero_vad = os.path.join(self.model_dir, "silero_vad.onnx")
        if not os.path.isfile(silero_vad):
            raise FileNotFoundError(f"Silero VAD model not found: {silero_vad}")

        print("[QWEN3+VAD] loading")
        print(" conv_frontend:", conv_frontend)
        print(" encoder:", encoder)
        print(" decoder:", decoder)
        print(" tokenizer:", tokenizer)
        print(" silero_vad:", silero_vad)

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=conv_frontend,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            num_threads=self.num_threads,
            sample_rate=self.sr,
            feature_dim=128,
            decoding_method="greedy_search",
            provider="cpu",
            max_total_len=512,
            max_new_tokens=128,
            temperature=1e-6,
            top_p=0.8,
            seed=42,
        )

        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=silero_vad,
                threshold=self.vad_threshold,
                min_silence_duration=self.vad_min_silence_s,
                min_speech_duration=self.vad_min_speech_s,
                window_size=self.vad_window_size,
                max_speech_duration=self.vad_max_speech_s,
            ),
            sample_rate=self.sr,
            num_threads=1,
            provider="cpu",
            debug=False,
        )
        self.vad = sherpa_onnx.VadModel.create(vad_config)

        self._vad_window = self.vad.window_size()
        self._min_speech_samples = self.vad.min_speech_duration_samples()
        self._min_silence_samples = self.vad.min_silence_duration_samples()

        print("[QWEN3+VAD] window_size:", self._vad_window)
        print("[QWEN3+VAD] min_speech_samples:", self._min_speech_samples)
        print("[QWEN3+VAD] min_silence_samples:", self._min_silence_samples)

        self.compute_time_s = 0.0
        self._remainder = np.zeros(0, dtype=np.float32)
        self._pre_buffer = deque(maxlen=self._pre_speech_samples)
        self._speech_buffer = []
        self._speech_active = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._decode_thread = None

        self._running = True

    def stop(self):
        self._running = False

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

    def accept_chunk(self, chunk: AudioChunk):
        if not self._running:
            return

        audio = chunk.data
        if audio is None or audio.size == 0:
            return

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        # ensure normalized range just in case
        audio = np.clip(audio, -1.0, 1.0)

        with self._lock:
            self._process_audio_locked(audio)

    def _process_audio_locked(self, audio: np.ndarray):
        # prepend leftover remainder from last call
        if self._remainder.size > 0:
            audio = np.concatenate([self._remainder, audio])

        # update pre-buffer sample-by-sample
        for x in audio:
            self._pre_buffer.append(float(x))

        i = 0
        n = audio.size

        while i + self._vad_window <= n:
            frame = audio[i:i + self._vad_window]
            i += self._vad_window

            is_speech = self.vad.is_speech(frame)

            if not self._speech_active:
                if is_speech:
                    self._speech_active = True
                    self._speech_samples = frame.size
                    self._silence_samples = 0
                    self._speech_buffer = [
                        np.array(self._pre_buffer, dtype=np.float32),
                        frame.copy(),
                    ]
                    self._emit({"type": "info", "message": "speech_start"})
            else:
                self._speech_buffer.append(frame.copy())

                if is_speech:
                    self._speech_samples += frame.size
                    self._silence_samples = 0
                else:
                    self._silence_samples += frame.size

                # force finalization if utterance is too long
                total_seg_samples = sum(x.size for x in self._speech_buffer)
                if total_seg_samples >= int(self.vad_max_speech_s * self.sr):
                    self._finalize_segment_locked(reason="max_speech_duration")
                    continue

                # normal endpoint
                if self._silence_samples >= self._min_silence_samples:
                    self._finalize_segment_locked(reason="silence")

        # keep incomplete tail for next chunk
        self._remainder = audio[i:]

    def _finalize_segment_locked(self, reason: str):
        if not self._speech_buffer:
            self._reset_segment_state_locked()
            return

        utterance = np.concatenate(self._speech_buffer)
        effective_speech = utterance.size - self._silence_samples

        self._emit({"type": "info", "message": f"speech_end ({reason})"})

        if effective_speech < self._min_speech_samples:
            self._emit({"type": "info", "message": "discarded short speech"})
            self._reset_segment_state_locked()
            return

        # avoid parallel decodes piling up
        if self._decode_thread is not None and self._decode_thread.is_alive():
            self._emit({"type": "info", "message": "decoder busy, dropped segment"})
            self._reset_segment_state_locked()
            return

        audio_to_decode = utterance.astype(np.float32, copy=False)
        self._decode_thread = threading.Thread(
            target=self._decode_async,
            args=(audio_to_decode,),
            daemon=True,
        )
        self._decode_thread.start()

        self._reset_segment_state_locked()

    def _reset_segment_state_locked(self):
        self._speech_buffer = []
        self._speech_active = False
        self._speech_samples = 0
        self._silence_samples = 0

    def _decode_async(self, audio: np.ndarray):
        if audio.size == 0:
            return

        try:
            stream = self.recognizer.create_stream()

            t0 = time.perf_counter()
            stream.accept_waveform(self.sr, audio)
            self.recognizer.decode_stream(stream)
            self.compute_time_s += (time.perf_counter() - t0)

            text = (stream.result.text or "").strip()
            self._emit({"type": "info", "message": f"Qwen3 decoded text length={len(text)}"})

            if text:
                self._emit({"type": "final", "text": text})
            else:
                self._emit({"type": "info", "message": "empty qwen3 output"})

        except Exception:
            import traceback
            self._emit({"type": "error", "message": traceback.format_exc()})

    def flush_final(self):
        """
        Optional: call this before stopping to force decoding of the current segment.
        """
        with self._lock:
            if self._speech_active and self._speech_buffer:
                self._finalize_segment_locked(reason="flush")