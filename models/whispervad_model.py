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


class WhisperVadModel(ASRModel):
    """
    Whisper + sherpa-onnx VAD (Silero) for live microphone/audio chunks.

    Notes:
    - This is VAD-segmented offline ASR, not true streaming Whisper.
    - Results are emitted when a speech segment ends or hits max_speech_duration.
    """

    def __init__(
        self,
        model_dir: str,
        sample_rate: int = 16000,
        num_threads: int = 3,
        whisper_language: str = "de",
        whisper_task: str = "transcribe",
        whisper_tail_paddings: int = -1,
        vad_threshold: float = 0.5,
        vad_min_silence_s: float = 0.5,
        vad_min_speech_s: float = 0.25,
        vad_window_size: int = 512,
        vad_max_speech_s: float = 20.0,
        pre_speech_ms: int = 200,
        provider: str = "cpu",
    ):
        self.model_dir = model_dir
        self.sr = sample_rate
        self.num_threads = num_threads
        self.provider = provider

        self.whisper_language = whisper_language
        self.whisper_task = whisper_task
        self.whisper_tail_paddings = whisper_tail_paddings

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

        # streaming / segmentation state
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

        # Common sherpa whisper layout:
        #   tiny.en-encoder.onnx / tiny.en-encoder.int8.onnx
        #   tiny.en-decoder.onnx / tiny.en-decoder.int8.onnx
        #   tiny.en-tokens.txt
        # or similarly for base/small/etc.
        encoder = pick_existing(
            f"{self.model_dir}/encoder.int8.onnx",
            f"{self.model_dir}/encoder.onnx",
            *self._glob_candidates("encoder"),
        )
        decoder = pick_existing(
            f"{self.model_dir}/decoder.int8.onnx",
            f"{self.model_dir}/decoder.onnx",
            *self._glob_candidates("decoder"),
        )
        tokens = pick_existing(
            f"{self.model_dir}/tokens.txt",
            *self._glob_candidates("tokens"),
        )

        silero_vad = pick_existing(
            f"{self.model_dir}/silero_vad.onnx",
        )

        print("[WHISPER+VAD] loading")
        print(" encoder:", encoder)
        print(" decoder:", decoder)
        print(" tokens:", tokens)
        print(" silero_vad:", silero_vad)

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            num_threads=self.num_threads,
            decoding_method="greedy_search",
            debug=False,
            language=self.whisper_language,
            task=self.whisper_task,
            tail_paddings=self.whisper_tail_paddings,
            provider=self.provider,
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
            provider=self.provider,
            debug=False,
        )
        self.vad = sherpa_onnx.VadModel.create(vad_config)

        self._vad_window = self.vad.window_size()
        self._min_speech_samples = self.vad.min_speech_duration_samples()
        self._min_silence_samples = self.vad.min_silence_duration_samples()

        print("[WHISPER+VAD] window_size:", self._vad_window)
        print("[WHISPER+VAD] min_speech_samples:", self._min_speech_samples)
        print("[WHISPER+VAD] min_silence_samples:", self._min_silence_samples)

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

        audio = np.clip(audio, -1.0, 1.0)

        with self._lock:
            self._process_audio_locked(audio)

    def _process_audio_locked(self, audio: np.ndarray):
        if self._remainder.size > 0:
            audio = np.concatenate([self._remainder, audio])

        # Keep short context before speech start
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

                total_seg_samples = sum(x.size for x in self._speech_buffer)

                # Force segmentation for long utterances
                if total_seg_samples >= int(self.vad_max_speech_s * self.sr):
                    self._finalize_segment_locked(reason="max_speech_duration")
                    continue

                if self._silence_samples >= self._min_silence_samples:
                    self._finalize_segment_locked(reason="silence")

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

        # Avoid multiple simultaneous offline decodes piling up
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
            dt = time.perf_counter() - t0

            self.compute_time_s += dt

            result = stream.result
            text = (result.text or "").strip()

            self._emit(
                {
                    "type": "info",
                    "message": f"Whisper decoded text length={len(text)} compute_time={dt:.3f}s",
                }
            )

            if text:
                self._emit({"type": "final", "text": text})
            else:
                self._emit({"type": "info", "message": "empty whisper output"})

        except Exception:
            import traceback
            self._emit({"type": "error", "message": traceback.format_exc()})

    def flush_final(self):
        """
        Force decoding of the currently active segment before stopping.
        """
        with self._lock:
            if self._speech_active and self._speech_buffer:
                self._finalize_segment_locked(reason="flush")

    def _glob_candidates(self, kind: str):
        """
        Accept both generic and sherpa whisper release naming, e.g.
        tiny.en-encoder.onnx, base-encoder.int8.onnx, etc.
        """
        out = []
        for name in os.listdir(self.model_dir):
            full = os.path.join(self.model_dir, name)
            if not os.path.isfile(full):
                continue

            lower = name.lower()

            if kind == "encoder" and "encoder" in lower and lower.endswith(".onnx"):
                out.append(full)
            elif kind == "decoder" and "decoder" in lower and lower.endswith(".onnx"):
                out.append(full)
            elif kind == "tokens" and lower.endswith("tokens.txt"):
                out.append(full)

        # deterministic order
        out.sort()
        return out