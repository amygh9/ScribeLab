import queue
import threading
import time
from typing import List, Optional

import numpy as np
import whisper

from core.types import AudioChunk
from models.model_base import ASRModel


class Whisper2(ASRModel):
    """
    Strategy A:
    - Keep a rolling audio buffer
    - Re-decode it periodically
    - Compare current hypothesis with previous hypothesis
    - Commit only the stable prefix
    - Emit the unstable suffix as partial

    This is a simpler, string/word-based approximation of the "local agreement"
    style used in streaming Whisper systems.
    """

    def __init__(
        self,
        language: str = "de",
        model_name: str = "small",
        sample_rate: int = 16000,
        min_buffer_seconds: float = 1.2,
        decode_period_seconds: float = 0.7,
        max_buffer_seconds: float = 6.0,
        stability_reserve_words: int = 2,
        silence_rms_threshold: float = 0.003,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4,
        temperature: float = 0.0,
    ):
        self.language = language
        self.model_name = model_name
        self.sr = sample_rate

        self.min_buffer_seconds = float(min_buffer_seconds)
        self.decode_period_seconds = float(decode_period_seconds)
        self.max_buffer_seconds = float(max_buffer_seconds)
        self.stability_reserve_words = int(stability_reserve_words)

        self.silence_rms_threshold = float(silence_rms_threshold)
        self.no_speech_threshold = float(no_speech_threshold)
        self.logprob_threshold = float(logprob_threshold)
        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.temperature = float(temperature)

        self._q_events = queue.Queue(maxsize=200)
        self._running = False

        self._lock = threading.Lock()
        self._buf: List[np.ndarray] = []
        self._processing = False
        self._last_decode_t = 0.0

        # Text state
        self._prev_words: List[str] = []
        self._committed_words: List[str] = []
        self._last_partial_text = ""

        self.model = None
        self.compute_time_s = 0.0

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def start(self):
        if self._running:
            return

        print(f"[WHISPER2] loading {self.model_name} (CPU)")
        self.model = whisper.load_model(self.model_name)

        self.compute_time_s = 0.0
        self._buf = []
        self._processing = False
        self._last_decode_t = 0.0
        self._prev_words = []
        self._committed_words = []
        self._last_partial_text = ""
        self._running = True

    def stop(self):
        self._running = False

    def accept_chunk(self, chunk: AudioChunk):
        if not self._running:
            return

        should_start = False
        now = time.perf_counter()

        with self._lock:
            self._buf.append(chunk.data.astype(np.float32, copy=False))
            self._trim_audio_buffer_locked()

            total_samples = sum(x.size for x in self._buf)
            dur_s = total_samples / float(self.sr)

            enough_audio = dur_s >= self.min_buffer_seconds
            due_for_decode = (now - self._last_decode_t) >= self.decode_period_seconds

            if enough_audio and due_for_decode and not self._processing:
                self._processing = True
                self._last_decode_t = now
                should_start = True

        if should_start:
            threading.Thread(target=self._process, daemon=True).start()

    def poll(self) -> Optional[dict]:
        try:
            return self._q_events.get_nowait()
        except queue.Empty:
            return None

    def get_compute_time_s(self) -> float:
        return self.compute_time_s

    def flush_final(self):
        """
        Optional manual flush to force one more decode and commit what looks stable.
        Useful before shutdown if you want to squeeze out the last text.
        """
        if not self._running:
            return

        should_start = False
        with self._lock:
            if not self._processing and sum(x.size for x in self._buf) > 0:
                self._processing = True
                should_start = True

        if should_start:
            self._process(force_flush=True)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _emit(self, ev: dict):
        try:
            self._q_events.put_nowait(ev)
        except queue.Full:
            pass

    def _trim_audio_buffer_locked(self):
        max_samples = int(self.max_buffer_seconds * self.sr)
        total = sum(x.size for x in self._buf)

        while self._buf and total > max_samples:
            removed = self._buf.pop(0)
            total -= removed.size

    def _concat_audio_snapshot(self) -> np.ndarray:
        with self._lock:
            if not self._buf:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._buf).astype(np.float32, copy=False)

    def _schedule_next_if_ready(self):
        should_start = False
        now = time.perf_counter()

        with self._lock:
            total_samples = sum(x.size for x in self._buf)
            dur_s = total_samples / float(self.sr)

            enough_audio = dur_s >= self.min_buffer_seconds
            due_for_decode = (now - self._last_decode_t) >= self.decode_period_seconds

            if self._running and enough_audio and due_for_decode:
                self._processing = True
                self._last_decode_t = now
                should_start = True
            else:
                self._processing = False

        if should_start:
            threading.Thread(target=self._process, daemon=True).start()

    def _process(self, force_flush: bool = False):
        try:
            audio = self._concat_audio_snapshot()

            if audio.size == 0:
                return

            audio = np.clip(audio, -1.0, 1.0)

            rms = float(np.sqrt(np.mean(audio * audio))) if audio.size > 0 else 0.0
            if rms < self.silence_rms_threshold and not force_flush:
                self._emit({
                    "type": "info",
                    "message": f"skip whisper decode: low energy rms={rms:.5f}",
                })
                return

            t0 = time.perf_counter()

            result = self.model.transcribe(
                audio,
                fp16=False,
                language=self.language,
                task="transcribe",
                temperature=self.temperature,
                condition_on_previous_text=False,
                no_speech_threshold=self.no_speech_threshold,
                compression_ratio_threshold=self.compression_ratio_threshold,
                logprob_threshold=self.logprob_threshold,
                verbose=False,
            )

            self.compute_time_s += (time.perf_counter() - t0)

            raw_text = (result.get("text") or "").strip()
            if not raw_text:
                self._handle_empty_result()
                return

            current_words = self._tokenize_words(raw_text)

            if not current_words:
                self._handle_empty_result()
                return

            self._update_hypothesis(current_words, force_flush=force_flush)

        except Exception as e:
            self._emit({"type": "error", "message": str(e)})
        finally:
            self._schedule_next_if_ready()

    def _handle_empty_result(self):
        if self._last_partial_text:
            self._last_partial_text = ""
            self._emit({"type": "partial", "text": ""})
        self._emit({"type": "info", "message": "empty whisper output"})

    def _update_hypothesis(self, current_words: List[str], force_flush: bool = False):
        prev_words = self._prev_words
        committed_words = self._committed_words

        stable_len = self._longest_common_prefix_len(prev_words, current_words)

        # Reserve the last few "stable" words so they are not committed too early.
        # This avoids prematurely finalizing words that may still change slightly.
        if force_flush:
            commit_target_len = stable_len
        else:
            commit_target_len = max(0, stable_len - self.stability_reserve_words)

        # Never commit less than already committed.
        commit_target_len = max(commit_target_len, len(committed_words))
        commit_target_len = min(commit_target_len, len(current_words))

        # Emit newly committed words as final append.
        if commit_target_len > len(committed_words):
            new_final_words = current_words[len(committed_words):commit_target_len]
            final_text = self._join_words(new_final_words).strip()
            if final_text:
                self._emit({"type": "final", "text": final_text})

            self._committed_words = current_words[:commit_target_len]
            committed_words = self._committed_words

        # Partial = current hypothesis tail after committed prefix.
        partial_words = current_words[len(committed_words):]
        partial_text = self._join_words(partial_words).strip()

        if partial_text != self._last_partial_text:
            self._last_partial_text = partial_text
            self._emit({"type": "partial", "text": partial_text})

        self._prev_words = current_words

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        # Simple whitespace tokenization is enough for a first streaming version.
        return text.split()

    @staticmethod
    def _join_words(words: List[str]) -> str:
        return " ".join(words)

    @staticmethod
    def _longest_common_prefix_len(a: List[str], b: List[str]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i