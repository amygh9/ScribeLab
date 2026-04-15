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
  raise FileNotFoundError("None exist:\n " + "\n ".join(paths))

class Qwen3WindowModel(ASRModel):
  def __init__(
    self,
    model_dir: str,
    target_seconds: float = 3.0,
    overlap_seconds: float = 0.0,
    sample_rate: int = 16000,
  ):
    self.model_dir = model_dir
    self.target_seconds = target_seconds
    self.overlap_seconds = overlap_seconds
    self.sr = sample_rate
    

    self._q_events = queue.Queue(maxsize=50)
    self._running = False

    self._lock = threading.Lock()
    self._buf = []
    self._processing = False

    self.recognizer = None
    self.compute_time_s = 0.0

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

    print("[QWEN3] loading")
    print(" conv_frontend:", conv_frontend)
    print(" encoder:", encoder)
    print(" decoder:", decoder)
    print(" tokenizer:", tokenizer)

    self.recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
      conv_frontend=conv_frontend,
      encoder=encoder,
      decoder=decoder,
      tokenizer=tokenizer,
      num_threads=3,
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
      self._emit({"type": "info", "message": f"Processing {dur:.2f}s of audio with Qwen3..."})
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

    

    try:  

      stream = self.recognizer.create_stream()  

    

      t0 = time.perf_counter()  

      stream.accept_waveform(self.sr, audio.astype(np.float32, copy=False))  

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

    

    keep = int(self.overlap_seconds * self.sr)  

    if keep > 0 and audio.size > keep:  

      with self._lock:  

        self._buf.append(audio[-keep:])  

    

    self._processing = False