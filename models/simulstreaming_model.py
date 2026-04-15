import json
import os
import queue
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from core.types import AudioChunk
from models.model_base import ASRModel


class SimulStreamingModel(ASRModel):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 43007,
        sample_rate: int = 16000,
        auto_start_server: bool = True,
        repo_root: Optional[str] = None,
        python_executable: str = "python3",
        server_module: str = "simulstreaming_whisper_server",
        server_args: Optional[list[str]] = None,
        connect_timeout_s: float = 60.0,
        startup_wait_s: float = 0.2,
    ):
        self.host = host
        self.port = int(port)
        self.sr = int(sample_rate)

        self.auto_start_server = bool(auto_start_server)
        self.repo_root = repo_root
        self.python_executable = python_executable
        self.server_module = server_module
        self.server_args = server_args or []

        self.connect_timeout_s = float(connect_timeout_s)
        self.startup_wait_s = float(startup_wait_s)

        self._q_events: queue.Queue = queue.Queue(maxsize=500)
        self._running = False
        self._connected = False
        self._flushing = False

        self._sock: Optional[socket.socket] = None
        self._sock_file = None
        self._server_proc: Optional[subprocess.Popen] = None

        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._last_partial_text = ""
        self.compute_time_s = 0.0

        self._server_ready = threading.Event()
        self._server_failed = threading.Event()
        self._server_error_lines: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            return

        self._running = True
        self._connected = False
        self._flushing = False
        self.compute_time_s = 0.0
        self._last_partial_text = ""
        self._server_ready.clear()
        self._server_failed.clear()
        self._server_error_lines = []

        if self.auto_start_server:
            self._start_server_subprocess()
        else:
            self._server_ready.set()

        self._wait_for_server_ready()
        self._connect_with_retry()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._emit({
            "type": "info",
            "message": f"SimulStreaming connected to {self.host}:{self.port}",
        })

    def stop(self):
        self._running = False

        # If there is still a partial around, promote it once before closing.
        self._promote_last_partial_to_final()

        try:
            with self._lock:
                if self._sock_file is not None:
                    try:
                        self._sock_file.close()
                    except Exception:
                        pass
                    self._sock_file = None

                if self._sock is not None:
                    try:
                        self._sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
        except Exception:
            pass

        if self._connected:
            self._emit({
                "type": "info",
                "message": f"SimulStreaming disconnected from {self.host}:{self.port}",
            })

        if self._server_proc is not None:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=5.0)
            except Exception:
                try:
                    self._server_proc.kill()
                except Exception:
                    pass
            finally:
                self._server_proc = None

        self._connected = False

    def accept_chunk(self, chunk: AudioChunk):
        if not self._running or not self._connected or self._flushing:
            return

        if int(chunk.sample_rate) != self.sr:
            self._emit({
                "type": "error",
                "message": f"sample rate mismatch: got {chunk.sample_rate}, expected {self.sr}",
            })
            return

        try:
            t0 = time.perf_counter()
            payload = self._float32_to_pcm16le(chunk.data)

            with self._lock:
                if self._sock is None:
                    return
                self._sock.sendall(payload)

            self.compute_time_s += (time.perf_counter() - t0)

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self._connected = False
            if not self._flushing:
                self._emit({"type": "error", "message": f"socket send failed: {e}"})

    def poll(self) -> Optional[dict]:
        try:
            return self._q_events.get_nowait()
        except queue.Empty:
            return None

    def get_compute_time_s(self) -> float:
        return self.compute_time_s

    def flush_final(self):
        self._flushing = True

        # Promote current partial to final immediately.
        self._promote_last_partial_to_final()

        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.shutdown(socket.SHUT_WR)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, ev: dict):
        try:
            self._q_events.put_nowait(ev)
        except queue.Full:
            pass

    def _promote_last_partial_to_final(self):
        if self._last_partial_text:
            self._emit({
                "type": "final",
                "text": self._last_partial_text,
            })
            self._last_partial_text = ""

    def _start_server_subprocess(self):
        if not self.repo_root:
            raise ValueError("repo_root is required when auto_start_server=True")

        repo_root = str(Path(self.repo_root).expanduser().resolve())

        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + ":" + env.get("PYTHONPATH", "")

        cmd = [
            self.python_executable,
            "-m",
            self.server_module,
            *self.server_args,
        ]

        self._server_proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

        time.sleep(self.startup_wait_s)

    def _wait_for_server_ready(self):
        deadline = time.time() + self.connect_timeout_s

        while self._running and time.time() < deadline:
            if self._server_failed.is_set():
                detail = "\n".join(self._server_error_lines[-20:])
                raise RuntimeError(f"SimulStreaming server failed to start.\n{detail}")

            if self._server_ready.is_set():
                return

            if self._server_proc is not None:
                rc = self._server_proc.poll()
                if rc is not None:
                    detail = "\n".join(self._server_error_lines[-20:])
                    raise RuntimeError(
                        f"SimulStreaming server exited before becoming ready (returncode={rc}).\n{detail}"
                    )

            if self._can_connect_once():
                self._server_ready.set()
                return

            time.sleep(0.1)

        raise TimeoutError(
            f"Timed out waiting for SimulStreaming server to become ready on {self.host}:{self.port}"
        )

    def _can_connect_once(self) -> bool:
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.5)
            s.close()
            return True
        except OSError:
            return False

    def _connect_with_retry(self):
        deadline = time.time() + self.connect_timeout_s
        last_err = None

        while self._running and time.time() < deadline:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=2.0)
                sock.settimeout(None)

                with self._lock:
                    self._sock = sock
                    self._sock_file = sock.makefile("r", encoding="utf-8", newline="\n")

                self._connected = True
                return

            except OSError as e:
                last_err = e

                if self._server_proc is not None:
                    rc = self._server_proc.poll()
                    if rc is not None:
                        detail = "\n".join(self._server_error_lines[-20:])
                        raise RuntimeError(
                            f"SimulStreaming server exited before client connection (returncode={rc}).\n{detail}"
                        )

                time.sleep(0.25)

        raise ConnectionError(
            f"could not connect to SimulStreaming server at {self.host}:{self.port}: {last_err}"
        )

    def _reader_loop(self):
        while self._running:
            try:
                line = self._readline_blocking()
                if line is None:
                    time.sleep(0.01)
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self._emit({"type": "info", "message": f"non-json server output: {line}"})
                    continue

                self._handle_server_message(msg)

            except Exception as e:
                if self._running:
                    self._emit({"type": "error", "message": f"reader loop failed: {e}"})
                break

    def _stderr_loop(self):
        if self._server_proc is None or self._server_proc.stderr is None:
            return

        try:
            for line in self._server_proc.stderr:
                line = line.rstrip()
                if not line:
                    continue

                self._server_error_lines.append(line)
                self._emit({"type": "info", "message": f"[simulstreaming] {line}"})

                lower = line.lower()
                if "listening on" in lower:
                    self._server_ready.set()
                if "traceback" in lower or "error" in lower or "exception" in lower:
                    if "futurewarning" not in lower and "warning" not in lower:
                        self._server_failed.set()

        except Exception as e:
            if self._running:
                self._emit({"type": "info", "message": f"stderr loop stopped: {e}"})

    def _readline_blocking(self) -> Optional[str]:
        with self._lock:
            f = self._sock_file

        if f is None:
            return None

        try:
            line = f.readline()
            if line == "":
                return None
            return line
        except Exception as e:
            if self._running and not self._flushing:
                self._emit({"type": "error", "message": f"socket read failed: {e}"})
            return None

    def _handle_server_message(self, msg: dict):
        text = (msg.get("text") or "").strip()
        is_final = bool(msg.get("is_final", False))

        if text and text != self._last_partial_text:
            self._last_partial_text = text
            self._emit({
                "type": "partial",
                "text": text,
                "start": msg.get("start"),
                "end": msg.get("end"),
                "raw": msg,
            })

        if is_final:
            self._promote_last_partial_to_final()

    @staticmethod
    def _float32_to_pcm16le(x: np.ndarray) -> bytes:
        x = np.asarray(x, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        return (x * 32767.0).astype("<i2").tobytes()