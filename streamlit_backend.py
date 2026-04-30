#!/usr/bin/env python3
import os
import json
import time
import threading
import re
from pathlib import Path
from dataclasses import asdict

PROJECT_ROOT = "/home/iat/projectaria_client_sdk_samples"
ASR_DEMO_DIR = Path(PROJECT_ROOT) / "asr_demo"
SESSIONS_DIR = ASR_DEMO_DIR / "sessions"
LOG_PATH = ASR_DEMO_DIR / "live_output.log"
LATEST_SESSION_PATH = ASR_DEMO_DIR / "latest_session.json"

from core.streaming_controller import StreamingController
from core.audio_preprocessing import PreprocessConfig

from models.vosk_model import VoskStreamingModel
from models.zipformer_model import ZipformerStreamingModel
from models.whisper_model import WhisperWindowModel
from models.wav2vec2_model import Wav2Vec2WindowModel
from models.qwen3_model import Qwen3WindowModel

from models.simulstreaming_model import SimulStreamingModel
from models.qwen3vad_model import Qwen3VadModel
from models.whispervad_model import WhisperVadModel

from metrics.metrics import (
    SessionAggregator,
    HypothesisSessionStore,
    LatencyTracker,
    OutputLatencyTracker,
)

GROUND_TRUTH_TEXT = """Die Sterilisation ist ein validiertes Verfahren zur Sicherstellung der Sterilität.
Dabei werden Mikroorganismen in einem kontrollierten Prozess inaktiviert.

Vor dem Sterilisationszyklus erfolgt die Reinigung mit einem geeigneten Reinigungsmittel.
Die Reinigung kann manuell oder in einem Reinigungs Desinfektionsgerät durchgeführt werden.
Anschließend erfolgt die chemische Desinfektion.

Die mikrobielle Kontamination wird als Bioburden beschrieben.
Bei der Belastungsprüfung werden koloniebildende Einheiten analysiert.

Der Sterilisationsprozess läuft in einer Kammer mit definierter Referenztemperatur.
Die Haltezeit, die Aufbaudauer und die Abbaudauer bestimmen die Zykluszeit.
Parameter wie Luftfeuchte werden während des Prozesses überwacht.

Bei der Verwendung von Filtern wird eine Filterintegritätsprüfung durchgeführt.

In strahlenbasierten Verfahren wird die Dosimetrie eingesetzt.
Dabei wird die absorbierte Strahlendosis überwacht, um die Sterilitätssicherheit zu gewährleisten.
Dies erfolgt mit Dosimeter und Dosimetriesystem.

Ein aseptischer Arbeitsbereich dient der Vermeidung extrinsischer und intrinsischer Kontamination.
"""

_WS_RE = re.compile(r"\s+")
_LETTER_END_RE = re.compile(r"[A-Za-zÄÖÜäöüß]$")
_LOWER_START_RE = re.compile(r"^[a-zäöüß]")


def clean_partial_text(text: str) -> str:
    t = (text or "").strip()
    t = _WS_RE.sub(" ", t)
    return t.strip()


def should_keep_partial(text: str) -> bool:
    t = clean_partial_text(text)
    if not t:
        return False
    if all(ch in "[](){}<>.,;:!?" for ch in t):
        return False
    return True


def should_join_without_space(current: str, new_text: str, delta_s: float, threshold_s: float = 0.30) -> bool:
    current = clean_partial_text(current)
    new_text = clean_partial_text(new_text)

    if not current or not new_text:
        return False
    if delta_s > threshold_s:
        return False

    # do not glue after punctuation
    if current.endswith((".", ",", ";", ":", "!", "?", ")", "]")):
        return False

    # likely same-word continuation, e.g. "sterili" + "sation"
    if _LETTER_END_RE.search(current) and _LOWER_START_RE.search(new_text):
        return True

    return False


def append_incremental_text(current: str, new_text: str, delta_s=None) -> str:
    current = clean_partial_text(current)
    new_text = clean_partial_text(new_text)

    if not new_text:
        return current
    if not current:
        return new_text
    if new_text == current:
        return current
    if current.endswith(new_text):
        return current

    max_k = min(len(current), len(new_text))
    overlap = 0
    for k in range(max_k, 0, -1):
        if current[-k:] == new_text[:k]:
            overlap = k
            break

    if overlap > 0:
        return (current + new_text[overlap:]).strip()

    if delta_s is not None and should_join_without_space(current, new_text, delta_s):
        return current + new_text

    if not current.endswith(" ") and not new_text.startswith((".", ",", ";", ":", "!", "?")):
        return current + " " + new_text
    return current + new_text


class PipelineSink:
    def __init__(self, model, latency_tracker=None):
        self._lock = threading.Lock()
        self.model = model
        self.latency_tracker = latency_tracker
        self.chunk_count = 0

    def set_model(self, model):
        with self._lock:
            self.model = model

    def accept_chunk(self, chunk):
        with self._lock:
            m = self.model
            lt = self.latency_tracker

        chunk_duration_s = len(chunk.data) / float(chunk.sample_rate)
        self.chunk_count += 1

        if self.chunk_count % 50 == 1:
            log_line(
                f"[CHUNK] n={self.chunk_count} "
                f"samples={len(chunk.data)} "
                f"sr={chunk.sample_rate} "
                f"duration={chunk_duration_s:.4f}s"
            )

        if lt is not None:
            lt.on_audio_chunk(chunk.rms, chunk_duration_s)

        if m:
            m.accept_chunk(chunk)
def build_model(name: str):
    name = name.lower().strip()

    if name == "zipformer":
        model_dir = os.path.join(PROJECT_ROOT, "ASR_models", "Zipformer", "de", "kroko_64l")
        print("Using Zipformer model_dir:", model_dir)
        return ZipformerStreamingModel(model_dir=model_dir)

    if name == "nemo":
        model_dir = os.path.join(PROJECT_ROOT, "ASR_models","Zipformer", "en", "nemo_ctc_80ms")
        print("Using Nemo model_dir:", model_dir)
        return NemoStreamingModel(model_dir=model_dir)

    if name == "parakeet":
        model_dir = os.path.join(
            PROJECT_ROOT,
            "sherpa-stt-onnx",
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        )
        print("Using Parakeet model_dir:", model_dir, flush=True)
        return ParakeetStreamingModel(
            model_dir=model_dir,
            num_threads=1,
            decode_interval_s=1.0,
            max_buffer_s=8.0,
            silence_rms_threshold=0.004,
            silence_finalize_s=1.0,
            min_final_chars=12,
        )

    if name == "vosk":
        model_path = os.path.join(PROJECT_ROOT, "ASR_models","vosk-model-small-de-0.15")
        print("Using Vosk model_path:", model_path)
        return VoskStreamingModel(model_path=model_path)

    if name == "whisper":
        return WhisperWindowModel(
            language="de",
            model_name="small",
            target_seconds=4.0,
            overlap_seconds=0.0,
            silence_rms_threshold=0.003,
            no_speech_threshold=0.8,
        )

    if name == "wav2vec2":
        return Wav2Vec2WindowModel(
            model_id="facebook/wav2vec2-large-xlsr-53-german",
            target_seconds=20.0,
            overlap_seconds=10.0,
        )

    if name == "qwen3":
        model_dir = os.path.join(
            PROJECT_ROOT,
            "ASR_models",
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        )
        print("Using Qwen3 model_dir:", model_dir, flush=True)
        return Qwen3WindowModel(
            model_dir=model_dir,
            target_seconds=3.0,
            overlap_seconds=0.0,
        )

    if name == "qwen3vad":
        model_dir = os.path.join(
            PROJECT_ROOT, "ASR_models", "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
        )
        print("Using Qwen3 model_dir:", model_dir, flush=True)
        return Qwen3VadModel(
            model_dir=model_dir,
            sample_rate=16000,
            num_threads=3,
            vad_threshold=0.5,
            vad_min_silence_s=0.5,
            vad_min_speech_s=0.25,
            vad_window_size=512,
            vad_max_speech_s=20.0,
            pre_speech_ms=200,
        )
    if name == "whispervad":
        model_dir = os.path.join(
            PROJECT_ROOT, "ASR_models", "sherpa-onnx-whisper-small"
        )
        print("Using Whisper VAD model_dir:", model_dir, flush=True)

        return WhisperVadModel(
            model_dir=model_dir,
            sample_rate=16000,
            num_threads=3,

            # Whisper settings
            whisper_language="de",
            whisper_task="transcribe",
            whisper_tail_paddings=-1,

            # VAD settings (tuned for faster demo)
            vad_threshold=0.5,
            vad_min_silence_s=0.3,   # ↓ faster endpoint than Qwen
            vad_min_speech_s=0.2,
            vad_window_size=512,
            vad_max_speech_s=10.0,   # ↓ shorter segments → faster output
            pre_speech_ms=200,
        )
    if name == "simulstreaming":
        repo_root = os.path.join(PROJECT_ROOT, "ASR_models", "SimulStreaming")
        print("Using SimulStreaming repo_root:", repo_root, flush=True)

        return SimulStreamingModel(
            host="127.0.0.1",
            port=43007,
            sample_rate=16000,
            auto_start_server=True,
            repo_root=repo_root,
            python_executable="python3",
            server_module="simulstreaming_whisper_server",
            server_args=[
                "--model_path", "small",
                "--language", "de",
                "--task", "transcribe",
                "--beams", "1",
            ],
        )

    raise ValueError(f"Unknown model '{name}'")


def log_line(text: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
        f.flush()


def main():
    model_name = os.environ.get("ASR_MODEL", "zipformer")
    interface = os.environ.get("ASR_INTERFACE", "usb")
    gain = float(os.environ.get("ASR_GAIN", "8.0"))
    denoise = os.environ.get("ASR_DENOISE", "false").lower() == "true"
    test_id = os.environ.get("ASR_TEST_ID", "test_001")
    person_id = os.environ.get("ASR_PERSON_ID", "person_001")
    gender = os.environ.get("ASR_GENDER", "not_specified")

    ASR_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    if LATEST_SESSION_PATH.exists():
        LATEST_SESSION_PATH.unlink()

    cfg = PreprocessConfig(gain=gain, enable_denoise=denoise)
    controller = StreamingController(preprocess_cfg=cfg)

    model = build_model(model_name)
    model.start()

    latency_tracker = LatencyTracker(speech_rms_threshold=0.0025)
    output_latency_tracker = OutputLatencyTracker()
    aggregator = SessionAggregator()
    hyp_store = HypothesisSessionStore(out_dir=str(SESSIONS_DIR))

    sink = PipelineSink(model, latency_tracker=latency_tracker)
    controller.set_sink(sink)

    simul_partial_hyp = ""
    last_partial = ""
    last_partial_time = None

    aggregator.start()
    latency_tracker.on_stream_start()
    output_latency_tracker.on_stream_start()

    log_line(f"[INFO] backend started model={model_name} interface={interface} denoise={denoise}")
    controller.start(interface=interface)

    try:
        while True:
            ev = model.poll()
            if ev:
                t = ev.get("type")

                if t == "partial":
                    text = clean_partial_text(ev.get("text") or "")
                    if text:
                        latency_tracker.on_partial()
                        output_latency_tracker.on_output("partial")
                        log_line(f"[PARTIAL] {text}")

                        if (
                            model_name == "simulstreaming"
                            and text != last_partial
                            and should_keep_partial(text)
                        ):
                            now = time.time()
                            delta_s = None if last_partial_time is None else (now - last_partial_time)
                            simul_partial_hyp = append_incremental_text(simul_partial_hyp, text, delta_s=delta_s)
                            log_line(f"[LIVE_HYP] {simul_partial_hyp}")
                            last_partial_time = now

                        last_partial = text

                elif t == "final":
                    text = (ev.get("text") or "").strip()
                    if text:
                        latency_tracker.on_final()
                        output_latency_tracker.on_output("final")
                        aggregator.add_final(text)
                        log_line(f"[FINAL] {text}")
                    last_partial = ""

                elif t == "error":
                    log_line(f"[ERROR] {ev.get('message', '')}")

                else:
                    log_line(f"[EVENT] {ev}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log_line(f"[ERROR] backend exception: {e}")
        raise
    finally:
        if model_name in {"zipformer", "simulstreaming", "qwen3vad"} and hasattr(model, "flush_final"):
            try:
                model.flush_final()
                deadline = time.time() + 0.5
                while time.time() < deadline:
                    ev2 = model.poll()
                    if not ev2:
                        time.sleep(0.01)
                        continue

                    t2 = ev2.get("type")
                    if t2 == "final":
                        txt2 = (ev2.get("text") or "").strip()
                        if txt2:
                            aggregator.add_final(txt2)
                            latency_tracker.on_final()
                            output_latency_tracker.on_output("final")
                            log_line(f"[FINAL] {txt2}")
                        break
                    elif t2 == "partial":
                        txt2 = clean_partial_text(ev2.get("text") or "")
                        if (
                            model_name == "simulstreaming"
                            and txt2
                            and txt2 != last_partial
                            and should_keep_partial(txt2)
                        ):
                            now = time.time()
                            delta_s = None if last_partial_time is None else (now - last_partial_time)
                            simul_partial_hyp = append_incremental_text(simul_partial_hyp, txt2, delta_s=delta_s)
                            log_line(f"[PARTIAL] {txt2}")
                            log_line(f"[LIVE_HYP] {simul_partial_hyp}")
                            last_partial = txt2
                            last_partial_time = now
                    elif t2 == "error":
                        log_line(f"[ERROR] {ev2.get('message', '')}")
                        break
            except Exception as e:
                log_line(f"[ERROR] flush failed: {e}")

        try:
            controller.stop()
        except Exception:
            pass

        aggregator.stop()
        latency_tracker.on_stream_stop()

        if hasattr(model, "get_compute_time_s"):
            latency_tracker.model_compute_time_s = model.get_compute_time_s()

        try:
            model.stop()
        except Exception:
            pass

        if model_name == "simulstreaming" and simul_partial_hyp.strip():
            hyp_text = simul_partial_hyp.strip()
        else:
            hyp_text = aggregator.full_hypothesis()

        session_id = aggregator.make_session_id(model_name, test_id)

        meta = {
            "interface": interface,
            "gain": gain,
            "denoise": denoise,
            "person_id": person_id,
            "gender": gender,
        }

        hyp_store.append_session(
            session_id=session_id,
            model=model_name,
            test_id=test_id,
            stream_start_utc=aggregator.stream_start_utc or "",
            stream_stop_utc=aggregator.stream_stop_utc or "",
            hyp_text=hyp_text,
            meta=meta,
        )

        payload = {
    
            "session_id": session_id,
            "model": model_name,
            "test_id": test_id,
            "person_id": person_id,
            "gender": gender,
            "ground_truth_text": GROUND_TRUTH_TEXT,
            "hyp_text": hyp_text,
            "meta": meta,
            "latency": asdict(latency_tracker.result()),
            "output_latency": asdict(output_latency_tracker.result()),
        }
        with open(LATEST_SESSION_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        log_line("[INFO] backend stopped")


if __name__ == "__main__":
    main()