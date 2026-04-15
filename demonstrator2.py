#!/usr/bin/env python3
# demonstrator.py
# Streaming demo with non-blocking command input

import os
import time
import threading
import queue
import re

PROJECT_ROOT = "/home/iat/projectaria_client_sdk_samples"
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "asr_demo", "sessions")

from core.streaming_controller import StreamingController
from core.audio_preprocessing import PreprocessConfig

from models.vosk_model import VoskStreamingModel
from models.zipformer_model import ZipformerStreamingModel
from models.whisper_model import WhisperWindowModel
from models.wav2vec2_model import Wav2Vec2WindowModel
from models.qwen3_model import Qwen3WindowModel
from models.simulstreaming_model import SimulStreamingModel
from models.nemo_model import NemoStreamingModel
from models.parakeet_model import ParakeetStreamingModel
from models.qwen3vad_model import Qwen3VadModel
from models.whispervad_model import WhisperVadModel

from metrics.metrics import (
    SessionAggregator,
    HypothesisSessionStore,
    MetricsStore,
    LatencyTracker,
    OutputLatencyTracker,
)

# -----------------------------
# Ground Truth (hardcoded)
# -----------------------------
GROUND_TRUTH_TEXT = """Sterilisation ist ein validiertes Verfahren zur Inaktivierung lebensfähiger Mikroorganismen.
Der Zustand der Sterilität beschreibt das Freisein von vermehrungsfähigen Keimen.
Da absolute Keimfreiheit praktisch nicht nachweisbar ist, wird das Sterilitätssicherheitsniveau
als Wahrscheinlichkeitsmaß verwendet.
Vor der Sterilisation erfolgt die Reinigung zur Entfernung sichtbarer Anschmutzungen sowie die
Desinfektion zur Reduktion der mikrobiellen Belastung.
Die vorhandene Ausgangsbelastung wird als Bioburden bezeichnet.
Die Wirksamkeit eines Sterilisationsverfahrens hängt von definierten Prozessparametern wie
Temperatur, Druck, Einwirkzeit, Luftfeuchte und Konzentration des sterilisierenden Agens ab.
Die Validierung umfasst Installationsqualifizierung, Funktionsqualifizierung und
Leistungsqualifizierung.
Die Verifizierung bestätigt die Einhaltung festgelegter Anforderungen.
Ein Sterilbarrieresystem verhindert das Eindringen von Mikroorganismen und ermöglicht eine
aseptische Bereitstellung.
Die parametrische Freigabe erfolgt auf Basis dokumentierter Prozessdaten ohne zusätzliche
Endproduktprüfung.
Die Aufbereitung von wiederverwendbaren Medizinprodukten beinhaltet Vorbereitung, Reinigung,
Desinfektion, Prüfung auf Sauberkeit und Funktionalität, Verpackung, Sterilisation und
sachgerechte Lagerung.
"""

# -----------------------------
# Helpers for SimulStreaming partial accumulation
# -----------------------------
_WS_RE = re.compile(r"\s+")


def clean_partial_text(text: str) -> str:
    t = (text or "").strip()
    t = _WS_RE.sub(" ", t)
    return t.strip()


def should_keep_partial(text: str) -> bool:
    t = clean_partial_text(text)
    if not t:
        return False

    # skip fragments that are only punctuation/brackets
    if all(ch in "[](){}<>.,;:!?" for ch in t):
        return False

    return True


def append_incremental_text(current: str, new_text: str) -> str:
    """
    Append a new partial chunk while avoiding simple duplication.
    This assumes partials arrive in order and are incremental chunks.
    """
    current = clean_partial_text(current)
    new_text = clean_partial_text(new_text)

    if not new_text:
        return current

    if not current:
        return new_text

    # exact duplicate
    if new_text == current:
        return current

    # already covered
    if current.endswith(new_text):
        return current

    # overlap handling: longest suffix of current == prefix of new_text
    max_k = min(len(current), len(new_text))
    overlap = 0
    for k in range(max_k, 0, -1):
        if current[-k:] == new_text[:k]:
            overlap = k
            break

    if overlap > 0:
        return (current + new_text[overlap:]).strip()

    # normal append with spacing
    if not current.endswith(" ") and not new_text.startswith((".", ",", ";", ":", "!", "?")):
        return current + " " + new_text
    return current + new_text


# -----------------------------
# Pipeline sink (audio -> model)
# -----------------------------
class PipelineSink:
    def __init__(self, model, latency_tracker=None):
        self._lock = threading.Lock()
        self.model = model
        self.latency_tracker = latency_tracker

    def set_model(self, model):
        with self._lock:
            self.model = model

    def accept_chunk(self, chunk):
        with self._lock:
            m = self.model
            lt = self.latency_tracker

        if lt is not None:
            chunk_duration_s = len(chunk.data) / float(chunk.sample_rate)
            lt.on_audio_chunk(chunk.rms, chunk_duration_s)

        if m:
            m.accept_chunk(chunk)


# -----------------------------
# Model builder
# -----------------------------
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


def print_help():
    print("\nCommands:")
    print("  start [usb|wifi]        start streaming (default usb)")
    print("  stop                    stop streaming")
    print("  model <name>            switch model: vosk|zipformer|whisper|wav2vec2|qwen3|qwen3vad|nemo|parakeet|simulstreaming")
    print("  testid <id>             set test id (e.g., test_001)")
    print("  metrics                 after stop: compute/store metrics for last session")
    print("  showhyp                 show concatenated hypothesis of last session")
    print("  gain <float>            set gain for preprocessing (applies on next start)")
    print("  denoise on|off          enable/disable denoise (applies on next start)")
    print("  status                  show current state")
    print("  help                    show this help")
    print("  quit                    exit\n")


# -----------------------------
# Command input thread
# -----------------------------
def command_reader(cmd_q: "queue.Queue[str]", stop_flag: threading.Event):
    while not stop_flag.is_set():
        try:
            cmd = input("> ").strip()
        except EOFError:
            stop_flag.set()
            return
        cmd_q.put(cmd)


def main():
    interface = "usb"
    gain = 8.0
    denoise = False
    test_id = "test_001"

    cfg = PreprocessConfig(gain=gain, enable_denoise=denoise)
    controller = StreamingController(preprocess_cfg=cfg)

    model_name = "zipformer"
    model = build_model(model_name)
    model.start()

    latency_tracker = LatencyTracker(speech_rms_threshold=0.0025)
    output_latency_tracker = OutputLatencyTracker()

    sink = PipelineSink(model, latency_tracker=latency_tracker)
    controller.set_sink(sink)

    os.makedirs(SESSIONS_DIR, exist_ok=True)
    aggregator = SessionAggregator()
    hyp_store = HypothesisSessionStore(out_dir=SESSIONS_DIR)
    metrics_store = MetricsStore(out_dir=SESSIONS_DIR, per_language="de")

    last_partial = ""
    simul_partial_hyp = ""
    last_session_id = None
    last_session_hyp = ""
    last_latency_result = None
    last_output_latency_result = None

    print("=== ASR Demonstrator ===")
    print(f"Default model: {model_name}")
    print(f"Sessions dir: {SESSIONS_DIR}")

    cmd_q: "queue.Queue[str]" = queue.Queue()
    stop_input = threading.Event()
    threading.Thread(target=command_reader, args=(cmd_q, stop_input), daemon=True).start()

    def do_start(new_interface: str):
        nonlocal controller, interface, last_partial, simul_partial_hyp, last_session_id, last_session_hyp
        nonlocal last_latency_result, last_output_latency_result

        interface = new_interface

        cfg2 = PreprocessConfig(gain=gain, enable_denoise=denoise)
        controller = StreamingController(preprocess_cfg=cfg2)
        controller.set_sink(sink)

        aggregator.start()
        latency_tracker.on_stream_start()
        output_latency_tracker.on_stream_start()

        last_partial = ""
        simul_partial_hyp = ""
        last_session_id = None
        last_session_hyp = ""
        last_latency_result = None
        last_output_latency_result = None

        controller.start(interface=interface)

    def do_stop():
        nonlocal last_session_id, last_session_hyp, last_latency_result, last_output_latency_result, simul_partial_hyp

        if not controller.running:
            print("[DEMO] already stopped")
            return

        # force one last final before stopping for models that support it
        if model_name in {"zipformer", "nemo", "parakeet", "qwen3vad"} and hasattr(model, "flush_final"):
            model.flush_final()

            deadline = time.time() + 1.0
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
                        print(f"[FINAL] {txt2}")
                    break
                elif t2 == "partial":
                    pass
                elif t2 == "error":
                    print(f"[ERROR] {ev2.get('message', '')}")
                    break

        controller.stop()
        aggregator.stop()
        latency_tracker.on_stream_stop()

        if hasattr(model, "get_compute_time_s"):
            latency_tracker.model_compute_time_s = model.get_compute_time_s()

        last_latency_result = latency_tracker.result()
        last_output_latency_result = output_latency_tracker.result()

        if model_name == "simulstreaming" and simul_partial_hyp.strip():
            hyp_text = simul_partial_hyp.strip()
        else:
            hyp_text = aggregator.full_hypothesis()

        last_session_hyp = hyp_text

        session_id = aggregator.make_session_id(model_name, test_id)
        last_session_id = session_id

        meta = {"interface": interface, "gain": gain, "denoise": denoise}
        hyp_store.append_session(
            session_id=session_id,
            model=model_name,
            test_id=test_id,
            stream_start_utc=aggregator.stream_start_utc or "",
            stream_stop_utc=aggregator.stream_stop_utc or "",
            hyp_text=hyp_text,
            meta=meta,
        )

        print(f"[DEMO] session saved: {session_id}")
        print(f"[DEMO] hypothesis length: {len(hyp_text)} chars")

    print_help()

    try:
        while True:
            ev = model.poll()
            if ev:
                t = ev.get("type")

                if t == "partial":
                    text = clean_partial_text(ev.get("text") or "")
                    if text and text != last_partial:
                        print(f"[PARTIAL] {text}")
                        last_partial = text
                        latency_tracker.on_partial()
                        output_latency_tracker.on_output("partial")

                        if controller.running and model_name == "simulstreaming" and should_keep_partial(text):
                            simul_partial_hyp = append_incremental_text(simul_partial_hyp, text)

                elif t == "final":
                    text = (ev.get("text") or "").strip()
                    if text:
                        print(f"[FINAL] {text}")
                        latency_tracker.on_final()
                        output_latency_tracker.on_output("final")
                        if controller.running:
                            aggregator.add_final(text)
                    last_partial = ""

                elif t == "error":
                    print(f"[ERROR] {ev.get('message', '')}")
                else:
                    print(f"[EVENT] {ev}")

            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd is not None:
                cmd = cmd.strip()
                if cmd:
                    parts = cmd.split()
                    c = parts[0].lower()

                    if c == "help":
                        print_help()

                    elif c in ("quit", "exit"):
                        break

                    elif c == "status":
                        print(
                            f"[STATUS] streaming={controller.running}, interface={interface}, "
                            f"model={model_name}, test_id={test_id}, gain={gain}, denoise={denoise}"
                        )
                        if last_session_id:
                            print(f"[STATUS] last_session_id={last_session_id} hyp_len={len(last_session_hyp)}")
                        if last_latency_result is not None:
                            print("[STATUS] processing latency:")
                            print(f"  stream_duration_s={last_latency_result.stream_duration_s}")
                            print(f"  audio_duration_s={last_latency_result.audio_duration_s}")
                            print(f"  model_compute_time_s={last_latency_result.model_compute_time_s}")
                            print(f"  rtf={last_latency_result.rtf}")
                            print(f"  stream_start_to_first_partial_s={last_latency_result.stream_start_to_first_partial_s}")
                            print(f"  stream_start_to_first_final_s={last_latency_result.stream_start_to_first_final_s}")
                            print(f"  speech_start_to_first_partial_s={last_latency_result.speech_start_to_first_partial_s}")
                            print(f"  speech_start_to_first_final_s={last_latency_result.speech_start_to_first_final_s}")
                            print(f"  stop_to_last_final_s={last_latency_result.stop_to_last_final_s}")
                        if last_output_latency_result is not None:
                            print("[STATUS] output latency:")
                            print(f"  start_to_first_output_s={last_output_latency_result.start_to_first_output_s}")
                            print(f"  start_to_first_partial_output_s={last_output_latency_result.start_to_first_partial_output_s}")
                            print(f"  start_to_first_final_output_s={last_output_latency_result.start_to_first_final_output_s}")
                            print(f"  num_outputs={last_output_latency_result.num_outputs}")
                            print(f"  mean_inter_output_s={last_output_latency_result.mean_inter_output_s}")
                            print(f"  min_inter_output_s={last_output_latency_result.min_inter_output_s}")
                            print(f"  max_inter_output_s={last_output_latency_result.max_inter_output_s}")
                            print(f"  last_output_time_from_start_s={last_output_latency_result.last_output_time_from_start_s}")

                    elif c == "testid":
                        if len(parts) < 2:
                            print("Usage: testid test_001")
                        else:
                            test_id = parts[1].strip()
                            print(f"[DEMO] test_id set -> {test_id}")

                    elif c == "start":
                        new_if = interface
                        if len(parts) >= 2:
                            new_if = parts[1].lower()
                            if new_if not in ("usb", "wifi"):
                                print("interface must be usb or wifi")
                                new_if = interface
                        do_start(new_if)

                    elif c == "stop":
                        do_stop()

                    elif c == "showhyp":
                        if not last_session_id:
                            print("[DEMO] No session yet. Run start -> stop first.")
                        else:
                            print("\n=== FULL HYPOTHESIS ===")
                            print(last_session_hyp if last_session_hyp else "(empty)")
                            print("=======================\n")

                    elif c == "metrics":
                        if controller.running:
                            print("[DEMO] Stop streaming first (stop)")
                            continue
                        if not last_session_id:
                            print("[DEMO] No session yet. Run start -> stop first.")
                            continue
                        if not last_session_hyp.strip():
                            print("[DEMO] Hypothesis empty.")
                            continue

                        gt_text = GROUND_TRUTH_TEXT.strip()
                        meta = {"interface": interface, "gain": gain, "denoise": denoise}

                        res = metrics_store.append(
                            session_id=last_session_id,
                            model=model_name,
                            test_id=test_id,
                            gt_text=gt_text,
                            hyp_text=last_session_hyp,
                            latency=last_latency_result,
                            output_latency=last_output_latency_result,
                            meta=meta,
                        )

                        print("\n=== METRICS ===")
                        print(f"GT words:  {len(gt_text.split())}")
                        print(f"HYP words: {len(last_session_hyp.split())}")
                        print(f"WER: {res.wer:.4f}")
                        print(f"CER: {res.cer:.4f}")
                        print(f"SER: {res.ser:.4f}")
                        if res.per is None:
                            print("PER: (not available)")
                        else:
                            print(f"PER: {res.per:.4f}")
                        print(
                            f"BERTScore-P: {res.bertscore_precision:.4f}"
                            if res.bertscore_precision is not None else "BERTScore-P: (not available)"
                        )
                        print(
                            f"BERTScore-R: {res.bertscore_recall:.4f}"
                            if res.bertscore_recall is not None else "BERTScore-R: (not available)"
                        )
                        print(
                            f"BERTScore-F1: {res.bertscore_f1:.4f}"
                            if res.bertscore_f1 is not None else "BERTScore-F1: (not available)"
                        )

                        if last_latency_result is not None:
                            print("\n=== PROCESSING LATENCY ===")
                            print(f"stream_duration_s: {last_latency_result.stream_duration_s}")
                            print(f"audio_duration_s: {last_latency_result.audio_duration_s}")
                            print(f"model_compute_time_s: {last_latency_result.model_compute_time_s}")
                            print(f"RTF: {last_latency_result.rtf}")
                            print(f"stream_start_to_first_partial_s: {last_latency_result.stream_start_to_first_partial_s}")
                            print(f"stream_start_to_first_final_s: {last_latency_result.stream_start_to_first_final_s}")
                            print(f"speech_start_to_first_partial_s: {last_latency_result.speech_start_to_first_partial_s}")
                            print(f"speech_start_to_first_final_s: {last_latency_result.speech_start_to_first_final_s}")
                            print(f"stop_to_last_final_s: {last_latency_result.stop_to_last_final_s}")

                        if last_output_latency_result is not None:
                            print("\n=== OUTPUT LATENCY ===")
                            print(f"start_to_first_output_s: {last_output_latency_result.start_to_first_output_s}")
                            print(f"start_to_first_partial_output_s: {last_output_latency_result.start_to_first_partial_output_s}")
                            print(f"start_to_first_final_output_s: {last_output_latency_result.start_to_first_final_output_s}")
                            print(f"num_outputs: {last_output_latency_result.num_outputs}")
                            print(f"mean_inter_output_s: {last_output_latency_result.mean_inter_output_s}")
                            print(f"min_inter_output_s: {last_output_latency_result.min_inter_output_s}")
                            print(f"max_inter_output_s: {last_output_latency_result.max_inter_output_s}")
                            print(f"last_output_time_from_start_s: {last_output_latency_result.last_output_time_from_start_s}")

                        print("Stored in sessions/metrics.csv\n")

                    elif c == "model":
                        if len(parts) < 2:
                            print("Usage: model vosk|zipformer|whisper|wav2vec2|qwen3|qwen3vad|nemo|parakeet|simulstreaming")
                        else:
                            new_name = parts[1].lower().strip()

                            if controller.running:
                                print("[DEMO] stopping stream before model switch")
                                do_stop()

                            model.stop()
                            model = build_model(new_name)
                            model.start()

                            sink.set_model(model)
                            model_name = new_name
                            last_partial = ""
                            simul_partial_hyp = ""
                            print(f"[DEMO] switched model -> {model_name}")

                    elif c == "gain":
                        if len(parts) < 2:
                            print("Usage: gain 8.0")
                        else:
                            try:
                                gain = float(parts[1])
                                print(f"[DEMO] gain set to {gain} (applies on next start)")
                            except ValueError:
                                print("gain must be a number")

                    elif c == "denoise":
                        if len(parts) < 2 or parts[1].lower() not in ("on", "off"):
                            print("Usage: denoise on|off")
                        else:
                            denoise = (parts[1].lower() == "on")
                            print(f"[DEMO] denoise set to {denoise} (applies on next start)")

                    else:
                        print("Unknown command. Type 'help'.")

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        stop_input.set()
        try:
            controller.stop()
        except Exception:
            pass
        try:
            model.stop()
        except Exception:
            pass
        print("bye")


if __name__ == "__main__":
    main()