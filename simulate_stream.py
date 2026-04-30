#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import soundfile as sf

from core.types import AudioChunk
from core.audio_preprocessing import PreprocessConfig, resample_linear

from streamlit_backend import (
    build_model,
    PipelineSink,
    GROUND_TRUTH_TEXT,
    clean_partial_text,
    should_keep_partial,
    append_incremental_text,
)

from metrics.metrics import (
    SessionAggregator,
    LatencyTracker,
    OutputLatencyTracker,
    MetricsStore,
)

PROJECT_ROOT = Path("/home/iat/projectaria_client_sdk_samples")
ASR_DEMO_DIR = PROJECT_ROOT / "asr_demo"
SESSIONS_DIR = ASR_DEMO_DIR / "sessions"
LIVE_METRICS_CSV_PATH = SESSIONS_DIR / "metrics.csv"

OUT_DIR = ASR_DEMO_DIR / "analysis" / "recording_stream_simulation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MODELS = ["zipformer", "vosk", "whispervad", "qwen3vad"]


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != target_sr:
        audio = resample_linear(audio, sr, target_sr)

    return audio.astype(np.float32)


def poll_model(
    model,
    model_name: str,
    aggregator: SessionAggregator,
    latency_tracker: LatencyTracker,
    output_latency_tracker: OutputLatencyTracker,
    simul_state: dict,
):
    ev = model.poll()
    if not ev:
        return

    t = ev.get("type")

    if t == "partial":
        text = clean_partial_text(ev.get("text") or "")
        if text:
            latency_tracker.on_partial()
            output_latency_tracker.on_output("partial")

            if (
                model_name == "simulstreaming"
                and text != simul_state["last_partial"]
                and should_keep_partial(text)
            ):
                now = time.time()
                last_time = simul_state["last_partial_time"]
                delta_s = None if last_time is None else now - last_time

                simul_state["simul_partial_hyp"] = append_incremental_text(
                    simul_state["simul_partial_hyp"],
                    text,
                    delta_s=delta_s,
                )

                simul_state["last_partial_time"] = now

            simul_state["last_partial"] = text

    elif t == "final":
        text = (ev.get("text") or "").strip()
        if text:
            latency_tracker.on_final()
            output_latency_tracker.on_output("final")
            aggregator.add_final(text)

        simul_state["last_partial"] = ""

    elif t == "error":
        print(f"[ERROR] {model_name}: {ev.get('message', '')}")


def drain_model_events(
    model,
    model_name: str,
    aggregator: SessionAggregator,
    latency_tracker: LatencyTracker,
    output_latency_tracker: OutputLatencyTracker,
    simul_state: dict,
    seconds: float = 0.01,
):
    end_t = time.time() + seconds

    while time.time() < end_t:
        poll_model(
            model=model,
            model_name=model_name,
            aggregator=aggregator,
            latency_tracker=latency_tracker,
            output_latency_tracker=output_latency_tracker,
            simul_state=simul_state,
        )
        time.sleep(0.002)


def flush_and_drain(
    model,
    model_name: str,
    aggregator: SessionAggregator,
    latency_tracker: LatencyTracker,
    output_latency_tracker: OutputLatencyTracker,
    simul_state: dict,
):
    if hasattr(model, "flush_final"):
        try:
            model.flush_final()
        except Exception as e:
            print(f"[WARN] flush_final failed for {model_name}: {e}")

    deadline = time.time() + 2.0

    while time.time() < deadline:
        poll_model(
            model=model,
            model_name=model_name,
            aggregator=aggregator,
            latency_tracker=latency_tracker,
            output_latency_tracker=output_latency_tracker,
            simul_state=simul_state,
        )
        time.sleep(0.01)


def run_one_model(
    model_name: str,
    audio: np.ndarray,
    cfg: PreprocessConfig,
    chunk_duration_s: float,
    realtime: bool,
    test_id: str,
    person_id: str,
    gender: str,
    source_audio: str,
) -> dict:
    print(f"\n[SIM] Running model: {model_name}")

    model = build_model(model_name)
    model.start()

    latency_tracker = LatencyTracker(speech_rms_threshold=0.0025)
    output_latency_tracker = OutputLatencyTracker()
    aggregator = SessionAggregator()

    sink = PipelineSink(model, latency_tracker=latency_tracker)

    simul_state = {
        "simul_partial_hyp": "",
        "last_partial": "",
        "last_partial_time": None,
    }

    audio_duration_s = len(audio) / float(cfg.target_sr)
    chunk_size = int(round(cfg.target_sr * chunk_duration_s))

    if chunk_size <= 0:
        raise ValueError("chunk_duration_s is too small")

    aggregator.start()
    latency_tracker.on_stream_start()
    output_latency_tracker.on_stream_start()

    t0 = time.time()

    for start in range(0, len(audio), chunk_size):
        chunk_audio = audio[start:start + chunk_size]

        if chunk_audio.size == 0:
            continue

        # same gain idea as live preprocessing
        chunk_audio = chunk_audio * float(cfg.gain)
        chunk_audio = np.clip(chunk_audio, -1.0, 1.0).astype(np.float32)

        rms = float(np.sqrt(np.mean(chunk_audio ** 2))) if chunk_audio.size > 0 else 0.0

        timestamp_s = t0 + start / float(cfg.target_sr)

        chunk = AudioChunk(
            data=chunk_audio,
            sample_rate=cfg.target_sr,
            rms=rms,
            timestamp_s=timestamp_s,
        )

        sink.accept_chunk(chunk)

        drain_model_events(
            model=model,
            model_name=model_name,
            aggregator=aggregator,
            latency_tracker=latency_tracker,
            output_latency_tracker=output_latency_tracker,
            simul_state=simul_state,
            seconds=0.006,
        )

        if realtime:
            time.sleep(chunk_duration_s)

    flush_and_drain(
        model=model,
        model_name=model_name,
        aggregator=aggregator,
        latency_tracker=latency_tracker,
        output_latency_tracker=output_latency_tracker,
        simul_state=simul_state,
    )

    aggregator.stop()
    latency_tracker.on_stream_stop()

    if hasattr(model, "get_compute_time_s"):
        latency_tracker.model_compute_time_s = model.get_compute_time_s()

    try:
        model.stop()
    except Exception:
        pass

    if model_name == "simulstreaming" and simul_state["simul_partial_hyp"].strip():
        hyp_text = simul_state["simul_partial_hyp"].strip()
    else:
        hyp_text = aggregator.full_hypothesis()

    latency_result = latency_tracker.result()
    output_latency_result = output_latency_tracker.result()

    session_id = aggregator.make_session_id(model_name, test_id)

    meta = {
        "interface": "recording_stream_simulation",
        "gain": cfg.gain,
        "denoise": cfg.enable_denoise,
        "person_id": person_id,
        "gender": gender,
        "chunk_duration_s": chunk_duration_s,
        "chunk_size_samples": chunk_size,
        "source_audio": source_audio,
        "target_sr": cfg.target_sr,
    }

    metrics_store = MetricsStore(
        out_dir=str(OUT_DIR),
        per_language="de",
    )

    metric_values = metrics_store.append(
        session_id=session_id,
        model=model_name,
        test_id=test_id,
        gt_text=GROUND_TRUTH_TEXT,
        hyp_text=hyp_text,
        latency=latency_result,
        output_latency=output_latency_result,
        meta=meta,
    )

    result = {
        "session_id": session_id,
        "model": model_name,
        "test_id": test_id,
        "person_id": person_id,
        "gender": gender,
        "source": "recording_stream_simulation",
        "gain": cfg.gain,
        "denoise": cfg.enable_denoise,
        "chunk_duration_s": chunk_duration_s,
        "chunk_size_samples": chunk_size,
        "target_sr": cfg.target_sr,
        "audio_duration_s": audio_duration_s,
        "source_audio": source_audio,
        "hyp_text": hyp_text,

        "wer": metric_values.wer,
        "cer": metric_values.cer,
        "ser": metric_values.ser,
        "per": metric_values.per,
        "bertscore_precision": metric_values.bertscore_precision,
        "bertscore_recall": metric_values.bertscore_recall,
        "bertscore_f1": metric_values.bertscore_f1,

        **asdict(latency_result),
        **{f"output_{k}": v for k, v in asdict(output_latency_result).items()},
    }

    print(
        f"[SIM] {model_name}: "
        f"WER={result['wer']:.4f}, "
        f"CER={result['cer']:.4f}, "
        f"SER={result['ser']:.4f}, "
        f"PER={result['per'] if result['per'] is not None else 'n/a'}, "
        f"BERT-F1={result['bertscore_f1'] if result['bertscore_f1'] is not None else 'n/a'}"
    )

    return result


def normalize_person_id(x: str) -> str:
    x = str(x).lower().strip()
    x = x.replace("person_", "")
    x = x.replace("person", "")
    x = x.strip("_ ")
    x = x.lstrip("0")
    return x


def find_person_rows(metrics_csv: Path, person_id: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)

    if "person_id" not in df.columns:
        raise ValueError("metrics.csv has no column 'person_id'")

    target = normalize_person_id(person_id)

    df["_person_norm"] = df["person_id"].astype(str).map(normalize_person_id)

    out = df[df["_person_norm"] == target].copy()
    out = out.drop(columns=["_person_norm"])

    if out.empty:
        raise ValueError(f"No rows found for person_id={person_id}")

    return out


def compare_with_live_person(sim_df: pd.DataFrame, live_person_df: pd.DataFrame) -> pd.DataFrame:
    live_cols = [
        "model",
        "wer",
        "cer",
        "ser",
        "per",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "rtf",
        "audio_duration_s",
        "model_compute_time_s",
    ]

    live_cols = [c for c in live_cols if c in live_person_df.columns]

    live_small = live_person_df[live_cols].copy()
    live_small = live_small.rename(
        columns={
            "wer": "live_wer",
            "cer": "live_cer",
            "ser": "live_ser",
            "per": "live_per",
            "bertscore_precision": "live_bertscore_precision",
            "bertscore_recall": "live_bertscore_recall",
            "bertscore_f1": "live_bertscore_f1",
            "rtf": "live_rtf",
            "audio_duration_s": "live_audio_duration_s",
            "model_compute_time_s": "live_model_compute_time_s",
        }
    )

    sim_small = sim_df.copy()
    sim_small = sim_small.rename(
        columns={
            "wer": "sim_wer",
            "cer": "sim_cer",
            "ser": "sim_ser",
            "per": "sim_per",
            "bertscore_precision": "sim_bertscore_precision",
            "bertscore_recall": "sim_bertscore_recall",
            "bertscore_f1": "sim_bertscore_f1",
            "rtf": "sim_rtf",
            "audio_duration_s": "sim_audio_duration_s",
            "model_compute_time_s": "sim_model_compute_time_s",
        }
    )

    merged = pd.merge(sim_small, live_small, on="model", how="left")

    for metric in [
        "wer",
        "cer",
        "ser",
        "per",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "rtf",
    ]:
        sim_col = f"sim_{metric}"
        live_col = f"live_{metric}"

        if sim_col in merged.columns and live_col in merged.columns:
            merged[sim_col] = pd.to_numeric(merged[sim_col], errors="coerce")
            merged[live_col] = pd.to_numeric(merged[live_col], errors="coerce")
            merged[f"delta_{metric}"] = merged[sim_col] - merged[live_col]

    return merged


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        help="Path to recorded reference audio, e.g. sessions/person11_reference.wav",
    )

    parser.add_argument(
        "--person-id",
        default="person_11",
        help="Person ID to compare against in sessions/metrics.csv",
    )

    parser.add_argument(
        "--gender",
        default="not_specified",
    )

    parser.add_argument(
        "--test-id",
        default="recording_sim",
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--denoise",
        action="store_true",
    )

    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.0053125,
        help="Default matches your measured live chunks: 85 samples / 16000 Hz",
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between chunks to simulate real wall-clock streaming",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=TARGET_MODELS,
        help="Example: --models zipformer vosk whispervad qwen3vad",
    )

    args = parser.parse_args()

    cfg = PreprocessConfig(
        gain=args.gain,
        enable_denoise=args.denoise,
    )

    audio = load_audio(args.audio, cfg.target_sr)

    print("[SIM] Recording stream simulation")
    print(f"[SIM] Audio: {args.audio}")
    print(f"[SIM] Target SR: {cfg.target_sr}")
    print(f"[SIM] Chunk duration: {args.chunk_duration:.7f} s")
    print(f"[SIM] Chunk samples: {int(round(cfg.target_sr * args.chunk_duration))}")
    print(f"[SIM] Gain: {cfg.gain}")
    print(f"[SIM] Denoise: {cfg.enable_denoise}")
    print(f"[SIM] Models: {args.models}")

    results = []

    for model_name in args.models:
        result = run_one_model(
            model_name=model_name,
            audio=audio,
            cfg=cfg,
            chunk_duration_s=args.chunk_duration,
            realtime=args.realtime,
            test_id=args.test_id,
            person_id=args.person_id,
            gender=args.gender,
            source_audio=str(args.audio),
        )
        results.append(result)

    sim_df = pd.DataFrame(results)

    sim_out = OUT_DIR / "recording_simulation_results.csv"
    sim_df.to_csv(sim_out, index=False)

    json_out = OUT_DIR / "recording_simulation_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    try:
        live_person_df = find_person_rows(LIVE_METRICS_CSV_PATH, args.person_id)

        comparison_df = compare_with_live_person(
            sim_df=sim_df,
            live_person_df=live_person_df,
        )

        comp_out = OUT_DIR / f"comparison_{args.person_id}.csv"
        comparison_df.to_csv(comp_out, index=False)

        show_cols = [
            "model",
            "sim_wer",
            "live_wer",
            "delta_wer",
            "sim_cer",
            "live_cer",
            "delta_cer",
            "sim_ser",
            "live_ser",
            "delta_ser",
            "sim_per",
            "live_per",
            "delta_per",
            "sim_bertscore_f1",
            "live_bertscore_f1",
            "delta_bertscore_f1",
        ]

        show_cols = [c for c in show_cols if c in comparison_df.columns]

        print("\nComparison with live metrics:")
        print(comparison_df[show_cols].to_string(index=False))

        print(f"\nComparison CSV: {comp_out}")

    except Exception as e:
        print(f"\n[WARN] Could not compare with live metrics: {e}")

    print("\n[DONE]")
    print(f"Simulation results CSV: {sim_out}")
    print(f"Simulation results JSON: {json_out}")
    print(f"Simulation MetricsStore CSV: {OUT_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()