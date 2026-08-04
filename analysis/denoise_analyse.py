from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path("/home/iat/projectaria_client_sdk_samples")
CSV_PATH = PROJECT_ROOT / "asr_demo" / "sessions" / "metrics.csv"
OUT_DIR = PROJECT_ROOT / "asr_demo" / "analysis" / "denoise_analysis_plot_permodel"

TARGET_PERSON = "person_013"

TARGET_MODELS = ["zipformer", "vosk", "whispervad", "qwen3vad"]

MODEL_LABELS = {
    "zipformer": "Kroko Zipformer",
    "vosk": "Vosk",
    "whispervad": "Whisper + VAD",
    "qwen3vad": "Qwen3-ASR + VAD",
}

MODEL_COLORS = {
    "zipformer": "#4C72B0",
    "vosk": "#55A868",
    "whispervad": "#C44E52",
    "qwen3vad": "#8172B3",
}

DENOISE_COLORS = {
    False: "#8C8C8C",
    True: "#4C72B0",
}

ACCURACY_METRICS = [
    ("wer", "WER"),
    ("cer", "CER"),
    ("per", "PER"),
    ("bertscore_f1", "BERT-F1"),
]


# ============================================================
# Helpers
# ============================================================

def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def model_color(model: str) -> str:
    return MODEL_COLORS.get(str(model), "#808080")


def parse_meta(meta_str: str) -> dict:
    if not isinstance(meta_str, str) or not meta_str.strip():
        return {}

    try:
        return json.loads(meta_str)
    except Exception:
        return {}


def to_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return None

    value_str = str(value).strip().lower()

    if value_str in {"true", "1", "yes", "on"}:
        return True

    if value_str in {"false", "0", "no", "off"}:
        return False

    return None


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "meta_json" in df.columns:
        meta = df["meta_json"].apply(parse_meta)

        if "denoise" not in df.columns:
            df["denoise"] = meta.apply(lambda x: x.get("denoise", None))

        if "interface" not in df.columns:
            df["interface"] = meta.apply(lambda x: x.get("interface", None))

        if "gain" not in df.columns:
            df["gain"] = meta.apply(lambda x: x.get("gain", None))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col, _ in ACCURACY_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "denoise" in df.columns:
        df["denoise"] = df["denoise"].apply(to_bool)

    for col in ["model", "person_id", "test_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def filter_person_and_models(df: pd.DataFrame) -> pd.DataFrame:
    if "person_id" not in df.columns:
        raise ValueError("Column 'person_id' not found in metrics.csv")

    if "model" not in df.columns:
        raise ValueError("Column 'model' not found in metrics.csv")

    if "denoise" not in df.columns:
        raise ValueError("Column 'denoise' not found in metrics.csv or meta_json")

    df = df[df["person_id"] == TARGET_PERSON].copy()
    df = df[df["model"].isin(TARGET_MODELS)].copy()
    df = df[df["denoise"].isin([True, False])].copy()

    return df


def keep_latest_per_model_and_denoise(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df.groupby(["model", "denoise"], as_index=False).tail(1)

    df = df.sort_values("timestamp")
    df = df.groupby(["model", "denoise"], as_index=False).tail(1)

    return df


def get_value(df_model: pd.DataFrame, denoise_value: bool, metric: str) -> Optional[float]:
    row = df_model[df_model["denoise"] == denoise_value]

    if row.empty or metric not in row.columns:
        return None

    value = row.iloc[-1][metric]

    if pd.isna(value):
        return None

    return float(value)


# ============================================================
# Plotting
# ============================================================

def plot_single_model_accuracy(df_model: pd.DataFrame, model: str, out_path: Path) -> None:
    labels = []
    off_vals = []
    on_vals = []

    for metric, label in ACCURACY_METRICS:
        off_value = get_value(df_model, False, metric)
        on_value = get_value(df_model, True, metric)

        if off_value is None and on_value is None:
            continue

        labels.append(label)
        off_vals.append(off_value if off_value is not None else np.nan)
        on_vals.append(on_value if on_value is not None else np.nan)

    if not labels:
        print(f"[SKIP] No accuracy data for {model}")
        return

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        off_vals,
        width=width,
        label="ohne Rauschunterdrückung",
        color="#8C8C8C",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.bar(
        x + width / 2,
        on_vals,
        width=width,
        label="mit Rauschunterdrückung",
        color=model_color(model),
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.xticks(x, labels)
    plt.ylabel("Metrikwert")
    plt.xlabel("Genauigkeitsmetrik")
    plt.title(f"{model_label(model)}: Einfluss der Rauschunterdrückung")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

    print(f"[OK] Saved {out_path}")


def plot_all_models_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    for ax, model in zip(axes, TARGET_MODELS):
        df_model = df[df["model"] == model].copy()

        labels = []
        off_vals = []
        on_vals = []

        for metric, label in ACCURACY_METRICS:
            off_value = get_value(df_model, False, metric)
            on_value = get_value(df_model, True, metric)

            if off_value is None and on_value is None:
                continue

            labels.append(label)
            off_vals.append(off_value if off_value is not None else np.nan)
            on_vals.append(on_value if on_value is not None else np.nan)

        if not labels:
            ax.set_visible(False)
            continue

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(
            x - width / 2,
            off_vals,
            width=width,
            label="ohne Rauschunterdrückung",
            color="#8C8C8C",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.bar(
            x + width / 2,
            on_vals,
            width=width,
            label="mit Rauschunterdrückung",
            color=model_color(model),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_title(model_label(model))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Metrikwert")
        ax.grid(True, axis="y", alpha=0.25)

    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=True,
    )

    fig.suptitle(
        f"",
        fontsize=15,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=250)
    plt.close()

    print(f"[OK] Saved {out_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"metrics.csv not found: {CSV_PATH}")

    df = load_metrics(CSV_PATH)
    df = filter_person_and_models(df)
    df = keep_latest_per_model_and_denoise(df)

    if df.empty:
        raise ValueError(f"No data found for {TARGET_PERSON} with target models and denoise values.")

    print("\nRuns used:")
    cols = [
        "timestamp",
        "person_id",
        "model",
        "denoise",
        "wer",
        "cer",
        "per",
        "bertscore_f1",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    print(df[existing_cols].to_string(index=False))

    for model in TARGET_MODELS:
        df_model = df[df["model"] == model].copy()

        if df_model.empty:
            print(f"[SKIP] No data for {model}")
            continue

        plot_single_model_accuracy(
            df_model,
            model,
            OUT_DIR / f"{model}_person013_denoise_accuracy.png",
        )

    plot_all_models_accuracy(
        df,
        OUT_DIR / "all_models_person013_denoise_accuracy.png",
    )

    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()