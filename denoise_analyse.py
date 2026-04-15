import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/iat/projectaria_client_sdk_samples")
CSV_PATH = PROJECT_ROOT / "asr_demo" / "sessions" / "metrics.csv"
OUT_DIR = PROJECT_ROOT / "asr_demo" / "analysis_plots_per_model"


def parse_meta(meta_str: str) -> dict:
    if not isinstance(meta_str, str) or not meta_str.strip():
        return {}
    try:
        return json.loads(meta_str)
    except Exception:
        return {}


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    meta = df["meta_json"].apply(parse_meta)
    df["denoise"] = meta.apply(lambda x: x.get("denoise", None))
    df["interface"] = meta.apply(lambda x: x.get("interface", None))
    df["gain"] = meta.apply(lambda x: x.get("gain", None))

    numeric_cols = [
        "wer",
        "cer",
        "ser",
        "per",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "rtf",
        "start_to_first_output_s",
        "start_to_first_partial_output_s",
        "start_to_first_final_output_s",
        "mean_inter_output_s",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_float(df[col])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def keep_latest_per_model_and_denoise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    df = df.groupby(["model", "denoise"], as_index=False).tail(1)
    return df


def get_value(df_model: pd.DataFrame, denoise_value: bool, col: str):
    row = df_model[df_model["denoise"] == denoise_value]
    if row.empty or col not in row.columns:
        return None
    val = row.iloc[0][col]
    return None if pd.isna(val) else float(val)


def plot_model_combined(df_model: pd.DataFrame, model_name: str, out_path: Path):
    metrics = [
        ("wer", "WER"),
        ("cer", "CER"),
        ("per", "PER"),
        ("bertscore_f1", "BERTScore-F1"),
        ("rtf", "RTF"),
        ("start_to_first_output_s", "FirstOutput"),
    ]

    labels = []
    off_vals = []
    on_vals = []

    for col, label in metrics:
        off_v = get_value(df_model, False, col)
        on_v = get_value(df_model, True, col)

        # nur Metriken plotten, für die mindestens ein Wert existiert
        if off_v is None and on_v is None:
            continue

        labels.append(label)
        off_vals.append(off_v if off_v is not None else 0.0)
        on_vals.append(on_v if on_v is not None else 0.0)

    if not labels:
        print(f"[SKIP] No plot data for model {model_name}")
        return

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], off_vals, width=width, label="denoise off")
    ax.bar([i + width / 2 for i in x], on_vals, width=width, label="denoise on")

    ax.set_title(f"{model_name}: Metric Comparison (off vs on)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_model_split(df_model: pd.DataFrame, model_name: str, out_path: Path):
    acc_metrics = [
        ("wer", "WER"),
        ("cer", "CER"),
        ("per", "PER"),
        ("bertscore_f1", "BERTScore-F1"),
    ]
    rt_metrics = [
        ("rtf", "RTF"),
        ("start_to_first_output_s", "FirstOutput"),
        ("mean_inter_output_s", "MeanInterOutput"),
    ]

    def collect(metrics):
        labels = []
        off_vals = []
        on_vals = []
        for col, label in metrics:
            off_v = get_value(df_model, False, col)
            on_v = get_value(df_model, True, col)
            if off_v is None and on_v is None:
                continue
            labels.append(label)
            off_vals.append(off_v if off_v is not None else 0.0)
            on_vals.append(on_v if on_v is not None else 0.0)
        return labels, off_vals, on_vals

    acc_labels, acc_off, acc_on = collect(acc_metrics)
    rt_labels, rt_off, rt_on = collect(rt_metrics)

    if not acc_labels and not rt_labels:
        print(f"[SKIP] No split plot data for model {model_name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    width = 0.35

    # Accuracy subplot
    if acc_labels:
        x = range(len(acc_labels))
        axes[0].bar([i - width / 2 for i in x], acc_off, width=width, label="denoise off")
        axes[0].bar([i + width / 2 for i in x], acc_on, width=width, label="denoise on")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(acc_labels)
        axes[0].set_title(f"{model_name}: Accuracy Metrics")
        axes[0].set_ylabel("value")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].legend()
    else:
        axes[0].set_visible(False)

    # Runtime subplot
    if rt_labels:
        x = range(len(rt_labels))
        axes[1].bar([i - width / 2 for i in x], rt_off, width=width, label="denoise off")
        axes[1].bar([i + width / 2 for i in x], rt_on, width=width, label="denoise on")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(rt_labels)
        axes[1].set_title(f"{model_name}: Runtime / Latency Metrics")
        axes[1].set_ylabel("value")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"metrics.csv not found: {CSV_PATH}")

    df = load_metrics(CSV_PATH)
    df = keep_latest_per_model_and_denoise(df)

    print("\nRuns used for per-model plots:")
    cols = ["timestamp", "model", "denoise", "wer", "cer", "per", "bertscore_f1", "rtf", "start_to_first_output_s"]
    existing_cols = [c for c in cols if c in df.columns]
    print(df[existing_cols].to_string(index=False))

    models = sorted(df["model"].dropna().unique())

    for model_name in models:
        df_model = df[df["model"] == model_name].copy()

        plot_model_combined(
            df_model,
            model_name,
            OUT_DIR / f"{model_name}_combined_metrics.png"
        )

        plot_model_split(
            df_model,
            model_name,
            OUT_DIR / f"{model_name}_split_metrics.png"
        )

    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()