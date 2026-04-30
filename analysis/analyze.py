from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

CSV_PATH = Path("/home/iat/projectaria_client_sdk_samples/asr_demo/sessions/metrics.csv")
OUT_DIR = Path("/home/iat/projectaria_client_sdk_samples/asr_demo/analysis/results")

TARGET_MODELS = ["zipformer", "vosk", "whispervad", "qwen3vad"]

NUMERIC_COLUMNS = [
    "gain",
    "wer",
    "cer",
    "ser",
    "per",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "stream_duration_s",
    "audio_duration_s",
    "model_compute_time_s",
    "rtf",
    "stream_start_to_first_partial_s",
    "stream_start_to_first_final_s",
    "speech_start_to_first_partial_s",
    "speech_start_to_first_final_s",
    "stop_to_last_final_s",
    "start_to_first_output_s",
    "start_to_first_partial_output_s",
    "start_to_first_final_output_s",
    "num_outputs",
    "mean_inter_output_s",
    "min_inter_output_s",
    "max_inter_output_s",
    "last_output_time_from_start_s",
]

LOWER_IS_BETTER = {
    "wer": True,
    "cer": True,
    "ser": True,
    "per": True,
    "rtf": True,
    "start_to_first_output_s": True,
    "start_to_first_partial_output_s": True,
    "start_to_first_final_output_s": True,
    "mean_inter_output_s": True,
    "min_inter_output_s": True,
    "max_inter_output_s": True,
    "speech_start_to_first_partial_s": True,
    "speech_start_to_first_final_s": True,
}

HIGHER_IS_BETTER = {
    "bertscore_precision": True,
    "bertscore_recall": True,
    "bertscore_f1": True,
}

OUT_SUBDIRS = {
    "tables": OUT_DIR / "tables",
    "reports": OUT_DIR / "reports",
    "accuracy": OUT_DIR / "accuracy",
    "streaming": OUT_DIR / "streaming",
    "tradeoffs": OUT_DIR / "tradeoffs",
    "fairness": OUT_DIR / "fairness",
    "robustness": OUT_DIR / "robustness",
    "speaking_speed": OUT_DIR / "speaking_speed",
    "distributions": OUT_DIR / "distributions",
    "correlations": OUT_DIR / "correlations",
}


# ============================================================
# Helpers
# ============================================================

def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder in OUT_SUBDIRS.values():
        folder.mkdir(parents=True, exist_ok=True)


def out_path(category: str, filename: str) -> Path:
    return OUT_SUBDIRS[category] / filename


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["model", "test_id", "person_id", "gender", "interface"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    if "model" in df.columns:
        df = df[df["model"].isin(TARGET_MODELS)].copy()

    if "gt_text" in df.columns and "audio_duration_s" in df.columns:
        df["gt_word_count"] = df["gt_text"].fillna("").apply(lambda x: len(str(x).split()))
        df["gt_char_count"] = df["gt_text"].fillna("").apply(lambda x: len(str(x)))
        df["words_per_second"] = np.where(
            (df["audio_duration_s"].notna()) & (df["audio_duration_s"] > 0),
            df["gt_word_count"] / df["audio_duration_s"],
            np.nan,
        )
        df["chars_per_second"] = np.where(
            (df["audio_duration_s"].notna()) & (df["audio_duration_s"] > 0),
            df["gt_char_count"] / df["audio_duration_s"],
            np.nan,
        )

    return df


def save_table(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(out_path("tables", filename), index=False)


def save_text(text: str, filename: str) -> None:
    out_path("reports", filename).write_text(text, encoding="utf-8")


def grouped_summary(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
    round_digits: int = 4,
) -> pd.DataFrame:
    available_metrics = [m for m in metric_cols if m in df.columns]
    if not available_metrics:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, dropna=False)[available_metrics].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    grouped = grouped.reset_index()

    grouped.columns = [
        "_".join([str(c) for c in col if c != ""]).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]

    numeric_cols = grouped.select_dtypes(include=[np.number]).columns
    grouped[numeric_cols] = grouped[numeric_cols].round(round_digits)

    return grouped


# ============================================================
# Summary Tables
# ============================================================

def simple_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
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
        "speech_start_to_first_partial_s",
        "speech_start_to_first_final_s",
        "mean_inter_output_s",
        "min_inter_output_s",
        "max_inter_output_s",
        "num_outputs",
        "audio_duration_s",
        "words_per_second",
    ]
    return grouped_summary(df, ["model"], metric_cols)


def model_gender_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "gender" not in df.columns:
        return pd.DataFrame()

    metric_cols = [
        "wer",
        "cer",
        "ser",
        "per",
        "bertscore_f1",
        "rtf",
        "start_to_first_output_s",
        "mean_inter_output_s",
        "audio_duration_s",
        "words_per_second",
    ]
    return grouped_summary(df, ["model", "gender"], metric_cols)


def model_person_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "person_id" not in df.columns:
        return pd.DataFrame()

    metric_cols = [
        "wer",
        "cer",
        "ser",
        "per",
        "bertscore_f1",
        "rtf",
        "start_to_first_output_s",
        "mean_inter_output_s",
        "audio_duration_s",
        "words_per_second",
    ]
    return grouped_summary(df, ["person_id", "model"], metric_cols)


# ============================================================
# Ranking
# ============================================================

def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    quality_metrics = [
        "wer",
        "cer",
        "per",
        "bertscore_f1",
    ]

    streaming_metrics = [
        "rtf",
        "start_to_first_output_s",
        "mean_inter_output_s",
    ]

    all_metrics = quality_metrics + streaming_metrics
    all_metrics = [m for m in all_metrics if m in df.columns]

    if not all_metrics:
        return pd.DataFrame()

    model_means = df.groupby("model")[all_metrics].mean(numeric_only=True)
    scored = pd.DataFrame(index=model_means.index)

    for metric in all_metrics:
        series = model_means[metric].copy()

        if series.isna().all():
            continue

        min_val = series.min()
        max_val = series.max()

        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            norm = pd.Series(1.0, index=series.index)
        else:
            norm = (series - min_val) / (max_val - min_val)

        if metric in LOWER_IS_BETTER:
            score = 1.0 - norm
        elif metric in HIGHER_IS_BETTER:
            score = norm
        else:
            score = norm

        scored[f"{metric}_score"] = score

    quality_score_cols = [
        f"{m}_score" for m in quality_metrics if f"{m}_score" in scored.columns
    ]

    streaming_score_cols = [
        f"{m}_score" for m in streaming_metrics if f"{m}_score" in scored.columns
    ]

    if quality_score_cols:
        scored["recognition_quality_score"] = scored[quality_score_cols].mean(axis=1)

    if streaming_score_cols:
        scored["streaming_score"] = scored[streaming_score_cols].mean(axis=1)

    scored = scored.reset_index()

    if "recognition_quality_score" in scored.columns:
        scored["recognition_quality_rank"] = (
            scored["recognition_quality_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    if "streaming_score" in scored.columns:
        scored["streaming_rank"] = (
            scored["streaming_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    scored = scored.sort_values(
        by=["recognition_quality_rank", "streaming_rank"],
        ascending=True,
    )

    return scored.round(4)


# ============================================================
# Plot Functions
# ============================================================

def plot_boxplot(df: pd.DataFrame, metric: str, ylabel: str, filename: Path) -> None:
    if metric not in df.columns:
        return

    labels = [model for model in TARGET_MODELS if model in df["model"].unique()]
    data = [df.loc[df["model"] == model, metric].dropna().values for model in labels]

    if not data:
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.title(f"{ylabel} by Model")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_bar_mean_std(df: pd.DataFrame, metric: str, ylabel: str, filename: Path) -> None:
    if metric not in df.columns:
        return

    stats = df.groupby("model")[metric].agg(["mean", "std"]).reindex(TARGET_MODELS)
    stats = stats.dropna(how="all")

    if stats.empty:
        return

    x = np.arange(len(stats))
    means = stats["mean"].values
    stds = stats["std"].fillna(0).values

    plt.figure(figsize=(10, 6))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, stats.index)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.title(f"{ylabel} by Model")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_tradeoff(df: pd.DataFrame, x_metric: str, y_metric: str, filename: Path) -> None:
    if x_metric not in df.columns or y_metric not in df.columns:
        return

    stats = df.groupby("model")[[x_metric, y_metric]].mean(numeric_only=True).reindex(TARGET_MODELS)
    stats = stats.dropna()

    if stats.empty:
        return

    plt.figure(figsize=(9, 7))

    for model in stats.index:
        x = stats.loc[model, x_metric]
        y = stats.loc[model, y_metric]
        plt.scatter(x, y, s=100)
        plt.annotate(model, (x, y), xytext=(5, 5), textcoords="offset points")

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f"Trade-off: {y_metric} vs {x_metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_pareto(df: pd.DataFrame, x_metric: str, y_metric: str, filename: Path) -> None:
    if x_metric not in df.columns or y_metric not in df.columns:
        return

    stats = df.groupby("model")[[x_metric, y_metric]].mean(numeric_only=True).reindex(TARGET_MODELS)
    stats = stats.dropna()

    if stats.empty:
        return

    plt.figure(figsize=(9, 7))

    for model in stats.index:
        x = stats.loc[model, x_metric]
        y = stats.loc[model, y_metric]
        plt.scatter(x, y, s=100)
        plt.annotate(model, (x, y), xytext=(5, 5), textcoords="offset points")

    sorted_points = stats.sort_values(x_metric)
    pareto_x = []
    pareto_y = []
    best_y = float("inf")

    for _, row in sorted_points.iterrows():
        if row[y_metric] < best_y:
            pareto_x.append(row[x_metric])
            pareto_y.append(row[y_metric])
            best_y = row[y_metric]

    plt.plot(pareto_x, pareto_y, linestyle="--", linewidth=1.5)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f"Pareto Front: {y_metric} vs {x_metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, filename: Path) -> None:
    cols = [
        "wer",
        "cer",
        "ser",
        "per",
        "bertscore_f1",
        "rtf",
        "start_to_first_output_s",
        "start_to_first_partial_output_s",
        "start_to_first_final_output_s",
        "mean_inter_output_s",
        "num_outputs",
        "audio_duration_s",
        "words_per_second",
    ]

    cols = [c for c in cols if c in df.columns]

    if len(cols) < 2:
        return

    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(11, 9))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_stability(df: pd.DataFrame, metric: str, filename: Path) -> None:
    if metric not in df.columns:
        return

    stats = df.groupby("model")[metric].agg(["mean", "std"]).reindex(TARGET_MODELS)
    stats = stats.dropna()

    if stats.empty:
        return

    plt.figure(figsize=(8, 6))

    for model in stats.index:
        x = stats.loc[model, "mean"]
        y = stats.loc[model, "std"]
        plt.scatter(x, y, s=100)
        plt.annotate(model, (x, y), xytext=(5, 5), textcoords="offset points")

    plt.xlabel(f"{metric} mean")
    plt.ylabel(f"{metric} standard deviation")
    plt.title(f"Stability vs Performance: {metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_grouped_bar_by_gender(df: pd.DataFrame, metric: str, ylabel: str, filename: Path) -> None:
    if "gender" not in df.columns or metric not in df.columns:
        return

    genders = sorted([g for g in df["gender"].dropna().unique() if g and g != "nan"])

    if not genders:
        return

    grouped = df.groupby(["model", "gender"])[metric].mean().unstack("gender").reindex(TARGET_MODELS)

    if grouped.empty:
        return

    x = np.arange(len(grouped.index))
    width = 0.8 / max(len(genders), 1)

    plt.figure(figsize=(11, 6))

    for i, gender in enumerate(genders):
        vals = grouped[gender].values if gender in grouped.columns else np.full(len(grouped.index), np.nan)
        plt.bar(
            x + i * width - ((len(genders) - 1) * width / 2),
            vals,
            width=width,
            label=gender,
        )

    plt.xticks(x, grouped.index)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.title(f"{ylabel} by Model and Gender")
    plt.legend(title="Gender")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_gender_gap(df: pd.DataFrame, metric: str, filename: Path) -> None:
    if "gender" not in df.columns or metric not in df.columns:
        return

    grouped = df.groupby(["model", "gender"])[metric].mean().unstack("gender").reindex(TARGET_MODELS)
    grouped = grouped.dropna(axis=1, how="all")

    if grouped.empty or grouped.shape[1] < 2:
        return

    gap = grouped.max(axis=1) - grouped.min(axis=1)
    gap = gap.dropna()

    if gap.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(gap.index, gap.values)
    plt.ylabel(f"{metric} gap")
    plt.xlabel("Model")
    plt.title(f"Gender Gap: {metric}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_person_lines(df: pd.DataFrame, metric: str, ylabel: str, filename: Path) -> None:
    if "person_id" not in df.columns or metric not in df.columns:
        return

    pivot = df.groupby(["person_id", "model"])[metric].mean().unstack("model")

    if pivot.empty:
        return

    pivot = pivot.reindex(columns=[m for m in TARGET_MODELS if m in pivot.columns])
    pivot = pivot.sort_index()

    plt.figure(figsize=(12, 6))

    for model in pivot.columns:
        plt.plot(pivot.index.astype(str), pivot[model], marker="o", label=model)

    plt.ylabel(ylabel)
    plt.xlabel("Person ID")
    plt.title(f"{ylabel} by Person")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_rtf_with_boundary(df: pd.DataFrame, filename: Path) -> None:
    if "rtf" not in df.columns:
        return

    stats = df.groupby("model")["rtf"].agg(["mean", "std"]).reindex(TARGET_MODELS)
    stats = stats.dropna(how="all")

    if stats.empty:
        return

    x = np.arange(len(stats))
    means = stats["mean"].values
    stds = stats["std"].fillna(0).values

    plt.figure(figsize=(10, 6))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(x, stats.index)
    plt.ylabel("RTF")
    plt.xlabel("Model")
    plt.title("Real-Time Factor by Model")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_speaking_speed_effect(df: pd.DataFrame, metric: str, filename: Path) -> None:
    if "words_per_second" not in df.columns or metric not in df.columns:
        return

    clean = df[["model", "words_per_second", metric]].dropna()

    if clean.empty:
        return

    plt.figure(figsize=(10, 7))

    for model in TARGET_MODELS:
        sub = clean[clean["model"] == model]

        if sub.empty:
            continue

        plt.scatter(sub["words_per_second"], sub[metric], s=60, label=model)

        if len(sub) >= 2:
            x = sub["words_per_second"].values
            y = sub[metric].values
            coeffs = np.polyfit(x, y, 1)
            xp = np.linspace(x.min(), x.max(), 100)
            yp = coeffs[0] * xp + coeffs[1]
            plt.plot(xp, yp)

    plt.xlabel("Speaking Speed Proxy (words per second)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Speaking Speed")
    plt.legend(title="Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_histogram(df: pd.DataFrame, metric: str, filename: Path, bins: int = 20) -> None:
    if metric not in df.columns:
        return

    plt.figure(figsize=(10, 6))

    for model in TARGET_MODELS:
        sub = df.loc[df["model"] == model, metric].dropna()

        if sub.empty:
            continue

        plt.hist(sub, bins=bins, alpha=0.5, label=model)

    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"Distribution of {metric}")
    plt.legend(title="Model")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_cdf(df: pd.DataFrame, metric: str, filename: Path) -> None:
    if metric not in df.columns:
        return

    plt.figure(figsize=(9, 6))

    plotted = False

    for model in TARGET_MODELS:
        values = df.loc[df["model"] == model, metric].dropna().sort_values()

        if values.empty:
            continue

        y = np.arange(1, len(values) + 1) / len(values)
        plt.plot(values, y, label=model)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel(metric)
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of {metric}")
    plt.legend(title="Model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ============================================================
# Markdown Report
# ============================================================

def build_markdown_report(
    df: pd.DataFrame,
    summary_model: pd.DataFrame,
    ranking: pd.DataFrame,
) -> str:
    lines = []

    lines.append("# ASR Evaluation Summary")
    lines.append("")
    lines.append(f"- Number of rows: {len(df)}")
    lines.append(f"- Models: {', '.join(sorted(df['model'].dropna().unique()))}")

    if "person_id" in df.columns:
        lines.append(f"- Number of persons: {df['person_id'].nunique()}")

    if "gender" in df.columns:
        lines.append(f"- Genders: {', '.join(sorted(df['gender'].dropna().astype(str).unique()))}")

    lines.append("")

    if not summary_model.empty:
        lines.append("## Model Summary")
        lines.append("")
        lines.append(summary_model.to_markdown(index=False))
        lines.append("")

    if not ranking.empty and "recognition_quality_score" in ranking.columns:
        lines.append("## Recognition Quality Ranking")
        lines.append("")
        lines.append(
            ranking[
                ["model", "recognition_quality_score", "recognition_quality_rank"]
            ]
            .sort_values("recognition_quality_rank")
            .to_markdown(index=False)
        )
        lines.append("")

    if not ranking.empty and "streaming_score" in ranking.columns:
        lines.append("## Streaming / Real-Time Ranking")
        lines.append("")
        lines.append(
            ranking[
                ["model", "streaming_score", "streaming_rank"]
            ]
            .sort_values("streaming_rank")
            .to_markdown(index=False)
        )
        lines.append("")

    if not ranking.empty:
        lines.append("## Key Findings")
        lines.append("")

        if "recognition_quality_rank" in ranking.columns:
            best_quality = ranking.sort_values("recognition_quality_rank").iloc[0]["model"]
            lines.append(f"- Best recognition quality: **{best_quality}**")

        if "streaming_rank" in ranking.columns:
            best_streaming = ranking.sort_values("streaming_rank").iloc[0]["model"]
            lines.append(f"- Best streaming performance: **{best_streaming}**")

        lines.append("")

    lines.append("## Suggested Interpretation")
    lines.append("")
    lines.append("### Recognition Quality")
    lines.append("- Lower WER, CER, SER and PER indicate fewer recognition errors.")
    lines.append("- Higher BERTScore-F1 indicates better semantic similarity.")
    lines.append("- The recognition quality ranking combines WER, CER, PER and BERTScore-F1.")
    lines.append("")
    lines.append("### Streaming / Real-Time Performance")
    lines.append("- Lower RTF indicates faster-than-real-time processing.")
    lines.append("- Lower start-to-first-output latency indicates faster initial responsiveness.")
    lines.append("- Lower mean inter-output latency indicates smoother streaming output.")
    lines.append("- The streaming ranking combines RTF, start-to-first-output latency and mean inter-output latency.")
    lines.append("")
    lines.append("### Stability and Robustness")
    lines.append("- Boxplots show the distribution and median of the metrics.")
    lines.append("- Stability plots compare mean error with standard deviation.")
    lines.append("- CDF plots show how many test cases remain below a certain error or latency threshold.")
    lines.append("")
    lines.append("### Trade-off")
    lines.append("- Pareto plots help identify models that offer strong accuracy-latency trade-offs.")
    lines.append("- A model can be strong in recognition quality while being weaker in streaming performance.")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_out_dir()

    df = load_data(CSV_PATH)

    if df.empty:
        raise ValueError("No data found after filtering. Check CSV path or model names.")

    # --------------------------------------------------------
    # Save cleaned data
    # --------------------------------------------------------
    save_table(df, "cleaned_metrics.csv")

    # --------------------------------------------------------
    # Summary tables
    # --------------------------------------------------------
    summary_model = simple_model_summary(df)
    summary_gender = model_gender_summary(df)
    summary_person = model_person_summary(df)
    ranking = rank_models(df)

    if not summary_model.empty:
        save_table(summary_model, "summary_by_model.csv")

    if not summary_gender.empty:
        save_table(summary_gender, "summary_by_model_gender.csv")

    if not summary_person.empty:
        save_table(summary_person, "summary_by_person_model.csv")

    if not ranking.empty:
        save_table(ranking, "model_ranking.csv")

    pivot_metrics = [
        "wer",
        "cer",
        "ser",
        "per",
        "bertscore_f1",
        "rtf",
        "start_to_first_output_s",
        "mean_inter_output_s",
    ]
    pivot_metrics = [m for m in pivot_metrics if m in df.columns]

    if pivot_metrics:
        pivot = (
            df.groupby("model")[pivot_metrics]
            .mean(numeric_only=True)
            .reindex(TARGET_MODELS)
            .round(4)
            .reset_index()
        )
        save_table(pivot, "report_main_table.csv")

    # --------------------------------------------------------
    # Accuracy plots
    # --------------------------------------------------------
    plot_boxplot(df, "wer", "WER", out_path("accuracy", "wer_boxplot.png"))
    plot_boxplot(df, "cer", "CER", out_path("accuracy", "cer_boxplot.png"))
    plot_boxplot(df, "per", "PER", out_path("accuracy", "per_boxplot.png"))
    plot_boxplot(df, "ser", "SER", out_path("accuracy", "ser_boxplot.png"))

    plot_bar_mean_std(df, "wer", "WER", out_path("accuracy", "wer_mean_std.png"))
    plot_bar_mean_std(df, "cer", "CER", out_path("accuracy", "cer_mean_std.png"))
    plot_bar_mean_std(df, "per", "PER", out_path("accuracy", "per_mean_std.png"))
    plot_bar_mean_std(df, "ser", "SER", out_path("accuracy", "ser_mean_std.png"))
    plot_bar_mean_std(df, "bertscore_f1", "BERTScore F1", out_path("accuracy", "bertscore_f1_mean_std.png"))
    plot_bar_mean_std(df, "bertscore_precision", "BERTScore Precision", out_path("accuracy", "bertscore_precision_mean_std.png"))
    plot_bar_mean_std(df, "bertscore_recall", "BERTScore Recall", out_path("accuracy", "bertscore_recall_mean_std.png"))

    # --------------------------------------------------------
    # Streaming plots
    # --------------------------------------------------------
    plot_rtf_with_boundary(df, out_path("streaming", "rtf_mean_std_with_realtime_boundary.png"))
    plot_bar_mean_std(df, "start_to_first_output_s", "Start to First Output (s)", out_path("streaming", "start_to_first_output_mean_std.png"))
    plot_bar_mean_std(df, "start_to_first_partial_output_s", "Start to First Partial Output (s)", out_path("streaming", "start_to_first_partial_mean_std.png"))
    plot_bar_mean_std(df, "start_to_first_final_output_s", "Start to First Final Output (s)", out_path("streaming", "start_to_first_final_mean_std.png"))
    plot_bar_mean_std(df, "speech_start_to_first_partial_s", "Speech Start to First Partial (s)", out_path("streaming", "speech_start_to_first_partial_mean_std.png"))
    plot_bar_mean_std(df, "speech_start_to_first_final_s", "Speech Start to First Final (s)", out_path("streaming", "speech_start_to_first_final_mean_std.png"))
    plot_boxplot(df, "mean_inter_output_s", "Mean Inter-Output Latency (s)", out_path("streaming", "mean_inter_output_boxplot.png"))
    plot_bar_mean_std(df, "num_outputs", "Number of Outputs", out_path("streaming", "num_outputs_mean_std.png"))

    # --------------------------------------------------------
    # Trade-off and Pareto plots
    # --------------------------------------------------------
    plot_tradeoff(df, "mean_inter_output_s", "wer", out_path("tradeoffs", "tradeoff_wer_vs_mean_inter_output.png"))
    plot_tradeoff(df, "start_to_first_output_s", "wer", out_path("tradeoffs", "tradeoff_wer_vs_start_to_first_output.png"))
    plot_tradeoff(df, "rtf", "wer", out_path("tradeoffs", "tradeoff_wer_vs_rtf.png"))
    plot_tradeoff(df, "rtf", "bertscore_f1", out_path("tradeoffs", "tradeoff_bertscore_f1_vs_rtf.png"))

    plot_pareto(df, "rtf", "wer", out_path("tradeoffs", "pareto_wer_vs_rtf.png"))
    plot_pareto(df, "start_to_first_output_s", "wer", out_path("tradeoffs", "pareto_wer_vs_start_to_first_output.png"))
    plot_pareto(df, "mean_inter_output_s", "wer", out_path("tradeoffs", "pareto_wer_vs_mean_inter_output.png"))

    # --------------------------------------------------------
    # Fairness / gender plots
    # --------------------------------------------------------
    plot_grouped_bar_by_gender(df, "wer", "WER", out_path("fairness", "gender_wer_mean.png"))
    plot_grouped_bar_by_gender(df, "cer", "CER", out_path("fairness", "gender_cer_mean.png"))
    plot_grouped_bar_by_gender(df, "per", "PER", out_path("fairness", "gender_per_mean.png"))
    plot_grouped_bar_by_gender(df, "bertscore_f1", "BERTScore F1", out_path("fairness", "gender_bertscore_f1_mean.png"))
    plot_grouped_bar_by_gender(df, "mean_inter_output_s", "Mean Inter-Output Latency (s)", out_path("fairness", "gender_mean_inter_output_mean.png"))

    plot_gender_gap(df, "wer", out_path("fairness", "gender_gap_wer.png"))
    plot_gender_gap(df, "cer", out_path("fairness", "gender_gap_cer.png"))
    plot_gender_gap(df, "per", out_path("fairness", "gender_gap_per.png"))
    plot_gender_gap(df, "bertscore_f1", out_path("fairness", "gender_gap_bertscore_f1.png"))
    plot_gender_gap(df, "mean_inter_output_s", out_path("fairness", "gender_gap_mean_inter_output.png"))

    # --------------------------------------------------------
    # Robustness / person-wise plots
    # --------------------------------------------------------
    plot_person_lines(df, "wer", "WER", out_path("robustness", "person_wer_lines.png"))
    plot_person_lines(df, "cer", "CER", out_path("robustness", "person_cer_lines.png"))
    plot_person_lines(df, "per", "PER", out_path("robustness", "person_per_lines.png"))
    plot_person_lines(df, "bertscore_f1", "BERTScore F1", out_path("robustness", "person_bertscore_f1_lines.png"))
    plot_person_lines(df, "mean_inter_output_s", "Mean Inter-Output Latency (s)", out_path("robustness", "person_mean_inter_output_lines.png"))

    plot_stability(df, "wer", out_path("robustness", "stability_wer_mean_vs_std.png"))
    plot_stability(df, "cer", out_path("robustness", "stability_cer_mean_vs_std.png"))
    plot_stability(df, "per", out_path("robustness", "stability_per_mean_vs_std.png"))
    plot_stability(df, "mean_inter_output_s", out_path("robustness", "stability_mean_inter_output_mean_vs_std.png"))
    plot_stability(df, "rtf", out_path("robustness", "stability_rtf_mean_vs_std.png"))

    # --------------------------------------------------------
    # Speaking speed plots
    # --------------------------------------------------------
    plot_bar_mean_std(df, "audio_duration_s", "Audio Duration (s)", out_path("speaking_speed", "audio_duration_mean_std.png"))
    plot_bar_mean_std(df, "words_per_second", "Words per Second", out_path("speaking_speed", "words_per_second_mean_std.png"))

    plot_speaking_speed_effect(df, "wer", out_path("speaking_speed", "speaking_speed_vs_wer.png"))
    plot_speaking_speed_effect(df, "cer", out_path("speaking_speed", "speaking_speed_vs_cer.png"))
    plot_speaking_speed_effect(df, "per", out_path("speaking_speed", "speaking_speed_vs_per.png"))
    plot_speaking_speed_effect(df, "mean_inter_output_s", out_path("speaking_speed", "speaking_speed_vs_mean_inter_output.png"))
    plot_speaking_speed_effect(df, "rtf", out_path("speaking_speed", "speaking_speed_vs_rtf.png"))

    # --------------------------------------------------------
    # Distribution plots
    # --------------------------------------------------------
    plot_histogram(df, "mean_inter_output_s", out_path("distributions", "hist_mean_inter_output.png"))
    plot_histogram(df, "rtf", out_path("distributions", "hist_rtf.png"))
    plot_histogram(df, "wer", out_path("distributions", "hist_wer.png"))
    plot_histogram(df, "cer", out_path("distributions", "hist_cer.png"))

    plot_cdf(df, "wer", out_path("distributions", "cdf_wer.png"))
    plot_cdf(df, "cer", out_path("distributions", "cdf_cer.png"))
    plot_cdf(df, "per", out_path("distributions", "cdf_per.png"))
    plot_cdf(df, "rtf", out_path("distributions", "cdf_rtf.png"))
    plot_cdf(df, "mean_inter_output_s", out_path("distributions", "cdf_mean_inter_output.png"))
    plot_cdf(df, "start_to_first_output_s", out_path("distributions", "cdf_start_to_first_output.png"))

    # --------------------------------------------------------
    # Correlation plots
    # --------------------------------------------------------
    plot_correlation_heatmap(df, out_path("correlations", "metric_correlation_heatmap.png"))

    # --------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------
    report_md = build_markdown_report(df, summary_model, ranking)
    save_text(report_md, "report_summary.md")

    print(f"Done. Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()