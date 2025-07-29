#!/usr/bin/env python3
"""
Flexible, fault-tolerant visualiser for mineral-rights classifier results.

Usage
-----
python flexible_visualiser.py <summary_json> [--all] [--confusion] [--metrics]
                               [--confidence] [--category] [--processing]
                               [--save-dir DIR]

If no flags are passed the script shows a quick metrics table + confusion matrix.
"""

import json, argparse, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

sns.set_style("whitegrid")
plt.rcParams["figure.autolayout"] = True
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ─────────────────────────── helpers ────────────────────────────
def safe_float(x, default=np.nan):
    """Convert to float; return `default` on failure (e.g. 'N/A')."""
    try:
        return float(x)
    except Exception:
        return default


def load_summary(json_path: Path) -> tuple[pd.DataFrame, dict]:
    """Return (results_df, summary_dict)."""
    data = json.loads(json_path.read_text())
    # Build DataFrame from detailed_results, falling back if missing
    df = pd.DataFrame(data.get("detailed_results", []))
    # Ensure critical columns exist
    for col in ("true_label", "predicted_label", "confidence", "category"):
        if col not in df.columns:
            df[col] = np.nan
    df["confidence"] = df["confidence"].apply(safe_float)
    df["correct"]    = df["true_label"] == df["predicted_label"]
    return df, data


def print_metrics_table(data: dict):
    perf = data["performance_metrics"]["performance_metrics"]
    cm   = data["performance_metrics"]["confusion_matrix"]
    header = (
        f"\n{30*'='} METRICS {30*'='}\n"
        f"Accuracy    : {perf['accuracy']:.3f}\n"
        f"Precision   : {perf['precision']:.3f}\n"
        f"Recall      : {perf['recall']:.3f}\n"
        f"F1-score    : {perf['f1_score']:.3f}\n"
        f"Specificity : {perf['specificity']:.3f}\n\n"
        "Confusion Matrix:\n"
        f"TP: {cm['true_positives']:3}  FP: {cm['false_positives']:3}\n"
        f"FN: {cm['false_negatives']:3}  TN: {cm['true_negatives']:3}\n"
        f"{65*'='}\n"
    )
    print(header)


# ─────────────────────────── plotting ────────────────────────────
def plot_confusion_matrix(data: dict, ax=None):
    cm = data["performance_metrics"]["confusion_matrix"]
    mat = np.array([[cm["true_negatives"], cm["false_positives"]],
                    [cm["false_negatives"], cm["true_positives"]]])
    ax = ax or plt.gca()
    sns.heatmap(mat,
                annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: No O&G", "Pred: O&G"],
                yticklabels=["Actual: No O&G", "Actual: O&G"],
                ax=ax)
    ax.set_title("Confusion Matrix")


def plot_metric_bars(data: dict, ax=None):
    perf = data["performance_metrics"]["performance_metrics"]

    # map pretty labels → JSON keys
    metrics = [
        ("Accuracy",    "accuracy"),
        ("Precision",   "precision"),
        ("Recall",      "recall"),
        ("F1-Score",    "f1_score"),     # ← fixed key
        ("Specificity", "specificity")
    ]

    labels  = [m[0] for m in metrics]
    values  = [perf.get(m[1], np.nan) for m in metrics]

    ax = ax or plt.gca()
    sns.barplot(x=labels, y=values, palette="husl", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Main Performance Metrics")

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}" if not np.isnan(v) else "N/A",
                ha="center", fontweight="bold")


def plot_confidence_hist(df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    sns.histplot(df[df.correct]["confidence"], color="green",
                 label="Correct", kde=False, bins=20, ax=ax, alpha=.7)
    sns.histplot(df[~df.correct]["confidence"], color="red",
                 label="Incorrect", kde=False, bins=20, ax=ax, alpha=.7)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()


def plot_accuracy_by_category(df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    tmp = df.groupby("category").correct.mean().reset_index()
    sns.barplot(data=tmp, x="category", y="correct", palette="viridis", ax=ax)
    ax.set_ylim(0,1); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Category")
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2, p.get_height()+0.02,
                f"{p.get_height():.2f}", ha="center")


def plot_processing_stats(data: dict, ax=None):
    pstats = data["performance_metrics"]["processing_stats"]
    keys = ["avg_time_per_doc", "avg_samples_used", "avg_confidence"]
    names = ["Avg Time", "Avg Samples", "Avg Conf"]
    values = [safe_float(pstats.get(k, np.nan)) for k in keys]
    ax = ax or plt.gca()
    sns.barplot(x=names, y=values, palette="Set2", ax=ax)
    ax.set_title("Processing Stats")
    for i, v in enumerate(values):
        txt = f"{v:.2f}" if not math.isnan(v) else "N/A"
        ax.text(i, (v if not math.isnan(v) else 0)+0.02, txt,
                ha="center", fontweight="bold")

# ------------------------------------------------------------------
# NEW plotting helpers
def plot_roc(df: pd.DataFrame, ax=None):
    """ROC curve using confidence as score."""
    if df.confidence.isna().all():
        return
    fpr, tpr, _ = roc_curve(df.true_label, df.confidence)
    roc_auc = auc(fpr, tpr)
    ax = ax or plt.gca()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")

def plot_pr(df: pd.DataFrame, ax=None):
    if df.confidence.isna().all():
        return
    precision, recall, _ = precision_recall_curve(df.true_label, df.confidence)
    pr_auc = auc(recall, precision)
    ax = ax or plt.gca()
    ax.plot(recall, precision, label=f"AUC={pr_auc:.2f}", color="tomato")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend()

def plot_threshold_sweep(df: pd.DataFrame, ax=None):
    if df.confidence.isna().all():
        return
    thresholds = np.linspace(0, 1, 21)
    acc, prec, rec = [], [], []
    for t in thresholds:
        pred = (df.confidence >= t).astype(int)
        acc.append((pred == df.true_label).mean())
        prec.append((pred & (pred == 1)).sum() / max(pred.sum(), 1))
        rec.append(((pred == 1) & (df.true_label == 1)).sum() /
                   max((df.true_label == 1).sum(), 1))
    ax = ax or plt.gca()
    ax.plot(thresholds, acc, label="Accuracy")
    ax.plot(thresholds, prec, label="Precision")
    ax.plot(thresholds, rec, label="Recall")
    ax.set_xlabel("Confidence threshold"); ax.set_ylim(0, 1.05)
    ax.set_title("Metric vs Threshold"); ax.legend()

def plot_samples_hist(df: pd.DataFrame, ax=None):
    if "samples_used" not in df.columns:
        return
    sns.histplot(df.samples_used.dropna(), bins=range(1, int(df.samples_used.max())+2),
                 edgecolor="black", ax=ax)
    ax.set_xlabel("Samples used"); ax.set_title("Samples-used Distribution")

def plot_pages_hist(df: pd.DataFrame, ax=None):
    if "pages_processed" not in df.columns:
        return
    sns.histplot(df.pages_processed.dropna(), bins=range(1, int(df.pages_processed.max())+2),
                 edgecolor="black", ax=ax)
    ax.set_xlabel("Pages processed"); ax.set_title("Pages-processed Distribution")

def plot_reclass_pie(df: pd.DataFrame, ax=None):
    sub = df[df.category == "reservs"]
    if sub.empty: return
    kept = (sub.predicted_label == 1).sum()
    re   = (sub.predicted_label == 0).sum()
    ax = ax or plt.gca()
    ax.pie([kept, re], labels=["Kept as reservs", "Re-classified"],
           autopct="%1.1f%%", colors=["#2ecc71", "#e67e22"], startangle=90)
    ax.set_title("Reclassification of original 'reservs' docs")

def plot_corr(df: pd.DataFrame, ax=None):
    want_cols = ["confidence", "samples_used", "pages_processed"]
    cols = [c for c in want_cols if c in df.columns]
    if len(cols) < 2:          # need at least two vars to correlate
        ax.text(0.5, 0.5, "Not enough numeric\ncolumns for correlation",
                ha="center", va="center")
        ax.axis("off")
        return

    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix")

# NEW ──────────────────────────────────────────────────────────────
def plot_accuracy_per_group(df: pd.DataFrame, group_col: str = "category", ax=None):
    """
    Generic accuracy-per-group bar-plot.
    `group_col` is any column present in the DataFrame.
    """
    if group_col not in df.columns:
        ax.text(0.5, 0.5, f"Column '{group_col}' not in data",
                ha="center", va="center")
        ax.axis("off")
        return
    tmp = df.groupby(group_col).apply(lambda g: (g.predicted_label == g.true_label).mean())
    tmp = tmp.sort_values(ascending=False)     # nicer ordering

    ax = ax or plt.gca()
    sns.barplot(x=tmp.index.astype(str), y=tmp.values,
                palette="mako", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by '{group_col}'")
    for i, v in enumerate(tmp.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

# 🟢 MOVE THIS GLOBAL DICTIONARY UP ⬆️  so it's defined *before* main()
plot_funcs = {
    "confusion":   plot_confusion_matrix,
    "metrics":     plot_metric_bars,
    "confidence":  plot_confidence_hist,
    "category":    plot_accuracy_by_category,
    "processing":  plot_processing_stats,
    "roc":         plot_roc,
    "pr":          plot_pr,
    "threshold":   plot_threshold_sweep,
    "samples":     plot_samples_hist,
    "pages":       plot_pages_hist,
    "reclass":     plot_reclass_pie,
    "corr":        plot_corr,
    "group_acc":  plot_accuracy_per_group,
}

# global set indicating which plot functions need the DataFrame
PLOT_NEEDS_DF = {
    "confidence", "category",
    "roc", "pr", "threshold",
    "samples", "pages", "reclass", "corr",
    "group_acc"
}

# ------------------------------------------------------------------
# UPDATE CLI flags
def build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("summary_json", help="Path to evaluation summary JSON")
    p.add_argument("--save-dir", default="visualisation_output", help="Folder to save PNGs")
    # quick flags
    p.add_argument("--all", action="store_true", help="Create every plot")
    p.add_argument("--confusion", action="store_true", help="Confusion matrix")
    p.add_argument("--metrics", action="store_true", help="Bar chart of metrics")
    p.add_argument("--confidence", action="store_true", help="Confidence histogram")
    p.add_argument("--category", action="store_true", help="Accuracy by category")
    p.add_argument("--processing", action="store_true", help="Processing stats")
    # new ones
    p.add_argument("--roc",         action="store_true", help="ROC curve")
    p.add_argument("--pr",          action="store_true", help="Precision–Recall curve")
    p.add_argument("--threshold",   action="store_true", help="Metric-vs-threshold sweep")
    p.add_argument("--samples",     action="store_true", help="Samples-used histogram")
    p.add_argument("--pages",       action="store_true", help="Pages-processed histogram")
    p.add_argument("--reclass",     action="store_true", help="Pie-chart of reclassification")
    p.add_argument("--corr",        action="store_true", help="Correlation matrix")
    p.add_argument("--group-acc",  action="store_true",
                   help="Accuracy bar-plot for the chosen group column")
    p.add_argument("--group-col",  default="category",
                   help="Column to group by when --group-acc is used")
    return p


# ─────────────────────────── main CLI ────────────────────────────
def main():
    args = build_arg_parser().parse_args()
    json_path = Path(args.summary_json).expanduser()
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    df, summary = load_summary(json_path)
    print_metrics_table(summary)

    # Decide what to plot
    flags = list(plot_funcs.keys())
    selected = {f: getattr(args, f) for f in flags}

    if args.all:                                # ➊ all plots
        selected = dict.fromkeys(flags, True)

    elif not any(selected.values()):            # ➋ nothing explicit → default
        selected = dict.fromkeys(flags, False)
        selected["confusion"] = selected["metrics"] = True

    # set up axes grid dynamically
    n_plots = sum(selected.values())
    if n_plots == 0:
        return
    cols = 2
    rows = int(math.ceil(n_plots/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    # axes could be ndarray or single Axes
    axes = np.atleast_1d(axes).flatten()

    axis_idx = 0
    for flag, do_it in selected.items():
        if do_it:
            source = df if flag in PLOT_NEEDS_DF else summary
            if flag == "group_acc":
                plot_accuracy_per_group(df, group_col=args.group_col, ax=axes[axis_idx])
            else:
                plot_funcs[flag](source, ax=axes[axis_idx])
            axis_idx += 1

    # remove unused axes
    for ax in axes[axis_idx:]:
        ax.remove()

    # save + show
    out_dir = Path(args.save_dir)
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"viz_{json_path.stem}.png"
    plt.suptitle("Oil & Gas Classification – Quick Dashboard", fontsize=16, fontweight="bold")
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"📊 figure saved to {out_file.resolve()}")

if __name__ == "__main__":
    main()