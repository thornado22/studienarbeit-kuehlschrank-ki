import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def create_barchart(
    df,
    title,
    x_name,
    y_name,
    top_n=None,
    bottom_n=None,
    cols=None
):
    counts = df.sum().sort_values(ascending=False)

    if cols:
        counts = counts[[c for c in cols if c in counts.index]]

    if top_n:
        counts = counts.head(top_n)

    if bottom_n: 
        counts = counts.tail(bottom_n)

    counts_df = counts.reset_index()
    counts_df.columns = ["Label", "Count"]

    # Percentage relative to number of rows
    total_rows = len(df)
    counts_df["Percent"] = 100 * counts_df["Count"] / total_rows

    base_font_size = 14
    title_size = 16
    axis_label_size = 14
    tick_label_size = 12
    annot_size = 12

    plt.rc('font', size=base_font_size)


    # Plot
    plt.figure(figsize=(max(6, len(counts_df) * 0.6), 5.5))

    ax = sns.barplot(
        data=counts_df,
        x="Label",
        y="Count",
        hue="Label",
        palette="viridis",
        legend=False
    )

    ax.set_xlabel(x_name, fontsize=axis_label_size)
    ax.set_ylabel(y_name, fontsize=axis_label_size)
    ax.set_title(title, fontsize=title_size)

    plt.xticks(rotation=45, ha="right", fontsize=tick_label_size)

    for i, row in counts_df.iterrows():
        ax.text(
            i,
            row["Count"],
            "{:.1f}%".format(row["Percent"]),
            ha="center",
            va="bottom",
            fontsize=annot_size
        )

    plt.tight_layout()
    plt.show()


def plot_f1(x, y, title, path, figsize=None, palette="viridis", fmt="{:.2f}"):
    # compute figure size similar to create_barchart when not provided
    if figsize is None:
        try:
            n = len(x)
        except Exception:
            n = 6
        figsize = (max(6, n * 0.6), 5.5)

    base_font_size = 14
    title_size = 16
    axis_label_size = 14
    tick_label_size = 12
    annot_size = 12

    plt.rc('font', size=base_font_size)

    plt.figure(figsize=figsize)
    ax = sns.barplot(x=x, y=y, palette=palette)

    ax.set_xlabel("Klasse", fontsize=axis_label_size)
    ax.set_ylabel("F1 Score", fontsize=axis_label_size)
    ax.set_title(title, fontsize=title_size)

    plt.xticks(rotation=45, ha="right", fontsize=tick_label_size)

    # annotate bars with values
    for i, v in enumerate(y):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=annot_size)

    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def parse_per_class_confusion_json(json_path: str | Path) -> Dict[str, Any]:
    """Load a per-class confusion JSON file generated during test evaluation."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_per_class_metrics(json_path: str | Path, metric: str = "f1") -> Tuple[List[str], List[float]]:
    """Return class names and one metric (precision, recall, f1, support, etc.) from a JSON file."""
    data = parse_per_class_confusion_json(json_path)
    per_class = data.get("per_class", {})

    if metric not in next(iter(per_class.values()), {}):
        raise KeyError(f"Metric '{metric}' not found in JSON data.")

    labels = list(per_class.keys())
    values = [float(per_class[name][metric]) for name in labels]
    return labels, values


def plot_metric_barchart_from_json(
    json_path: str | Path,
    metric: str = "f1",
    title: Optional[str] = None,
    x_name: str = "Label",
    y_name: Optional[str] = None,
    top_n: Optional[int] = None,
    bottom_n: Optional[int] = None,
    cols: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    fmt: str = "{:.2f}",
):
    """Plot a per-class metric (e.g. F1) from a per-class confusion JSON using the
    same styling/formatting as `create_barchart`.

    Returns (fig, ax).
    """
    labels, values = get_per_class_metrics(json_path, metric=metric)

    df = pd.DataFrame({"Label": labels, "Value": values})

    # Preserve requested column order if provided
    if cols:
        requested = [c for c in cols if c in df["Label"].values]
        if requested:
            df = df.set_index("Label").loc[requested].reset_index()

    if top_n:
        df = df.sort_values("Value", ascending=False).head(top_n)

    if bottom_n:
        df = df.sort_values("Value", ascending=False).tail(bottom_n)

    if title is None:
        title = f"{metric.upper()} pro Klasse"

    if y_name is None:
        y_name = metric.upper()

    base_font_size = 14
    title_size = 16
    axis_label_size = 14
    tick_label_size = 12
    annot_size = 12

    plt.rc('font', size=base_font_size)

    df = df.reset_index(drop=True)

    plt.figure(figsize=(max(6, len(df) * 0.6), 5.5))

    ax = sns.barplot(
        data=df,
        x="Label",
        y="Value",
        hue="Label",
        palette="viridis",
        legend=False
    )

    ax.set_xlabel(x_name, fontsize=axis_label_size)
    ax.set_ylabel(y_name, fontsize=axis_label_size)
    ax.set_title(title, fontsize=title_size)

    plt.xticks(rotation=45, ha="right", fontsize=tick_label_size)

    for i, row in df.iterrows():
        ax.text(
            i,
            row["Value"],
            fmt.format(row["Value"]),
            ha="center",
            va="bottom",
            fontsize=annot_size
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    return ax.get_figure(), ax


def build_binary_confusion_matrix_from_json(json_path: str | Path, class_name: Optional[str] = None) -> np.ndarray:
    """Build a 2x2 confusion matrix for one class or for the aggregated dataset."""
    data = parse_per_class_confusion_json(json_path)
    per_class = data.get("per_class", {})

    if class_name is not None:
        if class_name not in per_class:
            raise KeyError(f"Class '{class_name}' not found in JSON data.")
        stats = per_class[class_name]
        return np.array([[stats["tp"], stats["fp"]], [stats["fn"], stats["tn"]]], dtype=int)

    tp = sum(stats["tp"] for stats in per_class.values())
    fp = sum(stats["fp"] for stats in per_class.values())
    fn = sum(stats["fn"] for stats in per_class.values())
    tn = sum(stats["tn"] for stats in per_class.values())
    return np.array([[tp, fp], [fn, tn]], dtype=int)


def plot_confusion_matrix_from_json(
    json_path: str | Path,
    class_name: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
):
    """Plot a 2x2 confusion matrix for a selected class or the full dataset."""
    cm = build_binary_confusion_matrix_from_json(json_path)
    labels = ["Positiv", "Negativ"]

    base_font_size = 14
    title_size = 16
    axis_label_size = 14
    tick_label_size = 12
    annot_size = 12

    plt.rc('font', size=base_font_size)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    annot_labels = [
        [f"TP\n{int(cm[0,0])}", f"FP\n{int(cm[0,1])}"],
        [f"FN\n{int(cm[1,0])}", f"TN\n{int(cm[1,1])}"]
    ]
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": annot_size, "color": "black", "weight": "bold"}
    )
    plot_title = f"Confusion Matrix für {class_name}" if class_name else "Aggregierte Confusion Matrix"
    ax.set_title(plot_title, fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    import json

    file = "../tests/test_two_1.json"

    with open(file, "r") as f:
        src = json.load(f)

    val_class_f1 = src["results"]["val_class_f1"]
    classes = src["meta"]["classes"]

    plot_f1( classes, val_class_f1, "verkettete Klassifikation", "f1_test_two_1.png", figsize=(10, 5))