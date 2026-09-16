"""
scripts/visualize_results.py
============================
Reads the round_metrics.parquet file and generates a beautiful line chart grid
showing the F1 Score, Precision, Recall, and FPR evolution across all rounds.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def plot_metrics(parquet_path: Path):
    if not parquet_path.exists():
        logger.error(f"Cannot find metrics file at {parquet_path}")
        return None

    df = pd.read_parquet(parquet_path)
    if df.empty:
        logger.error("Metrics dataframe is empty.")
        return None

    # Set up seaborn style for beautiful plots
    sns.set_theme(style="whitegrid", palette="muted")
    
    # We want to plot 4 metrics for each persona across the rounds.
    metrics = ["f1", "precision", "recall", "fpr"]
    metric_titles = ["F1 Score (Overall Accuracy)", "Precision", "Recall (Catch Rate)", "False Positive Rate (FPR)"]
    personas = df["persona"].unique()
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
        
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]
        sns.lineplot(data=df, x="round", y=metric, hue="persona", marker="o", linewidth=2.5, ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Score")
        if metric == "fpr":
            ax.set_ylim(-0.05, max(0.5, df["fpr"].max() + 0.1))
        else:
            ax.set_ylim(0, 1.05)
            
        # Only show legend on the top plot to save space
        if i == 0:
            ax.legend(title="Persona", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.get_legend().remove()
            
    axes[-1].set_xlabel("Adversarial Round", fontsize=12)
    plt.xticks(sorted(df["round"].unique()))
    
    plt.tight_layout()
    
    # Save the plot in the same directory as the parquet file
    output_path = parquet_path.parent / "adversarial_metrics_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to round_metrics.parquet")
    args = parser.parse_args()
    
    out = plot_metrics(Path(args.metrics_path))
    if out:
        print(f"Plot saved to: {out}")
