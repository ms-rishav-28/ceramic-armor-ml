# src/interpretation/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")

def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def parity_plot(y_true, y_pred, title, out_path):
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.scatter(y_true, y_pred, s=16, alpha=0.6)
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def residual_plot(y_true, y_pred, title, out_path):
    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)
    ax.hist(res, bins=40, alpha=0.8)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def corr_heatmap(df, title, out_path):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def feature_importance_plot(feature_names, importance_values, title, out_path):
    """Plot feature importance as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Sort features by importance
    sorted_idx = np.argsort(importance_values)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = [importance_values[i] for i in sorted_idx]
    
    ax.barh(range(len(sorted_features)), sorted_importance)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
