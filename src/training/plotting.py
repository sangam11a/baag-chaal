"""
Publication-Quality Plotting Engine for Bagh-Chal Training & Ablation Results.

Formats figures according to Springer Nature (300 DPI, vector PDF & PNG, clean typography).
"""

from __future__ import annotations
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


def setup_springer_style():
    """
    Applies clean Springer Nature manuscript style to matplotlib.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 1.75,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def plot_training_dashboard(
    history: Dict[str, List[float]],
    save_path_prefix: str,
    title: str = "Bagh-Chal PPO Training Dynamics",
):
    """
    Generates a 4-panel publication dashboard:
    1. Loss Curves (Policy Loss & Value Loss)
    2. Self-Play Win/Draw Rates
    3. Asymmetry Dynamics (TBI & RBI)
    4. Evaluation Competitive Score
    Saves both PNG (300 DPI) and PDF.
    """
    setup_springer_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=300)

    episodes = history.get("episodes", list(range(len(history.get("loss", [])))))

    # Panel 1: Loss Curves
    ax1 = axes[0, 0]
    if "policy_loss" in history and history["policy_loss"]:
        ax1.plot(episodes, history["policy_loss"], label="Policy Loss", color="#1f77b4")
    if "value_loss" in history and history["value_loss"]:
        ax1.plot(episodes, history["value_loss"], label="Value Loss (MSE)", color="#ff7f0e")
    ax1.set_xlabel("Training Episodes")
    ax1.set_ylabel("Loss")
    ax1.set_title("(a) Policy & Value Loss")
    ax1.legend(loc="upper right")

    # Panel 2: Win & Draw Dynamics
    ax2 = axes[0, 1]
    if "tiger_win_rate" in history and history["tiger_win_rate"]:
        ax2.plot(episodes, history["tiger_win_rate"], label="Tiger WR", color="#d62728")
    if "goat_win_rate" in history and history["goat_win_rate"]:
        ax2.plot(episodes, history["goat_win_rate"], label="Goat WR", color="#2ca02c")
    if "draw_rate" in history and history["draw_rate"]:
        ax2.plot(episodes, history["draw_rate"], label="Draw Rate", color="#7f7f7f", linestyle=":")
    ax2.set_xlabel("Training Episodes")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("(b) Self-Play Win/Draw Dynamics")
    ax2.legend(loc="best")

    # Panel 3: Asymmetry Indices (TBI & RBI)
    ax3 = axes[1, 0]
    if "tiger_bias_index" in history and history["tiger_bias_index"]:
        ax3.plot(episodes, history["tiger_bias_index"], label="Tiger Bias Index (TBI)", color="#e377c2")
        ax3.axhline(0.0, color="black", linestyle="--", alpha=0.5, label="Equilibrium (TBI=0)")
    if "role_balance_index" in history and history["role_balance_index"]:
        ax3.plot(episodes, history["role_balance_index"], label="Role Balance Index (RBI)", color="#17becf")
    ax3.set_xlabel("Training Episodes")
    ax3.set_ylabel("Index")
    ax3.set_ylim(-1.05, 1.05)
    ax3.set_title("(c) Role Asymmetry Dynamics")
    ax3.legend(loc="best")

    # Panel 4: Benchmark Evaluation Competitive Score
    ax4 = axes[1, 1]
    eval_eps = history.get("eval_episodes", [])
    if "eval_competitive_score" in history and history["eval_competitive_score"]:
        ax4.plot(eval_eps, history["eval_competitive_score"], marker="o", markersize=4,
                 label="Competitive Score (vs Pool)", color="#9467bd")
    if "eval_tiger_wr" in history and history["eval_tiger_wr"]:
        ax4.plot(eval_eps, history["eval_tiger_wr"], linestyle="--", alpha=0.7,
                 label="Eval as Tiger WR", color="#d62728")
    if "eval_goat_wr" in history and history["eval_goat_wr"]:
        ax4.plot(eval_eps, history["eval_goat_wr"], linestyle="--", alpha=0.7,
                 label="Eval as Goat WR", color="#2ca02c")
    ax4.set_xlabel("Training Episodes")
    ax4.set_ylabel("Score / Win Rate")
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_title("(d) Benchmark Generalization")
    ax4.legend(loc="best")

    plt.suptitle(title, y=0.98, fontweight="bold")
    plt.tight_layout()

    # Save PNG and PDF
    png_path = f"{save_path_prefix}.png"
    pdf_path = f"{save_path_prefix}.pdf"
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved to: {png_path} and {pdf_path}")


def plot_ablation_comparison(
    ablation_data: Dict[str, Dict[str, Any]],
    save_path_prefix: str,
    title: str = "Representation & Architecture Ablation Comparison",
):
    """
    Plots comparative bar charts across representations/architectures:
    - Competitive Score
    - Final Tiger Bias Index
    - Average Episode Plies
    """
    setup_springer_style()
    models = list(ablation_data.keys())
    comp_scores = [ablation_data[m].get("competitive_score", 0.0) for m in models]
    tbi_scores = [ablation_data[m].get("tiger_bias_index", 0.0) for m in models]
    avg_plies = [ablation_data[m].get("avg_plies", 0.0) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # 1. Competitive Score
    ax1 = axes[0]
    bars1 = ax1.bar(models, comp_scores, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(models)])
    ax1.set_ylabel("Competitive Score")
    ax1.set_title("Generalization vs Baselines")
    ax1.set_ylim(0, 1.0)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

    # 2. Tiger Bias Index
    ax2 = axes[1]
    bars2 = ax2.bar(models, tbi_scores, color="#e377c2")
    ax2.axhline(0.0, color="black", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Tiger Bias Index (TBI)")
    ax2.set_title("Role Bias (+1=Tiger, 0=Bal)")
    ax2.set_ylim(-1.0, 1.0)

    # 3. Average Plies
    ax3 = axes[2]
    bars3 = ax3.bar(models, avg_plies, color="#17becf")
    ax3.set_ylabel("Average Moves per Game")
    ax3.set_title("Game Duration (Survival)")

    plt.suptitle(title, y=1.02, fontweight="bold")
    plt.tight_layout()

    png_path = f"{save_path_prefix}.png"
    pdf_path = f"{save_path_prefix}.pdf"
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation plots saved to: {png_path} and {pdf_path}")

