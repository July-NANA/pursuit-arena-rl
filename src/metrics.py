from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(values: list[float], window: int = 100) -> list[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float32)
    out: list[float] = []
    for i in range(len(arr)):
        left = max(0, i - window + 1)
        out.append(float(arr[left : i + 1].mean()))
    return out


def save_training_curves(history: dict[str, list[float]], assets_dir: Path, prefix: str = "") -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    pre = f"{prefix}_" if prefix else ""

    rewards = history["episode_rewards"]
    captures = history["capture_flags"]
    avg_distance = history["avg_distance"]

    capture_curve = rolling_mean([float(v) for v in captures], window=100)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(rewards, color="#4F8EF7", linewidth=1.0)
    ax.set_title("Reward Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(assets_dir / f"{pre}reward_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(capture_curve, color="#55BD8A", linewidth=1.2)
    ax.set_title("Capture Rate Curve (rolling=100)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Capture Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(assets_dir / f"{pre}capture_rate_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(avg_distance, color="#F59E4A", linewidth=1.0)
    ax.set_title("Average Distance per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(assets_dir / f"{pre}distance_curve.png", dpi=160)
    plt.close(fig)


def save_comparison_bar(results: dict[str, dict[str, float]], assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    labels = list(results.keys())
    capture = [results[k]["capture_rate"] for k in labels]
    steps = [results[k]["avg_steps_to_capture"] for k in labels]
    invalid = [results[k]["invalid_move_rate"] for k in labels]

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(x, capture, color="#55BD8A")
    axes[0].set_title("Capture Rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, steps, color="#4F8EF7")
    axes[1].set_title("Avg Steps to Capture")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    axes[2].bar(x, invalid, color="#EB6E6E")
    axes[2].set_title("Invalid Move Rate")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(assets_dir / "comparison_metrics_bar.png", dpi=160)
    plt.close(fig)
