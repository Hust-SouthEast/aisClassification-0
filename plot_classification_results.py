# -*- coding: utf-8 -*-
"""
根据 results_hybrid.txt 或传入数据，绘制各类别分类结果图（Recall / Precision / F1，及样本数 support）。
标签对应：0-Tanker, 1-Vehicle Carrier, ..., 13-Cargo Ship。
"""
import argparse
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

CLASS_NAMES = [
    "Tanker",           # 0
    "Vehicle Carrier",  # 1
    "Fishing Vessel",   # 2
    "Chemical Tanker",  # 3
    "Tug",              # 4
    "LNG Carrier",      # 5
    "Passenger Ship",   # 6
    "Ro-Ro Ship",       # 7
    "Reefer Ship",      # 8
    "Pleasure Craft",   # 9
    "Bulk Carrier",     # 10
    "Container Ship",   # 11
    "Other Cargo",     # 12
    "Cargo Ship",       # 13
]


def load_results(results_path: Path):
    """
    解析 results_hybrid.txt，支持两种格式：
    - 旧格式：accuracy=... / macro_f1=... / class_0: recall prec f1 support
    - 当前格式：Test Accuracy (总体): ... / Macro-F1: ... / 类别  0:  准确率(Recall)=... Precision=... F1=... support=...
    返回：{ "accuracy": float, "macro_f1": float, "per_class": [ (class_id, recall, precision, f1, support), ... ] }
    """
    out = {"accuracy": None, "macro_f1": None, "per_class": []}
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("accuracy="):
                out["accuracy"] = float(line.split("=", 1)[1])
            elif line.startswith("macro_f1="):
                out["macro_f1"] = float(line.split("=", 1)[1])
            elif "Test Accuracy" in line and ":" in line:
                out["accuracy"] = float(line.split(":", 1)[1].strip())
            elif "Macro-F1" in line and ":" in line:
                out["macro_f1"] = float(line.split(":", 1)[1].strip())
            elif "类别" in line and "准确率" in line:
                m = re.search(r"类别\s*(\d+)\s*:\s*准确率\(Recall\)=([\d.]+)\s+Precision=([\d.]+)\s+F1=([\d.]+)\s+support=(\d+)", line)
                if m:
                    c, recall, prec, f1, sup = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5))
                    out["per_class"].append((c, recall, prec, f1, sup))
            elif line.startswith("class_"):
                m = re.match(r"class_(\d+):\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
                if m:
                    c = int(m.group(1))
                    recall, prec, f1, sup = float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5))
                    out["per_class"].append((c, recall, prec, f1, sup))
    out["per_class"].sort(key=lambda x: x[0])
    return out


def plot_classification_results(per_class: list, class_names: list, out_path: Path, title_suffix: str = ""):
    """
    绘制各类别 Recall / Precision / F1 柱状图，并在柱上方或下方标注 support。
    """
    if not per_class or not plt:
        return
    n = len(per_class)
    labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c, *_ in per_class]
    recall = np.array([r for _, r, *_ in per_class])
    prec = np.array([p for _, _, p, *_ in per_class])
    f1 = np.array([f for _, _, _, f, _ in per_class])
    support = np.array([s for _, _, _, _, s in per_class])

    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(12, n * 0.55), 6))
    bars_r = ax.bar(x - w, recall, width=w, label="Recall", color="#2ecc71", alpha=0.9, edgecolor="none")
    bars_p = ax.bar(x, prec, width=w, label="Precision", color="#3498db", alpha=0.9, edgecolor="none")
    bars_f = ax.bar(x + w, f1, width=w, label="F1", color="#9b59b6", alpha=0.9, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(-0.12, 1.08)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="gray", linewidth=0.5)

    # 在每组柱子下方标注样本数 support
    for i in range(n):
        ax.text(x[i], -0.06, f"n={support[i]}", ha="center", va="top", fontsize=8, color="gray")

    if title_suffix:
        ax.set_title(f"Per-class Classification Results {title_suffix}", fontsize=12)
    else:
        ax.set_title("Per-class Classification Results (Recall / Precision / F1)", fontsize=12)

    plt.subplots_adjust(bottom=0.18)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制各类别分类结果图（Recall/Precision/F1）")
    parser.add_argument("--results", type=str, default="results_hybrid.txt", help="results_hybrid.txt 路径")
    parser.add_argument("--out", type=str, default="classification_results.png", help="输出图片路径")
    parser.add_argument("--title", type=str, default="", help="图标题后缀，可选")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    results_path = base_dir / args.results
    out_path = base_dir / args.out

    data = load_results(results_path)
    if not data or not data.get("per_class"):
        print(f"未找到有效数据：{results_path} 或其中无 per_class 行。请先运行 train_hybrid.py 生成该文件。")
        return

    plot_classification_results(
        data["per_class"],
        CLASS_NAMES,
        out_path,
        title_suffix=args.title.strip(),
    )
    if data.get("accuracy") is not None:
        print(f"Overall Accuracy: {data['accuracy']:.4f}")
    if data.get("macro_f1") is not None:
        print(f"Macro-F1: {data['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
