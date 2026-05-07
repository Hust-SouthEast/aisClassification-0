# -*- coding: utf-8 -*-
"""
根据 train_hybrid.py 运行后生成的文件，绘制或列出：
1. 训练收敛曲线（Transformer：loss / train_acc / val_acc）
2. 各类别性能指标对比（Recall、Precision、F1）
3. 超参数设置表
依赖：training_history.json、results_hybrid.txt、config_hybrid.json（需先运行 train_hybrid.py）
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# 与项目其他脚本一致的类别名（0-13）
CLASS_NAMES = [
    "Tanker", "Vehicle Carrier", "Fishing Vessel", "Chemical Tanker", "Tug",
    "LNG Carrier", "Passenger Ship", "Ro-Ro Ship", "Reefer Ship", "Pleasure Craft",
    "Bulk Carrier", "Container Ship", "Other Cargo", "Cargo Ship",
]


def load_history(base_dir: Path):
    p = base_dir / "training_history.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(base_dir: Path):
    p = base_dir / "results_hybrid.txt"
    if not p.exists():
        return None
    out = {"accuracy": None, "macro_f1": None, "per_class": []}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("accuracy="):
                out["accuracy"] = float(line.split("=", 1)[1])
            elif line.startswith("macro_f1="):
                out["macro_f1"] = float(line.split("=", 1)[1])
            elif line.startswith("class_"):
                m = re.match(r"class_(\d+):\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
                if m:
                    c = int(m.group(1))
                    recall, prec, f1, sup = float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5))
                    out["per_class"].append((c, recall, prec, f1, sup))
    out["per_class"].sort(key=lambda x: x[0])
    return out


def load_config(base_dir: Path):
    p = base_dir / "config_hybrid.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_convergence(history: list, out_path: Path):
    if not history or not plt:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(epochs, train_loss, color="C0", label="Train Loss")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, color="C1", label="Train Acc")
    ax2.plot(epochs, val_acc, color="C2", label="Val Acc")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Transformer Training Convergence", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_class_metrics(results: dict, out_path: Path, class_names: list):
    if not results or not results.get("per_class") or not plt:
        return
    per = results["per_class"]
    n = len(per)
    labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c, *_ in per]
    recall = [r for _, r, *_ in per]
    prec = [p for _, _, p, *_ in per]
    f1 = [f for _, _, _, f, _ in per]
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 5))
    ax.bar(x - w, recall, width=w, label="Recall", color="C0", alpha=0.9)
    ax.bar(x, prec, width=w, label="Precision", color="C1", alpha=0.9)
    ax.bar(x + w, f1, width=w, label="F1", color="C2", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.title("Per-class Performance (Test Set)")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_per_class_table(results: dict, class_names: list):
    if not results or not results.get("per_class"):
        return
    per = results["per_class"]
    print("\n各类别性能指标（Test）:")
    print("-" * 80)
    print(f"  {'Class':<20} {'Recall':>10} {'Precision':>10} {'F1':>10} {'Support':>8}")
    print("-" * 80)
    for c, recall, prec, f1, sup in per:
        name = class_names[c] if c < len(class_names) else f"Class {c}"
        print(f"  {name:<20} {recall:>10.4f} {prec:>10.4f} {f1:>10.4f} {sup:>8d}")
    print("-" * 80)
    if results.get("accuracy") is not None:
        print(f"  {'Overall Accuracy':<20} {results['accuracy']:.4f}")
    if results.get("macro_f1") is not None:
        print(f"  {'Macro-F1':<20} {results['macro_f1']:.4f}")


def print_or_plot_config(config: dict, out_path: Path = None):
    if not config:
        return
    trans = config.get("transformer", {})
    lgb = config.get("lgb", {})

    # 控制显示的键（排除冗长或重复的）
    trans_keys = ["data_root", "seq_len", "batch_size", "epochs", "lr", "d_model", "nhead", "num_layers",
                  "dim_ff", "dropout", "pool", "patience", "slide_stride", "num_segments", "segment_agg",
                  "lgb_rounds", "lgb_early_stop", "lgb_device"]
    lgb_keys = ["num_leaves", "max_depth", "learning_rate", "min_data_in_leaf", "feature_fraction",
                "bagging_fraction", "bagging_freq", "device"]

    rows = []
    for k in trans_keys:
        if k in trans:
            rows.append(("Transformer", k, str(trans[k])))
    for k in lgb_keys:
        if k in lgb:
            rows.append(("LightGBM", k, str(lgb[k])))

    print("\n超参数设置:")
    print("-" * 60)
    print(f"  {'Stage':<12} {'Parameter':<22} {'Value':<20}")
    print("-" * 60)
    for stage, param, value in rows:
        print(f"  {stage:<12} {param:<22} {str(value):<20}")
    print("-" * 60)

    if out_path and plt and rows:
        fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.35)))
        ax.axis("off")
        table = ax.table(
            cellText=[[r[1], r[2]] for r in rows],
            rowLabels=[r[0] for r in rows],
            colLabels=["Parameter", "Value"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        plt.title("Hyperparameters")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制/列出训练收敛曲线、各类别指标、超参表")
    parser.add_argument("--out_dir", type=str, default=".", help="输出目录，默认项目根目录")
    parser.add_argument("--no_plot", action="store_true", help="仅打印表格，不生成图片")
    parser.add_argument("--prefix", type=str, default="training_results", help="输出文件名前缀")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(base_dir)
    results = load_results(base_dir)
    config = load_config(base_dir)

    if history is None and results is None and config is None:
        print("未找到 training_history.json / results_hybrid.txt / config_hybrid.json，请先运行 train_hybrid.py")
        return

    # 1) 收敛曲线
    if history:
        if not args.no_plot and plt:
            plot_convergence(history, out_dir / f"{args.prefix}_convergence.png")
        else:
            print("\n训练收敛（最近 5 轮）:")
            for h in history[-5:]:
                print(f"  Epoch {h['epoch']}: loss={h['train_loss']:.4f} train_acc={h['train_acc']:.4f} val_acc={h['val_acc']:.4f}")

    # 2) 各类别性能
    if results:
        print_per_class_table(results, CLASS_NAMES)
        if not args.no_plot and plt:
            plot_per_class_metrics(results, out_dir / f"{args.prefix}_per_class.png", CLASS_NAMES)

    # 3) 超参表
    if config:
        out_hyper = (out_dir / f"{args.prefix}_hyperparams.png") if not args.no_plot and plt else None
        print_or_plot_config(config, out_hyper)


if __name__ == "__main__":
    main()
