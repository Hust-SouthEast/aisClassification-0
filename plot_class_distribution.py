import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def count_samples(data_root: Path):
    """
    统计 data_root 下每个子目录中的 .txt 文件数。
    返回 (labels, counts)，其中 labels 为子目录名（船型名），counts 为对应样本数。
    """
    if not data_root.is_dir():
        raise RuntimeError(f"数据目录不存在: {data_root}")

    subdirs = sorted([d for d in data_root.iterdir() if d.is_dir()], key=lambda x: x.name)
    if not subdirs:
        raise RuntimeError(f"在 {data_root} 下没有找到任何子目录。")

    labels, counts = [], []
    for d in subdirs:
        n_txt = len(list(d.glob("*.txt")))
        labels.append(d.name)
        counts.append(n_txt)

    return labels, counts


def plot_distribution(labels, counts, out_path: Path, title: str = "Sample Count per Ship Type"):
    """
    绘制柱状图：x 轴为船型（子目录名），y 轴为该目录下样本数。
    按样本数从左到右由高到低排序。
    """
    # 按样本数从大到小排序
    order = sorted(range(len(labels)), key=lambda i: counts[i], reverse=True)
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]

    n = len(labels)
    x = range(n)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 6))
    bars = ax.bar(x, counts, color="#1f77b4", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of samples")
    ax.set_title(title)

    # 在柱子上方标注数值
    for rect, v in zip(bars, counts):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            str(v),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot sample count distribution per ship type in data folder.")
    parser.add_argument("--data_root", type=str, default="data", help="Root folder containing ship-type subfolders.")
    parser.add_argument(
        "--out",
        type=str,
        default="class_distribution.png",
        help="Output image filename (saved next to this script).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    data_root = base_dir / args.data_root
    out_path = base_dir / args.out

    labels, counts = count_samples(data_root)
    plot_distribution(labels, counts, out_path)

    total = sum(counts)
    print("Class distribution:")
    for name, c in zip(labels, counts):
        ratio = (c / total) * 100 if total > 0 else 0.0
        print(f"  {name}: {c} samples ({ratio:.2f}%)")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

