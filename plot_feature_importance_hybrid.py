#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 hybrid（Transformer 深度特征 + 表格特征）LightGBM 的特征重要性图（不依赖 shap）。

输出为横向条形图：
- 横轴：Importance 值
- 纵轴：特征名（来自 feature_names.json）

用法（在项目根目录执行）::

    python plot_feature_importance_hybrid.py
    python plot_feature_importance_hybrid.py --top_k 30 --importance_type gain
    python plot_feature_importance_hybrid.py --model checkpoints/lgb_hybrid.txt \\
        --feature_names_json outputs_shap/feature_names.json --out_png fi.png

依赖：已训练好的 LightGBM txt、与模型特征顺序一致的 ``feature_names.json``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_feature_names(feature_names_json: Path) -> list[str]:
    with open(feature_names_json, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError(f"feature_names.json 内容不是 list：{feature_names_json}")
    return [str(x) for x in names]


def _display_name(name: str) -> str:
    if name.startswith("ft_"):
        return "trans_emb_" + name.split("ft_", 1)[1]
    return name


def _format_importance_value(v: float, importance_type: str) -> str:
    """Human-readable label for bar end (non-negative importance)."""
    if importance_type == "split":
        iv = int(round(float(v)))
        return f"{iv:,}" if iv >= 1000 else str(iv)
    x = float(v)
    if x >= 1e6:
        return f"{x:.3g}"
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 10:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
        return s
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def main():
    parser = argparse.ArgumentParser(description="Plot hybrid LightGBM feature importance (no shap).")
    parser.add_argument("--model", type=str, default="checkpoints/lgb_hybrid.txt", help="LightGBM 模型 txt")
    parser.add_argument("--feature_names_json", type=str, default="outputs_shap/feature_names.json", help="特征名 JSON")
    parser.add_argument("--out_png", type=str, default="", help="输出 PNG；默认写到 outputs_shap/ 下")
    parser.add_argument("--top_k", type=int, default=25, help="只显示前 K 个特征")
    parser.add_argument(
        "--importance_type",
        type=str,
        default="gain",
        choices=["gain", "split"],
        help="gain=按增益；split=按分裂次数",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = project_dir / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型：{args.model}（也未在 {project_dir} 下找到）")

    feature_names_path = Path(args.feature_names_json)
    if not feature_names_path.exists():
        feature_names_path = project_dir / args.feature_names_json
    if not feature_names_path.exists():
        raise FileNotFoundError(f"找不到 feature_names_json：{args.feature_names_json}")

    if args.out_png:
        out_png = Path(args.out_png)
    else:
        out_png = project_dir / "outputs_shap" / f"feature_importance_{args.importance_type}_top{args.top_k}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    feature_names = _load_feature_names(feature_names_path)

    booster = lgb.Booster(model_file=str(model_path))
    importances = booster.feature_importance(importance_type=args.importance_type)

    if len(importances) != len(feature_names):
        # 允许轻微不一致：截断到最小长度，并给出提示
        n = min(len(importances), len(feature_names))
        print(
            f"[警告] 特征数不一致：importances={len(importances)} vs feature_names={len(feature_names)}；"
            f"将截断到 {n}。"
        )
        importances = importances[:n]
        feature_names = feature_names[:n]

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(args.top_k)
    df["feature_display"] = df["feature"].map(_display_name)

    values = df["importance"].values.astype(np.float64)
    y = np.arange(len(df))
    n = len(df)

    fig_w = 12.0
    fig_h = max(5.2, 0.42 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("#F4F6F9")

    bar_color = "#2563EB"
    edge_color = "#1E40AF"
    bars = ax.barh(
        y,
        values,
        height=0.68,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.6,
        alpha=0.92,
        zorder=2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        df["feature_display"].values,
        fontsize=13,
        color="#1E293B",
        fontweight="bold",
    )
    ax.invert_yaxis()

    ax.set_xlabel(
        f"Importance ({args.importance_type})",
        fontsize=15,
        color="#334155",
        labelpad=10,
        fontweight="bold",
    )
    ax.set_title(
        f"Hybrid LightGBM — Feature Importance (Top {n}, {args.importance_type})",
        fontsize=15,
        fontweight="600",
        color="#0F172A",
        pad=16,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#94A3B8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=13, colors="#475569", width=1.0, length=5)
    ax.tick_params(axis="y", length=0)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color("#CBD5E1")
        ax.spines[s].set_linewidth(0.8)

    xmax = float(np.max(values)) if n else 1.0
    ax.set_xlim(0.0, xmax * 1.18 if xmax > 0 else 1.0)

    value_labels = [_format_importance_value(v, args.importance_type) for v in values]
    ax.bar_label(
        bars,
        labels=value_labels,
        padding=6,
        fontsize=10,
        color="#1E293B",
        fontweight="600",
        zorder=3,
    )

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=280, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

