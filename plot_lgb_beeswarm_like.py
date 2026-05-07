#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 LightGBM pred_contrib=True 计算多分类“SHAP-like”贡献值，并绘制 beeswarm-like 图：
横轴：SHAP value
纵轴：特征（按 mean(|SHAP|) 排序后取 Top-K）

不依赖 shap 包。
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
import matplotlib.transforms as mtransforms  # noqa: E402


def _ensure_xlim_includes_zero(ax: plt.Axes) -> None:
    xmin, xmax = ax.get_xlim()
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    span = xmax - xmin if xmax > xmin else 1.0
    pad = span * 0.04
    lo, hi = xmin, xmax
    # Leave room past 0 so “negative / positive impact” arrows are visible
    if lo > 0:
        lo = min(0.0, lo - pad)
        lo = min(lo, -span * 0.08)
    if hi < 0:
        hi = max(0.0, hi + pad)
        hi = max(hi, span * 0.08)
    ax.set_xlim(lo, hi)


def _add_x_axis_impact_arrows(ax: plt.Axes) -> None:
    """Arrows from x=0 toward negative/positive with English labels (below plot, clip off)."""
    _ensure_xlim_includes_zero(ax)
    xmin, xmax = ax.get_xlim()
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    span = xmax - xmin if xmax > xmin else 1.0
    frac = 0.12
    neg_tip = max(xmin, 0.0 - span * frac)
    pos_tip = min(xmax, 0.0 + span * frac)

    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_arrow = -0.15
    y_text = -0.10
    kw = dict(
        xycoords=blend,
        textcoords=blend,
        arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.2, mutation_scale=10),
        clip_on=False,
    )
    # Negative: from 0 toward left
    # ax.annotate("", xy=(neg_tip, y_arrow), xytext=(0, y_arrow), **kw)
    # ax.text(
    #     (neg_tip + 0) / 2.0,
    #     y_text,
    #     "Negative Impact",
    #     ha="center",
    #     va="bottom",
    #     transform=blend,
    #     fontsize=10,
    #     color="0.2",
    #     clip_on=False,
    # )
    # # Positive: from 0 toward right
    # ax.annotate("", xy=(pos_tip, y_arrow), xytext=(0, y_arrow), **kw)
    # ax.text(
    #     (pos_tip + 0) / 2.0,
    #     y_text,
    #     "Positive Impact",
    #     ha="center",
    #     va="bottom",
    #     transform=blend,
    #     fontsize=10,
    #     color="0.2",
    #     clip_on=False,
    # )


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


def _reshape_pred_contrib_multiclass(contrib_flat: np.ndarray, n_features: int, num_classes: int) -> np.ndarray:
    """
    contrib_flat: (n_samples, (n_features+1)*num_classes)
    return: (n_samples, n_features, num_classes)  （不包含bias项）
    """
    if contrib_flat.ndim != 2:
        raise ValueError(f"contrib_flat 期望 2D，实际 ndim={contrib_flat.ndim}, shape={contrib_flat.shape}")
    nf1 = n_features + 1
    if contrib_flat.shape[1] != nf1 * num_classes:
        raise ValueError(
            f"pred_contrib 展平维度不匹配：got={contrib_flat.shape[1]}, expected={(nf1 * num_classes)} "
            f"(n_features={n_features}, num_classes={num_classes})"
        )
    contrib_3d = contrib_flat.reshape(contrib_flat.shape[0], nf1, num_classes)
    return contrib_3d[:, :-1, :]


def _beeswarm_offsets(n: int, swarm_levels: int = 6, spread: float = 0.18) -> np.ndarray:
    """
    简单的“蜂群”y偏移：按样本顺序循环分配 offset。
    """
    offsets = np.zeros(n, dtype=np.float32)
    if swarm_levels <= 1:
        return offsets
    # 把 offset 分到若干 levels 上
    levels = np.arange(swarm_levels, dtype=np.float32)
    if swarm_levels > 1:
        levels = (levels / (swarm_levels - 1) - 0.5) * 2.0  # [-1,1]
    offsets_map = levels * (spread / 2.0)
    for k in range(n):
        offsets[k] = offsets_map[k % swarm_levels]
    return offsets


def main():
    parser = argparse.ArgumentParser(description="Plot beeswarm-like SHAP value summary (no shap dependency).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--out_png", type=str, default="", help="输出PNG路径；默认写到 outputs_shap/")
    parser.add_argument("--out_dir", type=str, default="outputs_shap", help="包含 X_{split}.csv / feature_names.json / lightgbm_model.txt")
    parser.add_argument("--lgb_model", type=str, default="checkpoints/lgb_hybrid.txt", help="混合 LightGBM 模型 txt（特征维=Transformer+表格）")

    parser.add_argument("--top_k", type=int, default=25, help="只显示前K个特征")
    parser.add_argument("--swarm_levels", type=int, default=6, help="蜂群y偏移的层数")
    parser.add_argument(
        "--x_scale",
        type=str,
        default="raw",
        choices=["raw", "mean_abs"],
        help="raw：横轴为原始 SHAP（全特征共用一条 x 轴）；"
        "mean_abs：横轴为 SHAP/mean(|SHAP|)，各特征量纲一致，避免大行挤压小行",
    )

    # 多分类的“全局图”选择策略：
    # default：每个样本-特征对取绝对值最大的那个类别的贡献值（避免正负抵消）
    parser.add_argument("--select_mode", type=str, default="max_abs_across_classes", choices=["max_abs_across_classes"])
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    out_dir = project_dir / args.out_dir
    feature_names_path = out_dir / "feature_names.json"
    X_path = out_dir / f"X_{args.split}.csv"

    if args.out_png:
        out_png = Path(args.out_png)
    else:
        out_png = out_dir / f"beeswarm_like_{args.split}_top{args.top_k}.png"

    if not feature_names_path.exists():
        raise FileNotFoundError(f"找不到：{feature_names_path}")
    if not X_path.exists():
        raise FileNotFoundError(f"找不到：{X_path}")

    feature_names = _load_feature_names(feature_names_path)
    X = pd.read_csv(X_path).values.astype(np.float32)

    # 载入 LightGBM
    lgb_path = Path(args.lgb_model)
    if not lgb_path.exists():
        lgb_path = project_dir / args.lgb_model
    if not lgb_path.exists():
        raise FileNotFoundError(f"找不到 LightGBM 模型：{args.lgb_model}")
    booster = lgb.Booster(model_file=str(lgb_path))

    # 贡献值
    contrib_flat = booster.predict(X, pred_contrib=True)
    contrib_flat = np.asarray(contrib_flat, dtype=np.float32)
    n_features = X.shape[1]
    nf1 = n_features + 1
    flat_dim = int(contrib_flat.shape[1])
    if flat_dim % nf1 != 0:
        raise ValueError(
            f"pred_contrib 展平维度无法整除：(n_samples={X.shape[0]}, flat_dim={flat_dim}) "
            f"不是 (n_features+1)*num_classes（n_features={n_features}, nf1={nf1}）。"
        )
    num_classes = flat_dim // nf1

    shap_per_class = _reshape_pred_contrib_multiclass(contrib_flat, n_features=X.shape[1], num_classes=num_classes)
    # shap_per_class: (n_samples, n_features, num_classes)

    # 选择用于绘图的 shap value（每个样本-特征选一个类别）
    if args.select_mode == "max_abs_across_classes":
        abs_vals = np.abs(shap_per_class)
        idx = abs_vals.argmax(axis=2)  # (n_samples, n_features)
        rows = np.arange(shap_per_class.shape[0])[:, None]
        cols = np.arange(shap_per_class.shape[1])[None, :]
        shap_selected = shap_per_class[rows, cols, idx]  # (n_samples, n_features)
    else:
        raise ValueError(f"未知 select_mode={args.select_mode}")

    # 排序
    mean_abs = np.mean(np.abs(shap_selected), axis=0)  # (n_features,)
    order = np.argsort(mean_abs)[::-1]
    top = min(args.top_k, len(feature_names))
    top_idx = order[:top]

    # 颜色：每个特征内 min-max 到 [0,1]，与 SHAP 官方 summary 一致，共用同一 colorbar 含义
    # 可视化：每个特征一个 y-level，横轴始终为同一个 Axes（一条 x 轴）
    plt.figure(figsize=(12, max(6, top * 0.25)))
    ax = plt.gca()

    eps = 1e-12
    for rank, feat_i in enumerate(top_idx):
        vals = shap_selected[:, feat_i].astype(np.float64, copy=False)
        if args.x_scale == "mean_abs":
            denom = float(mean_abs[feat_i]) + eps
            vals = vals / denom
        color_vals = X[:, feat_i]
        c_lo, c_hi = float(np.min(color_vals)), float(np.max(color_vals))
        if c_hi > c_lo:
            c_plot = (color_vals - c_lo) / (c_hi - c_lo)
        else:
            c_plot = np.full_like(color_vals, 0.5, dtype=np.float32)

        # offsets：按 shap 值排序后再分配，让形状更像 beeswarm
        sorted_order = np.argsort(vals)
        offsets = _beeswarm_offsets(len(vals), swarm_levels=args.swarm_levels, spread=0.22)
        y_sorted = np.empty(len(vals), dtype=np.float32)
        y_sorted[sorted_order] = rank + offsets

        ax.scatter(
            vals,
            y_sorted,
            c=c_plot,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=12,
            alpha=0.8,
            linewidths=0,
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(np.arange(top))
    ax.set_yticklabels([_display_name(feature_names[i]) for i in top_idx], fontsize=9)
    if args.x_scale == "mean_abs":
        ax.set_xlabel("SHAP / mean(|SHAP|) per feature")
    else:
        ax.set_xlabel("SHAP value")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()  # 和 SHAP summary plot 一致：重要特征在上方

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Feature value (low to high, within each row)")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    _add_x_axis_impact_arrows(ax)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

