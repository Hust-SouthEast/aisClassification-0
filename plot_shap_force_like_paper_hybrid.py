#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按论文描述风格绘制 hybrid 模型的 SHAP force plot（不依赖 shap 包）。

实现方式：
- 用 LightGBM pred_contrib=True 得到多分类下每个特征对每个类别的贡献；
- 将 bias 作为 base_value，其余特征贡献作为 shap_values；
- 使用自定义 matplotlib 图形复刻 force plot 的“base -> (正/负贡献) -> final”。
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
from matplotlib.patches import Rectangle  # noqa: E402

from data_loader import TABULAR_FEATURE_NAMES  # noqa: E402


def _load_feature_names(feature_names_json: Path) -> list[str]:
    with open(feature_names_json, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError(f"feature_names.json 内容不是 list：{feature_names_json}")
    return [str(x) for x in names]


def _get_ft_count(feature_names: list[str]) -> int:
    # 约定：Transformer 深度特征命名为 ft_*
    ft_cnt = 0
    for n in feature_names:
        if n.startswith("ft_"):
            ft_cnt += 1
        else:
            break
    return ft_cnt


def _choose_sample_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_index: int,
    cargo_id: int,
    require_correct: bool,
) -> int:
    if not require_correct:
        return sample_index
    if int(y_true[sample_index]) == cargo_id and int(y_pred[sample_index]) == cargo_id:
        return sample_index
    # 自动找一个正确分类的 cargo
    for i in range(len(y_true)):
        if int(y_true[i]) == cargo_id and int(y_pred[i]) == cargo_id:
            return i
    raise RuntimeError("在当前 split 中找不到正确分类的 cargo 样本，无法画 force plot。")


def _compute_class_contrib(booster: lgb.Booster, x_row: np.ndarray, class_id: int) -> tuple[float, np.ndarray]:
    """
    返回：
    - base_value：bias 项（该类的 base logit）
    - shap_vec：形状 (n_features,) 的贡献（去掉 bias）
    """
    # (1, (n_features+1)*num_classes)；最后一项为 bias
    contrib_flat = booster.predict(x_row.reshape(1, -1), pred_contrib=True)
    contrib_flat = np.asarray(contrib_flat)
    if contrib_flat.ndim != 2 or contrib_flat.shape[0] != 1:
        raise ValueError(f"pred_contrib 输出异常：shape={contrib_flat.shape}")
    n_features = x_row.shape[0]
    num_classes = int(contrib_flat.shape[1] / (n_features + 1))

    contrib_3d = contrib_flat.reshape(1, n_features + 1, num_classes)
    class_vec = contrib_3d[0, :, class_id]
    base_value = float(class_vec[-1])
    shap_vec = class_vec[:-1].astype(np.float64)
    return base_value, shap_vec


def _force_like_plot(
    feature_names: list[str],
    ft_cnt: int,
    x_row: np.ndarray,
    base_value: float,
    shap_vec: np.ndarray,
    top_k: int,
    force_include_indices: list[int],
    title: str,
    out_path: Path,
):
    num_features = shap_vec.shape[0]
    if num_features != len(feature_names):
        raise ValueError(f"特征维度不一致：len(feature_names)={len(feature_names)}, len(shap_vec)={num_features}")

    # 选择要显示的特征：强制包含指定 Transformer embedding 维度
    idx_set = set()
    for idx in force_include_indices:
        if 0 <= idx < num_features:
            idx_set.add(idx)

    abs_sorted = np.argsort(np.abs(shap_vec))[::-1]
    for idx in abs_sorted:
        if len(idx_set) >= top_k:
            break
        idx_set.add(int(idx))
    display_idx = sorted(list(idx_set))

    # 剩余项合并，保证 base + sum(display) == final
    full_sum = float(np.sum(shap_vec))
    display_sum = float(np.sum(shap_vec[display_idx])) if display_idx else 0.0
    remaining = full_sum - display_sum

    remaining_idx = None
    if abs(remaining) > 1e-12:
        remaining_idx = -1

    # 构造展示用列表（先负后正）
    items_pos = []
    items_neg = []
    for idx in display_idx:
        v = float(shap_vec[idx])
        if v >= 0:
            items_pos.append((idx, v))
        else:
            items_neg.append((idx, v))

    if remaining_idx is not None:
        if remaining >= 0:
            items_pos.append((remaining_idx, remaining))
        else:
            items_neg.append((remaining_idx, remaining))

    # 负向：按值从小到大（更负先画到更左）
    items_neg.sort(key=lambda t: t[1])
    # 正向：按值从大到小（更正先画到更右）
    items_pos.sort(key=lambda t: t[1], reverse=True)

    final_value = base_value + full_sum

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    y = 0.8
    height = 0.25

    # baseline
    ax.axvline(base_value, color="gray", linestyle="--", linewidth=1)

    # draw negatives (left)
    cursor = base_value
    for idx, v in items_neg:
        new_cursor = cursor + v  # v < 0
        x_left = min(cursor, new_cursor)
        width = abs(new_cursor - cursor)
        ax.add_patch(Rectangle((x_left, y), width, height, color="#3B82F6", alpha=0.8))
        label = "other" if idx == remaining_idx else feature_names[idx]
        ax.text(x_left + width / 2, y + height + 0.02, label, ha="center", va="bottom", fontsize=8, rotation=0)
        cursor = new_cursor

    # draw positives (right)
    cursor = base_value
    for idx, v in items_pos:
        new_cursor = cursor + v  # v >= 0
        x_left = cursor
        width = abs(new_cursor - cursor)
        ax.add_patch(Rectangle((x_left, y), width, height, color="#EF4444", alpha=0.8))
        label = "other" if idx == remaining_idx else feature_names[idx]
        ax.text(x_left + width / 2, y + height + 0.02, label, ha="center", va="bottom", fontsize=8, rotation=0)
        cursor = new_cursor

    # base/final text
    ax.text(base_value, y - 0.25, f"base={base_value:.3f}", ha="center", va="top", fontsize=9, color="gray")
    ax.text(final_value, y - 0.25, f"final={final_value:.3f}", ha="center", va="top", fontsize=9, color="black")

    # transform labels like Trans_emb_14 / Trans_emb_62
    # 这里只做视觉替换：把 ft_i 改成 Trans_emb_i
    xtick_min = min(base_value, final_value, cursor)
    xtick_max = max(base_value, final_value, cursor)
    ax.set_xlim(xtick_min - 0.5 * (abs(xtick_max - xtick_min) + 1e-6), xtick_max + 0.5 * (abs(xtick_max - xtick_min) + 1e-6))

    ax.set_title(title, fontsize=12)

    # 小图例
    ax.plot([], [], color="#3B82F6", linewidth=6, label="negative")
    ax.plot([], [], color="#EF4444", linewidth=6, label="positive")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot force-like SHAP contributions for hybrid LightGBM (no shap dependency).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--out_png", type=str, default="", help="输出 PNG 路径；默认写到 outputs_shap/")
    parser.add_argument("--out_dir", type=str, default="outputs_shap", help="包含 X_{split}.csv / y_{split}.csv / feature_names.json / lightgbm_model.txt")

    parser.add_argument("--lgb_model", type=str, default="checkpoints/lgb_hybrid.txt", help="混合阶段训练好的 LightGBM 模型")

    parser.add_argument("--sample_index", type=int, default=23, help="test set 内的样本行号（基于 X_{split}.csv 行）")
    parser.add_argument("--cargo_id", type=int, default=13, help="cargo ship 类别 id")
    parser.add_argument("--require_correct", action="store_true", help="若 sample_index 未正确分类 cargo，则自动找一个正确分类的 cargo")

    parser.add_argument("--top_k", type=int, default=10, help="force plot 显示的最多特征数（含强制包含的维度）")
    parser.add_argument("--force_include_trans_emb", type=str, default="14,62", help="必须显示的 Transformer embedding 维度列表，如：14,62")

    args = parser.parse_args()

    project_dir = Path(__file__).parent
    data_dir = project_dir / args.out_dir
    split = args.split

    feature_names_path = data_dir / "feature_names.json"
    X_path = data_dir / f"X_{split}.csv"
    y_path = data_dir / f"y_{split}.csv"

    if not feature_names_path.exists():
        raise FileNotFoundError(f"找不到：{feature_names_path}")
    if not X_path.exists():
        raise FileNotFoundError(f"找不到：{X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"找不到：{y_path}")

    feature_names = _load_feature_names(feature_names_path)
    ft_cnt = _get_ft_count(feature_names)

    X = pd.read_csv(X_path).values.astype(np.float32)
    y_true = pd.read_csv(y_path)["label"].values.astype(int) if "label" in pd.read_csv(y_path).columns else pd.read_csv(y_path).iloc[:, 0].values.astype(int)

    if args.sample_index < 0 or args.sample_index >= len(y_true):
        raise ValueError(f"--sample_index={args.sample_index} 超出范围 0..{len(y_true)-1}")

    lgb_path = Path(args.lgb_model)
    if not lgb_path.exists():
        # 允许相对路径（相对项目根）
        cand = project_dir / args.lgb_model
        if cand.exists():
            lgb_path = cand
        else:
            raise FileNotFoundError(f"找不到 LightGBM 模型：{args.lgb_model}（也未找到 {cand}）")

    booster = lgb.Booster(model_file=str(lgb_path))
    pred_proba = booster.predict(X)
    y_pred = np.argmax(pred_proba, axis=1).astype(int)
    num_classes = int(pred_proba.shape[1])

    # choose sample
    sample_i = _choose_sample_index(y_true, y_pred, args.sample_index, args.cargo_id, args.require_correct)
    true_c = int(y_true[sample_i])
    pred_c = int(y_pred[sample_i])

    # for correct cargo sample, explain cargo class (true == cargo_id)
    class_to_explain = args.cargo_id if true_c == args.cargo_id else pred_c

    # compute base + contributions for that class
    x_row = X[sample_i].astype(np.float32)
    base_value, shap_vec = _compute_class_contrib(booster, x_row, class_to_explain)

    # Force include transformer dims: map trans emb index -> ft index
    force_idx = []
    for s in args.force_include_trans_emb.split(","):
        s = s.strip()
        if not s:
            continue
        force_idx.append(int(s))
    # ft indices correspond directly because transformer feature names are ft_i
    # (0..ft_cnt-1); for safety clamp
    force_idx = [i for i in force_idx if 0 <= i < ft_cnt]

    trans_labels = [f"Trans_emb_{i}" for i in force_idx]

    if args.out_png:
        out_png = Path(args.out_png)
    else:
        out_png = data_dir / f"shap_force_like_sample{sample_i}_cargo{args.cargo_id}.png"

    # temporarily replace ft_i labels for readability
    display_feature_names = feature_names.copy()
    for i in range(ft_cnt):
        display_feature_names[i] = f"Trans_emb_{i}"

    title = (
        f"Force-like SHAP (hybrid LightGBM) | sample_index={sample_i}\n"
        f"true={true_c}, pred={pred_c}, explain_class={class_to_explain} | forced={','.join(trans_labels)}"
    )

    _force_like_plot(
        feature_names=display_feature_names,
        ft_cnt=ft_cnt,
        x_row=x_row,
        base_value=base_value,
        shap_vec=shap_vec,
        top_k=args.top_k,
        force_include_indices=force_idx,
        title=title,
        out_path=out_png,
    )

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

