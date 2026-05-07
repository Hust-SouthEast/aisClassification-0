# -*- coding: utf-8 -*-
"""
[补实验 5] 细化消融：表格特征分组 + fingerprint 维度

针对审稿意见：
  当前 Table 3 仅 3 行（Transformer only / LightGBM only / Full），过粗。
  审稿人希望看到"哪些特征贡献最大"以及"维度选择是否合理"。

做了什么：
  A) 表格特征分组消融（共用一份 fingerprint）：
     - kinematic：SOG/COG 一阶/二阶差分、加速度、转向率
     - geometric：total_dist / straight_dist / detour / hull / bbox / sinuosity / ...
     - statistical：基本统计扩展（mean/std/var/skew/kurt/quantiles）
     - frequency：FFT + 自相关
     - segment：seg1/seg2/seg3 分段统计
     - all_tabular：上面五组合并（=原版的 50+ 表格特征）
     - fingerprint_only：仅 64 维 Transformer fingerprint
     - fingerprint + 单组：fingerprint + 各组合
     - fingerprint + all_tabular：full（论文版）

  B) fingerprint 维度消融：
     d_model ∈ {32, 64, 128, 256}，重新预训练 Transformer，对比性能。
     -> 验证你选 64 是否合理，是否过拟合 / 欠拟合。

输出：
  results/exp5_tab_group_ablation.csv  (表格特征分组)
  results/exp5_dmodel_ablation.csv     (fingerprint 维度)
  results/exp5_tab_group_ablation.png  / exp5_dmodel_ablation.png

用法：
  python supplementary/exp5_feature_group_ablation.py --data_root data --epochs 50
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (
    DEFAULT_HP, set_seed, get_device, build_loaders,
    train_feature_transformer, extract_trajectory_fingerprints,
    train_lgb_classifier, evaluate_lgb,
)
from data_loader import TABULAR_FEATURE_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- 表格特征分组（按 TABULAR_FEATURE_NAMES 的命名规律划分） ----------
def build_feature_groups():
    """根据 TABULAR_FEATURE_NAMES 把列名分到 5 个组。"""
    groups = {
        "kinematic":   [],   # 一/二阶差分 / 加速度 / RoT
        "geometric":   [],   # 距离 / 凸包 / bbox / sinuosity / stop
        "statistical": [],   # 基础统计扩展
        "frequency":   [],   # FFT + autocorr
        "segment":     [],   # seg1/2/3
    }
    for i, name in enumerate(TABULAR_FEATURE_NAMES):
        n = name.lower()
        if any(k in n for k in ["fft", "_ac1", "_ac2", "_ac3", "_ac5"]):
            groups["frequency"].append(i)
        elif n.startswith("seg") or n == "dsog_half_diff":
            groups["segment"].append(i)
        elif any(k in n for k in [
            "total_dist", "straight_dist", "detour_index", "hull_area",
            "bbox", "bendiness", "sinuosity", "stop_points", "stop_time",
        ]):
            groups["geometric"].append(i)
        elif any(k in n for k in [
            "dsog", "dcog", "turn_count", "acc_", "rot_", "curv_",
            "dt_", "duration", "sampling_density", "stationary_ratio", "traj_len",
        ]):
            groups["kinematic"].append(i)
        else:
            # sog_*, cog_* 基础扩展统计
            groups["statistical"].append(i)
    return groups


def slice_by_group(X_tab, group_idx):
    return X_tab[:, group_idx] if group_idx else np.zeros((X_tab.shape[0], 0), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="特征分组 + 维度消融")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--out_dir", type=str, default="supplementary/results")
    parser.add_argument("--skip_dmodel", action="store_true",
                        help="只跑表格特征分组消融，跳过 d_model 网格")
    parser.add_argument("--dmodel_grid", type=int, nargs="+", default=[32, 64, 128, 256])
    args = parser.parse_args()

    hp = dict(DEFAULT_HP); hp["epochs"] = args.epochs
    device = get_device(args.no_cuda)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ============== A) 表格特征分组消融（固定 d_model=64 的 fingerprint） ==============
    print("[A/2] 训练 d_model=64 的 Feature Transformer（一次） ...")
    set_seed(args.seed)
    (
        train_loader, val_loader, _, scaler, n_classes, F, seq_len,
        idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
    ) = build_loaders(args.data_root, hp, seed=args.seed, repair_missing=True)
    y_all = np.array(y_list, dtype=np.int64)
    transformer, _, _ = train_feature_transformer(
        train_loader, val_loader, n_classes, F, device, hp, save_path=None, verbose=False,
    )
    ft_train = extract_trajectory_fingerprints(
        transformer, idx_train, X_list, scaler, seq_len,
        hp["num_segments"], hp["segment_agg"], device)
    ft_val = extract_trajectory_fingerprints(
        transformer, idx_val, X_list, scaler, seq_len,
        hp["num_segments"], hp["segment_agg"], device)
    ft_test = extract_trajectory_fingerprints(
        transformer, idx_test, X_list, scaler, seq_len,
        hp["num_segments"], hp["segment_agg"], device)
    Xt_train, Xt_val, Xt_test = X_tab_all[idx_train], X_tab_all[idx_val], X_tab_all[idx_test]
    y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]

    groups = build_feature_groups()
    print("  特征组划分：")
    for g, idx in groups.items():
        print(f"    {g}: {len(idx)} 维")

    configs = []
    # 单独表格组（不带 fingerprint）
    for g, idx in groups.items():
        if not idx:
            continue
        configs.append((f"tab[{g}]", slice_by_group(Xt_train, idx),
                                     slice_by_group(Xt_val, idx),
                                     slice_by_group(Xt_test, idx)))
    # 全部表格（不带 fingerprint）
    configs.append(("tab[all]", Xt_train, Xt_val, Xt_test))
    # 仅 fingerprint
    configs.append(("fingerprint_only", ft_train, ft_val, ft_test))
    # fingerprint + 单组
    for g, idx in groups.items():
        if not idx:
            continue
        configs.append((f"fp+tab[{g}]",
                        np.hstack([ft_train, slice_by_group(Xt_train, idx)]),
                        np.hstack([ft_val,   slice_by_group(Xt_val,   idx)]),
                        np.hstack([ft_test,  slice_by_group(Xt_test,  idx)])))
    # fingerprint + 全部表格（=论文版 full）
    configs.append(("fp+tab[all] (full)",
                    np.hstack([ft_train, Xt_train]),
                    np.hstack([ft_val,   Xt_val]),
                    np.hstack([ft_test,  Xt_test])))

    rows = []
    print("\n[A] 跑表格特征分组消融 ...")
    for name, Xtr, Xva, Xte in configs:
        m_lgb = train_lgb_classifier(Xtr, y_train, Xva, y_val, n_classes, hp)
        m = evaluate_lgb(m_lgb, Xte, y_test, n_classes)
        rows.append({"config": name, "n_features": Xtr.shape[1],
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        print(f"  {name:35s}  dim={Xtr.shape[1]:3d}  ACC={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")
    df_a = pd.DataFrame(rows)
    df_a.to_csv(out_dir / "exp5_tab_group_ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(df_a))
    ax.bar(x - 0.2, df_a["accuracy"], 0.4, label="Accuracy")
    ax.bar(x + 0.2, df_a["macro_f1"], 0.4, label="Macro-F1")
    ax.set_xticks(x); ax.set_xticklabels(df_a["config"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("Feature group ablation (fingerprint d_model=64)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp5_tab_group_ablation.png", dpi=150)
    plt.close(fig)
    print(f"[已保存] {out_dir / 'exp5_tab_group_ablation.csv'}")

    # ============== B) fingerprint 维度消融 ==============
    if args.skip_dmodel:
        print("[B] 跳过 d_model 消融")
        print("[完成] 实验 5：分组消融（仅 A）")
        return

    print(f"\n[B/2] fingerprint 维度消融：d_model ∈ {args.dmodel_grid}")
    rows_b = []
    for d in args.dmodel_grid:
        # nhead 必须能整除 d_model
        nhead = 4 if d % 4 == 0 else (2 if d % 2 == 0 else 1)
        local_hp = dict(hp); local_hp["d_model"] = d; local_hp["nhead"] = nhead
        print(f"\n  d_model={d}, nhead={nhead}")
        set_seed(args.seed)
        (
            tr, va, _, sc, _, F2, _,
            i_tr, i_va, i_te, Xl, yl, Xt,
        ) = build_loaders(args.data_root, local_hp, seed=args.seed, repair_missing=True)
        tf, _, _ = train_feature_transformer(tr, va, n_classes, F2, device, local_hp,
                                             save_path=None, verbose=False)
        ftr = extract_trajectory_fingerprints(tf, i_tr, Xl, sc, local_hp["seq_len"],
                                              local_hp["num_segments"], local_hp["segment_agg"], device)
        fva = extract_trajectory_fingerprints(tf, i_va, Xl, sc, local_hp["seq_len"],
                                              local_hp["num_segments"], local_hp["segment_agg"], device)
        fte = extract_trajectory_fingerprints(tf, i_te, Xl, sc, local_hp["seq_len"],
                                              local_hp["num_segments"], local_hp["segment_agg"], device)
        ya = np.array(yl, dtype=np.int64)
        Xtr2 = np.hstack([ftr, Xt[i_tr]]); Xva2 = np.hstack([fva, Xt[i_va]]); Xte2 = np.hstack([fte, Xt[i_te]])
        mdl = train_lgb_classifier(Xtr2, ya[i_tr], Xva2, ya[i_va], n_classes, local_hp)
        m = evaluate_lgb(mdl, Xte2, ya[i_te], n_classes)
        rows_b.append({"d_model": d, "nhead": nhead, "n_params_transformer": sum(p.numel() for p in tf.parameters()),
                       "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        print(f"    ACC={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")
    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(out_dir / "exp5_dmodel_ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_b["d_model"], df_b["accuracy"], "o-", label="Accuracy")
    ax.plot(df_b["d_model"], df_b["macro_f1"], "s-", label="Macro-F1")
    ax.set_xscale("log", base=2)
    ax.set_xticks(df_b["d_model"])
    ax.set_xticklabels(df_b["d_model"])
    ax.set_xlabel("d_model (fingerprint dimension)"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fingerprint dimension ablation")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp5_dmodel_ablation.png", dpi=150)
    plt.close(fig)
    print(f"[已保存] {out_dir / 'exp5_dmodel_ablation.csv'}")
    print("[完成] 实验 5：分组消融 + d_model 消融")


if __name__ == "__main__":
    main()
