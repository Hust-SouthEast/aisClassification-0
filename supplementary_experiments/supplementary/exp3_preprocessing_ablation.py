# -*- coding: utf-8 -*-
"""
[补实验 3] 预处理消融：插值方法 / 序列长度 / 修复开关

针对审稿意见：
  "data pre-processing procedures, handling of missing values and noise,
   exact list of '50+' tabular features, hyperparameter tuning strategy ...
   are described too briefly"

做了什么：
  A) 修复开关：repair_missing = True / False（关掉 scipy 插值，相当于 raw）
  B) 插值方法：linear / nearest / spline-cubic（修改 method 参数）
  C) 序列长度敏感性：seq_len = 64 / 128 / 256

为了控制时间成本，每个配置只跑 1 个 seed；结果用于"敏感性分析"，
而不是用于和 baseline 比强弱（那个已经在 exp1 里做了）。

输出：
  results/exp3_preprocessing_ablation.csv
  results/exp3_preprocessing_ablation.png  (柱状图)

用法：
  python supplementary/exp3_preprocessing_ablation.py --data_root data --epochs 30
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl
from common_utils import (
    DEFAULT_HP, set_seed, get_device, build_loaders,
    train_feature_transformer, extract_trajectory_fingerprints,
    train_lgb_classifier, evaluate_lgb,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def patch_interp_method(method):
    """通过 monkey-patch 让 data_loader.repair_trajectory_with_interpolation
    使用指定的插值方法（linear/nearest/cubic）。
    原函数对非 ('linear','nearest') 的 method 会回退到 'linear'，所以这里给它一个
    更灵活的版本。"""
    orig = dl.repair_trajectory_with_interpolation

    def patched(ts, arr, max_gap_seconds=300, method_arg="linear", min_valid_ratio=0.5):
        from scipy.interpolate import interp1d
        ts = np.asarray(ts, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(arr).all(axis=1)
        if valid.sum() < max(2, int(min_valid_ratio * len(ts))):
            out = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return ts.astype(np.float32), out.astype(np.float32)
        t_valid = ts[valid]
        arr_valid = arr[valid]
        kind = method  # 闭包：用外层 method 覆盖
        for j in range(4):
            col = arr[:, j].copy()
            if np.isfinite(col).sum() < 2:
                col = np.nan_to_num(col, nan=np.nanmean(col) if np.isfinite(col).any() else 0.0)
                arr[:, j] = col
                continue
            try:
                f = interp1d(
                    t_valid, arr_valid[:, j], kind=kind,
                    bounds_error=False,
                    fill_value=(arr_valid[:, j].min(), arr_valid[:, j].max()),
                )
            except ValueError:
                f = interp1d(t_valid, arr_valid[:, j], kind="linear",
                             bounds_error=False,
                             fill_value=(arr_valid[:, j].min(), arr_valid[:, j].max()))
            nan_mask = ~np.isfinite(col)
            if nan_mask.any():
                arr[nan_mask, j] = f(ts[nan_mask])
        if len(ts) >= 2 and max_gap_seconds > 0:
            dtv = np.diff(ts)
            dtv = np.where(dtv <= 0, np.median(dtv[dtv > 0]) if np.any(dtv > 0) else 1.0, dtv)
            step = min(float(np.median(dtv)), max_gap_seconds)
            gaps = np.where(dtv > max_gap_seconds)[0]
            if len(gaps) > 0:
                t_extra = []
                for i in gaps:
                    t_start, t_end = ts[i], ts[i + 1]
                    n_ins = max(1, int((t_end - t_start) / step))
                    t_extra.extend(np.linspace(t_start, t_end, n_ins + 1)[1:-1].tolist())
                if t_extra:
                    t_extra = np.array(t_extra, dtype=np.float64)
                    t_all = np.concatenate([ts, t_extra]); t_all = np.sort(t_all)
                    arr_all = np.zeros((len(t_all), 4), dtype=np.float64)
                    for j in range(4):
                        try:
                            f = interp1d(ts, arr[:, j], kind=kind, bounds_error=False, fill_value="extrapolate")
                        except ValueError:
                            f = interp1d(ts, arr[:, j], kind="linear", bounds_error=False, fill_value="extrapolate")
                        arr_all[:, j] = f(t_all)
                    ts = t_all; arr = arr_all
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return ts.astype(np.float32), arr.astype(np.float32)

    dl.repair_trajectory_with_interpolation = patched
    return orig


def restore_interp(orig):
    dl.repair_trajectory_with_interpolation = orig


def run_one_config(data_root, hp, repair_missing, interp_method, seq_len, seed, device):
    """跑一次完整 hybrid，返回 acc + macro_f1。"""
    hp = dict(hp); hp["seq_len"] = seq_len

    orig = None
    if repair_missing and interp_method != "linear":
        orig = patch_interp_method(interp_method)

    set_seed(seed)
    try:
        (
            train_loader, val_loader, _, scaler, n_classes, F, _,
            idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
        ) = build_loaders(data_root, hp, seed=seed, repair_missing=repair_missing)
        y_all = np.array(y_list, dtype=np.int64)
        transformer, _, _ = train_feature_transformer(
            train_loader, val_loader, n_classes, F, device, hp, save_path=None, verbose=False,
        )
        ft_train = extract_trajectory_fingerprints(
            transformer, idx_train, X_list, scaler,
            seq_len, hp["num_segments"], hp["segment_agg"], device)
        ft_val = extract_trajectory_fingerprints(
            transformer, idx_val, X_list, scaler,
            seq_len, hp["num_segments"], hp["segment_agg"], device)
        ft_test = extract_trajectory_fingerprints(
            transformer, idx_test, X_list, scaler,
            seq_len, hp["num_segments"], hp["segment_agg"], device)
        X_tr = np.hstack([ft_train, X_tab_all[idx_train]])
        X_va = np.hstack([ft_val,   X_tab_all[idx_val]])
        X_te = np.hstack([ft_test,  X_tab_all[idx_test]])
        lgb_model = train_lgb_classifier(
            X_tr, y_all[idx_train], X_va, y_all[idx_val], n_classes, hp,
        )
        m = evaluate_lgb(lgb_model, X_te, y_all[idx_test], n_classes)
    finally:
        if orig is not None:
            restore_interp(orig)

    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


def main():
    parser = argparse.ArgumentParser(description="预处理消融：插值方法/seq_len")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30,
                        help="预处理消融用较短 epochs 即可（30 足够看趋势）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--out_dir", type=str, default="supplementary/results")
    args = parser.parse_args()

    hp = dict(DEFAULT_HP); hp["epochs"] = args.epochs
    device = get_device(args.no_cuda)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    # A. 修复开关
    configs.append(("repair=False (raw)",       dict(repair_missing=False, interp_method="linear", seq_len=128)))
    configs.append(("repair=True linear",        dict(repair_missing=True,  interp_method="linear", seq_len=128)))
    configs.append(("repair=True nearest",       dict(repair_missing=True,  interp_method="nearest", seq_len=128)))
    configs.append(("repair=True cubic",         dict(repair_missing=True,  interp_method="cubic",   seq_len=128)))
    # B. seq_len
    configs.append(("seq_len=64 (linear)",       dict(repair_missing=True,  interp_method="linear", seq_len=64)))
    configs.append(("seq_len=128 (linear)*",     dict(repair_missing=True,  interp_method="linear", seq_len=128)))
    configs.append(("seq_len=256 (linear)",      dict(repair_missing=True,  interp_method="linear", seq_len=256)))

    rows = []
    for name, cfg in configs:
        print(f"\n>>> 配置：{name}  ({cfg})")
        m = run_one_config(args.data_root, hp, **cfg, seed=args.seed, device=device)
        print(f"    ACC={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}")
        rows.append({
            "config": name,
            "repair_missing": cfg["repair_missing"],
            "interp_method": cfg["interp_method"],
            "seq_len": cfg["seq_len"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp3_preprocessing_ablation.csv", index=False)
    print(f"\n[已保存] {out_dir / 'exp3_preprocessing_ablation.csv'}")

    # 柱状图
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["accuracy"], 0.4, label="Accuracy")
    ax.bar(x + 0.2, df["macro_f1"], 0.4, label="Macro-F1")
    ax.set_xticks(x); ax.set_xticklabels(df["config"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("Preprocessing & seq_len ablation")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_preprocessing_ablation.png", dpi=150)
    plt.close(fig)
    print(f"[已保存] {out_dir / 'exp3_preprocessing_ablation.png'}")
    print("[完成] 实验 3：预处理消融")


if __name__ == "__main__":
    main()
