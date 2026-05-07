# -*- coding: utf-8 -*-
"""
[补实验 7] 噪声 / 丢点鲁棒性

针对审稿意见：
  "handling of missing values and noise" 描述太简略；缺失值/噪声的影响没量化。

做了什么：
  在测试集（不在训练集！）上注入两类扰动，对训练好的 hybrid 模型直接评估：
  A) 高斯噪声：分别给 SOG / COG / 经纬度按 σ ∈ {1%, 5%, 10%, 20%} 加噪
  B) 随机丢点：drop_ratio ∈ {5%, 10%, 20%, 30%}（被丢的位置用 0 填）

记录每种扰动下的 Accuracy / Macro-F1 衰减曲线。

输出：
  results/exp7_noise_robustness.csv
  results/exp7_noise_robustness.png

用法：
  python supplementary/exp7_noise_robustness.py --data_root data --epochs 50
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from common_utils import (
    DEFAULT_HP, set_seed, get_device, build_loaders,
    train_feature_transformer, extract_trajectory_fingerprints,
    train_lgb_classifier, evaluate_lgb,
)
from data_loader import get_trajectory_multi_segments, sequences_to_tabular

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def perturb_trajectory(seq, noise_sigma=None, drop_ratio=None, rng=None):
    """对单条轨迹 (T, 4) 添加扰动。
    seq 列：[lon, lat, sog, cog]
    noise_sigma: dict 或 float。dict 形如 {'lonlat': 0.01, 'sog': 0.05, 'cog': 0.05}
                 (按各特征的尺度乘以原值的相对噪声)
                 float 表示对四维统一按相对值 σ 加噪。
    drop_ratio: 0..1，丢点概率，被丢的整行置零。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    seq = seq.copy().astype(np.float32)
    if noise_sigma is not None:
        if isinstance(noise_sigma, float):
            sigma_vec = np.array([noise_sigma] * 4, dtype=np.float32)
        else:
            sigma_vec = np.array([
                noise_sigma.get("lon", 0.0),
                noise_sigma.get("lat", 0.0),
                noise_sigma.get("sog", 0.0),
                noise_sigma.get("cog", 0.0),
            ], dtype=np.float32)
        # 相对噪声：以每列绝对值的中位数作为 scale，避免 0 值列
        scale = np.maximum(np.abs(np.median(seq, axis=0)), 1e-3)
        noise = rng.normal(0.0, 1.0, size=seq.shape).astype(np.float32) * sigma_vec * scale
        seq = seq + noise
    if drop_ratio and drop_ratio > 0:
        mask = rng.random(len(seq)) < drop_ratio
        seq[mask] = 0.0
    return seq


def evaluate_under_perturbation(transformer, lgb_model, indices, X_list, scaler,
                                seq_len, hp, X_tab_all, y_all, device,
                                perturb_kwargs):
    """在测试集上施加扰动，重新计算 fingerprint + tabular，得到 metrics。"""
    rng = np.random.default_rng(perturb_kwargs.get("seed", 0))
    # 1) 扰动后的轨迹（不动训练集）
    X_perturbed = []
    for i in indices:
        seq_p = perturb_trajectory(
            X_list[i],
            noise_sigma=perturb_kwargs.get("noise_sigma"),
            drop_ratio=perturb_kwargs.get("drop_ratio"),
            rng=rng,
        )
        X_perturbed.append(seq_p)

    # 2) 重新抽取多段 + 归一化
    segs_list = [get_trajectory_multi_segments(seq, seq_len, hp["num_segments"]) for seq in X_perturbed]
    segs = np.stack(segs_list, axis=0)
    n, ns, L, F = segs.shape
    segs = scaler.transform(segs.reshape(-1, F)).reshape(n, ns, L, F).astype(np.float32)
    segs_t = torch.from_numpy(segs).reshape(n * ns, L, F).to(device)
    transformer.eval()
    with torch.no_grad():
        ft = transformer.get_features(segs_t).detach().cpu().numpy()
    ft = ft.reshape(n, ns, -1)
    ft = ft.mean(axis=1) if hp["segment_agg"] == "mean" else ft.max(axis=1)

    # 3) 表格特征：从扰动后的轨迹重算（path_list=None -> 用 arange 时间戳）
    X_tab_pert = sequences_to_tabular(X_perturbed, path_list=None,
                                      max_points_per_trajectory=2048)
    X_hyb = np.hstack([ft, X_tab_pert])
    y = y_all[indices]
    n_classes = lgb_model.params.get("num_class", int(y.max() + 1)) if hasattr(lgb_model, "params") else int(y.max() + 1)
    proba = lgb_model.predict(X_hyb)
    pred = np.argmax(proba, axis=1)
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="噪声/丢点鲁棒性")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--out_dir", type=str, default="supplementary/results")
    args = parser.parse_args()

    hp = dict(DEFAULT_HP); hp["epochs"] = args.epochs
    device = get_device(args.no_cuda)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] 训练 hybrid 模型 ...")
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
        transformer, idx_train, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], device)
    ft_val = extract_trajectory_fingerprints(
        transformer, idx_val, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], device)
    lgb_model = train_lgb_classifier(
        np.hstack([ft_train, X_tab_all[idx_train]]), y_all[idx_train],
        np.hstack([ft_val,   X_tab_all[idx_val]]),   y_all[idx_val],
        n_classes, hp,
    )
    # clean baseline
    ft_test = extract_trajectory_fingerprints(
        transformer, idx_test, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], device)
    clean = evaluate_lgb(lgb_model, np.hstack([ft_test, X_tab_all[idx_test]]),
                         y_all[idx_test], n_classes)
    print(f"  Clean: ACC={clean['accuracy']:.4f}  F1={clean['macro_f1']:.4f}")

    print("\n[2/2] 注入扰动 ...")
    rows = [{"perturbation": "clean", "level": 0.0,
             "accuracy": clean["accuracy"], "macro_f1": clean["macro_f1"]}]

    # A) 高斯噪声（统一相对 σ，作用在 SOG/COG 和 lon/lat 上）
    for sigma in [0.01, 0.05, 0.10, 0.20]:
        m = evaluate_under_perturbation(
            transformer, lgb_model, idx_test, X_list, scaler, seq_len,
            hp, X_tab_all, y_all, device,
            {"noise_sigma": {"lon": sigma, "lat": sigma, "sog": sigma, "cog": sigma},
             "seed": args.seed},
        )
        rows.append({"perturbation": f"gaussian_noise", "level": sigma,
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        print(f"  noise σ={sigma:.2f}: ACC={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")

    # B) 丢点
    for ratio in [0.05, 0.10, 0.20, 0.30]:
        m = evaluate_under_perturbation(
            transformer, lgb_model, idx_test, X_list, scaler, seq_len,
            hp, X_tab_all, y_all, device,
            {"drop_ratio": ratio, "seed": args.seed},
        )
        rows.append({"perturbation": f"point_drop", "level": ratio,
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        print(f"  drop={ratio:.2f}: ACC={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp7_noise_robustness.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
        sub_n = df[df["perturbation"] == "gaussian_noise"]
        sub_d = df[df["perturbation"] == "point_drop"]
        clean_v = df[df["perturbation"] == "clean"][metric].iloc[0]
        ax.axhline(clean_v, color="gray", linestyle="--", alpha=0.6, label=f"clean ({clean_v:.3f})")
        ax.plot(sub_n["level"], sub_n[metric], "o-", label="Gaussian noise σ")
        ax.plot(sub_d["level"], sub_d[metric], "s-", label="Random point drop ratio")
        ax.set_xlabel("Perturbation level"); ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3); ax.legend()
        ax.set_title(f"Robustness — {metric}")
    fig.tight_layout()
    fig.savefig(out_dir / "exp7_noise_robustness.png", dpi=150)
    plt.close(fig)
    print(f"\n[已保存] {out_dir / 'exp7_noise_robustness.csv'}")
    print(f"[已保存] {out_dir / 'exp7_noise_robustness.png'}")
    print("[完成] 实验 7：噪声/丢点鲁棒性")


if __name__ == "__main__":
    main()
