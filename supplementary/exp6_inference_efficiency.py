# -*- coding: utf-8 -*-
"""
[补实验 6] 推理效率：batch_size 扫描 + CPU vs GPU + 模型大小

针对审稿人对"real-time / edge deployment"主张的关切——
当前论文只有 1.58 ms/batch 一个数字，太单薄。

做了什么：
  1) 对 hybrid 完整推理流程分别测：batch_size ∈ {1, 8, 32, 128, 512}
     - Transformer fingerprint 提取耗时
     - LightGBM 分类耗时
     - 端到端总耗时（fingerprint + LGBM）
  2) 同时在 CPU / GPU（如有）上各跑一次
  3) 报告：
     - ms/batch、ms/sample、samples/sec（吞吐量）
     - 模型参数量、磁盘占用（Transformer .pt + LightGBM .txt）

注意：每个 batch_size 下重复 200 次（去掉前 20 次预热），
      用 perf_counter，避免 cold-start 偏差。

输出：
  results/exp6_inference_efficiency.csv
  results/exp6_model_size.json
  results/exp6_inference_efficiency.png

用法：
  python supplementary/exp6_inference_efficiency.py --data_root data --epochs 30
"""
import os
import sys
import time
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
from data_loader import get_trajectory_multi_segments

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def measure_latency_fn(fn, n_repeat=200, n_warmup=20):
    """重复执行 fn，返回平均耗时（秒）。"""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeat


def main():
    parser = argparse.ArgumentParser(description="推理效率")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--n_repeat", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="supplementary/results")
    parser.add_argument("--cpu_only", action="store_true",
                        help="只测 CPU；默认会同时测 CPU + GPU(若可用)")
    args = parser.parse_args()

    hp = dict(DEFAULT_HP); hp["epochs"] = args.epochs
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_device = get_device(no_cuda=False)
    print(f"训练设备: {train_device}")

    # 1) 训练一次，得到 transformer + LightGBM
    print("[1/3] 训练 hybrid 模型 ...")
    set_seed(args.seed)
    (
        train_loader, val_loader, _, scaler, n_classes, F, seq_len,
        idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
    ) = build_loaders(args.data_root, hp, seed=args.seed, repair_missing=True)
    y_all = np.array(y_list, dtype=np.int64)
    transformer, _, _ = train_feature_transformer(
        train_loader, val_loader, n_classes, F, train_device, hp,
        save_path=str(out_dir / "exp6_transformer.pt"), verbose=False,
    )
    ft_train = extract_trajectory_fingerprints(
        transformer, idx_train, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], train_device)
    ft_val = extract_trajectory_fingerprints(
        transformer, idx_val, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], train_device)
    ft_test = extract_trajectory_fingerprints(
        transformer, idx_test, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], train_device)
    X_test_hyb = np.hstack([ft_test, X_tab_all[idx_test]])
    lgb_model = train_lgb_classifier(
        np.hstack([ft_train, X_tab_all[idx_train]]), y_all[idx_train],
        np.hstack([ft_val,   X_tab_all[idx_val]]),   y_all[idx_val],
        n_classes, hp,
    )
    lgb_path = out_dir / "exp6_lgb.txt"
    lgb_model.save_model(str(lgb_path))
    final_acc = evaluate_lgb(lgb_model, X_test_hyb, y_all[idx_test], n_classes)
    print(f"  [check] test ACC={final_acc['accuracy']:.4f} Macro-F1={final_acc['macro_f1']:.4f}")

    # 模型大小
    transformer_params = sum(p.numel() for p in transformer.parameters())
    transformer_size_kb = (out_dir / "exp6_transformer.pt").stat().st_size / 1024
    lgb_size_kb = lgb_path.stat().st_size / 1024
    size_info = {
        "transformer_params": int(transformer_params),
        "transformer_disk_KB": float(transformer_size_kb),
        "lightgbm_disk_KB": float(lgb_size_kb),
        "total_disk_KB": float(transformer_size_kb + lgb_size_kb),
    }
    with open(out_dir / "exp6_model_size.json", "w", encoding="utf-8") as f:
        json.dump(size_info, f, indent=2)
    print(f"  Transformer 参数量={transformer_params:,}  磁盘={transformer_size_kb:.1f} KB")
    print(f"  LightGBM 磁盘={lgb_size_kb:.1f} KB")

    # 2) 准备测试样本（取测试集所有轨迹的多段序列）
    print("\n[2/3] 准备推理输入 ...")
    test_segs_list = [get_trajectory_multi_segments(X_list[i], seq_len, hp["num_segments"]) for i in idx_test]
    test_segs = np.stack(test_segs_list, axis=0)  # (n, ns, L, F)
    n_traj, ns, L, F2 = test_segs.shape
    test_segs_norm = scaler.transform(test_segs.reshape(-1, F2)).reshape(test_segs.shape).astype(np.float32)

    devices = [("cpu", torch.device("cpu"))]
    if torch.cuda.is_available() and not args.cpu_only:
        devices.append(("cuda", torch.device("cuda")))

    rows = []
    print("\n[3/3] 测推理延迟 ...")
    for dev_name, dev in devices:
        # 把模型放到目标设备
        tf = transformer.to(dev).eval()
        for B in args.batch_sizes:
            # 取前 B 条轨迹（不够则循环填充）
            if n_traj >= B:
                idx_pick = np.arange(B)
            else:
                idx_pick = np.tile(np.arange(n_traj), B // n_traj + 1)[:B]
            segs_b = test_segs_norm[idx_pick]              # (B, ns, L, F)
            segs_flat_b = segs_b.reshape(B * ns, L, F2)
            x_t = torch.from_numpy(segs_flat_b).to(dev)
            x_tab_b = X_tab_all[idx_test][idx_pick]

            # ---- a) Transformer 提取 fingerprint ----
            def fn_trans():
                with torch.no_grad():
                    h = tf.get_features(x_t)
                    h = h.detach().cpu().numpy().reshape(B, ns, -1).mean(axis=1)
                return h

            t_trans = measure_latency_fn(fn_trans, n_repeat=args.n_repeat, n_warmup=10)

            # 预先得到 fingerprint，再单独测 LGBM
            with torch.no_grad():
                h = tf.get_features(x_t).detach().cpu().numpy().reshape(B, ns, -1).mean(axis=1)
            X_b = np.hstack([h, x_tab_b]).astype(np.float64)

            # ---- b) LightGBM 分类 ----
            def fn_lgb():
                lgb_model.predict(X_b)
            t_lgb = measure_latency_fn(fn_lgb, n_repeat=args.n_repeat, n_warmup=10)

            # ---- c) 端到端 ----
            def fn_e2e():
                with torch.no_grad():
                    hh = tf.get_features(x_t).detach().cpu().numpy().reshape(B, ns, -1).mean(axis=1)
                Xb = np.hstack([hh, x_tab_b]).astype(np.float64)
                lgb_model.predict(Xb)
            t_e2e = measure_latency_fn(fn_e2e, n_repeat=args.n_repeat, n_warmup=10)

            rows.append({
                "device": dev_name,
                "batch_size": B,
                "transformer_ms_per_batch": t_trans * 1000,
                "lightgbm_ms_per_batch":    t_lgb * 1000,
                "e2e_ms_per_batch":         t_e2e * 1000,
                "e2e_ms_per_sample":        t_e2e / B * 1000,
                "throughput_samples_per_sec": B / t_e2e,
            })
            print(f"  [{dev_name}] B={B:4d}  trans={t_trans*1000:7.3f} ms  "
                  f"lgb={t_lgb*1000:7.3f} ms  e2e={t_e2e*1000:7.3f} ms  "
                  f"({B/t_e2e:.0f} samples/s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp6_inference_efficiency.csv", index=False)
    print(f"\n[已保存] {out_dir / 'exp6_inference_efficiency.csv'}")

    # 折线图：不同 batch_size 下的 ms/sample
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for dev_name in df["device"].unique():
        sub = df[df["device"] == dev_name]
        axes[0].plot(sub["batch_size"], sub["e2e_ms_per_sample"], "o-", label=dev_name)
        axes[1].plot(sub["batch_size"], sub["throughput_samples_per_sec"], "s-", label=dev_name)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(args.batch_sizes); ax.set_xticklabels(args.batch_sizes)
        ax.grid(True, alpha=0.3); ax.legend()
    axes[0].set_xlabel("Batch size"); axes[0].set_ylabel("ms / sample (E2E)")
    axes[0].set_title("End-to-end latency per sample")
    axes[1].set_xlabel("Batch size"); axes[1].set_ylabel("samples / sec")
    axes[1].set_title("End-to-end throughput")
    fig.tight_layout()
    fig.savefig(out_dir / "exp6_inference_efficiency.png", dpi=150)
    plt.close(fig)
    print(f"[已保存] {out_dir / 'exp6_inference_efficiency.png'}")
    print("[完成] 实验 6：推理效率")


if __name__ == "__main__":
    main()
