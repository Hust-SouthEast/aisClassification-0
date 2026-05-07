# -*- coding: utf-8 -*-
"""
[补实验 1] 多种子重复 + 不确定性 + 配对显著性检验

针对审稿意见：
  "information on result variance, the number of repetitions, confidence intervals
   or statistical significance tests is missing"

做了什么：
  1) 对每个模型重复 N_RUNS 次（默认 5，可调 10），每次用不同 seed
  2) 报告 Accuracy / Macro-F1 的均值、标准差、95% 置信区间
  3) 在测试集上对 hybrid vs 每个 baseline 做 paired t-test 和 Wilcoxon signed-rank test
     -> 输出 p-value，证明 hybrid 显著优于 baseline

模型清单（最低必须）：
  - Feature-Transformer + LightGBM (full, ours)
  - Feature-Transformer only (transformer 分类头直接预测)
  - LightGBM only (仅 50+ 表格特征)

可选扩展：可在 BASELINES 里加 ResNet+XGBoost / LSTM / GRU 等，
        本脚本聚焦"必须做"的最小可发表集合，跑通后再扩。

输出：
  results/exp1_multi_seed_summary.csv  -> 各模型每个 seed 的 acc/macro_f1
  results/exp1_multi_seed_stats.csv    -> mean ± std + 95% CI + p-value
  results/exp1_multi_seed_raw.json     -> 全部原始指标，便于复现

用法：
  cd aisClassification-0
  python supplementary/exp1_multi_seed.py --data_root data --n_runs 5 --epochs 50
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (
    DEFAULT_HP, set_seed, get_device, build_loaders,
    train_feature_transformer, extract_trajectory_fingerprints,
    train_lgb_classifier, evaluate_lgb,
)


def run_one_seed(data_root, hp, seed, device, verbose=False):
    """一次 seed -> 返回三个模型的指标 + 测试集预测向量（供配对检验）。"""
    set_seed(seed)

    (
        train_loader, val_loader, _, scaler, n_classes, F, seq_len,
        idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
    ) = build_loaders(data_root, hp, seed=seed, repair_missing=True)
    y_all = np.array(y_list, dtype=np.int64)
    y_test = y_all[idx_test]

    # ---- 1) 预训练 Feature Transformer ----
    transformer, _, _ = train_feature_transformer(
        train_loader, val_loader, n_classes, F, device, hp, save_path=None, verbose=verbose,
    )

    # ---- 2) Transformer-only 在测试集上的预测 ----
    import torch
    transformer.eval()
    # 重建一个 test_loader（每条轨迹一个样本，与论文 test 保持一致）
    from torch.utils.data import TensorDataset, DataLoader
    test_seqs = np.stack(
        [_pad_first_seq(X_list[i], seq_len, scaler, F) for i in idx_test], axis=0
    )
    test_ds = TensorDataset(torch.from_numpy(test_seqs.astype(np.float32)),
                            torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=hp["batch_size"], shuffle=False)
    preds_trans, probas_trans = [], []
    with torch.no_grad():
        for x_b, _y in test_loader:
            x_b = x_b.to(device)
            logits = transformer(x_b)
            probas_trans.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds_trans.append(logits.argmax(1).cpu().numpy())
    pred_trans = np.concatenate(preds_trans)
    proba_trans = np.concatenate(probas_trans, axis=0)

    from sklearn.metrics import accuracy_score, f1_score
    metrics_trans = {
        "accuracy": float(accuracy_score(y_test, pred_trans)),
        "macro_f1": float(f1_score(y_test, pred_trans, average="macro", zero_division=0)),
        "pred": pred_trans, "proba": proba_trans,
    }

    # ---- 3) Hybrid: fingerprint + tabular -> LightGBM ----
    ft_train = extract_trajectory_fingerprints(
        transformer, idx_train, X_list, scaler,
        seq_len, hp["num_segments"], hp["segment_agg"], device,
    )
    ft_val = extract_trajectory_fingerprints(
        transformer, idx_val, X_list, scaler,
        seq_len, hp["num_segments"], hp["segment_agg"], device,
    )
    ft_test = extract_trajectory_fingerprints(
        transformer, idx_test, X_list, scaler,
        seq_len, hp["num_segments"], hp["segment_agg"], device,
    )
    X_tab_train, X_tab_val, X_tab_test = X_tab_all[idx_train], X_tab_all[idx_val], X_tab_all[idx_test]
    X_hyb_train = np.hstack([ft_train, X_tab_train])
    X_hyb_val   = np.hstack([ft_val,   X_tab_val])
    X_hyb_test  = np.hstack([ft_test,  X_tab_test])

    lgb_hybrid = train_lgb_classifier(
        X_hyb_train, y_all[idx_train], X_hyb_val, y_all[idx_val],
        n_classes, hp,
    )
    metrics_hybrid = evaluate_lgb(lgb_hybrid, X_hyb_test, y_test, n_classes)

    # ---- 4) LightGBM-only：仅 50+ 表格特征 ----
    lgb_only = train_lgb_classifier(
        X_tab_train, y_all[idx_train], X_tab_val, y_all[idx_val],
        n_classes, hp,
    )
    metrics_lgb = evaluate_lgb(lgb_only, X_tab_test, y_test, n_classes)

    return {
        "hybrid": metrics_hybrid,
        "transformer_only": metrics_trans,
        "lightgbm_only": metrics_lgb,
        "y_test": y_test,
        "n_classes": n_classes,
    }


def _pad_first_seq(seq, seq_len, scaler, F):
    out = np.zeros((seq_len, F), dtype=np.float32)
    T = min(len(seq), seq_len)
    out[:T] = seq[:T]
    out = scaler.transform(out.reshape(-1, F)).reshape(seq_len, F)
    return out


def summarize_runs(model_name, accs, f1s):
    accs = np.asarray(accs); f1s = np.asarray(f1s)
    n = len(accs)
    # 95% CI（基于 t 分布的双尾 95%）
    if n > 1:
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        acc_ci = t_crit * accs.std(ddof=1) / np.sqrt(n)
        f1_ci  = t_crit * f1s.std(ddof=1) / np.sqrt(n)
    else:
        acc_ci = 0.0; f1_ci = 0.0
    return {
        "model": model_name,
        "n_runs": n,
        "acc_mean": float(accs.mean()),
        "acc_std":  float(accs.std(ddof=1)) if n > 1 else 0.0,
        "acc_ci95": float(acc_ci),
        "f1_mean":  float(f1s.mean()),
        "f1_std":   float(f1s.std(ddof=1)) if n > 1 else 0.0,
        "f1_ci95":  float(f1_ci),
    }


def paired_tests(metric_a, metric_b, label):
    """对两个模型在同一组 seed 上做配对检验。"""
    a = np.asarray(metric_a); b = np.asarray(metric_b)
    if len(a) < 2:
        return {"label": label, "n": len(a), "t_stat": np.nan, "t_p": np.nan,
                "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan,
                "diff_mean": float(a.mean() - b.mean())}
    t_stat, t_p = scipy_stats.ttest_rel(a, b)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:  # 全部相等时 wilcoxon 报错
        w_stat, w_p = np.nan, np.nan
    return {
        "label": label, "n": len(a),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
        "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
        "diff_mean": float(a.mean() - b.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="多种子重复 + 显著性检验")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--n_runs", type=int, default=5, help="重复次数（推荐 5 或 10）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--out_dir", type=str, default="supplementary/results")
    args = parser.parse_args()

    hp = dict(DEFAULT_HP)
    hp["epochs"] = args.epochs

    device = get_device(args.no_cuda)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | n_runs={args.n_runs} | epochs={args.epochs}")

    all_runs = []  # list of dict per seed
    seeds = list(range(args.n_runs))  # 0..n-1
    for s in seeds:
        print(f"\n========== Seed {s+1}/{args.n_runs} (seed={s}) ==========")
        out = run_one_seed(args.data_root, hp, seed=s, device=device, verbose=False)
        record = {
            "seed": s,
            "hybrid_acc":         out["hybrid"]["accuracy"],
            "hybrid_macro_f1":    out["hybrid"]["macro_f1"],
            "trans_acc":          out["transformer_only"]["accuracy"],
            "trans_macro_f1":     out["transformer_only"]["macro_f1"],
            "lgb_acc":            out["lightgbm_only"]["accuracy"],
            "lgb_macro_f1":       out["lightgbm_only"]["macro_f1"],
        }
        # 同步保存预测向量，配对检验需要每个样本的命中情况
        record["_hybrid_hits"] = (np.asarray(out["hybrid"]["pred"]) == out["y_test"]).astype(int).tolist()
        record["_trans_hits"]  = (np.asarray(out["transformer_only"]["pred"]) == out["y_test"]).astype(int).tolist()
        record["_lgb_hits"]    = (np.asarray(out["lightgbm_only"]["pred"]) == out["y_test"]).astype(int).tolist()
        all_runs.append(record)
        print(f"  Hybrid: acc={record['hybrid_acc']:.4f} F1={record['hybrid_macro_f1']:.4f}")
        print(f"  Transformer-only: acc={record['trans_acc']:.4f} F1={record['trans_macro_f1']:.4f}")
        print(f"  LightGBM-only: acc={record['lgb_acc']:.4f} F1={record['lgb_macro_f1']:.4f}")

    # ---- 汇总表 ----
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in all_runs])
    df.to_csv(out_dir / "exp1_multi_seed_summary.csv", index=False)
    print(f"\n[已保存] {out_dir / 'exp1_multi_seed_summary.csv'}")

    # ---- mean / std / 95% CI ----
    stats_rows = [
        summarize_runs("Hybrid (Ours)",         df["hybrid_acc"],  df["hybrid_macro_f1"]),
        summarize_runs("Transformer-only",      df["trans_acc"],   df["trans_macro_f1"]),
        summarize_runs("LightGBM-only",         df["lgb_acc"],     df["lgb_macro_f1"]),
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "exp1_multi_seed_stats.csv", index=False)
    print(f"[已保存] {out_dir / 'exp1_multi_seed_stats.csv'}")
    print("\n=== 均值 ± std (95% CI) ===")
    for row in stats_rows:
        print(f"  {row['model']}: ACC={row['acc_mean']:.4f} ± {row['acc_std']:.4f} "
              f"(±{row['acc_ci95']:.4f})  F1={row['f1_mean']:.4f} ± {row['f1_std']:.4f} "
              f"(±{row['f1_ci95']:.4f})")

    # ---- 配对检验 ----
    tests = []
    tests.append(paired_tests(df["hybrid_acc"], df["trans_acc"],
                              "Hybrid vs Transformer-only (accuracy)"))
    tests.append(paired_tests(df["hybrid_macro_f1"], df["trans_macro_f1"],
                              "Hybrid vs Transformer-only (macro_f1)"))
    tests.append(paired_tests(df["hybrid_acc"], df["lgb_acc"],
                              "Hybrid vs LightGBM-only (accuracy)"))
    tests.append(paired_tests(df["hybrid_macro_f1"], df["lgb_macro_f1"],
                              "Hybrid vs LightGBM-only (macro_f1)"))
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(out_dir / "exp1_multi_seed_paired_tests.csv", index=False)
    print(f"[已保存] {out_dir / 'exp1_multi_seed_paired_tests.csv'}")
    print("\n=== 配对显著性检验 (p<0.05 表示 Hybrid 显著优于对比模型) ===")
    for t in tests:
        print(f"  {t['label']}: t={t['t_stat']:.3f} (p={t['t_p']:.4g})  "
              f"Wilcoxon p={t['wilcoxon_p']:.4g}  Δmean={t['diff_mean']:+.4f}")

    # ---- 原始 dump ----
    with open(out_dir / "exp1_multi_seed_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
    print(f"[已保存] {out_dir / 'exp1_multi_seed_raw.json'}")
    print("\n[完成] 实验 1：多种子重复 + 显著性检验")


if __name__ == "__main__":
    main()
