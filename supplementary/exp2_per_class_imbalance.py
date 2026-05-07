# -*- coding: utf-8 -*-
"""
[补实验 2] per-class 指标 + 类不平衡缓解策略对比

针对审稿意见：
  "more detailed analysis of minority classes ... per-class analysis,
   an error matrix and a discussion of rare classes"

做了什么：
  1) 在固定 seed 下完整跑 4 种策略：
     (a) baseline                    -> Hybrid 原版（无加权、无重采样）
     (b) class_weight = balanced     -> LightGBM 给 minority 类更高权重
     (c) is_unbalance=True           -> LightGBM 自带的不平衡选项
     (d) SMOTE 过采样                -> 在 (fingerprint+tabular) 拼接特征上做 SMOTE
  2) 输出每个策略的：
     - per-class precision/recall/F1/support 表
     - 混淆矩阵（保存为 png + csv）
     - 每个类别的 F1 对比柱状图，重点突出 LNG/Other Cargo/Pleasure Craft 三类 minority

输出：
  results/exp2_per_class_<strategy>.csv
  results/exp2_confusion_<strategy>.png/.csv
  results/exp2_minority_compare.csv
  results/exp2_per_class_f1_compare.png

用法：
  python supplementary/exp2_per_class_imbalance.py --data_root data --epochs 50
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (
    DEFAULT_HP, DEFAULT_CLASS_NAMES, set_seed, get_device, build_loaders,
    train_feature_transformer, extract_trajectory_fingerprints,
    train_lgb_classifier, evaluate_lgb,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def build_hybrid_features(args, hp, seed, device):
    """跑一次 Transformer 预训练 + 提取 hybrid 特征矩阵，返回所有需要的矩阵。"""
    set_seed(seed)
    (
        train_loader, val_loader, _, scaler, n_classes, F, seq_len,
        idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
    ) = build_loaders(args.data_root, hp, seed=seed, repair_missing=True)
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

    X_train = np.hstack([ft_train, X_tab_all[idx_train]])
    X_val   = np.hstack([ft_val,   X_tab_all[idx_val]])
    X_test  = np.hstack([ft_test,  X_tab_all[idx_test]])
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_all[idx_train], "y_val": y_all[idx_val], "y_test": y_all[idx_test],
        "n_classes": n_classes,
    }


def run_strategy(name, data, hp, n_classes, class_names, out_dir):
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    sample_weight = None
    extra_params = None
    if name == "baseline":
        pass
    elif name == "class_weight_balanced":
        cls_count = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        cls_count[cls_count == 0] = 1.0
        cls_w = (len(y_train) / (n_classes * cls_count))
        sample_weight = cls_w[y_train]
    elif name == "is_unbalance":
        extra_params = {"is_unbalance": True}
    elif name == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError("请先安装 imbalanced-learn: pip install imbalanced-learn")
        # k_neighbors 不能大于 minority 类样本数 - 1
        min_count = int(np.bincount(y_train).min())
        k = max(1, min(5, min_count - 1))
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    else:
        raise ValueError(f"未知策略：{name}")

    lgb_model = train_lgb_classifier(
        X_train, y_train, X_val, y_val, n_classes, hp,
        sample_weight=sample_weight, extra_params=extra_params,
    )
    metrics = evaluate_lgb(lgb_model, X_test, y_test, n_classes)

    # per-class CSV
    rows = []
    for c in range(n_classes):
        rows.append({
            "class_id": c,
            "class_name": class_names[c] if c < len(class_names) else f"class_{c}",
            "precision": metrics["precision_per_class"][c],
            "recall":    metrics["recall_per_class"][c],
            "f1":        metrics["f1_per_class"][c],
            "support":   metrics["support_per_class"][c],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"exp2_per_class_{name}.csv", index=False)

    # confusion matrix
    cm = confusion_matrix(y_test, metrics["pred"], labels=np.arange(n_classes))
    cm_df = pd.DataFrame(cm, index=class_names[:n_classes], columns=class_names[:n_classes])
    cm_df.to_csv(out_dir / f"exp2_confusion_{name}.csv")
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names[:n_classes], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names[:n_classes], fontsize=8)
    for i in range(n_classes):
        for j in range(n_classes):
            v = cm_norm[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {name}\nACC={metrics['accuracy']:.4f}  Macro-F1={metrics['macro_f1']:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"exp2_confusion_{name}.png", dpi=150)
    plt.close(fig)

    return {
        "strategy": name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["f1_per_class"],
        "per_class_recall": metrics["recall_per_class"],
        "per_class_precision": metrics["precision_per_class"],
        "support": metrics["support_per_class"],
    }


def main():
    parser = argparse.ArgumentParser(description="per-class 分析 + 类不平衡缓解")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--out_dir", type=str, default="supplementary/results")
    parser.add_argument("--skip_smote", action="store_true",
                        help="若未安装 imbalanced-learn 可跳过 SMOTE")
    args = parser.parse_args()

    hp = dict(DEFAULT_HP); hp["epochs"] = args.epochs
    device = get_device(args.no_cuda)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | seed={args.seed} | epochs={args.epochs}")

    # 1) 共用一份 hybrid 特征矩阵（同 transformer）
    print("\n[1/2] 训练 Feature Transformer 并构造 hybrid 特征矩阵 ...")
    data = build_hybrid_features(args, hp, args.seed, device)
    n_classes = data["n_classes"]
    class_names = DEFAULT_CLASS_NAMES[:n_classes] + [f"class_{i}" for i in range(len(DEFAULT_CLASS_NAMES), n_classes)]

    strategies = ["baseline", "class_weight_balanced", "is_unbalance"]
    if not args.skip_smote:
        strategies.append("smote")

    # 2) 跑 4 种策略
    print("\n[2/2] 跑各种不平衡缓解策略 ...")
    results = []
    for s in strategies:
        try:
            r = run_strategy(s, data, hp, n_classes, class_names, out_dir)
            print(f"  {s}: ACC={r['accuracy']:.4f}  Macro-F1={r['macro_f1']:.4f}")
            results.append(r)
        except ImportError as e:
            print(f"  跳过 {s}: {e}")

    # 3) per-class F1 对比柱状图
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_classes)
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        ax.bar(x + i * width, r["per_class_f1"], width, label=r["strategy"])
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(class_names[:n_classes], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.0)
    ax.set_title("Per-class F1 across imbalance-handling strategies")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_per_class_f1_compare.png", dpi=150)
    plt.close(fig)

    # 4) minority 类（按 support 升序前 3）对比 CSV
    if results:
        supports = np.asarray(results[0]["support"])
        minority_ids = np.argsort(supports)[:3]  # 取最少的 3 类
        rows = []
        for cid in minority_ids:
            for r in results:
                rows.append({
                    "class_id": int(cid),
                    "class_name": class_names[cid],
                    "support": int(supports[cid]),
                    "strategy": r["strategy"],
                    "precision": r["per_class_precision"][cid],
                    "recall":    r["per_class_recall"][cid],
                    "f1":        r["per_class_f1"][cid],
                })
        pd.DataFrame(rows).to_csv(out_dir / "exp2_minority_compare.csv", index=False)
        print(f"[已保存] {out_dir / 'exp2_minority_compare.csv'}")

    # 5) 总览表
    overview = pd.DataFrame([
        {"strategy": r["strategy"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
        for r in results
    ])
    overview.to_csv(out_dir / "exp2_overview.csv", index=False)
    print(f"\n=== 策略总览 ===\n{overview.to_string(index=False)}")
    print("\n[完成] 实验 2：per-class + 类不平衡缓解")


if __name__ == "__main__":
    main()
