# -*- coding: utf-8 -*-
"""
[补实验 8] 分类级 SHAP 解释 + 错误案例分析

针对审稿意见：
  当前 SHAP 是全局的（图 4），缺少 per-class 解释和误分类样本的实例级解释。
  审稿人希望看到 minority 类（LNG / Other Cargo / Pleasure Craft）的决策依据，
  以及高频混淆对（Cargo↔Container, Fishing↔Reefer）的错误案例。

做了什么：
  1) 用训练好的 hybrid LightGBM + TreeExplainer 计算 SHAP
  2) 对每个类别画 top-10 特征 importance 柱状图（按该类的 mean(|SHAP|) 排序）
  3) 列出 top-K 错误案例：在两个高频混淆对里各挑 5 例，
     输出每例的预测概率分布 + 关键特征贡献

输出：
  results/exp8_per_class_top_features.csv       全部类别的 top-10 特征及其 mean|SHAP|
  results/exp8_per_class_top_features.png       多子图柱状图
  results/exp8_misclassified_cases.csv          高频混淆对的错误样本
  results/exp8_misclassified_topshap_<...>.png  每个错误样本的 top-10 特征贡献

用法：
  pip install shap   # 若未装
  python supplementary/exp8_per_class_shap.py --data_root data --epochs 50
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
    train_lgb_classifier, evaluate_lgb, feature_names_combined,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    raise ImportError("请安装 shap: pip install shap")


def main():
    parser = argparse.ArgumentParser(description="per-class SHAP + 错误案例")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--n_error_per_pair", type=int, default=5,
                        help="每个混淆对挑多少个错误案例做实例级解释")
    parser.add_argument("--out_dir", type=str, default="supplementary/results")
    args = parser.parse_args()

    hp = dict(DEFAULT_HP); hp["epochs"] = args.epochs
    device = get_device(args.no_cuda)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 训练 hybrid 模型 ...")
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
    ft_test = extract_trajectory_fingerprints(
        transformer, idx_test, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], device)
    X_tr = np.hstack([ft_train, X_tab_all[idx_train]])
    X_va = np.hstack([ft_val,   X_tab_all[idx_val]])
    X_te = np.hstack([ft_test,  X_tab_all[idx_test]])
    y_tr = y_all[idx_train]; y_va = y_all[idx_val]; y_te = y_all[idx_test]
    lgb_model = train_lgb_classifier(X_tr, y_tr, X_va, y_va, n_classes, hp)
    metrics = evaluate_lgb(lgb_model, X_te, y_te, n_classes)
    print(f"  Test ACC={metrics['accuracy']:.4f}  Macro-F1={metrics['macro_f1']:.4f}")

    feat_names = feature_names_combined(hp["d_model"])

    # 2) per-class SHAP
    print("\n[2/3] 计算 SHAP（这一步对大数据可能慢，若时间紧可加 --top_k 5）...")
    explainer = shap.TreeExplainer(lgb_model)
    # 多分类：返回 list[(n, n_feat)] 长度=n_classes
    shap_values = explainer.shap_values(X_te)
    if isinstance(shap_values, list):
        sv_per_class = shap_values  # list of arrays
    else:  # 新版 shap 返回 (n, n_feat, n_classes)
        sv_per_class = [shap_values[:, :, c] for c in range(shap_values.shape[2])]

    # 每个类的 top-K 特征（按 mean|SHAP|）
    rows = []
    class_names = DEFAULT_CLASS_NAMES[:n_classes] + [f"class_{i}" for i in range(len(DEFAULT_CLASS_NAMES), n_classes)]
    for c in range(n_classes):
        mean_abs = np.mean(np.abs(sv_per_class[c]), axis=0)
        order = np.argsort(mean_abs)[::-1][:args.top_k]
        for rank, idx in enumerate(order):
            rows.append({
                "class_id": c,
                "class_name": class_names[c],
                "rank": rank + 1,
                "feature_idx": int(idx),
                "feature_name": feat_names[idx] if idx < len(feat_names) else f"f{idx}",
                "mean_abs_shap": float(mean_abs[idx]),
            })
    df_shap = pd.DataFrame(rows)
    df_shap.to_csv(out_dir / "exp8_per_class_top_features.csv", index=False)
    print(f"[已保存] {out_dir / 'exp8_per_class_top_features.csv'}")

    # 多子图柱状图：每类一个子图
    n_cols = 4
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.array(axes).reshape(-1)
    for c in range(n_classes):
        ax = axes[c]
        sub = df_shap[df_shap["class_id"] == c].sort_values("rank")
        ax.barh(sub["feature_name"][::-1], sub["mean_abs_shap"][::-1], color="steelblue")
        ax.set_title(class_names[c], fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
    for c in range(n_classes, len(axes)):
        axes[c].axis("off")
    fig.suptitle(f"Per-class top-{args.top_k} features by mean(|SHAP|)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "exp8_per_class_top_features.png", dpi=150)
    plt.close(fig)
    print(f"[已保存] {out_dir / 'exp8_per_class_top_features.png'}")

    # 3) 错误案例：找混淆矩阵里最容易混的几对，挑误分类样本做实例级 force-plot 替代图
    print("\n[3/3] 错误案例分析 ...")
    pred = metrics["pred"]
    err_mask = pred != y_te
    err_indices = np.where(err_mask)[0]
    # 统计错误对 (true, pred) 频次
    from collections import Counter
    cnt = Counter([(int(y_te[i]), int(pred[i])) for i in err_indices])
    top_pairs = cnt.most_common(3)  # 取前 3 个最频繁的错误对
    print(f"  最频繁错误对：{top_pairs}")

    err_rows = []
    for (t_cls, p_cls), n in top_pairs:
        case_indices = [i for i in err_indices if y_te[i] == t_cls and pred[i] == p_cls]
        # 挑前 n_error_per_pair 个
        for ci in case_indices[: args.n_error_per_pair]:
            sv_true = sv_per_class[t_cls][ci]
            sv_pred = sv_per_class[p_cls][ci]
            top_idx_pred = np.argsort(np.abs(sv_pred))[::-1][:args.top_k]

            err_rows.append({
                "test_idx": int(ci),
                "true_class": class_names[t_cls],
                "pred_class": class_names[p_cls],
                "proba_true": float(metrics["proba"][ci, t_cls]),
                "proba_pred": float(metrics["proba"][ci, p_cls]),
                "top_features_for_pred_class": ", ".join(
                    f"{feat_names[i]}({sv_pred[i]:+.3f})" for i in top_idx_pred[:5]
                ),
            })

            # 画 SHAP 对比柱状图：true class vs pred class 在该样本上的 top features
            top_idx_union = np.argsort(np.maximum(np.abs(sv_pred), np.abs(sv_true)))[::-1][:args.top_k]
            names_u = [feat_names[i] for i in top_idx_union]
            v_true = sv_true[top_idx_union]
            v_pred = sv_pred[top_idx_union]
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(names_u))
            ax.barh(x - 0.2, v_true, 0.4, label=f"contribution to TRUE class ({class_names[t_cls]})")
            ax.barh(x + 0.2, v_pred, 0.4, label=f"contribution to PRED class ({class_names[p_cls]})")
            ax.set_yticks(x); ax.set_yticklabels(names_u, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("SHAP value")
            ax.set_title(f"Misclassified sample (test idx={ci})\nTrue={class_names[t_cls]}, Pred={class_names[p_cls]}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fname = f"exp8_misclassified_topshap_idx{ci}_{class_names[t_cls].replace(' ', '_')}_to_{class_names[p_cls].replace(' ', '_')}.png"
            fig.savefig(out_dir / fname, dpi=150)
            plt.close(fig)

    pd.DataFrame(err_rows).to_csv(out_dir / "exp8_misclassified_cases.csv", index=False)
    print(f"[已保存] {out_dir / 'exp8_misclassified_cases.csv'}")
    print("[完成] 实验 8：per-class SHAP + 错误案例")


if __name__ == "__main__":
    main()
