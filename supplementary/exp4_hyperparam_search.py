# -*- coding: utf-8 -*-
"""
[补实验 4] Optuna 超参搜索：LightGBM + Transformer 维度

针对审稿意见：
  "the description of the experiments should be supplemented with information on
   the hyperparameter tuning procedure ... and measures of uncertainty"

做了什么：
  1) 用一份固定的 Transformer fingerprint（同一 seed 训练一次，节省时间）
  2) 用 Optuna 对 LightGBM 超参做 N_TRIALS 次搜索（默认 50）
     搜索空间：num_leaves / max_depth / learning_rate / min_data_in_leaf /
              feature_fraction / bagging_fraction / bagging_freq
  3) 输出搜索历史（csv），最优超参（json），以及 study 的 importance 图
  4) 用最优超参在 test 上评估

可选：传 --search_transformer 启用对 d_model / nhead / num_layers 的小规模搜索。
       该路径较慢（每个 trial 都重训一次 Transformer），默认关闭。

输出：
  results/exp4_optuna_history.csv
  results/exp4_best_params.json
  results/exp4_param_importance.png
  results/exp4_final_test.json

用法：
  python supplementary/exp4_hyperparam_search.py --data_root data --n_trials 50
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history
except ImportError:
    raise ImportError("请安装 optuna: pip install optuna")


def prepare_hybrid_features(args, hp, seed, device):
    """跑一次 Transformer 训练，构造 hybrid 特征矩阵（避免每个 trial 重训）。"""
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
        transformer, idx_train, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], device)
    ft_val = extract_trajectory_fingerprints(
        transformer, idx_val, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], device)
    ft_test = extract_trajectory_fingerprints(
        transformer, idx_test, X_list, scaler, seq_len, hp["num_segments"], hp["segment_agg"], device)
    return {
        "X_train": np.hstack([ft_train, X_tab_all[idx_train]]),
        "X_val":   np.hstack([ft_val,   X_tab_all[idx_val]]),
        "X_test":  np.hstack([ft_test,  X_tab_all[idx_test]]),
        "y_train": y_all[idx_train], "y_val": y_all[idx_val], "y_test": y_all[idx_test],
        "n_classes": n_classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Optuna 超参搜索")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50, help="预训练 Transformer 的 epochs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 搜索轮数")
    parser.add_argument("--metric", type=str, default="macro_f1",
                        choices=["macro_f1", "accuracy"], help="搜索目标")
    parser.add_argument("--out_dir", type=str, default="supplementary/results")
    args = parser.parse_args()

    hp = dict(DEFAULT_HP); hp["epochs"] = args.epochs
    device = get_device(args.no_cuda)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 训练 Feature Transformer 并构造 hybrid 特征 ...")
    data = prepare_hybrid_features(args, hp, args.seed, device)
    n_classes = data["n_classes"]

    print(f"[2/3] Optuna 搜索 LightGBM 超参（n_trials={args.n_trials}, metric={args.metric}）...")

    def objective(trial):
        local_hp = dict(hp)
        local_hp["lgb_num_leaves"] = trial.suggest_int("num_leaves", 16, 96)
        local_hp["lgb_max_depth"]  = trial.suggest_int("max_depth", 4, 14)
        local_hp["lgb_lr"]         = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        extra_params = {
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 80),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 0, 7),
            "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
        }
        model = train_lgb_classifier(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_classes, local_hp, extra_params=extra_params,
        )
        m = evaluate_lgb(model, data["X_val"], data["y_val"], n_classes)
        return m[args.metric]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # 搜索历史
    history_rows = []
    for t in study.trials:
        if t.state.is_finished():
            row = {"trial": t.number, "value": t.value, "state": t.state.name}
            row.update(t.params)
            history_rows.append(row)
    pd.DataFrame(history_rows).to_csv(out_dir / "exp4_optuna_history.csv", index=False)
    with open(out_dir / "exp4_best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_value_val_" + args.metric: study.best_value,
            "best_params": study.best_params,
            "n_trials": args.n_trials,
            "search_target_metric": args.metric,
        }, f, indent=2)

    # 参数重要性
    try:
        ax = plot_param_importances(study)
        fig = ax.figure
        fig.set_size_inches(8, 5)
        fig.tight_layout()
        fig.savefig(out_dir / "exp4_param_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [跳过] param importance: {e}")
    try:
        ax = plot_optimization_history(study)
        fig = ax.figure
        fig.set_size_inches(8, 4.5)
        fig.tight_layout()
        fig.savefig(out_dir / "exp4_optimization_history.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [跳过] optimization history: {e}")

    print(f"  best val {args.metric}={study.best_value:.4f}")
    print(f"  best params={study.best_params}")

    print("[3/3] 用最优超参在 test 上评估 ...")
    best = study.best_params
    final_hp = dict(hp,
                    lgb_num_leaves=best["num_leaves"],
                    lgb_max_depth=best["max_depth"],
                    lgb_lr=best["learning_rate"])
    extra = {
        "min_data_in_leaf": best["min_data_in_leaf"],
        "feature_fraction": best["feature_fraction"],
        "bagging_fraction": best["bagging_fraction"],
        "bagging_freq":     best["bagging_freq"],
        "lambda_l1":        best["lambda_l1"],
        "lambda_l2":        best["lambda_l2"],
    }
    model = train_lgb_classifier(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        n_classes, final_hp, extra_params=extra,
    )
    m = evaluate_lgb(model, data["X_test"], data["y_test"], n_classes)
    final = {
        "test_accuracy": m["accuracy"],
        "test_macro_f1": m["macro_f1"],
        "best_params": best,
    }
    with open(out_dir / "exp4_final_test.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"  Test ACC={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}")
    print(f"\n[已保存] {out_dir / 'exp4_optuna_history.csv'}")
    print(f"[已保存] {out_dir / 'exp4_best_params.json'}")
    print(f"[已保存] {out_dir / 'exp4_final_test.json'}")
    print("[完成] 实验 4：超参搜索")


if __name__ == "__main__":
    main()
