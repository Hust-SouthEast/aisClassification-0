#!/usr/bin/env bash
# 一键跑完所有补实验 (Linux/macOS)
set -e
cd "$(dirname "$0")/.."

echo "===== Exp 1: Multi-seed + significance ====="
python supplementary/exp1_multi_seed.py --data_root data --n_runs 5 --epochs 50

echo "===== Exp 2: Per-class + imbalance ====="
python supplementary/exp2_per_class_imbalance.py --data_root data --epochs 50

echo "===== Exp 3: Preprocessing ablation ====="
python supplementary/exp3_preprocessing_ablation.py --data_root data --epochs 30

echo "===== Exp 4: Hyperparameter search (Optuna) ====="
python supplementary/exp4_hyperparam_search.py --data_root data --n_trials 50 --epochs 50

echo "===== Exp 5: Feature group + d_model ablation ====="
python supplementary/exp5_feature_group_ablation.py --data_root data --epochs 50

echo "===== Exp 6: Inference efficiency ====="
python supplementary/exp6_inference_efficiency.py --data_root data --epochs 30

echo "===== Exp 7: Noise/drop robustness ====="
python supplementary/exp7_noise_robustness.py --data_root data --epochs 50

echo "===== Exp 8: Per-class SHAP + error analysis ====="
python supplementary/exp8_per_class_shap.py --data_root data --epochs 50

echo
echo "===== 全部补实验完成。结果保存在 supplementary/results/ ====="
