@echo off
REM ============================================================
REM 一键跑完所有补实验 (Windows)
REM ============================================================
REM 注意：实验 1 重复 5 次，每次完整训练 hybrid，最耗时；
REM       实验 4 默认 50 轮 Optuna 搜索；
REM       不想跑全量可单独执行某个 .py
REM ============================================================
setlocal
cd /d "%~dp0\.."

echo ===== Exp 1: Multi-seed + significance =====
python supplementary\exp1_multi_seed.py --data_root data --n_runs 5 --epochs 50

echo ===== Exp 2: Per-class + imbalance =====
python supplementary\exp2_per_class_imbalance.py --data_root data --epochs 50

echo ===== Exp 3: Preprocessing ablation =====
python supplementary\exp3_preprocessing_ablation.py --data_root data --epochs 30

echo ===== Exp 4: Hyperparameter search (Optuna) =====
python supplementary\exp4_hyperparam_search.py --data_root data --n_trials 50 --epochs 50

echo ===== Exp 5: Feature group + d_model ablation =====
python supplementary\exp5_feature_group_ablation.py --data_root data --epochs 50

echo ===== Exp 6: Inference efficiency =====
python supplementary\exp6_inference_efficiency.py --data_root data --epochs 30

echo ===== Exp 7: Noise/drop robustness =====
python supplementary\exp7_noise_robustness.py --data_root data --epochs 50

echo ===== Exp 8: Per-class SHAP + error analysis =====
python supplementary\exp8_per_class_shap.py --data_root data --epochs 50

echo.
echo ===== 全部补实验完成。结果保存在 supplementary\results\ =====
endlocal
