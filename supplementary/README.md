# 补实验脚本说明（针对审稿意见）

> 本目录下的脚本对应论文 *Feature Transformer + LightGBM Ensemble for Ship Trajectory Recognition* 的补实验，全部直接复用主目录的 `data_loader.py`、`train_hybrid.py` 中的 `FeatureTransformer`，**不动你原有代码**。结果统一写入 `supplementary/results/`。

## 安装额外依赖

`requirements.txt` 已经包含大部分依赖，补实验额外需要：

```bash
pip install pandas matplotlib scipy shap optuna imbalanced-learn
```

- `imbalanced-learn` 仅用于 Exp 2 的 SMOTE，可用 `--skip_smote` 跳过。
- `shap` 仅用于 Exp 8。
- `optuna` 仅用于 Exp 4。
- `scipy` 用于 Exp 1 的统计检验、Exp 3 的样条插值。

## 数据目录

脚本默认从 `--data_root data` 读取轨迹（即 `aisClassification-0/data/`），结构同你原版：14 个子目录，每个目录里若干 `.txt`。

## 一键运行

```bash
# Linux/macOS
bash supplementary/run_supplementary.sh

# Windows
supplementary\run_supplementary.bat
```

或单独跑某一个：

```bash
cd aisClassification-0
python supplementary/exp1_multi_seed.py --data_root data --n_runs 5 --epochs 50
```

## 实验对照审稿意见

| # | 脚本 | 对应审稿人关切 | 预计耗时（GPU） | 关键输出 |
|---|---|---|---|---|
| 1 | `exp1_multi_seed.py` | "result variance / CI / statistical significance test 缺失" | ≈ 5 × hybrid 训练时长 | `exp1_multi_seed_stats.csv` 含 mean ± std + 95% CI；`exp1_multi_seed_paired_tests.csv` 含 paired t-test / Wilcoxon p-value |
| 2 | `exp2_per_class_imbalance.py` | "minority class 分析 / per-class 表 / error matrix / 不平衡处理" | ≈ 4 × LightGBM 训练（Transformer 只训一次） | `exp2_per_class_<strategy>.csv`、`exp2_confusion_<strategy>.png/csv`、`exp2_per_class_f1_compare.png`、`exp2_minority_compare.csv` |
| 3 | `exp3_preprocessing_ablation.py` | "data pre-processing / interpolation / seq_len 描述太简略" | ≈ 7 × hybrid 训练 | `exp3_preprocessing_ablation.csv/.png`：repair on/off、linear/nearest/cubic、seq_len 64/128/256 |
| 4 | `exp4_hyperparam_search.py` | "hyperparameter tuning 描述不够" | ≈ Transformer 1 次 + Optuna n_trials 次（默认 50） | `exp4_optuna_history.csv`、`exp4_best_params.json`、`exp4_param_importance.png`、`exp4_final_test.json` |
| 5 | `exp5_feature_group_ablation.py` | 现 Table 3 消融过粗 | ≈ Transformer 1 + LightGBM ×11 + d_model 网格 ×4 | `exp5_tab_group_ablation.csv/.png`、`exp5_dmodel_ablation.csv/.png` |
| 6 | `exp6_inference_efficiency.py` | "1.58 ms/batch 太单薄" | ≈ 1 次 hybrid + 推理基准（5 分钟） | `exp6_inference_efficiency.csv/.png`、`exp6_model_size.json`（参数量+磁盘大小） |
| 7 | `exp7_noise_robustness.py` | "noise / missing values handling" | ≈ 1 次 hybrid + 8 次推理 | `exp7_noise_robustness.csv/.png` |
| 8 | `exp8_per_class_shap.py` | 当前 SHAP 仅全局，缺 per-class 与错误案例 | ≈ 1 次 hybrid + SHAP（数十秒） | `exp8_per_class_top_features.csv/.png`、`exp8_misclassified_cases.csv` 及若干 `exp8_misclassified_topshap_*.png` |

## 关键设计说明

1. **不重训 Transformer 的实验**：Exp 2 / Exp 4 / Exp 5(A) / Exp 8 都共用一份 fingerprint，避免重复跑预训练耗时。
2. **重训 Transformer 的实验**：Exp 1（多 seed）、Exp 3（不同插值/seq_len）、Exp 5(B)（不同 d_model）、Exp 6 / Exp 7（独立 baseline 模型）。
3. **vessel-wise split**：完全沿用 `get_sequence_data_hybrid` 的划分逻辑，不重新切数据，与原 paper 测试集一致。
4. **种子可控**：所有脚本接 `--seed`，并在 `common_utils.set_seed()` 中固定 numpy / torch / cudnn。Exp 1 用 seed = 0..4。
5. **结果统一写到 `supplementary/results/`**，不会覆盖你原来的 `results_hybrid.txt` 等文件。

## 跑完之后做什么

每个实验的输出已经按"论文表格/图"的尺寸来设计，可直接：

1. **正文 Table 2 改写** → 用 `exp1_multi_seed_stats.csv` 的 mean ± std + p-value，替换原来的单点数字。
2. **新增 per-class 表（建议放正文或附录）** → `exp2_per_class_baseline.csv`。
3. **新增不平衡缓解表** → `exp2_overview.csv` + minority 三类的 F1 提升用 `exp2_minority_compare.csv`。
4. **新增预处理/seq_len 敏感性表（附录 A）** → `exp3_preprocessing_ablation.csv`。
5. **新增超参搜索描述（附录 B）** → `exp4_optuna_history.csv` + `exp4_param_importance.png`。
6. **改写 Section 5.3 ablation** → `exp5_tab_group_ablation.csv` 的细化结果。
7. **改写 Section 5.5 inference efficiency** → `exp6_inference_efficiency.csv` 的 batch 扫描表 + `exp6_model_size.json`。
8. **新增 Section 5.x Noise robustness（建议加）** → `exp7_noise_robustness.png`。
9. **改写 Section 5.4 SHAP 解释** → `exp8_per_class_top_features.png` + 至少 1 个错误案例 `exp8_misclassified_topshap_*.png`。

## 常见问题

- **GPU 显存不够**：把 `--batch_size`（实际通过环境变量改 `DEFAULT_HP['batch_size']` 或 hp dict）调小到 32；或加 `--no_cuda` 改为 CPU（Transformer 训练会慢 5–10 倍）。
- **Exp 4 Optuna 太慢**：把 `--n_trials 50` 调成 `20` 或 `30`；如果只想看最优超参也可以用 `--n_trials 10`。
- **Exp 1 想做 10 次而不是 5 次**：`--n_runs 10`，时间翻倍。
- **想加更多 baseline 进入 Exp 1（比如 ResNet+XGBoost）**：在 `run_one_seed` 里复用 `train_resnet_xgboost.py` 的逻辑（你原版已有），把它包成一个函数加进来即可。
- **数据预处理插值用 spline-cubic 报错**：scipy 的 cubic 至少需要 4 个有效点，少数轨迹太短时脚本会自动 fallback 到 linear，不会 crash。

## 与原论文一致性自检

跑完 `exp1` 后，**Hybrid baseline 的 mean acc 应当大致接近你原 paper 报告的 82.42%**（±1–2%属正常方差）。如果差距很大，请检查：

- `data/` 是否包含全部 2196 条轨迹（每个子目录数与 paper Table 1 一致）；
- 默认超参是否被改过（看 `common_utils.DEFAULT_HP`）；
- `repair_missing` 是否一致（默认 True）。
