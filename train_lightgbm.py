# -*- coding: utf-8 -*-
"""
LightGBM 模型训练脚本 - AIS 14分类航迹（表格特征）
对比准确率与推理时效。LightGBM 使用 CPU/GPU 由 lightgbm 自动选择。
"""
import time
import argparse
import numpy as np
from pathlib import Path
from data_loader import get_tabular_data

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("请安装 lightgbm: pip install lightgbm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"], help="LightGBM 计算设备")
    parser.add_argument("--save", type=str, default="checkpoints/lightgbm.txt")
    args = parser.parse_args()

    data_root = Path(__file__).parent / args.data_root
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_tabular_data(str(data_root))
    print(f"表格特征维度 {X_train.shape[1]}")

    num_classes = 14
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "verbosity": 1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.n_estimators,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.save)

    # 准确率
    pred_proba = model.predict(X_test)
    pred = np.argmax(pred_proba, axis=1)
    test_acc = (pred == y_test).mean()
    print(f"\n===== LightGBM 最终结果 =====")
    print(f"Test Accuracy: {test_acc:.4f}")

    # 推理时效：单条与批量
    n_repeat = 200
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        model.predict(X_test)
    elapsed_batch = (time.perf_counter() - t0) / n_repeat
    n_test = X_test.shape[0]
    ms_batch = elapsed_batch * 1000
    ms_sample = elapsed_batch / n_test * 1000
    print(f"Inference: {ms_batch:.2f} ms/batch (n={n_test}), {ms_sample:.2f} ms/sample")

    with open("results_lightgbm.txt", "w", encoding="utf-8") as f:
        f.write(f"model=lightgbm\naccuracy={test_acc:.4f}\ninference_ms_per_batch={ms_batch:.2f}\ninference_ms_per_sample={ms_sample:.2f}\n")


if __name__ == "__main__":
    main()
