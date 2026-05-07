# -*- coding: utf-8 -*-
"""
补实验通用工具：
  - 复用 data_loader 的 hybrid pipeline 和 FeatureTransformer
  - 提供 set_seed / 一次完整 hybrid 训练 / hybrid 推理评估 / per-class 指标
  - 所有补实验脚本都从这里 import，避免重复造轮子

放在 aisClassification-0/supplementary/ 下，
运行时会自动把上一级目录加入 sys.path 以便复用 data_loader。
"""
import os
import sys
import json
import math
import time
import random
import numpy as np
from pathlib import Path

# 让我们能 import 上一级的 data_loader
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import lightgbm as lgb

from data_loader import (
    get_sequence_data_hybrid,
    get_trajectory_multi_segments,
    TABULAR_FEATURE_NAMES,
)
from train_hybrid import FeatureTransformer, train_epoch, evaluate, extract_features


# 14 个类别的中文/英文名（与论文 Table 1 一致；按目录字母序映射，可能与 ID 不严格对应，
# 仅用于结果展示和报表，正式论文里请按你 Table 1 的 ID 顺序覆盖）
DEFAULT_CLASS_NAMES = [
    "Bulk Carrier", "Cargo Ship", "Chemical Tanker", "Container Ship",
    "Fishing Ship", "LNG Carrier", "Other Cargo", "Passenger Ship",
    "Pleasure Craft", "Reefer Ship", "Ro-Ro Ship", "Tanker",
    "Tug", "Vehicle Carrier",
]


def set_seed(seed: int):
    """全局确定性种子。注意 torch.use_deterministic_algorithms 会拖慢训练，
    这里只做基本的种子固定，已经能让多 seed 实验有可比的方差估计。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 让 cudnn 行为更确定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------- 默认超参 ----------------
DEFAULT_HP = dict(
    seq_len=128,
    batch_size=64,
    epochs=50,
    lr=1e-3,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_ff=256,
    dropout=0.1,
    pool="last",
    patience=10,
    slide_stride=64,
    num_segments=3,
    segment_agg="mean",
    lgb_rounds=500,
    lgb_early_stop=50,
    lgb_device="cpu",
    lgb_num_leaves=31,
    lgb_max_depth=8,
    lgb_lr=0.05,
)


def build_loaders(data_root, hp: dict, seed: int, repair_missing=True):
    """统一入口：返回 hybrid pipeline 的全部产物。"""
    return get_sequence_data_hybrid(
        data_root=str(data_root),
        seq_len=hp["seq_len"],
        batch_size=hp["batch_size"],
        slide_stride=hp["slide_stride"],
        seed=seed,
        repair_missing=repair_missing,
    )


def train_feature_transformer(
    train_loader, val_loader, n_classes, F, device, hp: dict, save_path=None, verbose=True
):
    """跑一次 Feature Transformer 预训练，返回最佳模型和训练历史。"""
    model = FeatureTransformer(
        input_dim=F,
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_encoder_layers=hp["num_layers"],
        dim_feedforward=hp["dim_ff"],
        num_classes=n_classes,
        dropout=hp["dropout"],
        max_len=hp["seq_len"],
        pool=hp["pool"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    best_val = 0.0
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(1, hp["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(train_loss),
                        "train_acc": float(train_acc), "val_acc": float(val_acc)})
        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if verbose:
            print(f"  [Transformer] epoch {epoch}/{hp['epochs']}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        if no_improve >= hp["patience"]:
            if verbose:
                print(f"  早停于 epoch {epoch} (val 未提升 {hp['patience']} 轮)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, save_path)
    return model, history, best_val


def extract_trajectory_fingerprints(model, indices, X_list, scaler, seq_len, num_segments, agg, device):
    """与 train_hybrid 完全一致的多段 -> Transformer -> 聚合 流程。"""
    n_traj = len(indices)
    segs_list = [get_trajectory_multi_segments(X_list[i], seq_len, num_segments) for i in indices]
    segs = np.stack(segs_list, axis=0)
    n, ns, L, f = segs.shape
    segs = scaler.transform(segs.reshape(-1, f)).reshape(n, ns, L, f)
    segs_t = torch.from_numpy(segs.astype(np.float32))
    segs_flat = segs_t.reshape(n * ns, L, f)
    ft = extract_features(model, segs_flat, device)
    ft = ft.reshape(n, ns, -1)
    if agg == "mean":
        ft = ft.mean(axis=1)
    else:
        ft = ft.max(axis=1)
    return ft  # (n_traj, d_model)


def train_lgb_classifier(
    X_train, y_train, X_val, y_val, n_classes, hp: dict,
    sample_weight=None, extra_params=None,
):
    """统一的 LightGBM 训练。"""
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "device": hp["lgb_device"],
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
        "num_leaves": hp["lgb_num_leaves"],
        "max_depth": hp["lgb_max_depth"],
        "learning_rate": hp["lgb_lr"],
    }
    if extra_params:
        params.update(extra_params)
    callbacks = [lgb.early_stopping(hp["lgb_early_stop"], verbose=False), lgb.log_evaluation(0)]
    return lgb.train(
        params, train_data, num_boost_round=hp["lgb_rounds"],
        valid_sets=[val_data], callbacks=callbacks,
    )


def evaluate_lgb(lgb_model, X_test, y_test, n_classes):
    """返回 dict：accuracy / macro_f1 / per_class（precision/recall/f1/support）/ pred / proba。"""
    proba = lgb_model.predict(X_test)
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_test, pred, labels=np.arange(n_classes), zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "f1_per_class": f.tolist(),
        "support_per_class": s.tolist(),
        "pred": pred,
        "proba": proba,
    }


def run_full_hybrid(data_root, hp: dict, seed: int, device,
                    repair_missing=True, save_dir=None, verbose=True,
                    sample_weight_strategy=None, return_features=False):
    """
    一次完整的 hybrid 训练 + 测试集评估。
    sample_weight_strategy: None / "balanced" -> 用 LightGBM 的 class-weight。

    返回 metrics（dict） + 可选 (X_lgb_train/val/test, y_*, lgb_model, transformer)
    """
    set_seed(seed)
    if verbose:
        print(f"\n=== run_full_hybrid: seed={seed}, seq_len={hp['seq_len']}, "
              f"d_model={hp['d_model']}, repair_missing={repair_missing} ===")

    (
        train_loader, val_loader, _, scaler, n_classes, F, seq_len,
        idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
    ) = build_loaders(data_root, hp, seed=seed, repair_missing=repair_missing)
    y_all = np.array(y_list, dtype=np.int64)

    # 1) 预训练 Feature Transformer
    transformer, history, best_val = train_feature_transformer(
        train_loader, val_loader, n_classes, F, device, hp,
        save_path=None, verbose=verbose,
    )

    # 2) 提取 fingerprint
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

    X_tab_train = X_tab_all[idx_train]
    X_tab_val = X_tab_all[idx_val]
    X_tab_test = X_tab_all[idx_test]
    X_lgb_train = np.hstack([ft_train, X_tab_train])
    X_lgb_val = np.hstack([ft_val, X_tab_val])
    X_lgb_test = np.hstack([ft_test, X_tab_test])
    y_lgb_train = y_all[idx_train]
    y_lgb_val = y_all[idx_val]
    y_lgb_test = y_all[idx_test]

    # 3) LightGBM
    sample_weight = None
    extra_params = None
    if sample_weight_strategy == "balanced":
        # 用 sklearn 的 compute_class_weight 等价：n_samples / (n_classes * np.bincount(y))
        cls_count = np.bincount(y_lgb_train, minlength=n_classes).astype(np.float64)
        cls_count[cls_count == 0] = 1.0
        cls_w = (len(y_lgb_train) / (n_classes * cls_count))
        sample_weight = cls_w[y_lgb_train]

    lgb_model = train_lgb_classifier(
        X_lgb_train, y_lgb_train, X_lgb_val, y_lgb_val,
        n_classes, hp, sample_weight=sample_weight, extra_params=extra_params,
    )

    # 4) 评估
    metrics = evaluate_lgb(lgb_model, X_lgb_test, y_lgb_test, n_classes)
    metrics.update({
        "best_val_acc_transformer": float(best_val),
        "n_classes": int(n_classes),
        "n_train": int(len(y_lgb_train)),
        "n_val": int(len(y_lgb_val)),
        "n_test": int(len(y_lgb_test)),
        "y_test": y_lgb_test.tolist(),
    })
    if verbose:
        print(f"  >>> Test ACC={metrics['accuracy']:.4f}  Macro-F1={metrics['macro_f1']:.4f}")

    if save_dir is not None:
        sd = Path(save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        with open(sd / f"history_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    if return_features:
        return metrics, {
            "X_lgb_train": X_lgb_train, "X_lgb_val": X_lgb_val, "X_lgb_test": X_lgb_test,
            "y_lgb_train": y_lgb_train, "y_lgb_val": y_lgb_val, "y_lgb_test": y_lgb_test,
            "lgb_model": lgb_model, "transformer": transformer,
            "scaler": scaler, "X_tab_all": X_tab_all, "X_list": X_list, "y_list": y_list,
            "idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test,
            "n_classes": n_classes, "F": F, "n_tab": X_tab_all.shape[1],
            "d_model": hp["d_model"],
        }
    return metrics


def get_device(no_cuda=False):
    return torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")


def feature_names_combined(d_model, tabular_names=None):
    """构造 hybrid 特征列名：trans_emb_0..d_model-1 + 表格特征名。"""
    tabular_names = tabular_names or TABULAR_FEATURE_NAMES
    return [f"trans_emb_{i}" for i in range(d_model)] + list(tabular_names)
