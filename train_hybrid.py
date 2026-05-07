# -*- coding: utf-8 -*-
"""
Feature-Transformer + LightGBM 混合架构（方案 A：两阶段特征融合）
1) 预训练：用 Transformer 做轨迹分类，序列定长+归一化，位置编码+多头自注意力+池化得到轨迹指纹。
2) 特征导出：取 Transformer 倒数第二层输出（池化后、分类头前）作为深度特征。
3) LightGBM：将深度特征与原始统计表格特征拼接，训练 LightGBM 做最终分类。
"""
import json
import math
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score
import torch.nn as nn
from pathlib import Path

from data_loader import (
    get_sequence_data_hybrid,
    _pad_single,
    get_trajectory_multi_segments,
)

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("请安装 lightgbm: pip install lightgbm")

try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ---------- Feature-Transformer：位置编码 + 多头自注意力 + 池化 ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FeatureTransformer(nn.Module):
    """
    输入 (B, L, F) -> 线性映射 -> 位置编码 -> Transformer Encoder -> 池化 -> (B, D)。
    可选分类头用于预训练；get_features() 返回池化后的 D 维向量（倒数第二层）。
    """

    def __init__(
        self,
        input_dim=4,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        num_classes=14,
        dropout=0.1,
        max_len=512,
        pool="last",
    ):
        super().__init__()
        self.d_model = d_model
        self.pool = pool  # "last" | "gap"
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def _pool(self, x):
        # x: (B, T, D)
        if self.pool == "last":
            return x[:, -1, :]
        return x.mean(dim=1)

    def get_features(self, x):
        """倒数第二层输出，供 LightGBM 拼接。返回 (B, d_model)。"""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self._pool(x)

    def forward(self, x):
        h = self.get_features(x)
        return self.fc(h)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def extract_features(model, X_tensor, device, batch_size=256):
    """对序列张量 X_tensor (n, L, F) 提取 (n, d_model)。"""
    model.eval()
    n = X_tensor.shape[0]
    feats = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = X_tensor[start:end].to(device)
        h = model.get_features(x)
        feats.append(h.cpu().numpy())
    return np.vstack(feats)


def main():
    parser = argparse.ArgumentParser(description="Feature-Transformer + LightGBM 两阶段训练")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool", type=str, default="last", choices=["last", "gap"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--slide_stride", type=int, default=64)
    parser.add_argument("--num_segments", type=int, default=3, help="每条轨迹取几段(首/中/末)做特征聚合")
    parser.add_argument("--segment_agg", type=str, default="mean", choices=["mean", "max"], help="多段特征的聚合方式")
    parser.add_argument("--save_transformer", type=str, default="checkpoints/feature_transformer.pt")
    parser.add_argument(
        "--save_lgb",
        type=str,
        default="checkpoints/lgb_hybrid.txt",
        help="混合阶段训练好的 LightGBM 模型保存路径（供 SHAP/导出脚本使用，特征维=Transformer深度+表格）",
    )
    parser.add_argument("--lgb_rounds", type=int, default=500)
    parser.add_argument("--lgb_early_stop", type=int, default=50)
    parser.add_argument("--lgb_device", type=str, default="cpu", choices=["gpu", "cpu"], help="LightGBM 设备，无 OpenCL 时用 cpu")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--use_optuna", action="store_true", help="使用 Optuna 搜索 LightGBM 超参数")
    parser.add_argument("--optuna_trials", type=int, default=30, help="Optuna 搜索轮数")
    parser.add_argument("--no_repair", action="store_true", help="关闭 AIS 轨迹插值修复")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(__file__).parent / args.data_root

    # ---------- 1) 加载数据（归一化序列 + 轨迹索引 + 表格特征） ----------
    (
        train_loader, val_loader, test_loader, scaler, num_classes, F, seq_len,
        idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
    ) = get_sequence_data_hybrid(
        str(data_root),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        slide_stride=args.slide_stride,
        repair_missing=not args.no_repair,
    )
    y_all = np.array(y_list, dtype=np.int64)
    n_tab = X_tab_all.shape[1]
    print(f"序列 seq_len={seq_len}, 特征 dim={F}, 表格特征数={n_tab}")
    print(f"轨迹划分: 训练 {len(idx_train)}, 验证 {len(idx_val)}, 测试 {len(idx_test)}")

    # ---------- 2) 预训练 Transformer（轨迹分类） ----------
    model = FeatureTransformer(
        input_dim=F,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        num_classes=num_classes,
        dropout=args.dropout,
        max_len=args.seq_len,
        pool=args.pool,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.save_transformer).parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    no_improve = 0
    history = []  # 用于收敛曲线：epoch, train_loss, train_acc, val_acc
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(train_loss), "train_acc": float(train_acc), "val_acc": float(val_acc)})
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({"model": model.state_dict(), "args": args}, args.save_transformer)
        else:
            no_improve += 1
        print(f"Transformer Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        if no_improve >= args.patience:
            print(f"早停：验证准确率连续 {args.patience} 轮未提升")
            break

    # 加载最佳 Transformer
    ckpt = torch.load(args.save_transformer, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # ---------- 3) 按轨迹取多段（首/中/末），归一化，过 Transformer 后聚合为一条轨迹指纹 ----------
    def multi_segments_and_normalize(indices):
        """返回 (n_traj, num_segments, seq_len, F) 归一化后张量"""
        n_traj = len(indices)
        segs_list = [get_trajectory_multi_segments(X_list[i], seq_len, args.num_segments) for i in indices]
        segs = np.stack(segs_list, axis=0)
        n, ns, L, f = segs.shape
        segs = scaler.transform(segs.reshape(-1, f)).reshape(n, ns, L, f)
        return torch.from_numpy(segs.astype(np.float32))

    def extract_trajectory_features(indices, agg=args.segment_agg):
        """多段 -> Transformer -> 聚合 -> (n_traj, d_model)"""
        segs = multi_segments_and_normalize(indices)
        n, ns, L, f = segs.shape
        segs_flat = segs.reshape(n * ns, L, f)
        ft = extract_features(model, segs_flat, device)
        ft = ft.reshape(n, ns, -1)
        if agg == "mean":
            ft = ft.mean(axis=1)
        else:
            ft = ft.max(axis=1)[0]
        return ft

    print(f"轨迹特征: 每轨迹 {args.num_segments} 段(首/中/末), 聚合方式={args.segment_agg}")
    ft_train = extract_trajectory_features(idx_train)
    ft_val = extract_trajectory_features(idx_val)
    ft_test = extract_trajectory_features(idx_test)

    X_tab_train = X_tab_all[idx_train]
    X_tab_val = X_tab_all[idx_val]
    X_tab_test = X_tab_all[idx_test]

    X_lgb_train = np.hstack([ft_train, X_tab_train])
    X_lgb_val = np.hstack([ft_val, X_tab_val])
    X_lgb_test = np.hstack([ft_test, X_tab_test])
    y_lgb_train = y_all[idx_train]
    y_lgb_val = y_all[idx_val]
    y_lgb_test = y_all[idx_test]
    print(f"LightGBM 输入维度: Transformer {ft_train.shape[1]} + 表格 {X_tab_train.shape[1]} = {X_lgb_train.shape[1]}")

    # ---------- 4) 训练 LightGBM（可选 Optuna 超参搜索） ----------
    train_data = lgb.Dataset(X_lgb_train, label=y_lgb_train)
    val_data = lgb.Dataset(X_lgb_val, label=y_lgb_val, reference=train_data)

    def _train_lgb(params, rounds=None, callbacks=None):
        p = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "device": args.lgb_device,
            "verbosity": -1,
            "seed": 42,
            "feature_pre_filter": False,  # 允许 Optuna 动态调整 min_data_in_leaf，避免 LightGBM 报错
            **params,
        }
        cb = callbacks or [lgb.early_stopping(args.lgb_early_stop, verbose=False), lgb.log_evaluation(0)]
        return lgb.train(p, train_data, num_boost_round=rounds or args.lgb_rounds, valid_sets=[val_data], callbacks=cb)

    if args.use_optuna and _HAS_OPTUNA:
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "num_leaves": trial.suggest_int("num_leaves", 20, 64),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
            }
            model_lgb = _train_lgb(params)
            pred = np.argmax(model_lgb.predict(X_lgb_val), axis=1)
            # 最小化验证错误率 (1 - accuracy)，best value 0.22 表示验证准确率约 78%
            return 1.0 - (pred == y_lgb_val).mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=True)
        best_params = study.best_params
        print(f"Optuna 最佳超参: {best_params}")
        lgb_params = {
            "num_leaves": best_params["num_leaves"],
            "max_depth": best_params["max_depth"],
            "learning_rate": best_params["learning_rate"],
            "min_data_in_leaf": best_params["min_data_in_leaf"],
            "feature_fraction": best_params["feature_fraction"],
            "bagging_fraction": best_params["bagging_fraction"],
            "bagging_freq": best_params["bagging_freq"],
        }
    else:
        if args.use_optuna and not _HAS_OPTUNA:
            print("未安装 Optuna，使用固定超参。pip install optuna 以启用搜索。")
        lgb_params = {
            "num_leaves": 31,
            "max_depth": 8,
            "learning_rate": 0.05,
        }
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "device": args.lgb_device,
        "verbosity": 1,
        "seed": 42,
        "feature_pre_filter": False,  # 与 _train_lgb 一致，避免 min_data_in_leaf 相关报错
        **lgb_params,
    }
    callbacks = [lgb.early_stopping(args.lgb_early_stop, verbose=True), lgb.log_evaluation(50)]
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=args.lgb_rounds,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    Path(args.save_lgb).parent.mkdir(parents=True, exist_ok=True)
    lgb_model.save_model(args.save_lgb)
    print(f"已保存混合 LightGBM 模型：{args.save_lgb}（特征维={X_lgb_train.shape[1]}）")

    # ---------- 5) 测试集评估：整体 + 每类 Precision/Recall/F1，Macro-F1，推理耗时 ----------
    pred_proba = lgb_model.predict(X_lgb_test)
    pred = np.argmax(pred_proba, axis=1)
    y_true = y_lgb_test
    test_acc = (pred == y_true).mean()
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(num_classes), zero_division=0
    )

    n_repeat = 200
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        lgb_model.predict(X_lgb_test)
    elapsed = (time.perf_counter() - t0) / n_repeat
    ms_batch = elapsed * 1000
    ms_sample = elapsed / len(y_lgb_test) * 1000

    print("\n===== Feature-Transformer + LightGBM 最终结果 =====")
    print(f"Test Accuracy (总体): {test_acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Inference: {ms_batch:.2f} ms/batch, {ms_sample:.2f} ms/sample")
    print("\n各类别 准确率(Recall) / Precision / F1 / 样本数:")
    print("-" * 60)
    for c in range(num_classes):
        print(f"  类别 {c:2d}:  准确率(Recall)={recall_per[c]:.4f}  Precision={precision_per[c]:.4f}  F1={f1_per[c]:.4f}  support={int(support_per[c])}")
    print("-" * 60)

    with open("results_hybrid.txt", "w", encoding="utf-8") as f:
        f.write(f"model=feature_transformer_lightgbm\naccuracy={test_acc:.4f}\nmacro_f1={macro_f1:.4f}\n")
        f.write(f"inference_ms_per_batch={ms_batch:.2f}\ninference_ms_per_sample={ms_sample:.2f}\n\n")
        f.write("per_class: recall(准确率) precision f1 support\n")
        for c in range(num_classes):
            f.write(f"  class_{c}: {recall_per[c]:.4f} {precision_per[c]:.4f} {f1_per[c]:.4f} {int(support_per[c])}\n")

    # 保存训练收敛历史与超参，供 plot_training_results.py 使用
    base_dir = Path(__file__).parent
    with open(base_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    config = {
        "transformer": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()},
        "lgb": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in params.items()},
    }
    with open(base_dir / "config_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
