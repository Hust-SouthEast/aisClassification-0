# -*- coding: utf-8 -*-
"""
Feature-Transformer + LightGBM 消融实验
对比三种输入：
  - 完整：Transformer 深度特征 + 表格特征（与 train_hybrid 一致）
  - 消融1：仅 Transformer 深度特征
  - 消融2：仅人工统计表格特征（Tabular features）
"""
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, f1_score

from data_loader import get_sequence_data_hybrid, _pad_single

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("请安装 lightgbm: pip install lightgbm")


# ---------- 与 train_hybrid 相同的 Transformer 定义 ----------
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
        self.pool = pool
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
        if self.pool == "last":
            return x[:, -1, :]
        return x.mean(dim=1)

    def get_features(self, x):
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
    model.eval()
    n = X_tensor.shape[0]
    feats = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = X_tensor[start:end].to(device)
        h = model.get_features(x)
        feats.append(h.cpu().numpy())
    return np.vstack(feats)


def run_lgb_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, early_stop=50, rounds=500):
    """训练 LightGBM 并返回 pred, test_acc, macro_f1, precision_per, recall_per, f1_per, support_per, ms_batch, ms_sample."""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 8,
        "learning_rate": 0.05,
        "device": "cpu",
        "verbosity": 0,
        "seed": 42,
    }
    callbacks = [lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=rounds,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    pred_proba = model.predict(X_test)
    pred = np.argmax(pred_proba, axis=1)
    y_true = y_test
    test_acc = (pred == y_true).mean()
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(num_classes), zero_division=0
    )
    n_rep = 200
    t0 = time.perf_counter()
    for _ in range(n_rep):
        model.predict(X_test)
    ms_batch = (time.perf_counter() - t0) / n_rep * 1000
    ms_sample = ms_batch / len(y_test)
    return {
        "pred": pred,
        "test_acc": test_acc,
        "macro_f1": macro_f1,
        "precision_per": precision_per,
        "recall_per": recall_per,
        "f1_per": f1_per,
        "support_per": support_per,
        "ms_batch": ms_batch,
        "ms_sample": ms_sample,
    }


def print_and_save_result(name, res, num_classes, file_handle=None):
    def out(s=""):
        print(s)
        if file_handle:
            file_handle.write(s + "\n")

    out(f"  Test Accuracy: {res['test_acc']:.4f}")
    out(f"  Macro-F1:      {res['macro_f1']:.4f}")
    out(f"  Inference:     {res['ms_batch']:.2f} ms/batch, {res['ms_sample']:.2f} ms/sample")
    out("  各类别 Recall / Precision / F1 / support:")
    for c in range(num_classes):
        out(f"    类别 {c:2d}: Recall={res['recall_per'][c]:.4f}  Precision={res['precision_per'][c]:.4f}  F1={res['f1_per'][c]:.4f}  support={int(res['support_per'][c])}")


def main():
    parser = argparse.ArgumentParser(description="Transformer+LightGBM 消融实验")
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
    parser.add_argument("--save_transformer", type=str, default="checkpoints/feature_transformer.pt")
    parser.add_argument("--load_transformer", type=str, default=None, help="若提供则跳过 Transformer 训练，直接加载")
    parser.add_argument("--lgb_rounds", type=int, default=500)
    parser.add_argument("--lgb_early_stop", type=int, default=50)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(__file__).parent / args.data_root
    num_classes = 14

    # ---------- 1) 加载数据 ----------
    (
        train_loader, val_loader, test_loader, scaler, _, F, seq_len,
        idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
    ) = get_sequence_data_hybrid(
        str(data_root),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        slide_stride=args.slide_stride,
    )
    y_all = np.array(y_list, dtype=np.int64)
    X_tab_train = X_tab_all[idx_train]
    X_tab_val = X_tab_all[idx_val]
    X_tab_test = X_tab_all[idx_test]
    y_train = y_all[idx_train]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]
    print(f"序列 seq_len={seq_len}, 特征 dim={F}, 表格特征数={X_tab_train.shape[1]}")
    print(f"轨迹划分: 训练 {len(idx_train)}, 验证 {len(idx_val)}, 测试 {len(idx_test)}\n")

    # ---------- 2) Transformer：训练或加载，并提取深度特征 ----------
    if args.load_transformer and Path(args.load_transformer).exists():
        print(f"加载已训练 Transformer: {args.load_transformer}")
        ckpt = torch.load(args.load_transformer, map_location=device, weights_only=False)
        saved = getattr(ckpt, "args", None) or ckpt.get("args")
        if saved is not None:
            model = FeatureTransformer(
                input_dim=F,
                d_model=getattr(saved, "d_model", args.d_model),
                nhead=getattr(saved, "nhead", args.nhead),
                num_encoder_layers=getattr(saved, "num_layers", args.num_layers),
                dim_feedforward=getattr(saved, "dim_ff", args.dim_ff),
                num_classes=num_classes,
                dropout=getattr(saved, "dropout", args.dropout),
                max_len=getattr(saved, "seq_len", args.seq_len),
                pool=getattr(saved, "pool", args.pool),
            ).to(device)
        else:
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
        model.load_state_dict(ckpt["model"], strict=True)
    else:
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
        Path(args.save_transformer).parent.mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_acc = 0.0
        no_improve = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                torch.save({"model": model.state_dict(), "args": args}, args.save_transformer)
            else:
                no_improve += 1
            if epoch % 10 == 0 or no_improve == 0:
                print(f"Transformer Epoch {epoch}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
            if no_improve >= args.patience:
                print("Transformer 早停")
                break
        ckpt = torch.load(args.save_transformer, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)

    def first_segments_and_normalize(indices):
        segs = np.stack([_pad_single(X_list[i], seq_len) for i in indices], axis=0)
        n, L, f = segs.shape
        segs = scaler.transform(segs.reshape(-1, f)).reshape(n, L, f)
        return torch.from_numpy(segs.astype(np.float32))

    X_first_train = first_segments_and_normalize(idx_train)
    X_first_val = first_segments_and_normalize(idx_val)
    X_first_test = first_segments_and_normalize(idx_test)
    ft_train = extract_features(model, X_first_train, device)
    ft_val = extract_features(model, X_first_val, device)
    ft_test = extract_features(model, X_first_test, device)

    # ---------- 3) 消融1：仅 Transformer 深度特征 ----------
    print("=" * 60)
    print("消融1：仅使用 Transformer 提取的深度特征")
    print("=" * 60)
    res_transformer_only = run_lgb_and_eval(
        ft_train, y_train, ft_val, y_val, ft_test, y_test,
        num_classes, early_stop=args.lgb_early_stop, rounds=args.lgb_rounds,
    )
    print_and_save_result("消融1-仅Transformer特征", res_transformer_only, num_classes)
    print()

    # ---------- 4) 消融2：仅表格统计特征 ----------
    print("=" * 60)
    print("消融2：仅使用人工提取的统计特征（Tabular features）")
    print("=" * 60)
    res_tabular_only = run_lgb_and_eval(
        X_tab_train, y_train, X_tab_val, y_val, X_tab_test, y_test,
        num_classes, early_stop=args.lgb_early_stop, rounds=args.lgb_rounds,
    )
    print_and_save_result("消融2-仅Tabular特征", res_tabular_only, num_classes)
    print()

    # ---------- 5) 完整模型：Transformer + Tabular（对照） ----------
    print("=" * 60)
    print("对照：Transformer 深度特征 + 表格特征（完整）")
    print("=" * 60)
    X_full_train = np.hstack([ft_train, X_tab_train])
    X_full_val = np.hstack([ft_val, X_tab_val])
    X_full_test = np.hstack([ft_test, X_tab_test])
    res_full = run_lgb_and_eval(
        X_full_train, y_train, X_full_val, y_val, X_full_test, y_test,
        num_classes, early_stop=args.lgb_early_stop, rounds=args.lgb_rounds,
    )
    print_and_save_result("完整-Transformer+Tabular", res_full, num_classes)
    print()

    # ---------- 6) 汇总写入文件 ----------
    with open("results_hybrid_ablation.txt", "w", encoding="utf-8") as f:
        f.write("Feature-Transformer + LightGBM 消融实验\n")
        f.write("=" * 60 + "\n\n")
        f.write("消融1：仅 Transformer 深度特征\n")
        print_and_save_result("消融1", res_transformer_only, num_classes, f)
        f.write("\n消融2：仅 Tabular 统计特征\n")
        print_and_save_result("消融2", res_tabular_only, num_classes, f)
        f.write("\n对照：完整（Transformer + Tabular）\n")
        print_and_save_result("完整", res_full, num_classes, f)
        f.write("\n汇总:\n")
        f.write(f"  仅Transformer - Accuracy: {res_transformer_only['test_acc']:.4f}  Macro-F1: {res_transformer_only['macro_f1']:.4f}\n")
        f.write(f"  仅Tabular    - Accuracy: {res_tabular_only['test_acc']:.4f}  Macro-F1: {res_tabular_only['macro_f1']:.4f}\n")
        f.write(f"  完整融合     - Accuracy: {res_full['test_acc']:.4f}  Macro-F1: {res_full['macro_f1']:.4f}\n")
    print("结果已写入 results_hybrid_ablation.txt")


if __name__ == "__main__":
    main()
