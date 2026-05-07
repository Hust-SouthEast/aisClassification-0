# -*- coding: utf-8 -*-
"""
ResNet(1D) + XGBoost 混合架构 - AIS 14分类
1) 预训练：用 1D ResNet 对轨迹序列做分类，定长+归一化，提取池化后特征。
2) 特征导出：取 ResNet 倒数第二层（全局池化后、分类头前）作为深度特征。
3) XGBoost：将深度特征与表格统计特征拼接，训练 XGBoost 做最终分类。
"""
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from data_loader import get_sequence_data_hybrid, _pad_single

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("请安装 xgboost: pip install xgboost")

from sklearn.metrics import precision_recall_fscore_support, f1_score


# ---------- 1D ResNet：残差块 + 全局池化 ----------
class ResidualBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet1D(nn.Module):
    """
    输入 (B, L, F) -> 转为 (B, F, L) -> 1D Conv + 残差块 -> 全局平均池化 -> (B, feat_dim)。
    get_features() 返回池化后的向量；forward() 为分类输出。
    """

    def __init__(self, input_dim=4, base_ch=64, num_blocks=(2, 2, 2), num_classes=14):
        super().__init__()
        self.in_ch = input_dim
        self.feat_dim = base_ch * 8  # 最后一层通道数
        self.conv1 = nn.Conv1d(input_dim, base_ch, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_ch)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_ch, base_ch, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_ch, base_ch * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_ch * 2, base_ch * 4, num_blocks[2], stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_ch * 4, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock1d(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1d(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def get_features(self, x):
        """(B, L, F) -> (B, feat_dim)。"""
        if x.dim() == 3 and x.shape[2] == self.in_ch:
            x = x.transpose(1, 2)  # (B, F, L)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return x.squeeze(-1)

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
    """(n, L, F) -> (n, feat_dim)。"""
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
    parser = argparse.ArgumentParser(description="ResNet1D + XGBoost 两阶段训练")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--slide_stride", type=int, default=64)
    parser.add_argument("--save_resnet", type=str, default="checkpoints/resnet1d.pt")
    parser.add_argument("--xgb_rounds", type=int, default=500)
    parser.add_argument("--xgb_early_stop", type=int, default=50)
    parser.add_argument("--xgb_max_depth", type=int, default=8)
    parser.add_argument("--xgb_eta", type=float, default=0.05)
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
    n_tab = X_tab_all.shape[1]
    print(f"序列 seq_len={seq_len}, 特征 dim={F}, 表格特征数={n_tab}")
    print(f"轨迹划分: 训练 {len(idx_train)}, 验证 {len(idx_val)}, 测试 {len(idx_test)}")

    # ---------- 2) 预训练 ResNet1D ----------
    model = ResNet1D(
        input_dim=F,
        base_ch=args.base_ch,
        num_blocks=(2, 2, 2),
        num_classes=num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.save_resnet).parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({"model": model.state_dict(), "args": args}, args.save_resnet)
        else:
            no_improve += 1
        print(f"ResNet Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        if no_improve >= args.patience:
            print(f"早停：验证准确率连续 {args.patience} 轮未提升")
            break

    ckpt = torch.load(args.save_resnet, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # ---------- 3) 按轨迹首段提取深度特征 ----------
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

    X_tab_train = X_tab_all[idx_train]
    X_tab_val = X_tab_all[idx_val]
    X_tab_test = X_tab_all[idx_test]

    X_xgb_train = np.hstack([ft_train, X_tab_train])
    X_xgb_val = np.hstack([ft_val, X_tab_val])
    X_xgb_test = np.hstack([ft_test, X_tab_test])
    y_xgb_train = y_all[idx_train]
    y_xgb_val = y_all[idx_val]
    y_xgb_test = y_all[idx_test]
    print(f"XGBoost 输入维度: ResNet {ft_train.shape[1]} + 表格 {X_tab_train.shape[1]} = {X_xgb_train.shape[1]}")

    # ---------- 4) 训练 XGBoost ----------
    dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
    dval = xgb.DMatrix(X_xgb_val, label=y_xgb_val)
    params = {
        "objective": "multi:softmax",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "max_depth": args.xgb_max_depth,
        "eta": args.xgb_eta,
        "seed": 42,
    }
    evals = [(dtrain, "train"), (dval, "val")]
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.xgb_rounds,
        evals=evals,
        early_stopping_rounds=args.xgb_early_stop,
        verbose_eval=50,
    )

    # ---------- 5) 测试集评估：准确率、Macro-F1、每类 Recall/Precision/F1、推理耗时 ----------
    dtest = xgb.DMatrix(X_xgb_test)
    pred = xgb_model.predict(dtest).astype(np.int32)
    y_true = y_xgb_test
    test_acc = (pred == y_true).mean()
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(num_classes), zero_division=0
    )

    n_repeat = 200
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        xgb_model.predict(dtest)
    elapsed = (time.perf_counter() - t0) / n_repeat
    ms_batch = elapsed * 1000
    ms_sample = elapsed / len(y_xgb_test) * 1000

    print("\n===== ResNet + XGBoost 最终结果 =====")
    print(f"Test Accuracy (总体): {test_acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Inference: {ms_batch:.2f} ms/batch, {ms_sample:.2f} ms/sample")
    print("\n各类别 准确率(Recall) / Precision / F1 / 样本数:")
    print("-" * 60)
    for c in range(num_classes):
        print(f"  类别 {c:2d}:  准确率(Recall)={recall_per[c]:.4f}  Precision={precision_per[c]:.4f}  F1={f1_per[c]:.4f}  support={int(support_per[c])}")
    print("-" * 60)

    with open("results_resnet_xgboost.txt", "w", encoding="utf-8") as f:
        f.write(f"model=resnet_xgboost\naccuracy={test_acc:.4f}\nmacro_f1={macro_f1:.4f}\n")
        f.write(f"inference_ms_per_batch={ms_batch:.2f}\ninference_ms_per_sample={ms_sample:.2f}\n\n")
        f.write("per_class: recall(准确率) precision f1 support\n")
        for c in range(num_classes):
            f.write(f"  class_{c}: {recall_per[c]:.4f} {precision_per[c]:.4f} {f1_per[c]:.4f} {int(support_per[c])}\n")


if __name__ == "__main__":
    main()
