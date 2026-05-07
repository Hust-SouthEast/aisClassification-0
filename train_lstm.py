# -*- coding: utf-8 -*-
"""
LSTM 模型训练脚本 - AIS 14分类航迹
对比准确率与推理时效
"""
import os
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from data_loader import get_sequence_data


class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_size=128, num_layers=2, num_classes=14, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, _) = self.lstm(x)
        out = out[:, -1, :]  # 取最后时间步
        return self.fc(out)


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
def measure_inference_time(model, loader, device, warmup=10, repeat=100):
    model.eval()
    x_sample, _ = next(iter(loader))
    x_sample = x_sample.to(device)
    for _ in range(warmup):
        _ = model(x_sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = model(x_sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    batch_size = x_sample.size(0)
    return (elapsed / repeat) * 1000, (elapsed / repeat / batch_size) * 1000  # ms/batch, ms/sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--save", type=str, default="checkpoints/lstm.pt")
    parser.add_argument("--slide_stride", type=int, default=64, help="训练集滑窗步长(仅对>seq_len的轨迹)")
    parser.add_argument("--patience", type=int, default=10, help="早停：验证准确率连续多少轮不提升则停止")
    parser.add_argument("--no_cuda", action="store_true", help="禁用 GPU")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(__file__).parent / args.data_root

    train_loader, val_loader, test_loader, _, num_classes, input_dim, seq_len = get_sequence_data(
        str(data_root), seq_len=args.seq_len, batch_size=args.batch_size, slide_stride=args.slide_stride
    )
    n_train = len(train_loader.dataset)
    print(f"序列长度 seq_len = {seq_len} 个点/条, 每点 {input_dim} 维特征 (经度,纬度,速度,航向), 训练样本数 = {n_train}")

    model = LSTMModel(
        input_dim=input_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(Path(args.save).parent, exist_ok=True)
    best_acc = 0.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({"model": model.state_dict(), "args": args}, args.save)
        else:
            no_improve += 1
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        if no_improve >= args.patience:
            print(f"早停：验证准确率连续 {args.patience} 轮未提升，停止训练")
            break

    ckpt = torch.load(args.save, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_acc = evaluate(model, test_loader, device)
    ms_batch, ms_sample = measure_inference_time(model, test_loader, device)

    print("\n===== LSTM 最终结果 =====")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Inference: {ms_batch:.2f} ms/batch, {ms_sample:.2f} ms/sample")
    with open("results_lstm.txt", "w", encoding="utf-8") as f:
        f.write(f"model=lstm\naccuracy={test_acc:.4f}\ninference_ms_per_batch={ms_batch:.2f}\ninference_ms_per_sample={ms_sample:.2f}\n")


if __name__ == "__main__":
    main()
