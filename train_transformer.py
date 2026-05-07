# -*- coding: utf-8 -*-
"""
纯 Transformer 模型训练脚本 - AIS 14分类航迹
对比准确率与推理时效
"""
import os
import math
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from data_loader import get_sequence_data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=256,
                 num_classes=14, dropout=0.1, max_len=512):
        super().__init__()
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

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, T, d_model)
        x = x[:, -1, :]
        return self.fc(x)


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
    return (elapsed / repeat) * 1000, (elapsed / repeat / batch_size) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save", type=str, default="checkpoints/transformer.pt")
    parser.add_argument("--slide_stride", type=int, default=64, help="训练集滑窗步长")
    parser.add_argument("--patience", type=int, default=10, help="早停：验证准确率连续多少轮不提升则停止")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(__file__).parent / args.data_root

    train_loader, val_loader, test_loader, _, num_classes, input_dim, seq_len = get_sequence_data(
        str(data_root), seq_len=args.seq_len, batch_size=args.batch_size, slide_stride=args.slide_stride
    )
    print(f"序列 seq_len={seq_len}, 每点 {input_dim} 维 (经度,纬度,速度,航向)")

    model = TransformerModel(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        num_classes=num_classes,
        dropout=args.dropout,
        max_len=args.seq_len,
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

    print("\n===== Transformer 最终结果 =====")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Inference: {ms_batch:.2f} ms/batch, {ms_sample:.2f} ms/sample")
    with open("results_transformer.txt", "w", encoding="utf-8") as f:
        f.write(f"model=transformer\naccuracy={test_acc:.4f}\ninference_ms_per_batch={ms_batch:.2f}\ninference_ms_per_sample={ms_sample:.2f}\n")


if __name__ == "__main__":
    main()
