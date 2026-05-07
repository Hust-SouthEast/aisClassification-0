#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 Feature-Transformer + LightGBM（train_hybrid.py）对应的：
1) LightGBM 模型文件（可用于后续 SHAP 解释）
2) 与 SHAP 值一一对应的特征矩阵（原始值）：Transformer 深度特征 + 表格特征
3) （可选）直接计算并保存 SHAP 值（需要安装 shap）

输出目录默认：outputs_shap/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data_loader import (
    TABULAR_FEATURE_NAMES,
    get_sequence_data_hybrid,
    get_trajectory_multi_segments,
)

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("请先安装 lightgbm：pip install lightgbm") from e


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
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


def _save_csv(path: Path, X: np.ndarray, header: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(X, columns=header)
        df.to_csv(path, index=False, encoding="utf-8")
    except Exception:
        if header:
            np.savetxt(path, X, delimiter=",", header=",".join(header), comments="", fmt="%.10g")
        else:
            np.savetxt(path, X, delimiter=",", fmt="%.10g")


def main():
    parser = argparse.ArgumentParser(description="导出 hybrid 模型/数据（可选计算 SHAP）")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--slide_stride", type=int, default=64)
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--segment_agg", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_repair", action="store_true", help="关闭 AIS 轨迹插值修复（需与训练保持一致）")

    parser.add_argument("--transformer_ckpt", type=str, default="checkpoints/feature_transformer.pt")
    parser.add_argument("--config_hybrid", type=str, default="config_hybrid.json", help="可选：读取 train_hybrid.py 保存的超参")
    parser.add_argument("--lgb_model_in", type=str, default="", help="可选：直接加载已训练好的 LightGBM model txt")

    parser.add_argument("--out_dir", type=str, default="outputs_shap")
    parser.add_argument("--export_split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--compute_shap", action="store_true", help="计算并保存 SHAP（需要 pip install shap）")
    parser.add_argument("--shap_max_samples", type=int, default=2000, help="用于计算 SHAP 的最大样本数（从 export_split 中抽样）")
    args = parser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    base_dir = Path(__file__).parent
    data_root = base_dir / args.data_root
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        _train_loader,
        _val_loader,
        _test_loader,
        scaler,
        num_classes,
        F,
        seq_len,
        idx_train,
        idx_val,
        idx_test,
        X_list,
        y_list,
        X_tab_all,
    ) = get_sequence_data_hybrid(
        str(data_root),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        slide_stride=args.slide_stride,
        repair_missing=not args.no_repair,
    )

    # 读取 Transformer checkpoint 并实例化模型（优先用 ckpt 里的 args，保证一致性）
    ckpt_path = base_dir / args.transformer_ckpt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", None)
    d_model = int(getattr(ckpt_args, "d_model", 64)) if ckpt_args is not None else 64
    nhead = int(getattr(ckpt_args, "nhead", 4)) if ckpt_args is not None else 4
    num_layers = int(getattr(ckpt_args, "num_layers", 2)) if ckpt_args is not None else 2
    dim_ff = int(getattr(ckpt_args, "dim_ff", 256)) if ckpt_args is not None else 256
    dropout = float(getattr(ckpt_args, "dropout", 0.1)) if ckpt_args is not None else 0.1
    pool = str(getattr(ckpt_args, "pool", "last")) if ckpt_args is not None else "last"

    model = FeatureTransformer(
        input_dim=F,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=dim_ff,
        num_classes=num_classes,
        dropout=dropout,
        max_len=seq_len,
        pool=pool,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_all = np.array(y_list, dtype=np.int64)

    def multi_segments_and_normalize(indices: np.ndarray):
        segs_list = [get_trajectory_multi_segments(X_list[int(i)], seq_len, args.num_segments) for i in indices]
        segs = np.stack(segs_list, axis=0)  # (n, ns, L, F)
        n, ns, L, f = segs.shape
        segs = scaler.transform(segs.reshape(-1, f)).reshape(n, ns, L, f)
        return torch.from_numpy(segs.astype(np.float32))

    def extract_trajectory_features(indices: np.ndarray):
        segs = multi_segments_and_normalize(indices)
        n, ns, L, f = segs.shape
        segs_flat = segs.reshape(n * ns, L, f)
        ft = extract_features(model, segs_flat, device)
        ft = ft.reshape(n, ns, -1)
        if args.segment_agg == "mean":
            ft = ft.mean(axis=1)
        else:
            ft = ft.max(axis=1)
        return ft.astype(np.float32)

    def build_X(indices: np.ndarray):
        ft = extract_trajectory_features(indices)
        X_tab = X_tab_all[indices]
        X = np.hstack([ft, X_tab]).astype(np.float32)
        y = y_all[indices].astype(np.int64)
        return X, y, ft.shape[1], X_tab.shape[1]

    # 选择导出 split
    if args.export_split == "train":
        export_indices = np.array(idx_train)
        split_name = "train"
    elif args.export_split == "val":
        export_indices = np.array(idx_val)
        split_name = "val"
    elif args.export_split == "test":
        export_indices = np.array(idx_test)
        split_name = "test"
    else:
        export_indices = np.concatenate([np.array(idx_train), np.array(idx_val), np.array(idx_test)])
        split_name = "all"

    X_export, y_export, n_deep, n_tab = build_X(export_indices)

    deep_names = [f"ft_{i}" for i in range(n_deep)]
    tab_names = list(TABULAR_FEATURE_NAMES) if len(TABULAR_FEATURE_NAMES) == n_tab else [f"tab_{i}" for i in range(n_tab)]
    feature_names = deep_names + tab_names

    # 保存特征矩阵（原始值）与标签、索引映射
    _save_csv(out_dir / f"X_{split_name}.csv", X_export, header=feature_names)
    _save_csv(out_dir / f"y_{split_name}.csv", y_export.reshape(-1, 1), header=["label"])
    with open(out_dir / f"indices_{split_name}.json", "w", encoding="utf-8") as f:
        json.dump({"indices": export_indices.tolist()}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # LightGBM：加载已有模型 or 重新训练一个（用于 SHAP）
    # 注意：checkpoints/lightgbm.txt 多为「纯表格」模型（特征数远小于混合维数），不可混用。
    n_feat_hybrid = int(X_export.shape[1])
    booster: lgb.Booster | None = None
    if args.lgb_model_in:
        ckpt_lgb = base_dir / args.lgb_model_in
        b_try = lgb.Booster(model_file=str(ckpt_lgb))
        if b_try.num_feature() == n_feat_hybrid:
            booster = b_try
        else:
            print(
                f"[警告] 指定的 LightGBM 特征数为 {b_try.num_feature()}，与当前混合特征维数 {n_feat_hybrid} 不一致 "
                f"（例如误用了纯表格的 checkpoints/lightgbm.txt）。将自动在 train/val 上重新训练混合 LightGBM。"
            )

    if booster is None:
        cfg_path = base_dir / args.config_hybrid
        lgb_params_user = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                lgb_params_user = cfg.get("lgb", {}) if isinstance(cfg, dict) else {}
            except Exception:
                lgb_params_user = {}

        # 兼容：config_hybrid.json 里可能包含 objective/num_class 等；这里统一覆盖关键项
        params = {
            "objective": "multiclass",
            "num_class": int(num_classes),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": 42,
            "feature_pre_filter": False,
        }
        # 只保留 LightGBM 能识别的基本参数（避免把字符串化的复杂字段塞进去）
        for k, v in lgb_params_user.items():
            if k in params:
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                params[k] = v

        # 训练集/验证集：按 train/val 的轨迹级特征重新构建（保证与训练一致）
        X_train, y_train, _, _ = build_X(np.array(idx_train))
        X_val, y_val, _, _ = build_X(np.array(idx_val))
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

    model_out_path = out_dir / "lightgbm_model.txt"
    booster.save_model(str(model_out_path))

    # 可选：计算 SHAP（TreeExplainer）
    if args.compute_shap:
        try:
            import shap  # type: ignore
        except ImportError as e:
            raise ImportError("需要安装 shap 才能计算：pip install shap") from e

        n = X_export.shape[0]
        if n > args.shap_max_samples:
            rng = np.random.default_rng(42)
            sel = rng.choice(n, size=args.shap_max_samples, replace=False)
            explain_row_indices = sel
            X_explain = X_export[sel]
            y_explain = y_export[sel]
        else:
            explain_row_indices = np.arange(n, dtype=np.int64)
            X_explain = X_export
            y_explain = y_export

        _save_csv(out_dir / f"X_explain_{split_name}.csv", X_explain, header=feature_names)
        _save_csv(out_dir / f"y_explain_{split_name}.csv", y_explain.reshape(-1, 1), header=["label"])
        with open(out_dir / f"explain_row_indices_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump({"explain_row_indices": explain_row_indices.tolist()}, f, ensure_ascii=False, indent=2)

        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_explain)

        # multiclass: shap_values 通常为 list[n_class]，每个 (n, n_feat)
        save_path = out_dir / f"shap_values_{split_name}.npz"
        payload = {}
        if isinstance(shap_values, list):
            for i, sv in enumerate(shap_values):
                payload[f"class_{i}"] = np.asarray(sv, dtype=np.float32)
        else:
            payload["values"] = np.asarray(shap_values, dtype=np.float32)

        ev = getattr(explainer, "expected_value", None)
        if ev is not None:
            payload["expected_value"] = np.asarray(ev)
        np.savez_compressed(save_path, **payload)

        with open(out_dir / f"shap_meta_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split_name,
                    "n_samples": int(X_explain.shape[0]),
                    "n_features": int(X_explain.shape[1]),
                    "multiclass": bool(isinstance(shap_values, list)),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print(f"已导出到：{out_dir}")
    print(f"- 特征矩阵: X_{split_name}.csv")
    print(f"- 标签: y_{split_name}.csv")
    print(f"- LightGBM 模型: lightgbm_model.txt")
    if args.compute_shap:
        print(f"- SHAP 值: shap_values_{split_name}.npz（以及 X_explain_{split_name}.csv、explain_row_indices_{split_name}.json）")


if __name__ == "__main__":
    main()

