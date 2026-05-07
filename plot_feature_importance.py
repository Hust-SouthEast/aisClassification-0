# -*- coding: utf-8 -*-
"""
Plot feature importance for the Feature-Transformer + LightGBM hybrid model.
Loads data and Transformer same as train_hybrid, trains LightGBM with feature names,
then plots importance (gain) sorted descending. All labels in English.
"""
import argparse
import numpy as np
import torch
from pathlib import Path

from data_loader import (
    get_sequence_data_hybrid,
    get_trajectory_multi_segments,
    sequences_to_tabular,
    TABULAR_FEATURE_NAMES,
)
from train_hybrid import FeatureTransformer, extract_features

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("Install lightgbm: pip install lightgbm")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Install matplotlib: pip install matplotlib")


def get_feature_names(d_model):
    """Feature names: Trans_emb_xx then tabular (same order as train_hybrid)."""
    trans = [f"Trans_emb_{i:02d}" for i in range(d_model)]
    return trans + list(TABULAR_FEATURE_NAMES)


def main():
    parser = argparse.ArgumentParser(description="Plot LightGBM feature importance (hybrid model)")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_transformer", type=str, default="checkpoints/feature_transformer.pt")
    parser.add_argument("--out", type=str, default="feature_importance.png")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--slide_stride", type=int, default=64)
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--segment_agg", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=20, help="Plot top K features (0 = all)")
    parser.add_argument("--importance_type", type=str, default="gain", choices=["gain", "split"])
    parser.add_argument("--no_repair", action="store_true", help="Disable AIS interpolation repair")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(__file__).parent / args.data_root
    ckpt_path = Path(__file__).parent / args.save_transformer

    if not ckpt_path.exists():
        print(f"Transformer checkpoint not found: {ckpt_path}")
        print("Run train_hybrid.py first to train and save the model.")
        return

    # ---------- 1) Load data ----------
    (
        _train_loader, _val_loader, _test_loader, scaler, num_classes, F, seq_len,
        idx_train, idx_val, idx_test, X_list, y_list, X_tab_all,
    ) = get_sequence_data_hybrid(
        str(data_root),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        slide_stride=args.slide_stride,
        repair_missing=not args.no_repair,
    )
    y_all = np.array(y_list, dtype=np.int64)

    # ---------- 2) Load Transformer ----------
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    saved_args = ckpt.get("args", None)
    d_model = (getattr(saved_args, "d_model", None) if saved_args is not None else None) or args.d_model
    num_segments = (getattr(saved_args, "num_segments", None) if saved_args is not None else None) or args.num_segments
    segment_agg = (getattr(saved_args, "segment_agg", None) if saved_args is not None else None) or args.segment_agg

    model = FeatureTransformer(
        input_dim=F,
        d_model=d_model,
        nhead=getattr(saved_args, "nhead", 4),
        num_encoder_layers=getattr(saved_args, "num_layers", 2),
        dim_feedforward=getattr(saved_args, "dim_ff", 256),
        num_classes=num_classes,
        dropout=getattr(saved_args, "dropout", 0.1),
        max_len=seq_len,
        pool=getattr(saved_args, "pool", "last"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---------- 3) Extract trajectory features (same as train_hybrid) ----------
    def multi_segments_and_normalize(indices):
        n_traj = len(indices)
        segs_list = [get_trajectory_multi_segments(X_list[i], seq_len, num_segments) for i in indices]
        segs = np.stack(segs_list, axis=0)
        n, ns, L, f = segs.shape
        segs = scaler.transform(segs.reshape(-1, f)).reshape(n, ns, L, f)
        return torch.from_numpy(segs.astype(np.float32))

    def extract_trajectory_features(indices, agg=segment_agg):
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

    ft_train = extract_trajectory_features(idx_train)
    ft_val = extract_trajectory_features(idx_val)
    X_tab_train = X_tab_all[idx_train]
    X_tab_val = X_tab_all[idx_val]
    X_lgb_train = np.hstack([ft_train, X_tab_train])
    X_lgb_val = np.hstack([ft_val, X_tab_val])
    y_lgb_train = y_all[idx_train]
    y_lgb_val = y_all[idx_val]

    feature_names = get_feature_names(ft_train.shape[1])
    assert len(feature_names) == X_lgb_train.shape[1], (
        f"Feature name count {len(feature_names)} != data dim {X_lgb_train.shape[1]}"
    )

    # ---------- 4) Train LightGBM with feature names ----------
    train_data = lgb.Dataset(X_lgb_train, label=y_lgb_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_lgb_val, label=y_lgb_val, reference=train_data, feature_name=feature_names)

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 8,
        "learning_rate": 0.05,
        "device": "cpu",  # avoid GPU for reproducible importance
        "verbosity": 0,
        "seed": 42,
    }
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # ---------- 5) Feature importance (gain or split) ----------
    importance = lgb_model.feature_importance(importance_type=args.importance_type)
    names = lgb_model.feature_name()
    order = np.argsort(importance)[::-1]
    imp_sorted = importance[order]
    names_sorted = [names[i] for i in order]

    top_k = args.top_k if args.top_k > 0 else len(imp_sorted)
    imp_plot = imp_sorted[:top_k]
    names_plot = names_sorted[:top_k]

    # Group colors: Transformer embedding vs Tabular
    def is_trans_emb(name):
        return isinstance(name, str) and name.startswith("Trans_emb_")
    colors = ["#1f77b4" if is_trans_emb(n) else "#ff7f0e" for n in names_plot]  # blue = Trans_emb, orange = Tabular

    # ---------- 6) Plot (horizontal bar, top = most important, grouped color) ----------
    fig, ax = plt.subplots(figsize=(10, max(5, top_k * 0.25)))
    y_pos = np.arange(len(names_plot), dtype=float)[::-1]
    bars = ax.barh(y_pos, imp_plot, height=0.7, color=colors, alpha=0.9, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_plot, fontsize=9)
    ax.set_xlabel("Importance (gain)", fontsize=11)
    ax.set_title("Feature importance (Feature-Transformer + LightGBM)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    # Legend for groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", alpha=0.9, edgecolor="black", label="Transformer embedding"),
        Patch(facecolor="#ff7f0e", alpha=0.9, edgecolor="black", label="Tabular"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path = Path(__file__).parent / args.out
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Top 10: {names_sorted[:10]}")


if __name__ == "__main__":
    main()
