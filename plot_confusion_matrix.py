# -*- coding: utf-8 -*-
"""
Plot 14×14 confusion matrix heatmap for the Feature-Transformer + LightGBM model.
Uses the same data/model as train_hybrid: loads checkpoint, builds test features,
trains LightGBM, predicts on test set, then plots confusion matrix with color intensity.
Labels in English.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix

from data_loader import (
    get_sequence_data_hybrid,
    get_trajectory_multi_segments,
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


CLASS_NAMES = [
    "Tanker",            # 0
    "Vehicle Carrier",   # 1
    "Fishing Vessel",    # 2
    "Chemical Tanker",   # 3
    "Tug",               # 4
    "LNG Carrier",       # 5
    "Passenger Ship",    # 6
    "Ro-Ro Ship",        # 7
    "Reefer Ship",       # 8
    "Pleasure Craft",    # 9
    "Bulk Carrier",      # 10
    "Container Ship",    # 11
    "Other Cargo",       # 12
    "Cargo Ship",        # 13
]


def main():
    parser = argparse.ArgumentParser(description="Plot 14×14 confusion matrix heatmap")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_transformer", type=str, default="checkpoints/feature_transformer.pt")
    parser.add_argument("--out", type=str, default="confusion_matrix.png")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--slide_stride", type=int, default=64)
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--segment_agg", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--no_repair", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--cmap", type=str, default="Blues", help="Colormap name (e.g. Blues, YlOrRd)")
    parser.add_argument("--annot", action="store_true", default=True, help="Show count in each cell")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(__file__).parent / args.data_root
    ckpt_path = Path(__file__).parent / args.save_transformer

    if not ckpt_path.exists():
        print(f"Transformer checkpoint not found: {ckpt_path}. Run train_hybrid.py first.")
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
    d_model = (getattr(saved_args, "d_model", None) if saved_args is not None else None) or 64
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

    # ---------- 3) Extract trajectory features ----------
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

    # ---------- 4) Train LightGBM (same defaults as train_hybrid) ----------
    train_data = lgb.Dataset(X_lgb_train, label=y_lgb_train)
    val_data = lgb.Dataset(X_lgb_val, label=y_lgb_val, reference=train_data)
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
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    lgb_model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data], callbacks=callbacks,
    )

    # ---------- 5) Predict test set and compute confusion matrix ----------
    pred_proba = lgb_model.predict(X_lgb_test)
    y_pred = np.argmax(pred_proba, axis=1)
    y_true = y_lgb_test

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # ---------- 6) Plot 14×14 heatmap ----------
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap=args.cmap, aspect="equal", vmin=0, vmax=cm.max() if cm.size else 1)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))

    if len(CLASS_NAMES) >= num_classes:
        tick_labels = CLASS_NAMES[:num_classes]
    else:
        tick_labels = [f"Class {i}" for i in range(num_classes)]
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if args.annot:
        thresh = (cm.max() - cm.min()) / 2.0 + cm.min() if cm.size else 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion Matrix (Feature-Transformer + LightGBM)", fontsize=12)
    fig.colorbar(im, ax=ax, label="Count", shrink=0.8)
    fig.tight_layout()
    out_path = Path(__file__).parent / args.out
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Test accuracy: {(y_pred == y_true).mean():.4f}")


if __name__ == "__main__":
    main()
