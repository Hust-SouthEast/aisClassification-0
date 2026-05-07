# -*- coding: utf-8 -*-
"""
t-SNE / UMAP 2D visualization comparing three feature types (same data, colored by class):
  - Tabular only (statistical features)
  - Transformer only (trajectory embedding)
  - Concatenated (Transformer + Tabular)
Two rows: t-SNE and UMAP; three columns: the three feature types. Points colored by true class.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from data_loader import get_sequence_data_hybrid, get_trajectory_multi_segments
from train_hybrid import FeatureTransformer, extract_features

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

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
    parser = argparse.ArgumentParser(description="t-SNE/UMAP: Tabular vs Transformer vs Concatenated")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_transformer", type=str, default="checkpoints/feature_transformer.pt")
    parser.add_argument("--out", type=str, default="tsne_umap_compare.png")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--slide_stride", type=int, default=64)
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--segment_agg", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--max_points", type=int, default=2000, help="Subsample for speed (0 = use all)")
    parser.add_argument("--tsne_perp", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--umap_nn", type=int, default=15)
    parser.add_argument("--no_repair", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--show", action="store_true", help="Show figure windows after saving")
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
    n_traj = len(y_list)
    indices = np.arange(n_traj)

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

    # ---------- 3) Extract trajectory features for all trajectories ----------
    def multi_segments_and_normalize(indices_list):
        n_traj = len(indices_list)
        segs_list = [get_trajectory_multi_segments(X_list[i], seq_len, num_segments) for i in indices_list]
        segs = np.stack(segs_list, axis=0)
        n, ns, L, f = segs.shape
        segs = scaler.transform(segs.reshape(-1, f)).reshape(n, ns, L, f)
        return torch.from_numpy(segs.astype(np.float32))

    def extract_trajectory_features(indices_list, agg=segment_agg):
        segs = multi_segments_and_normalize(indices_list)
        n, ns, L, f = segs.shape
        segs_flat = segs.reshape(n * ns, L, f)
        ft = extract_features(model, segs_flat, device)
        ft = ft.reshape(n, ns, -1)
        if agg == "mean":
            ft = ft.mean(axis=1)
        else:
            ft = ft.max(axis=1)[0]
        return ft

    print("Extracting Transformer features for all trajectories...")
    ft_all = extract_trajectory_features(indices)
    X_tab_all_arr = X_tab_all
    X_concat = np.hstack([ft_all, X_tab_all_arr])
    y = y_all

    # Subsample for speed
    if args.max_points > 0 and len(y) > args.max_points:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(len(y), size=args.max_points, replace=False)
        ft_all = ft_all[idx_sub]
        X_tab_all_arr = X_tab_all_arr[idx_sub]
        X_concat = X_concat[idx_sub]
        y = y[idx_sub]
        print(f"Subsampled to {len(y)} points for visualization.")
    n_pts = len(y)

    # Replace NaN/Inf so t-SNE/UMAP do not fail (tabular can have NaNs from stats)
    imp_tab = SimpleImputer(strategy="median")
    imp_ft = SimpleImputer(strategy="median")
    imp_con = SimpleImputer(strategy="median")
    X_tab_all_arr = np.asarray(imp_tab.fit_transform(X_tab_all_arr), dtype=np.float64)
    ft_all = np.asarray(imp_ft.fit_transform(ft_all), dtype=np.float64)
    X_concat = np.asarray(imp_con.fit_transform(X_concat), dtype=np.float64)
    # Also replace any remaining inf
    for arr in (X_tab_all_arr, ft_all, X_concat):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize per feature set (helps t-SNE/UMAP)
    scaler_tab = StandardScaler()
    scaler_ft = StandardScaler()
    scaler_con = StandardScaler()
    X_tab_s = scaler_tab.fit_transform(X_tab_all_arr)
    X_ft_s = scaler_ft.fit_transform(ft_all)
    X_con_s = scaler_con.fit_transform(X_concat)

    # ---------- 4) t-SNE ----------
    print("Running t-SNE (Tabular)...")
    tsne_tab = TSNE(n_components=2, perplexity=min(args.tsne_perp, n_pts - 1), random_state=42, max_iter=args.tsne_iter, verbose=0)
    emb_tsne_tab = tsne_tab.fit_transform(X_tab_s)
    print("Running t-SNE (Transformer)...")
    tsne_ft = TSNE(n_components=2, perplexity=min(args.tsne_perp, n_pts - 1), random_state=43, max_iter=args.tsne_iter, verbose=0)
    emb_tsne_ft = tsne_ft.fit_transform(X_ft_s)
    print("Running t-SNE (Concatenated)...")
    tsne_con = TSNE(n_components=2, perplexity=min(args.tsne_perp, n_pts - 1), random_state=44, max_iter=args.tsne_iter, verbose=0)
    emb_tsne_con = tsne_con.fit_transform(X_con_s)

    # ---------- 5) UMAP ----------
    if _HAS_UMAP:
        print("Running UMAP (Tabular)...")
        reducer_tab = umap.UMAP(n_components=2, n_neighbors=min(args.umap_nn, n_pts - 1), min_dist=0.1, random_state=42, verbose=False)
        emb_umap_tab = reducer_tab.fit_transform(X_tab_s)
        print("Running UMAP (Transformer)...")
        reducer_ft = umap.UMAP(n_components=2, n_neighbors=min(args.umap_nn, n_pts - 1), min_dist=0.1, random_state=43, verbose=False)
        emb_umap_ft = reducer_ft.fit_transform(X_ft_s)
        print("Running UMAP (Concatenated)...")
        reducer_con = umap.UMAP(n_components=2, n_neighbors=min(args.umap_nn, n_pts - 1), min_dist=0.1, random_state=44, verbose=False)
        emb_umap_con = reducer_con.fit_transform(X_con_s)
    else:
        emb_umap_tab = emb_umap_ft = emb_umap_con = None
        print("UMAP not installed (pip install umap-learn); plotting t-SNE only.")

    # ---------- 6) Plot: two separate figures (t-SNE window, UMAP window) ----------
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(num_classes)]
    out_base = Path(__file__).parent / args.out
    out_tsne = out_base.parent / (out_base.stem + "_tsne" + out_base.suffix)
    out_umap = out_base.parent / (out_base.stem + "_umap" + out_base.suffix)

    def draw_three_panels(axes, emb_tab, emb_ft, emb_con, titles, suptitle):
        for i, (ax, emb, title) in enumerate(zip(axes, [emb_tab, emb_ft, emb_con], titles)):
            for c in range(num_classes):
                mask = y == c
                if not np.any(mask):
                    continue
                if len(CLASS_NAMES) >= num_classes:
                    label = CLASS_NAMES[c]
                else:
                    label = f"Class {c}"
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    c=[colors[c]],
                    label=label,
                    s=12,
                    alpha=0.7,
                    edgecolors="none",
                )
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.legend(loc="best", fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.text(0.5, -0.18, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="center", fontsize=12)
        plt.suptitle(suptitle, fontsize=12)
        plt.tight_layout()

    # Figure 1: t-SNE
    fig_tsne, axes_tsne = plt.subplots(1, 3, figsize=(12, 5))
    draw_three_panels(
        axes_tsne,
        emb_tsne_tab, emb_tsne_ft, emb_tsne_con,
        ["Tabular only", "Transformer only", "Concatenated"],
        "t-SNE: Tabular vs Transformer vs Concatenated (colored by class)",
    )
    plt.savefig(str(out_tsne), dpi=150, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()
    print(f"Saved: {out_tsne}")

    # Figure 2: UMAP
    if _HAS_UMAP:
        fig_umap, axes_umap = plt.subplots(1, 3, figsize=(12, 5))
        draw_three_panels(
            axes_umap,
            emb_umap_tab, emb_umap_ft, emb_umap_con,
            ["Tabular only", "Transformer only", "Concatenated"],
            "UMAP: Tabular vs Transformer vs Concatenated (colored by class)",
        )
        plt.savefig(str(out_umap), dpi=150, bbox_inches="tight")
        if args.show:
            plt.show()
        plt.close()
        print(f"Saved: {out_umap}")


if __name__ == "__main__":
    main()
