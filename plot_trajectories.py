# -*- coding: utf-8 -*-
"""
Plot trajectories from a folder (default: plot_traj). One subplot per trajectory file,
all subplots in one figure. Labels in English.
"""
import argparse
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data_loader import parse_trajectory_file


def _clean_trajectory(lon, lat, max_jump_factor=10.0, window=3):
    """
    简单预处理：
    1) 计算相邻点直线距离，去掉过大的跳变点（> max_jump_factor * 中位距离）作为异常点；
    2) 对剩余点做小窗口移动平均平滑，去除高频噪声。
    """
    lon = np.asarray(lon, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float32)
    if lon.size < 3:
        return lon, lat

    # 计算相邻点欧氏距离（经纬度近似，绘图足够）
    d = np.hypot(np.diff(lon), np.diff(lat))
    med = np.median(d[d > 0]) if np.any(d > 0) else 0.0
    if med > 0:
        thresh = max_jump_factor * med
        keep = np.ones(lon.size, dtype=bool)
        jump_idx = np.where(d > thresh)[0]
        # 去掉跳变后的那个点
        keep[jump_idx + 1] = False
        lon = lon[keep]
        lat = lat[keep]
        if lon.size < 3:
            return lon, lat

    # 简单移动平均平滑
    if window >= 3 and lon.size > window:
        k = window // 2
        kernel = np.ones(window, dtype=np.float32) / window
        lon_s = np.convolve(lon, kernel, mode="same")
        lat_s = np.convolve(lat, kernel, mode="same")
        # 首尾用原值替换，避免边缘偏移
        lon_s[:k] = lon[:k]
        lon_s[-k:] = lon[-k:]
        lat_s[:k] = lat[:k]
        lat_s[-k:] = lat[-k:]
        lon, lat = lon_s, lat_s

    return lon, lat


def load_trajectories_from_folder(traj_dir, max_points=2000):
    """Load all .txt trajectory files from traj_dir. Return list of (filename_stem, (lon, lat))."""
    traj_dir = Path(traj_dir)
    if not traj_dir.is_dir():
        return []
    out = []
    for f in sorted(traj_dir.glob("*.txt")):
        arr = parse_trajectory_file(str(f), max_points=max_points)
        if len(arr) < 2:
            continue
        lon_raw, lat_raw = arr[:, 0].copy(), arr[:, 1].copy()
        lon, lat = _clean_trajectory(lon_raw, lat_raw)
        if len(lon) < 2:
            continue
        out.append((f.stem, (lon, lat)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot one trajectory per file in a folder (one figure)")
    parser.add_argument("--traj_dir", type=str, default="plot_traj", help="Folder containing trajectory .txt files")
    parser.add_argument("--out", type=str, default="trajectories_plot_traj.png")
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--n_cols", type=int, default=0, help="Number of columns (0 = auto from sqrt(n))")
    parser.add_argument("--show", action="store_true", help="Show window after saving")
    args = parser.parse_args()

    traj_dir = Path(__file__).parent / args.traj_dir
    trajectories = load_trajectories_from_folder(traj_dir, max_points=args.max_points)

    if not trajectories:
        print(f"No trajectories found in {traj_dir} (expect .txt files)")
        return

    n_plots = len(trajectories)
    if args.n_cols > 0:
        n_cols = args.n_cols
    else:
        n_cols = max(1, int(math.ceil(math.sqrt(n_plots))))
    n_rows = math.ceil(n_plots / n_cols)

    ship_type_names = [
        "Tanker",            # 0
        "Vehicle Carrier",   # 1
        "Bulk Carrier",      # 2  (与 Fishing Vessel 对调)
        "Chemical Tanker",   # 3
        "Tug",               # 4
        "LNG Carrier",       # 5
        "Passenger Ship",    # 6
        "Ro-Ro Ship",        # 7
        "Reefer Ship",       # 8
        "Pleasure Craft",    # 9
        "Fishing Vessel",    # 10
        "Container Ship",    # 11
        "Other Cargo",       # 12
        "Cargo Ship",        # 13
    ]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 每个子图使用不同颜色
    color_cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", None)
    if not color_cycle:
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                       "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                       "#bcbd22", "#17becf"]

    for i in range(n_plots):
        _, (lon, lat) = trajectories[i]
        ax = axes[i]
        color = color_cycle[i % len(color_cycle)]
        ax.plot(lon, lat, color=color, linewidth=1.2, alpha=0.9)
        ax.scatter(lon[0], lat[0], color="green", s=30, zorder=5, label="Start")
        ax.scatter(lon[-1], lat[-1], color="red", s=30, marker="s", zorder=5, label="End")
        # 子图标题：按 0-13 船型名称映射，超出则用 classN
        if i < len(ship_type_names):
            title = ship_type_names[i]
        else:
            title = f"class{i + 1}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_path = Path(__file__).parent / args.out
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path} ({n_plots} trajectories)")
    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
