# -*- coding: utf-8 -*-
"""
AIS 14分类航迹数据加载与预处理
数据格式：每行 T时间戳 经度 纬度 速度 航向
特征：经度、纬度、速度、航向（不做归一化、不使用时序特征）。
"""
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

try:
    from scipy.interpolate import interp1d
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def parse_trajectory_file(filepath, max_points=None):
    """解析单条航迹文件，返回 (N, 4)：经度、纬度、速度、航向"""
    points = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                lon, lat, speed, course = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                points.append([lon, lat, speed, course])
            except (ValueError, IndexError):
                continue
    arr = np.array(points, dtype=np.float32) if points else np.zeros((1, 4), dtype=np.float32)
    if max_points and len(arr) > max_points:
        arr = arr[:max_points]
    return arr


def _parse_trajectory_file_with_ts(filepath, max_points=None):
    """
    解析单条航迹文件，保留时间戳；无效或缺失行用 NaN 占位，便于后续插值修复。
    返回 (ts, arr)：ts (N,), arr (N, 4) 经度、纬度、速度、航向。
    """
    ts_list, points = [], []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                t = float(parts[0].lstrip("T"))
                lon, lat, speed, course = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                ts_list.append(t)
                points.append([lon, lat, speed, course])
            except (ValueError, IndexError):
                if ts_list:
                    ts_list.append(ts_list[-1] + 1.0)
                    points.append([np.nan, np.nan, np.nan, np.nan])
    if not ts_list:
        return np.array([0.0], dtype=np.float32), np.zeros((1, 4), dtype=np.float32)
    ts = np.array(ts_list, dtype=np.float32)
    arr = np.array(points, dtype=np.float32)
    if len(ts) >= 2 and np.any(np.diff(ts) < 0):
        order = np.argsort(ts)
        ts = ts[order]
        arr = arr[order]
    if max_points and len(ts) > max_points:
        ts = ts[:max_points]
        arr = arr[:max_points]
    return ts, arr


def repair_trajectory_with_interpolation(
    ts, arr, max_gap_seconds=300, method="linear", min_valid_ratio=0.5
):
    """
    使用线性/样条插值修复 AIS 轨迹缺失与短时间隙。
    - 对每列用 scipy.interpolate 填充 NaN。
    - 若存在时间间隙 > max_gap_seconds，在间隙内按均匀时间步插值补点（短缺失修复）。
    返回修复后的 (ts_out, arr_out)，arr_out 为 (T, 4)，无 NaN。
    """
    if not _HAS_SCIPY:
        # 无 scipy 时仅做前向填充
        out = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return ts, out.astype(np.float32)
    ts = np.asarray(ts, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(arr).all(axis=1)
    if valid.sum() < max(2, int(min_valid_ratio * len(ts))):
        out = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return ts.astype(np.float32), out.astype(np.float32)
    # 先仅在有效点上插值，填充 NaN 位置
    t_valid = ts[valid]
    arr_valid = arr[valid]
    for j in range(4):
        col = arr[:, j].copy()
        if np.isfinite(col).sum() < 2:
            col = np.nan_to_num(col, nan=np.nanmean(col) if np.isfinite(col).any() else 0.0)
            arr[:, j] = col
            continue
        f = interp1d(
            t_valid, arr_valid[:, j],
            kind=method if method in ("linear", "nearest") else "linear",
            bounds_error=False,
            fill_value=(arr_valid[:, j].min(), arr_valid[:, j].max()),
        )
        nan_mask = ~np.isfinite(col)
        if nan_mask.any():
            arr[nan_mask, j] = f(ts[nan_mask])
    # 短缺失：在时间间隙内插值补点
    if len(ts) >= 2 and max_gap_seconds > 0:
        dt = np.diff(ts)
        dt = np.where(dt <= 0, np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0, dt)
        step = min(float(np.median(dt)), max_gap_seconds)
        gaps = np.where(dt > max_gap_seconds)[0]
        if len(gaps) > 0:
            t_extra = []
            for i in gaps:
                t_start, t_end = ts[i], ts[i + 1]
                n_ins = max(1, int((t_end - t_start) / step))
                t_extra.extend(np.linspace(t_start, t_end, n_ins + 1)[1:-1].tolist())
            if t_extra:
                t_extra = np.array(t_extra, dtype=np.float64)
                t_all = np.concatenate([ts, t_extra])
                t_all = np.sort(t_all)
                arr_all = np.zeros((len(t_all), 4), dtype=np.float64)
                for j in range(4):
                    f = interp1d(ts, arr[:, j], kind="linear", bounds_error=False, fill_value="extrapolate")
                    arr_all[:, j] = f(t_all)
                ts = t_all
                arr = arr_all
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return ts.astype(np.float32), arr.astype(np.float32)


def load_all_trajectories(data_root, max_points_per_trajectory=256, return_paths=False, repair_missing=True):
    """
    加载所有类别下的航迹。
    data_root: 如 'data'，其下为 0,1,...,13 子目录
    repair_missing: 若 True 且 return_paths=True，对每条轨迹用插值修复缺失/短间隙（需 scipy）。
    返回:
      - return_paths=False: X_list (list of (T,4)), y_list (list of int labels)
      - return_paths=True:  X_list, y_list, path_list (list of str filepaths)
    """
    data_root = Path(data_root)
    X_list, y_list, p_list = [], [], []

    # 支持两种目录结构：
    # 1) data_root/0,1,...,13 这样的数字目录
    # 2) data_root/Bulk Carrier, data_root/Container Ship 等任意以船型命名的目录
    # 实现方式：优先使用实际存在的子目录，按名称排序，顺序映射为 0..n-1 标签。
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    # 兼容旧版本：若没有任何子目录，则尝试 0..13 数字目录
    if not class_dirs:
        for cid in range(14):
            d = data_root / str(cid)
            if d.is_dir():
                class_dirs.append(d)
        class_dirs = sorted(class_dirs, key=lambda x: x.name)

    for class_id, class_dir in enumerate(class_dirs):
        for f in class_dir.glob("*.txt"):
            path = str(f)
            if repair_missing and return_paths and _HAS_SCIPY:
                ts, arr = _parse_trajectory_file_with_ts(path, max_points=None)
                ts, arr = repair_trajectory_with_interpolation(ts, arr, max_gap_seconds=300, method="linear")
                if max_points_per_trajectory and len(arr) > max_points_per_trajectory:
                    arr = arr[:max_points_per_trajectory]
            else:
                arr = parse_trajectory_file(path, max_points=max_points_per_trajectory)
            if len(arr) < 2:  # 至少2个点
                continue
            X_list.append(arr.astype(np.float32))
            y_list.append(class_id)
            p_list.append(path)

    if return_paths:
        return X_list, y_list, p_list
    return X_list, y_list


def pad_or_truncate(seq_list, seq_len, pad_value=0.0):
    """将变长序列列表变为 (n, seq_len, feat_dim) 的数组，不足补 pad_value，过长截断"""
    feat_dim = seq_list[0].shape[1]
    n = len(seq_list)
    out = np.full((n, seq_len, feat_dim), pad_value, dtype=np.float32)
    for i, seq in enumerate(seq_list):
        T = min(len(seq), seq_len)
        out[i, :T] = seq[:T]
    return out


def _pad_single(seq, seq_len, pad_value=0.0):
    """单条序列填充或截断为 (seq_len, F)，不足补 pad_value"""
    F = seq.shape[1]
    out = np.full((seq_len, F), pad_value, dtype=np.float32)
    T = min(len(seq), seq_len)
    out[:T] = seq[:T]
    return out


def get_trajectory_multi_segments(seq, seq_len, num_segments=3):
    """
    从一条轨迹中取多段（首段、中段、末段），每段定长 seq_len，不足补 0。
    返回 (num_segments, seq_len, F)。
    """
    T = len(seq)
    segs = []
    if num_segments == 1:
        segs.append(_pad_single(seq, seq_len))
    else:
        # 首、中、末的起始下标
        starts = [
            0,
            max(0, (T - seq_len) // 2),
            max(0, T - seq_len),
        ]
        for k in range(min(num_segments, 3)):
            start = starts[k]
            end = min(start + seq_len, T)
            seg = seq[start:end] if end > start else seq[:1]
            segs.append(_pad_single(seg, seq_len))
        while len(segs) < num_segments:
            segs.append(segs[-1].copy())
    return np.stack(segs, axis=0)


def _parse_times_from_file(filepath, max_points=None):
    """从原始文件解析时间戳序列（秒）。若解析失败则退化为等间隔 1。"""
    ts_list = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            try:
                ts = float(parts[0].lstrip("T"))
                ts_list.append(ts)
            except ValueError:
                continue
    if not ts_list:
        return np.arange(0, 1, dtype=np.float32)
    ts = np.array(ts_list, dtype=np.float32)
    if max_points and len(ts) > max_points:
        ts = ts[:max_points]
    # 确保单调不减（若乱序则排序）
    if len(ts) >= 2 and np.any(np.diff(ts) < 0):
        ts = np.sort(ts)
    return ts


def _wrap_angle_deg(x):
    """将角度差包裹到 [-180, 180]。"""
    return (x + 180.0) % 360.0 - 180.0


def _haversine_m(lon1, lat1, lon2, lat2):
    """两点球面距离（米）。输入为度。"""
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _convex_hull_area_m2(lon, lat):
    """
    近似计算凸包面积（m^2）：将经纬度做等距矩形近似投影到平面后算凸包多边形面积。
    纯 numpy 实现（Andrew monotonic chain）。
    """
    if len(lon) < 3:
        return 0.0
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lat0 = np.deg2rad(np.mean(lat))
    x = (np.deg2rad(lon) * 6371000.0) * np.cos(lat0)
    y = np.deg2rad(lat) * 6371000.0
    pts = np.stack([x, y], axis=1)
    # 去重
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return 0.0
    # 多边形面积
    xh, yh = hull[:, 0], hull[:, 1]
    area = 0.5 * np.abs(np.dot(xh, np.roll(yh, -1)) - np.dot(yh, np.roll(xh, -1)))
    return float(area)


def _safe_stats(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


def _safe_stats_extended(x):
    """扩展统计：+ var, range, IQR, skewness, kurtosis, q10, q90。"""
    base = _safe_stats(x)
    x = np.asarray(x, dtype=np.float32)
    if x.size < 2:
        base["var"] = 0.0
        base["range"] = 0.0
        base["iqr"] = 0.0
        base["skew"] = 0.0
        base["kurtosis"] = 0.0
        base["q10"] = base["min"]
        base["q90"] = base["max"]
        return base
    base["var"] = float(np.var(x))
    base["range"] = float(base["max"] - base["min"])
    base["iqr"] = float(base["q75"] - base["q25"])
    base["q10"] = float(np.quantile(x, 0.10))
    base["q90"] = float(np.quantile(x, 0.90))
    if _HAS_SCIPY and base["std"] > 1e-10:
        base["skew"] = float(scipy_stats.skew(x))
        base["kurtosis"] = float(scipy_stats.kurtosis(x))
    else:
        base["skew"] = 0.0
        base["kurtosis"] = 0.0
    return base


def _safe_autocorr(x, lag):
    x = np.asarray(x, dtype=np.float32)
    if x.size <= lag:
        return 0.0
    x0 = x[:-lag]
    x1 = x[lag:]
    s0, s1 = np.std(x0), np.std(x1)
    if s0 <= 0 or s1 <= 0 or not np.isfinite(s0) or not np.isfinite(s1):
        return 0.0
    cov = np.mean((x0 - np.mean(x0)) * (x1 - np.mean(x1)))
    c = cov / (s0 * s1)
    return 0.0 if not np.isfinite(c) else float(np.clip(c, -1.0, 1.0))


def _fft_topk_mag(x, k=5):
    """返回 rFFT 的前 k 个非 DC 幅值（不足补0）。"""
    x = np.asarray(x, dtype=np.float32)
    if x.size < 4:
        return [0.0] * k
    x = x - np.mean(x)
    spec = np.fft.rfft(x)
    mag = np.abs(spec).astype(np.float32)
    if mag.size > 0:
        mag[0] = 0.0  # 去 DC
    out = mag[1 : 1 + k].tolist()
    if len(out) < k:
        out += [0.0] * (k - len(out))
    return [float(v) for v in out]


# 扩展表格特征数量（与下方 feats 列表长度一致，空轨迹时使用）
_TABULAR_N_FEATURES = 103

# 表格特征名称（与 sequences_to_tabular 的 feats 顺序一致，供特征重要性等使用）
TABULAR_FEATURE_NAMES = [
    "sog_mean", "sog_median", "sog_std", "sog_var", "sog_min", "sog_max", "sog_range", "sog_iqr",
    "sog_skew", "sog_kurtosis", "sog_q10", "sog_q25", "sog_q75", "sog_q90",
    "cog_mean", "cog_median", "cog_std", "cog_var", "cog_min", "cog_max", "cog_range", "cog_iqr",
    "cog_skew", "cog_kurtosis", "cog_q10", "cog_q25", "cog_q75", "cog_q90",
    "dsog_mean", "dsog_std", "dsog_min", "dsog_max", "dsog_skew",
    "dcog_mean", "dcog_std", "dcog_min", "dcog_max", "dcog_skew",
    "turn_count_15", "dcog_std_angle", "dcog_abs_max",
    "acc_mean", "acc_std", "rot_mean", "rot_std", "rot_abs_max",
    "curv_mean", "curv_std", "curv_max",
    "dt_mean", "dt_std", "dt_min", "dt_max", "duration", "sampling_density", "stationary_ratio",
    "total_dist", "straight_dist", "detour_index", "hull_area", "bbox_area", "bbox_aspect",
    "bendiness", "sinuosity", "stop_points", "stop_time",
    "sog_fft1", "sog_fft2", "sog_fft3", "sog_fft4", "sog_fft5",
    "cog_fft1", "cog_fft2", "cog_fft3", "cog_fft4", "cog_fft5",
    "dsog_fft1", "dsog_fft2", "dsog_fft3", "dsog_fft4", "dsog_fft5",
    "sog_ac1", "sog_ac2", "sog_ac3", "sog_ac5", "cog_ac1", "cog_ac2", "cog_ac3", "cog_ac5",
    "seg1_sog_mean", "seg1_sog_std", "seg2_sog_mean", "seg2_sog_std", "seg3_sog_mean", "seg3_sog_std",
    "seg1_cog_mean", "seg1_cog_std", "seg2_cog_mean", "seg2_cog_std", "seg3_cog_mean", "seg3_cog_std",
    "dsog_half_diff", "traj_len",
]


def sequences_to_tabular(X_list, path_list=None, max_points_per_trajectory=256):
    """
    将轨迹序列转为表格特征（扩展版）：基本统计+扩展、一阶/二阶差分、几何/形状、
    角度与漂移、频域/自相关、时间/间隔、分段/局部统计。
    时间戳仅在此处（若提供 path_list）从原文件解析。
    """
    rows = []
    for idx, seq in enumerate(X_list):
        if len(seq) == 0:
            rows.append(np.zeros(_TABULAR_N_FEATURES, dtype=np.float32))
            continue
        lon, lat, spd, course = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3]
        T = len(seq)

        if path_list is not None:
            ts = _parse_times_from_file(path_list[idx], max_points=max_points_per_trajectory)
            ts = ts[:T] if len(ts) >= T else np.pad(ts, (0, T - len(ts)), mode="edge")
        else:
            ts = np.arange(T, dtype=np.float32)
        dt = np.diff(ts).astype(np.float32)
        dt = np.where(dt <= 0, 1.0, dt)

        d_sog = np.diff(spd).astype(np.float32) if T > 1 else np.zeros((0,), dtype=np.float32)
        d_cog = _wrap_angle_deg(np.diff(course).astype(np.float32)) if T > 1 else np.zeros((0,), dtype=np.float32)

        if T > 1:
            step_dist = _haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:]).astype(np.float32)
        else:
            step_dist = np.zeros((0,), dtype=np.float32)
        total_dist = float(np.sum(step_dist)) if step_dist.size else 0.0
        straight_dist = float(_haversine_m(lon[0], lat[0], lon[-1], lat[-1])) if T > 1 else 0.0
        detour_index = float(total_dist / (straight_dist + 1e-6)) if T > 1 else 0.0
        duration = float(ts[-1] - ts[0]) if T > 1 else 0.0
        sampling_density = float(T / (duration + 1e-6))
        stationary_ratio = float(np.mean(spd < 1.0))

        dt_stats = _safe_stats(dt)
        dt_stats_ext = _safe_stats_extended(dt) if dt.size >= 2 else dt_stats

        sog_stats = _safe_stats_extended(spd)
        cog_stats = _safe_stats_extended(course)
        dsog_stats = _safe_stats_extended(d_sog) if d_sog.size >= 2 else _safe_stats(d_sog)
        dcog_stats = _safe_stats_extended(d_cog) if d_cog.size >= 2 else _safe_stats(d_cog)

        turn_count_15 = float(np.sum(np.abs(d_cog) > 15.0)) if d_cog.size else 0.0
        dcog_std = float(np.std(d_cog)) if d_cog.size else 0.0
        dcog_abs_max = float(np.max(np.abs(d_cog))) if d_cog.size else 0.0

        acc = (d_sog / dt) if d_sog.size else np.zeros((0,), dtype=np.float32)
        acc_stats = _safe_stats(acc)
        rot = (np.deg2rad(np.abs(d_cog)) / dt) if (d_cog.size and dt.size) else np.zeros((0,), dtype=np.float32)
        rot_stats = _safe_stats(rot)
        rot_abs_max = float(np.max(np.abs(rot))) if rot.size else 0.0

        if d_cog.size and step_dist.size:
            dcog_rad = np.deg2rad(np.abs(d_cog)).astype(np.float32)
            curvature = dcog_rad / (step_dist + 1e-3)
        else:
            curvature = np.zeros((0,), dtype=np.float32)
        curv_stats = _safe_stats(curvature)
        curv_max = float(np.max(curvature)) if curvature.size else 0.0

        bbox_w = float(np.max(lon) - np.min(lon)) if T > 0 else 0.0
        bbox_h = float(np.max(lat) - np.min(lat)) if T > 0 else 0.0
        bbox_area = bbox_w * bbox_h if T > 0 else 0.0
        bbox_aspect = float((max(bbox_w, bbox_h) + 1e-6) / (min(bbox_w, bbox_h) + 1e-6)) if T > 0 else 0.0
        hull_area = _convex_hull_area_m2(lon, lat)
        total_turn_angle = float(np.sum(np.abs(d_cog))) if d_cog.size else 0.0
        bendiness = float(total_turn_angle / (total_dist + 1e-6)) if total_dist > 0 else 0.0
        sinuosity = detour_index

        stop_mask = spd < 1.0
        stop_points = float(np.sum(stop_mask))
        stop_time = float(np.sum(dt[(stop_mask[:-1] & stop_mask[1:])])) if T > 1 else 0.0

        sog_fft5 = _fft_topk_mag(spd, k=5)
        cog_fft5 = _fft_topk_mag(course, k=5)
        dsog_fft5 = _fft_topk_mag(d_sog, k=5) if d_sog.size >= 4 else [0.0] * 5
        sog_ac1, sog_ac2, sog_ac3 = _safe_autocorr(spd, 1), _safe_autocorr(spd, 2), _safe_autocorr(spd, 3)
        sog_ac5 = _safe_autocorr(spd, 5) if T > 5 else 0.0
        cog_ac1, cog_ac2, cog_ac3 = _safe_autocorr(course, 1), _safe_autocorr(course, 2), _safe_autocorr(course, 3)
        cog_ac5 = _safe_autocorr(course, 5) if T > 5 else 0.0

        n3 = max(1, T // 3)
        seg1 = slice(0, n3)
        seg2 = slice(n3, 2 * n3)
        seg3 = slice(2 * n3, T)
        seg1_sog = _safe_stats(spd[seg1])
        seg2_sog = _safe_stats(spd[seg2])
        seg3_sog = _safe_stats(spd[seg3])
        seg1_cog = _safe_stats(course[seg1])
        seg2_cog = _safe_stats(course[seg2])
        seg3_cog = _safe_stats(course[seg3])
        half = T // 2
        if d_sog.size and half >= 1:
            dsog_first = np.mean(d_sog[: half])
            dsog_second = np.mean(d_sog[half:]) if half < len(d_sog) else 0.0
            dsog_half_diff = float(dsog_first - dsog_second)
        else:
            dsog_half_diff = 0.0

        feats = [
            # 基本统计扩展 SOG
            sog_stats["mean"], sog_stats["median"], sog_stats["std"], sog_stats["var"], sog_stats["min"], sog_stats["max"],
            sog_stats["range"], sog_stats["iqr"], sog_stats["skew"], sog_stats["kurtosis"],
            sog_stats["q10"], sog_stats["q25"], sog_stats["q75"], sog_stats["q90"],
            # 基本统计扩展 COG
            cog_stats["mean"], cog_stats["median"], cog_stats["std"], cog_stats["var"], cog_stats["min"], cog_stats["max"],
            cog_stats["range"], cog_stats["iqr"], cog_stats["skew"], cog_stats["kurtosis"],
            cog_stats["q10"], cog_stats["q25"], cog_stats["q75"], cog_stats["q90"],
            # 一阶差分 ΔSOG
            dsog_stats["mean"], dsog_stats["std"], dsog_stats["min"], dsog_stats["max"], dsog_stats["skew"],
            # 一阶差分 ΔCOG（最小角度差）
            dcog_stats["mean"], dcog_stats["std"], dcog_stats["min"], dcog_stats["max"], dcog_stats["skew"],
            turn_count_15, dcog_std, dcog_abs_max,
            # 二阶：加速度、转向率 RoT
            acc_stats["mean"], acc_stats["std"],
            rot_stats["mean"], rot_stats["std"], rot_abs_max,
            # 曲率
            curv_stats["mean"], curv_stats["std"], curv_max,
            # 时间间隔
            dt_stats["mean"], dt_stats["std"], dt_stats_ext["min"], dt_stats_ext["max"], duration,
            sampling_density, stationary_ratio,
            # 几何/形状
            total_dist, straight_dist, detour_index, hull_area, bbox_area, bbox_aspect, bendiness, sinuosity,
            stop_points, stop_time,
            # 频域 FFT
            *sog_fft5, *cog_fft5, *dsog_fft5,
            # 自相关 lag 1,2,3,5
            sog_ac1, sog_ac2, sog_ac3, sog_ac5, cog_ac1, cog_ac2, cog_ac3, cog_ac5,
            # 分段：前/中/后 1/3 的 SOG/COG mean+std
            seg1_sog["mean"], seg1_sog["std"], seg2_sog["mean"], seg2_sog["std"], seg3_sog["mean"], seg3_sog["std"],
            seg1_cog["mean"], seg1_cog["std"], seg2_cog["mean"], seg2_cog["std"], seg3_cog["mean"], seg3_cog["std"],
            dsog_half_diff,
            float(T),
        ]
        rows.append(feats)

    return np.array(rows, dtype=np.float32)


def get_sequence_data(
    data_root,
    seq_len=128,
    max_points_per_trajectory=2048,
    test_ratio=0.15,
    val_ratio=0.15,
    batch_size=64,
    seed=42,
    slide_stride=64,
):
    """
    获取用于 LSTM/GRU/Transformer/CNN-LSTM 的 DataLoader。
    - 按轨迹(文件)划分 train/val/test，保证测试集与训练集来自不同 csv，无泄露。
    - 训练集中：长度>seq_len 的轨迹用滑窗切成多个 seq_len 样本；长度<=seq_len 的做一条填充样本。
    - 验证/测试集：每条轨迹只取一个样本(前 seq_len 点或填充)，不滑窗。
    特征：经度、纬度、速度、航向（原始值，不做归一化）。
    返回: train_loader, val_loader, test_loader, scaler, n_classes, input_dim, seq_len
    """
    X_list, y_list = load_all_trajectories(
        data_root,
        max_points_per_trajectory=max_points_per_trajectory,
        return_paths=False,
    )
    if not X_list:
        raise RuntimeError(f"在目录 {data_root} 下没有找到任何轨迹文件，请检查数据子目录和文件命名。")

    y = np.array(y_list, dtype=np.int64)
    n_traj = len(X_list)
    F = X_list[0].shape[1]
    n_classes = int(np.max(y) + 1)

    # 按轨迹索引做分层划分，保证同一 csv 只出现在一个集合
    indices = np.arange(n_traj)
    idx_train, idx_test = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=y
    )
    y_train = y[idx_train]
    idx_train, idx_val = train_test_split(
        idx_train, test_size=val_ratio / (1 - test_ratio), random_state=seed, stratify=y_train
    )

    # 训练集：滑窗扩样（仅对长度>seq_len 的轨迹）
    train_seqs, train_labels = [], []
    for i in idx_train:
        seq = X_list[i]  # (T, F)
        T = len(seq)
        if T > seq_len:
            for start in range(0, T - seq_len + 1, slide_stride):
                train_seqs.append(seq[start : start + seq_len].copy())
                train_labels.append(y_list[i])
        else:
            train_seqs.append(_pad_single(seq, seq_len))
            train_labels.append(y_list[i])

    # 验证集 / 测试集：每条轨迹只取一个样本（前 seq_len 点或填充）
    val_seqs = [_pad_single(X_list[i], seq_len) for i in idx_val]
    val_labels = y[idx_val]
    test_seqs = [_pad_single(X_list[i], seq_len) for i in idx_test]
    test_labels = y[idx_test]

    X_train = np.stack(train_seqs, axis=0)
    y_train = np.array(train_labels, dtype=np.int64)
    X_val = np.stack(val_seqs, axis=0)
    y_val = np.array(val_labels, dtype=np.int64)
    X_test = np.stack(test_seqs, axis=0)
    y_test = np.array(test_labels, dtype=np.int64)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, None, n_classes, F, seq_len


def get_sequence_data_hybrid(
    data_root,
    seq_len=128,
    max_points_per_trajectory=2048,
    test_ratio=0.15,
    val_ratio=0.15,
    batch_size=64,
    seed=42,
    slide_stride=64,
    repair_missing=True,
):
    """
    用于 Feature-Transformer + LightGBM 混合架构。
    - 与 get_sequence_data 相同的轨迹划分与滑窗，但序列做 MinMaxScaler 归一化（仅用训练集拟合）。
    - repair_missing: 是否对轨迹做插值修复（短缺失/NaN）。
    - 额外返回：idx_train, idx_val, idx_test（轨迹索引）、X_list, y_list、X_tab_all（每条轨迹的表格特征，与 X_list 同序）。
    返回: train_loader, val_loader, test_loader, scaler, n_classes, F, seq_len,
          idx_train, idx_val, idx_test, X_list, y_list, X_tab_all
    """
    X_list, y_list, p_list = load_all_trajectories(
        data_root,
        max_points_per_trajectory=max_points_per_trajectory,
        return_paths=True,
        repair_missing=repair_missing,
    )
    if not X_list:
        raise RuntimeError(f"在目录 {data_root} 下没有找到任何轨迹文件，请检查数据子目录和文件命名。")
    X_tab_all = sequences_to_tabular(X_list, path_list=p_list, max_points_per_trajectory=max_points_per_trajectory)
    y = np.array(y_list, dtype=np.int64)
    n_traj = len(X_list)
    F = X_list[0].shape[1]
    n_classes = int(np.max(y) + 1)

    indices = np.arange(n_traj)
    idx_train, idx_test = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=y)
    y_train = y[idx_train]
    idx_train, idx_val = train_test_split(
        idx_train, test_size=val_ratio / (1 - test_ratio), random_state=seed, stratify=y_train
    )

    train_seqs, train_labels = [], []
    for i in idx_train:
        seq = X_list[i]
        T = len(seq)
        if T > seq_len:
            for start in range(0, T - seq_len + 1, slide_stride):
                train_seqs.append(seq[start : start + seq_len].copy())
                train_labels.append(y_list[i])
        else:
            train_seqs.append(_pad_single(seq, seq_len))
            train_labels.append(y_list[i])

    val_seqs = [_pad_single(X_list[i], seq_len) for i in idx_val]
    val_labels = y[idx_val]
    test_seqs = [_pad_single(X_list[i], seq_len) for i in idx_test]
    test_labels = y[idx_test]

    X_train_raw = np.stack(train_seqs, axis=0)
    y_train = np.array(train_labels, dtype=np.int64)
    X_val_raw = np.stack(val_seqs, axis=0)
    y_val = np.array(val_labels, dtype=np.int64)
    X_test_raw = np.stack(test_seqs, axis=0)
    y_test = np.array(test_labels, dtype=np.int64)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_raw.reshape(-1, F))
    X_train = scaler.transform(X_train_raw.reshape(-1, F)).reshape(X_train_raw.shape)
    X_val = scaler.transform(X_val_raw.reshape(-1, F)).reshape(X_val_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, F)).reshape(X_test_raw.shape)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        n_classes,
        F,
        seq_len,
        idx_train,
        idx_val,
        idx_test,
        X_list,
        y_list,
        X_tab_all,
    )


def get_tabular_data(data_root, max_points_per_trajectory=256, test_ratio=0.15, val_ratio=0.15, seed=42):
    """
    获取用于 LightGBM 的表格数据（原始值，不做归一化）。
    返回: X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    X_list, y_list, p_list = load_all_trajectories(
        data_root,
        max_points_per_trajectory=max_points_per_trajectory,
        return_paths=True,
    )
    if not X_list:
        raise RuntimeError(f"在目录 {data_root} 下没有找到任何轨迹文件，请检查数据子目录和文件命名。")
    X_tab = sequences_to_tabular(X_list, path_list=p_list, max_points_per_trajectory=max_points_per_trajectory)
    y = np.array(y_list, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio / (1 - test_ratio), random_state=seed, stratify=y_train
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, None


if __name__ == "__main__":
    # 快速检查：按轨迹划分 + 训练集滑窗扩样
    r = Path(__file__).parent / "data"
    train_loader, val_loader, test_loader, scaler, n_cls, feat_dim, slen = get_sequence_data(
        str(r), seq_len=128, batch_size=32, slide_stride=64
    )
    print("训练样本数(滑窗后):", len(train_loader.dataset), "验证:", len(val_loader.dataset), "测试:", len(test_loader.dataset))
    for x, y in train_loader:
        print("Sequence batch:", x.shape, y.shape)
        break
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_tabular_data(str(r))
    print("Tabular X_train:", X_train.shape, "y_train:", y_train.shape)
