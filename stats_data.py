# -*- coding: utf-8 -*-
"""
统计 data 文件夹：各类型船舶样本数量与占比；可选：样本起止时间、两点时间间隔。
子目录按名称排序（支持 0,1,...,13 或 船型名 如 Tanker, Cargo Ship）。
支持 .txt 与 .csv。
"""
import argparse
from pathlib import Path
from datetime import datetime, timezone


def get_class_dirs(data_root: Path):
    """返回 data_root 下所有子目录（按名称排序），每个子目录代表一种船型。"""
    if not data_root.is_dir():
        return []
    dirs = sorted([d for d in data_root.iterdir() if d.is_dir()], key=lambda x: x.name)
    return dirs


def parse_timestamps_from_file(filepath):
    """
    从文件中解析时间戳序列（秒，可为 Unix 时间戳）。
    返回 list of float，若无法解析则返回空列表。
    """
    ts_list = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                raw = parts[0].lstrip("T").strip()
                ts = float(raw)
                ts_list.append(ts)
            except ValueError:
                continue
    return ts_list


def stats_one_file(filepath):
    """
    统计单个文件：点数、起止时间、相邻两点时间间隔的最小/最大（秒）。
    返回: (n_points, t_min, t_max, dt_min, dt_max) 或 None（无效文件）
    """
    ts_list = parse_timestamps_from_file(filepath)
    if len(ts_list) < 2:
        return None
    ts = sorted(ts_list)  # 按时间排序
    t_min, t_max = ts[0], ts[-1]
    dt_list = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    dt_list = [x for x in dt_list if x > 0]  # 忽略 0 或负间隔
    if not dt_list:
        return (len(ts_list), t_min, t_max, None, None)
    return (len(ts_list), t_min, t_max, min(dt_list), max(dt_list))


def format_ts(sec):
    """将秒时间戳转为可读时间（UTC）。若在合理 Unix 范围内则显示日期。"""
    if sec is None:
        return "N/A"
    try:
        if 1e9 <= sec <= 2e9:  # 常见 Unix 秒
            dt = datetime.fromtimestamp(sec, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC") + f" ({sec:.0f})"
        return f"{sec:.2f} (秒)"
    except Exception:
        return f"{sec}"


def main():
    parser = argparse.ArgumentParser(description="统计 data 下各类型船舶样本数量与占比")
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
    parser.add_argument("--no_time", action="store_true", help="不统计起止时间与时间间隔（仅样本数与占比）")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    data_root = base_dir / args.data_root
    if not data_root.is_dir():
        print(f"未找到目录: {data_root}")
        return

    class_dirs = get_class_dirs(data_root)
    if not class_dirs:
        print(f"在 {data_root} 下没有找到任何子目录。")
        return

    # 统计：各类型样本数、总文件数、可选的时间统计
    total_files = 0
    t_global_min = None
    t_global_max = None
    dt_global_min = None
    dt_global_max = None
    # 列表 (类型名, 样本数)，顺序与 class_dirs 一致
    rows = []

    for class_dir in class_dirs:
        count = 0
        for ext in ("*.txt", "*.csv"):
            for f in class_dir.glob(ext):
                total_files += 1
                count += 1
                if not args.no_time:
                    res = stats_one_file(f)
                    if res is None:
                        continue
                    n_pts, t_min, t_max, dt_min, dt_max = res
                    if t_global_min is None or t_min < t_global_min:
                        t_global_min = t_min
                    if t_global_max is None or t_max > t_global_max:
                        t_global_max = t_max
                    if dt_min is not None:
                        if dt_global_min is None or dt_min < dt_global_min:
                            dt_global_min = dt_min
                    if dt_max is not None:
                        if dt_global_max is None or dt_max > dt_global_max:
                            dt_global_max = dt_max
        rows.append((class_dir.name, count))

    # 输出：各类型船舶样本数量与占比
    print("=" * 70)
    print("data 文件夹 — 各类型船舶样本数量与占比")
    print("=" * 70)
    print(f"数据目录: {data_root}")
    print(f"样本文件总数（.txt + .csv）: {total_files}")
    print()
    print("各类型样本数与占比:")
    print("-" * 70)
    print(f"  {'类型（子目录名）':<36} {'样本数':>10}  {'占比':>8}")
    print("-" * 70)
    for name, n in rows:
        pct = (100.0 * n / total_files) if total_files else 0.0
        print(f"  {name:<36} {n:>10d}  {pct:>6.2f}%")
    print("-" * 70)
    print(f"  {'合计':<36} {total_files:>10d}  {100.0:>6.2f}%")
    print("=" * 70)

    if not args.no_time and (t_global_min is not None or dt_global_min is not None):
        print()
        print("样本起止时间（全数据集）:")
        print(f"  起始: {format_ts(t_global_min)}")
        print(f"  结束: {format_ts(t_global_max)}")
        print("相邻两点时间间隔（秒）:")
        print(f"  最小间隔: {dt_global_min if dt_global_min is not None else 'N/A'}")
        print(f"  最大间隔: {dt_global_max if dt_global_max is not None else 'N/A'}")


if __name__ == "__main__":
    main()
