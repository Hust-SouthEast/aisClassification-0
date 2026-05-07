# -*- coding: utf-8 -*-
"""
将 data 下每个类别子目录中的轨迹样本文件重命名为 1.txt、2.txt、3.txt …

说明：
- 每个类别目录独立编号（各自从 1 开始）。
- 采用「先改为临时名、再改为目标名」的两步重命名，避免 Windows 上覆盖冲突。
- 默认按文件名排序；可加 --natural 做数字自然序（如 2.txt 在 10.txt 前）。

用法示例：
  python rename_samples_by_class.py --data_root data --dry-run
  python rename_samples_by_class.py --data_root data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def natural_sort_key(name: str):
    """将字符串按「数字段按数值比较」排序，例如 a2.txt < a10.txt。"""
    parts = re.split(r"(\d+)", name)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def main():
    parser = argparse.ArgumentParser(description="按类别将 data 下样本重命名为 1.txt, 2.txt, …")
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录（其下为各类子文件夹）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的操作，不真正改名")
    parser.add_argument(
        "--natural",
        action="store_true",
        help="按自然序排序文件名（否则按普通字典序）",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".txt",
        help="要重命名的扩展名，默认 .txt",
    )
    args = parser.parse_args()

    base = Path(__file__).parent / args.data_root
    if not base.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{base}")

    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"

    class_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"{base} 下没有子目录（类别文件夹）。")

    for class_dir in class_dirs:
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()]
        if not files:
            print(f"[跳过] {class_dir.name}/ 下无 *{ext} 文件")
            continue

        if args.natural:
            files.sort(key=lambda p: natural_sort_key(p.name))
        else:
            files.sort(key=lambda p: p.name.lower())

        n = len(files)
        temp_names = [class_dir / f"__rename_tmp_{i:06d}{ext}" for i in range(n)]

        print(f"\n[{class_dir.name}] 共 {n} 个文件 -> 1{ext} .. {n}{ext}")

        if args.dry_run:
            for i, src in enumerate(files):
                print(f"  {src.name} -> {i + 1}{ext}")
            continue

        # 第一步：全部改为临时名（若临时名已存在则先报错，避免误覆盖）
        for src, tmp in zip(files, temp_names):
            if tmp.exists():
                raise FileExistsError(f"临时文件已存在，请先清理：{tmp}")
            src.rename(tmp)

        # 第二步：临时名 -> 1.txt, 2.txt, ...
        for i, tmp in enumerate(temp_names):
            dest = class_dir / f"{i + 1}{ext}"
            if dest.exists():
                raise FileExistsError(f"目标已存在（不应发生）：{dest}")
            tmp.rename(dest)

        print(f"  完成。")

    print("\n全部处理结束。")


if __name__ == "__main__":
    main()
