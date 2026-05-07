#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 hybrid（Transformer特征 + 表格特征）对应的 SHAP 解释结果导出到 Excel。

输出 Excel 每个 sheet 对应一个样本；sheet 内包含：
- feature_name：特征名称
- feature_value：该样本的特征原始值（对应 X_*.csv 列）
- shap_value：该样本在指定类（pred 或 true）下的 SHAP 值
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError as e:
    raise ImportError("需要安装 pandas：pip install pandas") from e

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("需要安装 lightgbm：pip install lightgbm") from e


def _read_csv_matrix(csv_path: Path) -> np.ndarray:
    # 兼容两种格式：
    # - pandas 写出的 CSV：第一行是表头
    # - numpy.savetxt + header：第一行是表头（但数值从下一行开始）
    df = pd.read_csv(csv_path)
    return df.values.astype(np.float32)


def _load_feature_names(feature_names_json: Path) -> list[str]:
    with open(feature_names_json, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError(f"feature_names.json 内容不是 list：{feature_names_json}")
    return [str(x) for x in names]


def _load_labels(y_csv: Path) -> np.ndarray:
    df = pd.read_csv(y_csv)
    # 允许列名 label 或第一列
    if "label" in df.columns:
        y = df["label"].values
    else:
        y = df.iloc[:, 0].values
    return y.astype(np.int64)


def _pick_samples_preset(y_true: np.ndarray, y_pred: np.ndarray, id_cargo: int, id_fishing: int, minority_ids: list[int]):
    idx_cargo = None
    for i in range(len(y_true)):
        if y_true[i] == id_cargo and y_pred[i] == id_cargo:
            idx_cargo = int(i)
            break

    idx_fishing_mis = None
    for i in range(len(y_true)):
        if y_true[i] == id_fishing and y_pred[i] != id_fishing:
            idx_fishing_mis = int(i)
            break

    idx_minority = None
    for cid in minority_ids:
        for i in range(len(y_true)):
            if y_true[i] == cid:
                idx_minority = int(i)
                break
        if idx_minority is not None:
            break

    return [
        ("cargo_correct", idx_cargo),
        ("fishing_misclassified", idx_fishing_mis),
        ("minority_sample", idx_minority),
    ]


def _pick_one_sample_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> list[tuple[str, int]]:
    """
    每个类别选 1 条轨迹（相对当前 X 的行号）：
    优先：真值=该类 且 预测正确；
    否则：真值=该类的第一条。
    若该 split 中完全没有该类，则跳过（并应在调用处打印警告）。
    """
    items: list[tuple[str, int]] = []
    for c in range(num_classes):
        idx_ok = None
        for i in range(len(y_true)):
            if int(y_true[i]) == c and int(y_pred[i]) == c:
                idx_ok = int(i)
                break
        if idx_ok is not None:
            items.append((f"class_{c:02d}", idx_ok))
            continue
        idx_any = None
        for i in range(len(y_true)):
            if int(y_true[i]) == c:
                idx_any = int(i)
                break
        if idx_any is not None:
            items.append((f"class_{c:02d}_mis", idx_any))
    return items


def _load_lgb_booster_for_matrix(
    project_dir: Path,
    primary_path: Path,
    n_features_x: int,
) -> tuple[lgb.Booster, Path]:
    """
    加载与 X 矩阵列数一致的 LightGBM 模型。
    常见错误：outputs_shap/lightgbm_model.txt 来自纯表格 checkpoints/lightgbm.txt（维数远小于混合特征）。
    """
    candidates: list[Path] = []
    if primary_path.exists():
        candidates.append(primary_path.resolve())
    hybrid_ckpt = (project_dir / "checkpoints" / "lgb_hybrid.txt").resolve()
    if hybrid_ckpt not in candidates and hybrid_ckpt.exists():
        candidates.append(hybrid_ckpt)

    tried: list[tuple[Path, int]] = []
    for p in candidates:
        b = lgb.Booster(model_file=str(p))
        nf = b.num_feature()
        tried.append((p, nf))
        if nf == n_features_x:
            if primary_path.exists() and p.resolve() != primary_path.resolve():
                print(
                    f"[提示] {primary_path} 与当前 X 列数 {n_features_x} 不匹配，"
                    f"已自动改用：{p}"
                )
            return b, p

    lines = "\n".join(f"  - {tp}: num_feature={nf}" for tp, nf in tried)
    raise RuntimeError(
        f"LightGBM 模型特征数与 X 不一致：X 有 {n_features_x} 列。\n"
        f"已尝试的路径：\n{lines}\n\n"
        "原因：混合模型输入 = Transformer 深度特征 + 表格特征（例如 64+103=167），"
        "而 checkpoints/lightgbm.txt 多为「仅表格」模型（维数更少）。\n\n"
        "解决办法（任选其一）：\n"
        "  1) 重新运行 train_hybrid.py（会自动保存 checkpoints/lgb_hybrid.txt），再运行本脚本；\n"
        "  2) 运行 export_shap_hybrid.py 时不要指定错误的 --lgb_model_in，或指定 train_hybrid 保存的 lgb_hybrid.txt；\n"
        "  3) 手动指定：python export_shap_excel_hybrid.py --lgb_model checkpoints/lgb_hybrid.txt"
    )


def _normalize_shap_values(shap_values, num_classes: int) -> list[np.ndarray]:
    # 返回 list[class_id]，每个元素 shape=(n_samples, n_features)
    if isinstance(shap_values, list):
        if len(shap_values) != num_classes:
            # 有些版本可能返回 list 长度 < num_classes；尽量按实际长度处理
            return [np.asarray(x) for x in shap_values]
        return [np.asarray(x) for x in shap_values]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_classes, n_samples, n_features) 或 (n_samples, n_classes, n_features)
        if arr.shape[0] == num_classes:
            return [arr[c] for c in range(num_classes)]
        if arr.shape[1] == num_classes:
            return [arr[:, c, :] for c in range(num_classes)]
        # 部分 shap + LightGBM 多分类返回 (n_samples, n_features, n_classes)
        if arr.shape[2] == num_classes:
            return [arr[:, :, c] for c in range(num_classes)]
    raise ValueError(f"无法识别 shap_values 的形状/类型：type={type(shap_values)}, shape={getattr(shap_values,'shape',None)}")


def _compute_shap_values_lightgbm_native(booster: lgb.Booster, X: np.ndarray, num_classes: int) -> list[np.ndarray]:
    """
    不依赖 shap 包，直接用 LightGBM pred_contrib=True 输出贡献值作为（树模型）SHAP 值。

    多分类情况下，LightGBM 返回 shape=(n_samples, (n_features+1)*num_classes) 的二维数组，
    其中最后一维的 (n_features+1) 包含 bias 项。
    """
    contrib = booster.predict(X, pred_contrib=True)
    contrib = np.asarray(contrib)
    if contrib.ndim != 2:
        raise ValueError(f"LightGBM pred_contrib 期望 2D，实际 ndim={contrib.ndim}, shape={contrib.shape}")
    n_samples, flat_dim = contrib.shape
    nf1 = X.shape[1] + 1  # +bias
    if flat_dim != nf1 * num_classes:
        raise ValueError(
            f"LightGBM pred_contrib 展平维度不匹配：flat_dim={flat_dim}, nf1*num_classes={nf1*num_classes} "
            f"(X_features={X.shape[1]}, num_classes={num_classes})"
        )

    contrib_3d = contrib.reshape(n_samples, nf1, num_classes)
    # 每个类：去掉 bias（最后一项） => (n_samples, n_features)
    out: list[np.ndarray] = []
    for c in range(num_classes):
        out.append(contrib_3d[:, :-1, c].astype(np.float32))
    return out


def main():
    parser = argparse.ArgumentParser(description="导出 hybrid SHAP（feature_name/feature_value/shap_value）到 Excel")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="用于选择 X_*.csv 对应 split")
    parser.add_argument("--out_excel", type=str, default="", help="输出 Excel 路径；默认写到 outputs_shap/ 下")

    parser.add_argument("--out_dir", type=str, default="outputs_shap", help="包含 X_{split}.csv / y_{split}.csv / feature_names.json / lightgbm_model.txt")
    parser.add_argument("--lgb_model", type=str, default="", help="LightGBM 模型 txt；默认使用 out_dir/lightgbm_model.txt")
    parser.add_argument("--shap_npz", type=str, default="", help="如已存在 shap_values_{split}.npz，可直接读取导出；否则需要安装 shap 并计算")

    parser.add_argument(
        "--sample_selector",
        type=str,
        default="preset",
        choices=["preset", "index", "all_classes"],
        help="preset=3个典型样本；index=自定义行号；all_classes=每个类别各1条（适合全类对比/作图）",
    )
    parser.add_argument("--sample_indices", type=str, default="", help="当 sample_selector=index 时使用，如：0,15,27（相对 X_{split}.csv 的行号）")

    parser.add_argument("--shap_class", type=str, default="pred", choices=["pred", "true"], help="使用哪一类的 SHAP 值解释该样本")
    parser.add_argument("--top_k", type=int, default=0, help="只导出前 K 个特征；0 表示全部（不会做排序，只是截断前 K 列）")

    # 类别 ID（与项目中约定一致）
    parser.add_argument("--id_cargo", type=int, default=13)
    parser.add_argument("--id_fishing", type=int, default=2)
    parser.add_argument("--id_lng", type=int, default=5)
    parser.add_argument("--id_pleasure", type=int, default=9)
    parser.add_argument("--id_other_cargo", type=int, default=12)

    parser.add_argument("--no_compute_if_missing", action="store_true", help="若 shap_npz 不存在则直接报错，不计算")
    parser.add_argument("--no_cuda", action="store_true", help="保留接口；shap.TreeExplainer 通常在 CPU")

    args = parser.parse_args()

    project_dir = Path(__file__).parent
    out_dir = project_dir / args.out_dir
    split = args.split

    feature_names_path = out_dir / "feature_names.json"
    X_path = out_dir / f"X_{split}.csv"
    y_path = out_dir / f"y_{split}.csv"
    lgb_model_path = Path(args.lgb_model) if args.lgb_model else (out_dir / "lightgbm_model.txt")

    if not feature_names_path.exists():
        raise FileNotFoundError(f"找不到：{feature_names_path}")
    if not X_path.exists():
        raise FileNotFoundError(f"找不到：{X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"找不到：{y_path}")
    if not lgb_model_path.exists():
        raise FileNotFoundError(f"找不到：{lgb_model_path}")

    feature_names = _load_feature_names(feature_names_path)
    X = _read_csv_matrix(X_path)
    y_true = _load_labels(y_path)

    if X.shape[1] != len(feature_names):
        raise ValueError(f"特征维度不一致：X={X.shape[1]} vs feature_names={len(feature_names)}")
    if X.shape[0] != y_true.shape[0]:
        raise ValueError(f"样本数不一致：X={X.shape[0]} vs y={y_true.shape[0]}")

    # 输出文件
    if args.out_excel:
        out_excel_path = Path(args.out_excel)
    elif args.sample_selector == "all_classes":
        out_excel_path = out_dir / f"shap_feature_export_{split}_all_classes.xlsx"
    else:
        out_excel_path = out_dir / f"shap_feature_export_{split}.xlsx"
    out_excel_path.parent.mkdir(parents=True, exist_ok=True)

    # 载入 LightGBM（必须与 X 列数一致；自动尝试 checkpoints/lgb_hybrid.txt）
    booster, lgb_used_path = _load_lgb_booster_for_matrix(project_dir, lgb_model_path, X.shape[1])
    pred_proba = booster.predict(X)
    y_pred = np.argmax(pred_proba, axis=1).astype(np.int64)
    # 以模型输出维度为准（测试集可能缺少数类样本，unique(y) 会少于真实类别数）
    num_classes = int(pred_proba.shape[1])

    # 选择样本
    if args.sample_selector == "preset":
        minority_ids = [args.id_lng, args.id_pleasure, args.id_other_cargo]
        chosen = _pick_samples_preset(y_true, y_pred, args.id_cargo, args.id_fishing, minority_ids)
        sample_items = [(name, idx) for name, idx in chosen if idx is not None]
        if not sample_items:
            raise RuntimeError("未能在当前 split 找到符合条件的样本（cargo_correct/fishing_misclassified/minority）。")
    elif args.sample_selector == "all_classes":
        sample_items = _pick_one_sample_per_class(y_true, y_pred, num_classes)
        covered = {int(y_true[i]) for _, i in sample_items}
        missing = [c for c in range(num_classes) if c not in covered]
        if missing:
            print(f"[警告] 当前 split={split} 中以下类别没有样本，已跳过：{missing}")
        if not sample_items:
            raise RuntimeError(f"当前 split 没有任何可用类别样本，无法导出（num_classes={num_classes}）。")
        print(f"[all_classes] 共导出 {len(sample_items)} 个类别样本（每类最多 1 条），输出：{out_excel_path.name}")
    else:
        if not args.sample_indices:
            raise ValueError("--sample_indices 不能为空，例如 --sample_selector index --sample_indices 0,15")
        idxs = [int(s.strip()) for s in args.sample_indices.split(",") if s.strip()]
        sample_items = [(f"sample_{i}", i) for i in idxs]

    sample_indices = [idx for _, idx in sample_items]
    X_selected = X[sample_indices]

    # 读取/计算 shap values（只对所选样本计算/读取）
    shap_npz_path = Path(args.shap_npz) if args.shap_npz else (out_dir / f"shap_values_{split}.npz")
    shap_values_list: list[np.ndarray] | None = None
    explain_pos_map: dict[int, int] = {}
    if shap_npz_path.exists():
        npz = np.load(shap_npz_path, allow_pickle=True)
        keys = list(npz.keys())
        # 期望是 export_shap_hybrid.py 的输出格式：class_{i} 或 values
        if any(k.startswith("class_") for k in keys):
            shap_values_list = [None] * num_classes  # type: ignore[list-item]
            for cid in range(num_classes):
                key = f"class_{cid}"
                if key not in npz:
                    raise ValueError(f"{shap_npz_path} 缺少键 {key}（当前检测到 keys={keys[:10]}...）")
                shap_values_list[cid] = np.asarray(npz[key], dtype=np.float32)
        elif "values" in keys:
            shap_values_list = _normalize_shap_values(npz["values"], num_classes=num_classes)
        else:
            shap_values_list = None

        # 映射：shap 数组的行顺序 -> X_{split}.csv 的行索引
        explain_row_indices_path = out_dir / f"explain_row_indices_{split}.json"
        if explain_row_indices_path.exists():
            with open(explain_row_indices_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            explain_row_indices = payload.get("explain_row_indices", None)
            if not isinstance(explain_row_indices, list):
                raise ValueError(f"无法解析 explain_row_indices：{explain_row_indices_path}")
            explain_pos_map = {int(orig_idx): pos for pos, orig_idx in enumerate(explain_row_indices)}
        else:
            # 兼容：如果没有 explain_row_indices，默认 shap 是对整份 X 计算的
            # 此时 shap 的样本维度应等于 X.shape[0]
            n_explain = int(shap_values_list[0].shape[0]) if shap_values_list is not None else -1
            if n_explain != X.shape[0]:
                raise FileNotFoundError(
                    f"找不到 {explain_row_indices_path}，且 shap 样本数不匹配（X={X.shape[0]} vs shap={n_explain}）。"
                    "请确保用 export_shap_hybrid.py 生成 shap 时同时落盘 explain_row_indices_*，或直接在本脚本中重新计算（需安装 shap）。"
                )
            explain_pos_map = {i: i for i in range(X.shape[0])}
    if shap_values_list is None:
        if args.no_compute_if_missing:
            raise FileNotFoundError(f"找不到 shap_values：{shap_npz_path} 且 --no_compute_if_missing=true")
        # 优先：如果 shap 包可用就走 TreeExplainer；否则用 LightGBM 原生贡献值 pred_contrib=True。
        shap_computed = False
        try:
            import shap  # type: ignore

            explainer = shap.TreeExplainer(booster)
            shap_values_selected = explainer.shap_values(X_selected)
            shap_values_list = _normalize_shap_values(shap_values_selected, num_classes=num_classes)
            shap_computed = True
        except ImportError:
            shap_computed = False

        if not shap_computed:
            shap_values_list = _compute_shap_values_lightgbm_native(booster, X_selected, num_classes=num_classes)

        # 计算路径：解释样本顺序 = X_selected 的顺序 = sample_indices 的顺序
        explain_pos_map = {int(idx): pos for pos, idx in enumerate(sample_indices)}

    # 将 shap_class=p...映射到正确的 class_id
    # 注意：多分类 shap_values_list[class_id][sample_pos, feat_id]
    # 对应 shap_class=pred/true
    with pd.ExcelWriter(out_excel_path) as writer:
        # meta
        meta_rows = [
            ("split", split),
            ("sample_selector", args.sample_selector),
            ("X_csv", str(X_path)),
            ("lgb_model_arg", str(lgb_model_path)),
            ("lgb_model_used", str(lgb_used_path)),
            ("shap_npz_used", str(shap_npz_path) if shap_npz_path.exists() else ""),
            ("shap_class_mode", args.shap_class),
            ("num_classes_model", int(num_classes)),
            ("num_features", int(X.shape[1])),
        ]
        pd.DataFrame(meta_rows, columns=["key", "value"]).to_excel(writer, sheet_name="meta", index=False)

        info_rows = []
        for sheet_name, idx in sample_items:
            info_rows.append(
                {
                    "sheet": sheet_name[:31],
                    "idx_in_X_csv": int(idx),
                    "y_true": int(y_true[idx]),
                    "y_pred": int(y_pred[idx]),
                }
            )
        pd.DataFrame(info_rows).to_excel(writer, sheet_name="sample_info", index=False)

        for sheet_name, idx in sample_items:
            # Excel sheet 名限制 31 字符
            safe_sheet = sheet_name[:31]
            x_row = X[idx]
            true_c = int(y_true[idx])
            pred_c = int(y_pred[idx])
            c_use = pred_c if args.shap_class == "pred" else true_c

            # 取该类对应的 shap
            if c_use >= len(shap_values_list):
                raise ValueError(f"shap_values_list 没有类 {c_use} 的结果：可用类数量={len(shap_values_list)}")
            if idx not in explain_pos_map:
                raise ValueError(
                    f"所选样本 idx={idx} 不在 shap 的解释样本集合中（请确认 shap_npz 对应同一个 split，且未设置过小 shap_max_samples）。"
                )
            shap_pos = explain_pos_map[idx]
            shap_vec = shap_values_list[c_use][shap_pos].astype(np.float64)

            if args.top_k > 0:
                k = min(args.top_k, len(feature_names))
                feature_names_out = feature_names[:k]
                x_out = np.asarray(x_row).reshape(-1)[:k]
                shap_out = np.asarray(shap_vec).reshape(-1)[:k]
            else:
                feature_names_out = feature_names
                x_out = np.asarray(x_row).reshape(-1)
                shap_out = np.asarray(shap_vec).reshape(-1)

            # 保险：确保三列同长度（有些 shap 多维输出在特定版本下可能导致形状偏差）
            n_common = min(len(feature_names_out), len(x_out), len(shap_out))
            if n_common != len(feature_names_out) or n_common != len(x_out) or n_common != len(shap_out):
                if n_common > 0:
                    print(
                        f"[警告] sheet={safe_sheet} 特征长度不一致："
                        f"len(feature_names_out)={len(feature_names_out)}, len(x_out)={len(x_out)}, len(shap_out)={len(shap_out)}；已取前 {n_common} 项。"
                    )
            feature_names_out = feature_names_out[:n_common]
            x_out = np.asarray(x_out)[:n_common]
            shap_out = np.asarray(shap_out)[:n_common]

            df = pd.DataFrame(
                {
                    "feature_name": feature_names_out,
                    "feature_value": x_out,
                    "shap_value": shap_out,
                }
            )
            # 在表格顶部额外写入几行信息
            # （用单独 sheet 展示更干净，这里仍然写入到 sheet 的第一列前）
            # 为了保持简单，不插入额外行，只是用同一 sheet 里的 header 信息不足；因此写入到 sheet 外 meta。
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

    print(f"已导出：{out_excel_path}")
    print("包含：feature_name / feature_value / shap_value")


if __name__ == "__main__":
    main()

