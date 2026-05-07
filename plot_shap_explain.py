import argparse
from pathlib import Path

import numpy as np
import torch

from data_loader import (
    get_sequence_data_hybrid,
    get_trajectory_multi_segments,
    TABULAR_FEATURE_NAMES,
)
from train_hybrid import FeatureTransformer, extract_features

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("请先安装 lightgbm: pip install lightgbm")

try:
    import shap
except ImportError:
    raise ImportError("请先安装 shap: pip install shap")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("请先安装 matplotlib: pip install matplotlib")


# 约定的类别 ID（与训练时的数据标签保持一致）
# 若你的数据标签编号与这里不同，可在命令行参数中覆盖。
DEFAULT_ID_CARGO = 13
DEFAULT_ID_FISHING = 2
DEFAULT_ID_LNG = 5
DEFAULT_ID_PLEASURE = 9
DEFAULT_ID_OTHER_CARGO = 12


def build_lgb_inputs(
    data_root: Path,
    seq_len: int,
    batch_size: int,
    slide_stride: int,
    num_segments: int,
    segment_agg: str,
    device: torch.device,
    no_repair: bool = False,
):
    """
    基本流程与 train_hybrid/plot_confusion_matrix 保持一致：
    - 加载数据
    - 加载 Transformer checkpoint（已训练好的 feature_transformer.pt）
    - 提取每条轨迹的 Transformer 特征 + 表格特征，拼接成 LightGBM 输入。
    """
    from train_hybrid import FeatureTransformer  # 避免循环引用问题

    (
        _train_loader,
        _val_loader,
        _test_loader,
        scaler,
        num_classes,
        F,
        seq_len_loaded,
        idx_train,
        idx_val,
        idx_test,
        X_list,
        y_list,
        X_tab_all,
    ) = get_sequence_data_hybrid(
        str(data_root),
        seq_len=seq_len,
        batch_size=batch_size,
        slide_stride=slide_stride,
        repair_missing=not no_repair,
    )

    y_all = np.array(y_list, dtype=np.int64)

    ckpt_path = Path(__file__).parent / "checkpoints" / "feature_transformer.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"未找到 Transformer 检查点: {ckpt_path}，请先运行 train_hybrid.py")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    saved_args = ckpt.get("args", None)
    d_model = (getattr(saved_args, "d_model", None) if saved_args is not None else None) or 64

    model = FeatureTransformer(
        input_dim=F,
        d_model=d_model,
        nhead=getattr(saved_args, "nhead", 4),
        num_encoder_layers=getattr(saved_args, "num_layers", 2),
        dim_feedforward=getattr(saved_args, "dim_ff", 256),
        num_classes=num_classes,
        dropout=getattr(saved_args, "dropout", 0.1),
        max_len=seq_len_loaded,
        pool=getattr(saved_args, "pool", "last"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def multi_segments_and_normalize(indices):
        n_traj = len(indices)
        segs_list = [get_trajectory_multi_segments(X_list[i], seq_len_loaded, num_segments) for i in indices]
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

    # 构造特征名称（前半部分是 Transformer，后半部分是表格特征）
    n_tf = ft_train.shape[1]
    n_tab = X_tab_train.shape[1]
    tf_names = [f"TF_{i}" for i in range(n_tf)]
    all_feature_names = tf_names + list(TABULAR_FEATURE_NAMES[:n_tab])

    return (
        X_lgb_train,
        X_lgb_val,
        X_lgb_test,
        y_lgb_train,
        y_lgb_val,
        y_lgb_test,
        num_classes,
        all_feature_names,
        n_tab,
    )


def select_indices_for_explanation(
    y_true,
    y_pred,
    id_cargo: int,
    id_fishing: int,
    minority_ids,
):
    """
    选择三个典型样本：
    - 一个正确分类的 cargo ship
    - 一个误分类的 fishing vessel
    - 一个少数类样本（优先 LNG / Pleasure / Other Cargo）
    返回 (idx_cargo, idx_fishing_mis, idx_minority)，找不到则为 None。
    """
    idx_cargo = None
    for i in range(len(y_true)):
        if y_true[i] == id_cargo and y_pred[i] == id_cargo:
            idx_cargo = i
            break

    idx_fishing_mis = None
    for i in range(len(y_true)):
        if y_true[i] == id_fishing and y_pred[i] != id_fishing:
            idx_fishing_mis = i
            break

    idx_minority = None
    for cid in minority_ids:
        for i in range(len(y_true)):
            if y_true[i] == cid:
                idx_minority = i
                break
        if idx_minority is not None:
            break

    return idx_cargo, idx_fishing_mis, idx_minority


def save_force_plot(expected_value, shap_values, X, feature_names, out_path: Path, title: str):
    """
    保存单样本 SHAP force plot（条形图形式）为 PNG。
    """
    plt.figure(figsize=(10, 4))
    shap.plots._waterfall.waterfall_legacy(  # 使用旧版 waterfall/force 风格的条形图
        expected_value,
        shap_values,
        features=X,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Use SHAP (TreeExplainer) to explain LightGBM in hybrid model.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--slide_stride", type=int, default=64)
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--segment_agg", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--no_repair", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    # 类别 ID 可覆盖，默认约定与项目中的 0-13 映射一致
    parser.add_argument("--id_cargo", type=int, default=DEFAULT_ID_CARGO)
    parser.add_argument("--id_fishing", type=int, default=DEFAULT_ID_FISHING)
    parser.add_argument("--id_lng", type=int, default=DEFAULT_ID_LNG)
    parser.add_argument("--id_pleasure", type=int, default=DEFAULT_ID_PLEASURE)
    parser.add_argument("--id_other_cargo", type=int, default=DEFAULT_ID_OTHER_CARGO)
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="shap_hybrid",
        help="输出文件前缀（将生成 *_force_*.png 和 *_summary.png）",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    base_dir = Path(__file__).parent
    data_root = base_dir / args.data_root

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        num_classes,
        feature_names,
        n_tab,
    ) = build_lgb_inputs(
        data_root=data_root,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        slide_stride=args.slide_stride,
        num_segments=args.num_segments,
        segment_agg=args.segment_agg,
        device=device,
        no_repair=args.no_repair,
    )

    # 训练一个与 train_hybrid 相同设置的 LightGBM，用于 SHAP 分析
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        "objective": "multiclass",
        "num_class": int(num_classes),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 8,
        "verbosity": -1,
        "seed": 42,
        "device": "cpu",
    }
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # 预测测试集，用于选择典型样本
    pred_proba = lgb_model.predict(X_test)
    y_pred = np.argmax(pred_proba, axis=1)

    id_cargo = args.id_cargo
    id_fishing = args.id_fishing
    minority_ids = [args.id_lng, args.id_pleasure, args.id_other_cargo]

    idx_cargo, idx_fishing_mis, idx_minority = select_indices_for_explanation(
        y_true=y_test,
        y_pred=y_pred,
        id_cargo=id_cargo,
        id_fishing=id_fishing,
        minority_ids=minority_ids,
    )

    print("Selected indices for SHAP explanation:")
    print(f"  Correct Cargo Ship index:      {idx_cargo}")
    print(f"  Misclassified Fishing index:   {idx_fishing_mis}")
    print(f"  Minority-class sample index:   {idx_minority}")

    # TreeExplainer（基于训练集做背景）
    explainer = shap.TreeExplainer(lgb_model)
    shap_values_all = explainer.shap_values(X_test)  # list of (n_samples, n_features)

    out_prefix = base_dir / args.out_prefix

    # 1) Cargo Ship - 正确分类样本
    if idx_cargo is not None:
        c = int(y_pred[idx_cargo])
        shap_cargo = shap_values_all[c][idx_cargo]
        x_cargo = X_test[idx_cargo]
        out_path = out_prefix.with_name(out_prefix.stem + "_force_cargo.png")
        save_force_plot(
            explainer.expected_value[c],
            shap_cargo,
            x_cargo,
            feature_names,
            out_path,
            title=f"SHAP Force Plot - Correct Cargo Sample (class {c})",
        )
        print(f"Saved Cargo force plot: {out_path}")
    else:
        print("Warning: 未找到正确分类的 Cargo Ship 样本，跳过该 force plot。")

    # 2) Fishing Vessel - 误分类样本
    if idx_fishing_mis is not None:
        true_c = int(y_test[idx_fishing_mis])
        pred_c = int(y_pred[idx_fishing_mis])
        shap_fish = shap_values_all[pred_c][idx_fishing_mis]
        x_fish = X_test[idx_fishing_mis]
        out_path = out_prefix.with_name(out_prefix.stem + "_force_fishing_mis.png")
        save_force_plot(
            explainer.expected_value[pred_c],
            shap_fish,
            x_fish,
            feature_names,
            out_path,
            title=f"SHAP Force Plot - Misclassified Fishing Sample (true={true_c}, pred={pred_c})",
        )
        print(f"Saved Misclassified Fishing force plot: {out_path}")
    else:
        print("Warning: 未找到被误分类的 Fishing Vessel 样本，跳过该 force plot。")

    # 3) 少数类样本（例如 LNG / Pleasure / Other Cargo）
    if idx_minority is not None:
        c_min = int(y_pred[idx_minority])
        shap_min = shap_values_all[c_min][idx_minority]
        x_min = X_test[idx_minority]
        out_path = out_prefix.with_name(out_prefix.stem + "_force_minority.png")
        save_force_plot(
            explainer.expected_value[c_min],
            shap_min,
            x_min,
            feature_names,
            out_path,
            title=f"SHAP Force Plot - Minority-class Sample (pred={c_min})",
        )
        print(f"Saved Minority-class force plot: {out_path}")
    else:
        print("Warning: 未找到少数类样本（LNG/Pleasure/Other Cargo），跳过该 force plot。")

    # 全局：更美观的 beeswarm summary plot
    # 1) 整体（Transformer + 表格特征），只显示最重要的前 25 个特征，画布更宽，字号适中。
    matplotlib.rcParams.update({"font.size": 10})
    shap.summary_plot(
        shap_values_all,
        X_test,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=25,
        plot_size=(12, 6),
        color_bar=False,
    )
    out_summary = out_prefix.with_name(out_prefix.stem + "_summary_beeswarm.png")
    plt.tight_layout()
    plt.savefig(str(out_summary), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP beeswarm summary plot: {out_summary}")

    # 2) 仅表格统计特征的 beeswarm（去掉大量 TF_* 维度，更利于阅读）
    n_total = len(feature_names)
    n_tf = n_total - n_tab
    if n_tab > 0 and n_tf >= 0:
        feature_names_tab = feature_names[n_tf:]
        X_test_tab = X_test[:, n_tf:]
        # 将多分类的 shap list 按类别求平均，得到 (n_samples, n_tab) 单矩阵
        shap_values_tab_list = [sv[:, n_tf:] for sv in shap_values_all]
        shap_values_tab = np.mean(np.stack(shap_values_tab_list, axis=-1), axis=-1)

        try:
            shap.summary_plot(
                shap_values_tab,
                X_test_tab,
                feature_names=feature_names_tab,
                plot_type="dot",
                show=False,
                max_display=25,
                plot_size=(10, 6),
                color_bar=False,
            )
            out_summary_tab = out_prefix.with_name(out_prefix.stem + "_summary_beeswarm_tabular.png")
            plt.tight_layout()
            plt.savefig(str(out_summary_tab), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved SHAP beeswarm summary plot (tabular only): {out_summary_tab}")
        except AssertionError as e:
            # 某些 shap 版本在多分类/裁剪特征时内部形状检查过于严格，这里直接跳过表格-only 图，避免程序崩溃
            plt.close()
            print(f"Skip tabular-only beeswarm due to SHAP shape AssertionError: {e}")


if __name__ == "__main__":
    main()

