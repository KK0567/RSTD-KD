#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ECU-IoFT 实验三：概率校准与攻击概率可靠性分析。

用途：
1) 读取已有 student/teacher 预测 CSV，不重新训练模型；
2) 使用验证集拟合 Platt Scaling 与 Isotonic Regression；
3) 在 train/val/test 上输出 Brier Score、ECE、Reliability Curve；
4) 对比校准前后的阈值扫描稳定性；
5) 可选：用攻击概率直接划分 Low/Medium/High，检查风险分层稳定性。

默认面向 Distilled Student：
  --pred-dir output1/student_risk_cascade_v1
  --prob-col student_prob_binary

运行示例：
python 4.calibration_reliability_analysis_ecu.py ^
  --pred-dir output1/student_risk_cascade_v1 ^
  --output-dir output1/exp3_probability_calibration_student ^
  --prob-col student_prob_binary

如分析 Teacher：
python 4.calibration_reliability_analysis_ecu.py ^
  --pred-dir output1/teacher_risk_cascade_v1 ^
  --output-dir output1/exp3_probability_calibration_teacher ^
  --prob-col teacher_prob_binary
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# -----------------------------
# 基础工具
# -----------------------------
def safe_json_dump(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def load_split_preds(pred_dir: str | Path, prefix: str) -> Dict[str, pd.DataFrame]:
    pred_dir = Path(pred_dir)
    out = {}
    for split in ["train", "val", "test"]:
        path = pred_dir / f"{prefix}_risk_preds_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"未找到预测文件：{path}")
        out[split] = pd.read_csv(path)
    return out


def infer_prefix(pred_dir: str | Path) -> str:
    pred_dir = Path(pred_dir)
    if (pred_dir / "student_risk_preds_val.csv").exists():
        return "student"
    if (pred_dir / "teacher_risk_preds_val.csv").exists():
        return "teacher"
    raise FileNotFoundError(
        f"{pred_dir} 下未找到 student_risk_preds_val.csv 或 teacher_risk_preds_val.csv"
    )


# -----------------------------
# 概率校准器
# -----------------------------
class IdentityCalibrator:
    def fit(self, p: np.ndarray, y: np.ndarray):
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return clip_prob(p)


class PlattLogitCalibrator:
    """对原始概率取 logit 后做 LogisticRegression，即标准 Platt scaling 形式。"""

    def __init__(self, seed: int = 42):
        self.model = LogisticRegression(solver="lbfgs", random_state=seed, max_iter=1000)

    def fit(self, p: np.ndarray, y: np.ndarray):
        x = logit(p).reshape(-1, 1)
        self.model.fit(x, y.astype(int))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        x = logit(p).reshape(-1, 1)
        return clip_prob(self.model.predict_proba(x)[:, 1])


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, p: np.ndarray, y: np.ndarray):
        self.model.fit(clip_prob(p), y.astype(int))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return clip_prob(self.model.predict(clip_prob(p)))


# -----------------------------
# 可靠性指标
# -----------------------------
def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[float, pd.DataFrame]:
    """
    ECE = sum_k |B_k|/n * |acc(B_k)-conf(B_k)|.
    对二分类，这里的 acc(B_k) 用 bin 内真实攻击比例表示，conf(B_k) 用平均攻击概率表示。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = clip_prob(y_prob)
    n = len(y_true)
    if n == 0:
        raise ValueError("空数组无法计算 ECE")

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        edges[0] = 0.0
        edges[-1] = 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin_id": i,
                "bin_left": lo,
                "bin_right": hi,
                "count": 0,
                "mean_confidence": np.nan,
                "attack_rate": np.nan,
                "abs_gap": np.nan,
                "weighted_gap": 0.0,
            })
            continue
        conf = float(y_prob[mask].mean())
        rate = float(y_true[mask].mean())
        gap = abs(rate - conf)
        weighted = (count / n) * gap
        ece += weighted
        rows.append({
            "bin_id": i,
            "bin_left": lo,
            "bin_right": hi,
            "count": count,
            "mean_confidence": conf,
            "attack_rate": rate,
            "abs_gap": gap,
            "weighted_gap": weighted,
        })
    return float(ece), pd.DataFrame(rows)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = clip_prob(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    _, bins = expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return bins.dropna(subset=["mean_confidence", "attack_rate"]).copy()


# -----------------------------
# 阈值扫描与稳定性
# -----------------------------
def scan_binary_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float],
    c_fp: float,
    c_fn: float,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = clip_prob(y_prob)
    rows = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        cost = c_fp * fpr + c_fn * fnr
        rows.append({
            "threshold": float(th),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "cost": float(cost),
        })
    return pd.DataFrame(rows)


def threshold_stability_summary(scan_df: pd.DataFrame, eps_ratio: float = 0.01) -> Dict[str, float]:
    df = scan_df.sort_values("threshold").reset_index(drop=True)
    costs = df["cost"].values.astype(float)
    ths = df["threshold"].values.astype(float)
    min_idx = int(np.nanargmin(costs))
    min_cost = float(costs[min_idx])
    best_th = float(ths[min_idx])
    tol = min_cost * (1.0 + eps_ratio) + 1e-12
    near = df.loc[df["cost"] <= tol, "threshold"].values
    near_width = float(near.max() - near.min()) if len(near) else 0.0
    total_variation = float(np.abs(np.diff(costs)).sum()) if len(costs) > 1 else 0.0
    local_minima = 0
    for i in range(1, len(costs) - 1):
        if costs[i] <= costs[i - 1] and costs[i] <= costs[i + 1]:
            local_minima += 1
    return {
        "best_threshold": best_th,
        "min_cost": min_cost,
        "near_optimal_width_1pct": near_width,
        "cost_total_variation": total_variation,
        "local_minima_count": int(local_minima),
    }


# -----------------------------
# 风险分层：用概率阈值直接划分 Low/Medium/High
# -----------------------------
def predict_risk_by_probability(y_prob: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    p = clip_prob(y_prob)
    return np.where(p < tau1, 0, np.where(p < tau2, 1, 2)).astype(int)


def risk_metrics_from_prob(y_risk: np.ndarray, y_prob: np.ndarray, tau1: float, tau2: float) -> Dict[str, float]:
    y_pred = predict_risk_by_probability(y_prob, tau1, tau2)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_risk.astype(int), y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    return {
        "tau1": float(tau1),
        "tau2": float(tau2),
        "accuracy": float(accuracy_score(y_risk.astype(int), y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_risk.astype(int), y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "high_recall": float(precision_recall_fscore_support(
            y_risk.astype(int), y_pred, labels=[2], average="macro", zero_division=0
        )[1]),
    }


def select_risk_thresholds_on_val(
    y_risk: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray,
    metric: str = "f1_macro",
) -> Dict[str, float]:
    best = None
    for tau1 in grid:
        for tau2 in grid:
            if tau2 <= tau1:
                continue
            m = risk_metrics_from_prob(y_risk, y_prob, tau1, tau2)
            if best is None or m[metric] > best[metric]:
                best = m
    if best is None:
        raise RuntimeError("未找到有效风险阈值，请检查 grid")
    return best


# -----------------------------
# 绘图
# -----------------------------
def plot_reliability_curve(points_by_method: Dict[str, pd.DataFrame], out_path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfect calibration")
    for name, df in points_by_method.items():
        if df.empty:
            continue
        ax.plot(df["mean_confidence"], df["attack_rate"], marker="o", linewidth=1.6, label=name)
    ax.set_xlabel("Mean predicted attack probability")
    ax.set_ylabel("Observed attack frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=600)
    fig.savefig(out_path_base.with_suffix(".pdf"))
    fig.savefig(out_path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_threshold_scan(scan_by_method: Dict[str, pd.DataFrame], out_path_base: Path, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, df in scan_by_method.items():
        ax.plot(df["threshold"], df["cost"], linewidth=1.8, label=name)
    ax.set_xlabel("Attack probability threshold")
    ax.set_ylabel("Task-aware cost")
    ax.set_title(title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=600)
    fig.savefig(out_path_base.with_suffix(".pdf"))
    fig.savefig(out_path_base.with_suffix(".svg"))
    plt.close(fig)


# -----------------------------
# 主流程
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="概率校准与攻击概率可靠性分析")
    parser.add_argument("--pred-dir", type=str, default="output1/teacher_risk_direct_v2")
    parser.add_argument("--output-dir", type=str, default="output1/exp3_probability_calibration_teacher")
    parser.add_argument("--prefix", type=str, default="teacher", choices=["auto", "student", "teacher"])
    parser.add_argument("--prob-col", type=str, default="teacher_prob_binary")
    parser.add_argument("--label-col", type=str, default="y_bin")
    parser.add_argument("--risk-label-col", type=str, default="y_risk")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-fp-logistics", type=float, default=3.0)
    parser.add_argument("--cost-fn-logistics", type=float, default=1.0)
    parser.add_argument("--cost-fp-inspection", type=float, default=1.0)
    parser.add_argument("--cost-fn-inspection", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    prefix = infer_prefix(args.pred_dir) if args.prefix == "auto" else args.prefix
    dfs = load_split_preds(args.pred_dir, prefix)

    for split, df in dfs.items():
        missing = [c for c in [args.label_col, args.prob_col, args.risk_label_col] if c not in df.columns]
        if missing:
            raise ValueError(f"{split} 缺少列 {missing}；当前列为：{df.columns.tolist()}")

    # 关键原则：校准器只在 validation split 上拟合；test 只用于最终报告。
    y_val = dfs["val"][args.label_col].values.astype(int)
    p_val_raw = dfs["val"][args.prob_col].values.astype(float)

    calibrators = {
        "Uncalibrated": IdentityCalibrator().fit(p_val_raw, y_val),
        "Platt": PlattLogitCalibrator(seed=args.seed).fit(p_val_raw, y_val),
        "Isotonic": IsotonicCalibrator().fit(p_val_raw, y_val),
    }

    metric_rows: List[Dict] = []
    reliability_rows: List[pd.DataFrame] = []
    calibrated_pred_tables: Dict[str, pd.DataFrame] = {}

    for split, df in dfs.items():
        y = df[args.label_col].values.astype(int)
        p_raw = df[args.prob_col].values.astype(float)
        y_risk = df[args.risk_label_col].values.astype(int)
        out_pred = df.copy()

        for name, cal in calibrators.items():
            p_cal = cal.predict(p_raw)
            col = f"prob_attack_{name.lower()}"
            out_pred[col] = p_cal.astype(np.float32)

            ece, ece_bins = expected_calibration_error(y, p_cal, n_bins=args.n_bins, strategy="uniform")
            brier = float(brier_score_loss(y, p_cal))
            bm = binary_metrics(y, p_cal, threshold=0.5)
            metric_rows.append({
                "split": split,
                "method": name,
                "brier_score": brier,
                "ece": ece,
                **{k: v for k, v in bm.items() if k not in ["brier_score"]},
            })
            ece_bins.insert(0, "method", name)
            ece_bins.insert(0, "split", split)
            reliability_rows.append(ece_bins)

        calibrated_pred_tables[split] = out_pred
        out_pred.to_csv(out_dir / f"calibrated_predictions_{split}.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "calibration_metrics.csv", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(out_dir / "reliability_bins.csv", index=False)

    # Reliability Curve：建议论文主图用 test，val 可作为补充图。
    for split in ["val", "test"]:
        points_by_method = {}
        y = dfs[split][args.label_col].values.astype(int)
        p_raw = dfs[split][args.prob_col].values.astype(float)
        for name, cal in calibrators.items():
            points_by_method[name] = reliability_points(y, cal.predict(p_raw), n_bins=args.n_bins)
        plot_reliability_curve(points_by_method, out_dir / f"reliability_curve_{split}")

    # 阈值扫描：分别模拟两个任务代价。
    thresholds = np.round(np.arange(0.0, 1.0 + 1e-9, args.threshold_step), 6)
    scan_rows: List[pd.DataFrame] = []
    stability_rows: List[Dict] = []
    task_cfgs = {
        "logistics_fp_sensitive": (args.cost_fp_logistics, args.cost_fn_logistics),
        "inspection_fn_sensitive": (args.cost_fp_inspection, args.cost_fn_inspection),
    }

    for split in ["val", "test"]:
        y = dfs[split][args.label_col].values.astype(int)
        p_raw = dfs[split][args.prob_col].values.astype(float)
        for task_name, (c_fp, c_fn) in task_cfgs.items():
            scan_by_method = {}
            for name, cal in calibrators.items():
                p = cal.predict(p_raw)
                scan = scan_binary_thresholds(y, p, thresholds, c_fp=c_fp, c_fn=c_fn)
                scan.insert(0, "method", name)
                scan.insert(0, "task", task_name)
                scan.insert(0, "split", split)
                scan_rows.append(scan)

                scan_by_method[name] = scan
                st = threshold_stability_summary(scan)
                stability_rows.append({"split": split, "task": task_name, "method": name, **st})

            plot_threshold_scan(
                scan_by_method,
                out_dir / f"threshold_scan_{task_name}_{split}",
                title_suffix=f"{task_name} / {split}",
            )

    pd.concat(scan_rows, ignore_index=True).to_csv(out_dir / "threshold_scan.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(out_dir / "threshold_stability_summary.csv", index=False)

    # 风险分层稳定性：在 val 上为每种概率版本选择 tau1/tau2，再固定应用到 test。
    # 注意：这只能证明“攻击概率阈值型风险评分”的稳定性，不能替代攻击类型/风险头的语义风险分层。
    grid = np.round(np.arange(0.05, 0.96, 0.01), 6)
    risk_rows: List[Dict] = []
    val_y_risk = dfs["val"][args.risk_label_col].values.astype(int)
    test_y_risk = dfs["test"][args.risk_label_col].values.astype(int)
    val_p_raw = dfs["val"][args.prob_col].values.astype(float)
    test_p_raw = dfs["test"][args.prob_col].values.astype(float)

    for name, cal in calibrators.items():
        val_p = cal.predict(val_p_raw)
        test_p = cal.predict(test_p_raw)
        best_val = select_risk_thresholds_on_val(val_y_risk, val_p, grid=grid, metric="f1_macro")
        tau1, tau2 = best_val["tau1"], best_val["tau2"]
        test_m = risk_metrics_from_prob(test_y_risk, test_p, tau1=tau1, tau2=tau2)
        risk_rows.append({"split": "val", "method": name, "selected_on": "val", **best_val})
        risk_rows.append({"split": "test", "method": name, "selected_on": "val", **test_m})

    pd.DataFrame(risk_rows).to_csv(out_dir / "probability_risk_stratification_summary.csv", index=False)

    summary = {
        "input": {
            "pred_dir": str(args.pred_dir),
            "prefix": prefix,
            "prob_col": args.prob_col,
            "label_col": args.label_col,
            "risk_label_col": args.risk_label_col,
        },
        "calibration_protocol": {
            "calibration_fit_split": "val",
            "reported_splits": ["train", "val", "test"],
            "methods": ["Uncalibrated", "Platt", "Isotonic"],
            "metrics": ["Brier Score", "ECE", "Reliability Curve", "Threshold Stability"],
            "note": "严格论文表述中，应以 test 上的 Brier/ECE/Reliability Curve 作为最终可靠性结论；val 主要用于拟合校准器和选择阈值。",
        },
        "outputs": {
            "calibration_metrics": "calibration_metrics.csv",
            "reliability_bins": "reliability_bins.csv",
            "reliability_curves": ["reliability_curve_val.png/pdf/svg", "reliability_curve_test.png/pdf/svg"],
            "threshold_scan": "threshold_scan.csv",
            "threshold_stability_summary": "threshold_stability_summary.csv",
            "probability_risk_stratification_summary": "probability_risk_stratification_summary.csv",
        },
    }
    safe_json_dump(summary, out_dir / "exp3_calibration_manifest.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[完成] 实验三概率校准与可靠性分析结果已保存到：", out_dir)


if __name__ == "__main__":
    main()
