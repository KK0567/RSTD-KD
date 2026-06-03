#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ECU-IoFT Teacher v2：binary + direct-risk + attack-type auxiliary.

核心修改：
1) 保留二分类攻击概率 teacher_prob_binary；
2) 新增直接风险模型 teacher_risk_prob_0/1/2；
3) 攻击类型模型只作为辅助蒸馏信息，不再作为最终风险分层的唯一依据；
4) 在验证集上选择风险决策阈值 tau_attack / tau_high，并固定应用到测试集；
5) 输出文件仍保持 teacher_risk_preds_train/val/test.csv，兼容后续 student v2。

运行示例：
python 2.teacher_risk_direct_v2.py ^
  --data-dir Dataset_w25/ecu_attack_risk_windows_v3 ^
  --output-dir output1/teacher_risk_direct_v2
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

ALWAYS_DROP_EXACT = {
    "y_bin", "y_attack_type", "y_attack_type_name", "y_risk", "y_risk_name",
    "attack_packet_count", "attack_packet_ratio",
    "window_id", "row_id", "binary_label", "label", "split", "session_id",
    "time_start", "time_end", "packet_id_start", "packet_id_end",
    "attack_scenario_meta", "packet_attack_type_mode_meta",
}
ALWAYS_DROP_PREFIX = ("cnt_",)
ALWAYS_DROP_CONTAINS = ("label", "target")

ATTACK_TO_RISK = {
    1: 1,  # Wifi Deauthentication Attack -> medium
    2: 2,  # WPA2-PSK WIFI Cracking Attack -> high
    3: 2,  # TELLO API Exploit -> high
}
RISK_ID_TO_NAME = {0: "low", 1: "medium", 2: "high"}
ATTACK_NAME_MAP = {
    1: "Wifi Deauthentication Attack",
    2: "WPA2-PSK WIFI Cracking Attack",
    3: "TELLO API Exploit",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_json_dump(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_required_columns(df: pd.DataFrame, name: str) -> None:
    required = ["window_id", "y_bin", "y_attack_type", "y_attack_type_name", "y_risk"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{name} 缺少必要列: {miss}")


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_id"] = out["window_id"].astype(np.int64)
    out["binary_label"] = out["y_bin"].astype(np.int64)
    out["label"] = out["y_attack_type_name"].astype(str)
    return out


def load_prepared_splits(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    paths = [base / "train_windows.csv", base / "val_windows.csv", base / "test_windows.csv"]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f"{data_dir} 下缺少 train/val/test_windows.csv")
    train_df = pd.read_csv(paths[0])
    val_df = pd.read_csv(paths[1])
    test_df = pd.read_csv(paths[2])
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _ensure_required_columns(df, name)
    return _standardize_df(train_df), _standardize_df(val_df), _standardize_df(test_df)


def load_feature_cols(data_dir: str, train_df: pd.DataFrame) -> List[str]:
    fpath = Path(data_dir) / "feature_columns.json"
    if fpath.exists():
        with open(fpath, "r", encoding="utf-8") as f:
            raw_cols = json.load(f)
    else:
        raw_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]

    keep: List[str] = []
    for c in raw_cols:
        if c not in train_df.columns:
            continue
        if c in ALWAYS_DROP_EXACT:
            continue
        if any(c.startswith(p) for p in ALWAYS_DROP_PREFIX):
            continue
        lc = c.lower()
        if any(k in lc for k in ALWAYS_DROP_CONTAINS):
            continue
        if not pd.api.types.is_numeric_dtype(train_df[c]):
            continue
        keep.append(c)
    if not keep:
        raise ValueError("过滤后 feature_cols 为空")
    return keep


def fill_missing_with_train_medians(train_df: pd.DataFrame, other_dfs: List[pd.DataFrame], feature_cols: List[str]):
    med = train_df[feature_cols].median(numeric_only=True).to_dict()
    train_df = train_df.copy()
    train_df[feature_cols] = train_df[feature_cols].fillna(med)
    outs = []
    for df in other_dfs:
        d = df.copy()
        d[feature_cols] = d[feature_cols].fillna(med)
        outs.append(d)
    return train_df, outs, med


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    w_map = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    return np.array([w_map[int(v)] for v in y], dtype=np.float32)


def build_xyw(df: pd.DataFrame, feature_cols: List[str], target_col: str):
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)
    w = balanced_sample_weight(y)
    return X, y, w


def build_attack_xyw(df: pd.DataFrame, feature_cols: List[str]):
    sub = df.loc[df["binary_label"] == 1].copy()
    X = sub[feature_cols].values.astype(np.float32)
    y = sub["y_attack_type"].values.astype(np.int64)
    w = balanced_sample_weight(y)
    return sub, X, y, w


def make_hgb(args, task: str, seed_offset: int = 0) -> HistGradientBoostingClassifier:
    if task == "binary":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf_binary,
            l2_regularization=args.l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=args.seed + seed_offset,
        )
    if task == "risk":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=args.risk_learning_rate,
            max_iter=args.risk_max_iter,
            max_leaf_nodes=args.risk_max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf_risk,
            l2_regularization=args.risk_l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=args.seed + seed_offset,
        )
    if task == "attack":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=args.attack_learning_rate,
            max_iter=args.attack_max_iter,
            max_leaf_nodes=args.attack_max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf_attack,
            l2_regularization=args.attack_l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=args.seed + seed_offset,
        )
    raise ValueError(task)


def stratified_bootstrap_indices(y: np.ndarray, frac: float, rng: np.random.RandomState) -> np.ndarray:
    parts = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = max(1, int(round(len(idx) * frac)))
        parts.append(rng.choice(idx, size=n, replace=True))
    boot = np.concatenate(parts)
    rng.shuffle(boot)
    return boot


def train_binary_ensemble(train_df: pd.DataFrame, feature_cols: List[str], args):
    X, y, w = build_xyw(train_df, feature_cols, "binary_label")
    models = []
    for i in range(args.ensemble_size):
        rng = np.random.RandomState(args.seed + 1009 * (i + 1))
        boot_idx = stratified_bootstrap_indices(y, frac=args.bootstrap_frac, rng=rng)
        model = make_hgb(args, "binary", seed_offset=i)
        model.fit(X[boot_idx], y[boot_idx], sample_weight=w[boot_idx])
        models.append(model)
    return models


def train_risk_model(train_df: pd.DataFrame, feature_cols: List[str], args):
    X, y, w = build_xyw(train_df, feature_cols, "y_risk")
    if args.risk_model_type == "hgb":
        model = make_hgb(args, "risk", seed_offset=37)
        model.fit(X, y, sample_weight=w)
        return model
    if args.risk_model_type == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=args.tree_n_estimators,
            max_depth=None,
            min_samples_leaf=args.tree_min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed + 37,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model
    if args.risk_model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=args.tree_n_estimators,
            max_depth=None,
            min_samples_leaf=args.tree_min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed + 37,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model
    raise ValueError(args.risk_model_type)


def train_attack_model(train_df: pd.DataFrame, feature_cols: List[str], args):
    _, X, y, w = build_attack_xyw(train_df, feature_cols)
    model = make_hgb(args, "attack", seed_offset=17)
    model.fit(X, y, sample_weight=w)
    return model


def predict_binary_ensemble(models, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].values.astype(np.float32)
    probs = [m.predict_proba(X)[:, list(m.classes_).index(1)] for m in models]
    return np.mean(np.stack(probs, axis=0), axis=0)


def predict_proba_fixed_classes(model, X: np.ndarray, classes: List[int]) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
    model_classes = list(model.classes_)
    for j, c in enumerate(classes):
        if c in model_classes:
            out[:, j] = raw[:, model_classes.index(c)]
    row_sum = out.sum(axis=1, keepdims=True)
    out = np.divide(out, np.clip(row_sum, 1e-12, None))
    return out


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    p_macro, r_macro, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_bin, r_bin, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1m),
        "precision_binary": float(p_bin),
        "recall_binary": float(r_bin),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "threshold": float(threshold),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")
    return out


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p, r, f1m, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    _, high_recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1m),
        "high_recall": float(high_recall),
    }


def risk_decision(bin_prob: np.ndarray, risk_prob: np.ndarray, tau_attack: float, tau_high: float) -> np.ndarray:
    """
    直接风险概率决策。
    先由二分类概率过滤正常窗口，再用 high probability 控制高风险召回。
    """
    p_med = risk_prob[:, 1]
    p_high = risk_prob[:, 2]
    pred_attack_side = np.where(p_high >= tau_high, 2, np.where(p_med >= p_high, 1, 2))
    pred = np.where(bin_prob < tau_attack, 0, pred_attack_side)
    return pred.astype(np.int64)


def select_risk_thresholds_on_val(y_true: np.ndarray, bin_prob: np.ndarray, risk_prob: np.ndarray, args) -> Dict[str, float]:
    grid_attack = np.arange(args.search_attack_min, args.search_attack_max + 1e-9, args.search_step)
    grid_high = np.arange(args.search_high_min, args.search_high_max + 1e-9, args.search_step)
    best = None
    for tau_attack in grid_attack:
        for tau_high in grid_high:
            pred = risk_decision(bin_prob, risk_prob, tau_attack=float(tau_attack), tau_high=float(tau_high))
            m = compute_multiclass_metrics(y_true, pred)
            # 主目标 Macro-F1；同分时优先 high_recall，再优先 balanced_accuracy。
            score = m["f1_macro"] + args.high_recall_weight * m["high_recall"] + 0.05 * m["balanced_accuracy"]
            cand = {"tau_attack": float(tau_attack), "tau_high": float(tau_high), "score": float(score), **m}
            if best is None or cand["score"] > best["score"]:
                best = cand
    assert best is not None
    return best


def evaluate_and_dump(split_name: str, df: pd.DataFrame, binary_models, risk_model, attack_model,
                      feature_cols: List[str], out_dir: str, thresholds: Dict[str, float]):
    X_all = df[feature_cols].values.astype(np.float32)
    bin_prob = predict_binary_ensemble(binary_models, df, feature_cols)
    risk_prob = predict_proba_fixed_classes(risk_model, X_all, [0, 1, 2])
    attack_prob = predict_proba_fixed_classes(attack_model, X_all, [1, 2, 3])

    final_risk_pred = risk_decision(
        bin_prob,
        risk_prob,
        tau_attack=float(thresholds["tau_attack"]),
        tau_high=float(thresholds["tau_high"]),
    )
    bin_pred = (bin_prob >= float(thresholds["tau_attack"])).astype(np.int64)
    attack_pred = np.array([1, 2, 3], dtype=np.int64)[np.argmax(attack_prob, axis=1)]
    attack_pred = np.where(bin_pred == 0, 0, attack_pred)

    y_risk = df["y_risk"].values.astype(np.int64)
    y_bin = df["binary_label"].values.astype(np.int64)

    binary_metrics = compute_binary_metrics(y_bin, bin_prob, threshold=float(thresholds["tau_attack"]))
    risk_metrics = compute_multiclass_metrics(y_risk, final_risk_pred)
    risk_metrics["labels"] = [0, 1, 2]
    risk_metrics["confusion_matrix"] = confusion_matrix(y_risk, final_risk_pred, labels=[0, 1, 2]).tolist()

    out = pd.DataFrame({
        "row_id": df["row_id"].values.astype(np.int64),
        "window_id": df["window_id"].values.astype(np.int64),
        "session_id": df["session_id"].astype(str).values if "session_id" in df.columns else np.array([""] * len(df)),
        "y_bin": y_bin,
        "y_attack_type": df["y_attack_type"].values.astype(np.int64),
        "y_attack_type_name": df["y_attack_type_name"].astype(str).values,
        "y_risk": y_risk,
        "y_risk_name": df["y_risk_name"].astype(str).values if "y_risk_name" in df.columns else np.array([RISK_ID_TO_NAME[v] for v in y_risk]),
        "teacher_prob_binary": bin_prob.astype(np.float32),
        "teacher_pred_binary": bin_pred.astype(np.int64),
        "teacher_risk_prob_0": risk_prob[:, 0].astype(np.float32),
        "teacher_risk_prob_1": risk_prob[:, 1].astype(np.float32),
        "teacher_risk_prob_2": risk_prob[:, 2].astype(np.float32),
        "teacher_attack_prob_1": attack_prob[:, 0].astype(np.float32),
        "teacher_attack_prob_2": attack_prob[:, 1].astype(np.float32),
        "teacher_attack_prob_3": attack_prob[:, 2].astype(np.float32),
        "teacher_pred_attack_type": attack_pred.astype(np.int64),
        "teacher_pred_attack_type_name": np.array([ATTACK_NAME_MAP.get(int(v), "No Attack") for v in attack_pred]),
        "teacher_pred_risk": final_risk_pred.astype(np.int64),
        "teacher_pred_risk_name": np.array([RISK_ID_TO_NAME[int(v)] for v in final_risk_pred]),
    })
    out.to_csv(Path(out_dir) / f"teacher_risk_preds_{split_name}.csv", index=False)
    return binary_metrics, risk_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ECU-IoFT teacher v2: binary + direct risk + attack auxiliary")
    parser.add_argument("--data-dir", type=str, default="Dataset/ecu_attack_risk_windows_v3")
    parser.add_argument("--output-dir", type=str, default="output1/teacher_risk_direct_v2")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bootstrap-frac", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf-binary", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--risk-model-type", type=str, default="hgb", choices=["hgb", "extra_trees", "rf"])
    parser.add_argument("--risk-max-iter", type=int, default=220)
    parser.add_argument("--risk-learning-rate", type=float, default=0.025)
    parser.add_argument("--risk-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf-risk", type=int, default=5)
    parser.add_argument("--risk-l2-regularization", type=float, default=0.5)
    parser.add_argument("--tree-n-estimators", type=int, default=500)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=1)
    parser.add_argument("--attack-max-iter", type=int, default=220)
    parser.add_argument("--attack-learning-rate", type=float, default=0.025)
    parser.add_argument("--attack-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf-attack", type=int, default=1)
    parser.add_argument("--attack-l2-regularization", type=float, default=0.5)
    parser.add_argument("--search-step", type=float, default=0.01)
    parser.add_argument("--search-attack-min", type=float, default=0.05)
    parser.add_argument("--search-attack-max", type=float, default=0.95)
    parser.add_argument("--search-high-min", type=float, default=0.05)
    parser.add_argument("--search-high-max", type=float, default=0.95)
    parser.add_argument("--high-recall-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_df, val_df, test_df = load_prepared_splits(args.data_dir)
    feature_cols = load_feature_cols(args.data_dir, train_df)
    train_df, [val_df, test_df], fill_values = fill_missing_with_train_medians(train_df, [val_df, test_df], feature_cols)

    binary_models = train_binary_ensemble(train_df, feature_cols, args)
    risk_model = train_risk_model(train_df, feature_cols, args)
    attack_model = train_attack_model(train_df, feature_cols, args)

    with open(Path(args.output_dir) / "teacher_binary_models.pkl", "wb") as f:
        pickle.dump(binary_models, f)
    with open(Path(args.output_dir) / "teacher_risk_model.pkl", "wb") as f:
        pickle.dump(risk_model, f)
    with open(Path(args.output_dir) / "teacher_attack_model.pkl", "wb") as f:
        pickle.dump(attack_model, f)

    # validation 上选阈值，test 严格只评估
    val_X = val_df[feature_cols].values.astype(np.float32)
    val_bin_prob = predict_binary_ensemble(binary_models, val_df, feature_cols)
    val_risk_prob = predict_proba_fixed_classes(risk_model, val_X, [0, 1, 2])
    thresholds = select_risk_thresholds_on_val(val_df["y_risk"].values.astype(np.int64), val_bin_prob, val_risk_prob, args)

    safe_json_dump({
        "feature_cols": feature_cols,
        "fill_values": fill_values,
        "thresholds_selected_on_val": thresholds,
        "risk_model_type": args.risk_model_type,
        "attack_to_risk": ATTACK_TO_RISK,
        "decision_rule": "if teacher_prob_binary < tau_attack -> low; else if teacher_risk_prob_2 >= tau_high -> high; else medium/high by p1 vs p2",
    }, Path(args.output_dir) / "preprocess_info.json")

    summary = {
        "task_design": {
            "teacher": "binary ensemble + direct risk model + attack-type auxiliary model",
            "final_risk": "direct risk probabilities with validation-selected tau_attack/tau_high",
            "why": "avoid high-risk collapse caused by attack-type-to-risk mapping errors",
        },
        "thresholds_selected_on_val": thresholds,
        "binary": {},
        "final_risk": {},
    }

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        binary_metrics, risk_metrics = evaluate_and_dump(
            split_name, df, binary_models, risk_model, attack_model, feature_cols, args.output_dir, thresholds
        )
        summary["binary"][split_name] = binary_metrics
        summary["final_risk"][split_name] = risk_metrics

    safe_json_dump(summary, Path(args.output_dir) / "teacher_risk_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
