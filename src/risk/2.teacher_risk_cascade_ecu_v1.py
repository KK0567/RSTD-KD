#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
    train_path = base / "train_windows.csv"
    val_path = base / "val_windows.csv"
    test_path = base / "test_windows.csv"
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"{data_dir} 下缺少 train/val/test_windows.csv")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
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


def build_binary_xyw(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].values.astype(np.int64)
    n_neg = max(1, int((y == 0).sum()))
    n_pos = max(1, int((y == 1).sum()))
    total = n_neg + n_pos
    w_neg = total / (2.0 * n_neg)
    w_pos = total / (2.0 * n_pos)
    w = np.where(y == 1, w_pos, w_neg).astype(np.float32)
    return X, y, w


def build_attack_xyw(df: pd.DataFrame, feature_cols: List[str]):
    sub = df.loc[df["binary_label"] == 1].copy()
    X = sub[feature_cols].values.astype(np.float32)
    y = sub["y_attack_type"].values.astype(np.int64)
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    w_map = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    w = np.array([w_map[int(v)] for v in y], dtype=np.float32)
    return sub, X, y, w


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
    p, r, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1m),
    }


def stratified_bootstrap_indices(y: np.ndarray, frac: float, rng: np.random.RandomState) -> np.ndarray:
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos = max(1, int(round(len(idx_pos) * frac)))
    n_neg = max(1, int(round(len(idx_neg) * frac)))
    boot_pos = rng.choice(idx_pos, size=n_pos, replace=True)
    boot_neg = rng.choice(idx_neg, size=n_neg, replace=True)
    boot = np.concatenate([boot_pos, boot_neg])
    rng.shuffle(boot)
    return boot


def train_teacher_binary_ensemble(train_df: pd.DataFrame, feature_cols: List[str], args) -> List[HistGradientBoostingClassifier]:
    X, y, w = build_binary_xyw(train_df, feature_cols)
    models = []
    for i in range(args.ensemble_size):
        rng = np.random.RandomState(args.seed + 1009 * (i + 1))
        boot_idx = stratified_bootstrap_indices(y, frac=args.bootstrap_frac, rng=rng)
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf_binary,
            l2_regularization=args.l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=args.seed + i,
        )
        model.fit(X[boot_idx], y[boot_idx], sample_weight=w[boot_idx])
        models.append(model)
    return models


def predict_teacher_binary_ensemble(models: List[HistGradientBoostingClassifier], df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].values.astype(np.float32)
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(np.stack(probs, axis=0), axis=0)


def train_attack_model(train_df: pd.DataFrame, feature_cols: List[str], args) -> HistGradientBoostingClassifier:
    _, X, y, w = build_attack_xyw(train_df, feature_cols)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.attack_learning_rate,
        max_iter=args.attack_max_iter,
        max_leaf_nodes=args.attack_max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf_attack,
        l2_regularization=args.attack_l2_regularization,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=args.seed + 17,
    )
    model.fit(X, y, sample_weight=w)
    return model


def attack_probs_to_risk_pred(attack_pred: np.ndarray) -> np.ndarray:
    return np.array([ATTACK_TO_RISK[int(a)] for a in attack_pred], dtype=np.int64)


def evaluate_and_dump(split_name: str, df: pd.DataFrame, binary_models, attack_model, feature_cols: List[str], out_dir: str, bin_threshold: float):
    bin_prob = predict_teacher_binary_ensemble(binary_models, df, feature_cols)
    bin_pred = (bin_prob >= bin_threshold).astype(np.int64)

    X_all = df[feature_cols].values.astype(np.float32)
    attack_prob = attack_model.predict_proba(X_all)
    attack_pred = attack_model.predict(X_all).astype(np.int64)

    final_risk_pred = np.where(bin_pred == 0, 0, attack_probs_to_risk_pred(attack_pred))
    y_risk = df["y_risk"].values.astype(np.int64)
    y_bin = df["binary_label"].values.astype(np.int64)

    binary_metrics = compute_binary_metrics(y_bin, bin_prob, threshold=bin_threshold)
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
        "teacher_attack_prob_1": attack_prob[:, list(attack_model.classes_).index(1)].astype(np.float32),
        "teacher_attack_prob_2": attack_prob[:, list(attack_model.classes_).index(2)].astype(np.float32),
        "teacher_attack_prob_3": attack_prob[:, list(attack_model.classes_).index(3)].astype(np.float32),
        "teacher_pred_attack_type": attack_pred.astype(np.int64),
        "teacher_pred_attack_type_name": np.array(df["y_attack_type_name"].map({1:"Wifi Deauthentication Attack",2:"WPA2-PSK WIFI Cracking Attack",3:"TELLO API Exploit"}).fillna("unknown")) if False else attack_pred,
        "teacher_pred_risk": final_risk_pred.astype(np.int64),
        "teacher_pred_risk_name": np.array([RISK_ID_TO_NAME[v] for v in final_risk_pred]),
    })
    # overwrite attack type name with stable mapping
    type_name_map = {1: "Wifi Deauthentication Attack", 2: "WPA2-PSK WIFI Cracking Attack", 3: "TELLO API Exploit"}
    out["teacher_pred_attack_type_name"] = out["teacher_pred_attack_type"].map(type_name_map)
    out.to_csv(Path(out_dir) / f"teacher_risk_preds_{split_name}.csv", index=False)
    return binary_metrics, risk_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ECU-IoFT teacher cascade: binary + attack-type -> low/medium/high risk")
    parser.add_argument("--data-dir", type=str, default="Dataset_w25/ecu_attack_risk_windows_v3")
    parser.add_argument("--output-dir", type=str, default="output1/teacher_risk_cascade_v1")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--bootstrap-frac", type=float, default=0.80)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf-binary", type=int, default=40)
    parser.add_argument("--l2-regularization", type=float, default=3.0)
    parser.add_argument("--attack-max-iter", type=int, default=160)
    parser.add_argument("--attack-learning-rate", type=float, default=0.03)
    parser.add_argument("--attack-max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf-attack", type=int, default=2)
    parser.add_argument("--attack-l2-regularization", type=float, default=1.0)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_df, val_df, test_df = load_prepared_splits(args.data_dir)
    feature_cols = load_feature_cols(args.data_dir, train_df)
    train_df, [val_df, test_df], fill_values = fill_missing_with_train_medians(train_df, [val_df, test_df], feature_cols)

    binary_models = train_teacher_binary_ensemble(train_df, feature_cols, args)
    attack_model = train_attack_model(train_df, feature_cols, args)

    with open(Path(args.output_dir) / "teacher_binary_models.pkl", "wb") as f:
        pickle.dump(binary_models, f)
    with open(Path(args.output_dir) / "teacher_attack_model.pkl", "wb") as f:
        pickle.dump(attack_model, f)

    safe_json_dump({
        "feature_cols": feature_cols,
        "fill_values": fill_values,
        "binary_threshold": args.binary_threshold,
        "attack_to_risk": ATTACK_TO_RISK,
    }, Path(args.output_dir) / "preprocess_info.json")

    summary = {
        "task_design": {
            "teacher": "binary ensemble + attack-type model",
            "final_risk": "if binary=normal -> low else map(pred_attack_type) to medium/high",
        },
        "binary": {},
        "final_risk": {},
    }

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        binary_metrics, risk_metrics = evaluate_and_dump(
            split_name, df, binary_models, attack_model, feature_cols, args.output_dir, args.binary_threshold
        )
        summary["binary"][split_name] = binary_metrics
        summary["final_risk"][split_name] = risk_metrics

    safe_json_dump(summary, Path(args.output_dir) / "teacher_risk_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
