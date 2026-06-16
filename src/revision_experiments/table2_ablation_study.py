#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSTD-KD Component Ablation Study (5-Fold Cross-Validation)
Component Ablation Study for RSTD-KD with 5-Fold CV

Reviewer Responses:
  Reviewer 1 #3: Add ablation for sliding window length, Platt calibration,
                 and task cost weight configuration
  Reviewer 2 #6: Window size w=32 sensitivity analysis

Ablation Variants (reported on each fold's test set, final results are 5-fold mean +/- std):
  1. Full RSTD-KD          -- Full method
  2. w/o Knowledge Dist.   -- Remove Knowledge Distillation (set KD loss weight to zero)
  3. w/o Platt Calibration -- Remove Platt probability calibration (use raw softmax probabilities)
  4. w/o Cost-aware Thresh -- Remove validation set threshold search (use fixed default threshold 0.5)
  5. w=16 / w=64           -- Different window sizes

Metrics: Accuracy, Balanced Accuracy, Macro-F1, High-state Recall,
         Brier Score, ECE, NLL

Usage:
  Run this file directly (python ablation_study_rstd_kd.py)
  All outputs are saved to OUTPUT_DIR
"""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
#  Path Configuration
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_CSV = str(PROJECT_ROOT / "data" / "ECU-IoFT-Dataset.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "table2_ablation")

# ================================================================
#  Global Hyperparameters
# ================================================================
SEED = 42
N_FOLDS = 5
WINDOW_SIZES = [16, 32, 64]
MAIN_WINDOW = 32

# Teacher
TEACHER_ENSEMBLE_SIZE = 5
TEACHER_BOOTSTRAP_FRAC = 0.90
TEACHER_MAX_ITER = 180
TEACHER_LR = 0.025
TEACHER_MAX_LEAF = 31
TEACHER_MIN_LEAF_BIN = 20
TEACHER_L2 = 1.0
TEACHER_RISK_MAX_ITER = 220
TEACHER_RISK_LR = 0.025
TEACHER_RISK_MAX_LEAF = 31
TEACHER_RISK_MIN_LEAF = 5
TEACHER_RISK_L2 = 0.5
TEACHER_ATK_MAX_ITER = 220
TEACHER_ATK_LR = 0.025
TEACHER_ATK_MAX_LEAF = 31
TEACHER_ATK_MIN_LEAF = 1
TEACHER_ATK_L2 = 0.5

# Student
STUDENT_EPOCHS = 160
STUDENT_BATCH = 128
STUDENT_LR = 4e-4
STUDENT_WD = 3e-4
STUDENT_PATIENCE = 24
STUDENT_TEMP = 2.0
STUDENT_HIDDEN1 = 128
STUDENT_HIDDEN2 = 64
STUDENT_DROPOUT1 = 0.12
STUDENT_DROPOUT2 = 0.08

# Threshold search
SEARCH_STEP = 0.02
SEARCH_MIN = 0.05
SEARCH_MAX = 0.95
HIGH_RECALL_WEIGHT = 0.10
DEFAULT_TAU_ATTACK = 0.50
DEFAULT_TAU_HIGH = 0.50

# ================================================================
#  Label Mapping
# ================================================================
ATTACK_TYPE_MAP = {
    "No Attack": 0, "Wifi Deauthentication Attack": 1,
    "WPA2-PSK WIFI Cracking Attack": 2, "TELLO API Exploit": 3,
}
ATTACK_ID_TO_NAME = {v: k for k, v in ATTACK_TYPE_MAP.items()}
RISK_MAP = {
    "No Attack": (0, "normal"), "Wifi Deauthentication Attack": (1, "medium"),
    "WPA2-PSK WIFI Cracking Attack": (2, "high"), "TELLO API Exploit": (2, "high"),
}
RISK_ID_TO_NAME = {0: "low", 1: "medium", 2: "high"}

ALWAYS_DROP_EXACT = {
    "y_bin", "y_attack_type", "y_attack_type_name", "y_risk", "y_risk_name",
    "attack_packet_count", "attack_packet_ratio",
    "window_id", "row_id", "binary_label", "label", "split", "session_id",
    "time_start", "time_end", "packet_id_start", "packet_id_end",
    "attack_scenario_meta", "packet_attack_type_mode_meta",
}
ALWAYS_DROP_PREFIX = ("cnt_",)
ALWAYS_DROP_CONTAINS = ("label", "target")

INFO_TOKENS = [
    "ack", "deauthentication", "authentication", "eapol", "beacon",
    "probe", "request", "response", "data", "null", "qos", "udp",
    "icmp", "exploit", "api", "sae", "handshake",
]
PROTO_VALUES = ["802.11", "UDP", "EAPOL", "ICMP"]


# ================================================================
#  Utility Functions
# ================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def dominant_attack_label(labels: pd.Series) -> str:
    vc = labels.value_counts()
    if vc.empty:
        return "No Attack"
    if len(vc) == 1:
        return str(vc.index[0])
    items = sorted(vc.items(), key=lambda kv: (kv[1], kv[0] != "No Attack"), reverse=True)
    return str(items[0][0])


def ratio_bool(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) > 0 else 0.0


def q(arr: np.ndarray, v: float) -> float:
    return float(np.quantile(arr, v)) if arr.size > 0 else 0.0


def calc_entropy(values: pd.Series) -> float:
    vc = values.value_counts(normalize=True)
    if vc.empty:
        return 0.0
    p = vc.values.astype(np.float64)
    return float(-(p * np.log2(np.clip(p, 1e-12, 1.0))).sum())


def normalize_string(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def parse_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


# ================================================================
#  Window Data Construction (same as original script)
# ================================================================
def build_features_one_window(sub: pd.DataFrame, window_id: int) -> Dict:
    lengths = sub["Length"].astype(float).values
    info = normalize_string(sub["Info"])
    proto = normalize_string(sub["Protocol"])
    src = normalize_string(sub["Source"])
    dst = normalize_string(sub["Destination"])
    atk = normalize_string(sub["Type of Attack"])
    typ = normalize_string(sub["Type"])
    dt = sub["_dt"]
    dominant = dominant_attack_label(atk)
    if dominant not in ATTACK_TYPE_MAP:
        raise ValueError(f"Unknown attack type: {dominant}")
    y_attack_type = ATTACK_TYPE_MAP[dominant]
    y_bin = int(dominant != "No Attack")
    y_risk, y_risk_name = RISK_MAP[dominant]
    info_len = info.str.len().values.astype(float)
    t_ns = dt.astype("int64").to_numpy(dtype=np.int64)
    t_sec = t_ns.astype(np.float64) / 1e9
    inter_arrival = np.clip(np.diff(t_sec), 0.0, None) if len(t_sec) > 1 else np.array([], dtype=np.float64)
    duration_raw = float((dt.iloc[-1] - dt.iloc[0]).total_seconds())
    packet_rate = float(len(sub) / max(duration_raw, 1e-6))
    if len(lengths) > 1:
        length_diff = np.diff(lengths)
        x_idx = np.arange(len(lengths), dtype=np.float64)
        length_slope = float(np.polyfit(x_idx, lengths.astype(np.float64), 1)[0])
    else:
        length_diff = np.array([], dtype=np.float64)
        length_slope = 0.0
    src_ne = src[src != ""].dropna()
    dst_ne = dst[dst != ""].dropna()
    proto_ne = proto[proto != ""].dropna()
    feat = {
        "window_id": int(window_id), "row_id": int(window_id),
        "packet_id_start": int(sub.index[0]), "packet_id_end": int(sub.index[-1]),
        "time_start": str(dt.iloc[0]), "time_end": str(dt.iloc[-1]),
        "window_packet_count": int(len(sub)), "duration_sec": duration_raw,
        "packet_rate": packet_rate,
        "inter_arrival_mean": float(inter_arrival.mean()) if inter_arrival.size else 0.0,
        "inter_arrival_std": float(inter_arrival.std(ddof=0)) if inter_arrival.size else 0.0,
        "inter_arrival_min": float(inter_arrival.min()) if inter_arrival.size else 0.0,
        "inter_arrival_max": float(inter_arrival.max()) if inter_arrival.size else 0.0,
        "inter_arrival_q25": q(inter_arrival, 0.25), "inter_arrival_q50": q(inter_arrival, 0.50),
        "inter_arrival_q75": q(inter_arrival, 0.75),
        "length_mean": float(lengths.mean()), "length_std": float(lengths.std(ddof=0)),
        "length_min": float(lengths.min()), "length_max": float(lengths.max()),
        "length_q25": q(lengths, 0.25), "length_q50": q(lengths, 0.50), "length_q75": q(lengths, 0.75),
        "length_first": float(lengths[0]) if lengths.size else 0.0,
        "length_last": float(lengths[-1]) if lengths.size else 0.0,
        "length_slope": length_slope,
        "length_diff_mean": float(length_diff.mean()) if length_diff.size else 0.0,
        "length_diff_std": float(length_diff.std(ddof=0)) if length_diff.size else 0.0,
        "length_absdiff_mean": float(np.abs(length_diff).mean()) if length_diff.size else 0.0,
        "length_absdiff_max": float(np.abs(length_diff).max()) if length_diff.size else 0.0,
        "info_len_mean": float(info_len.mean()), "info_len_std": float(info_len.std(ddof=0)),
        "info_len_max": float(info_len.max()),
        "src_unique": int(src_ne.nunique()), "dst_unique": int(dst_ne.nunique()),
        "proto_unique": int(proto_ne.nunique()),
        "src_entropy": calc_entropy(src_ne), "dst_entropy": calc_entropy(dst_ne),
        "proto_entropy": calc_entropy(proto_ne),
        "missing_source_ratio": ratio_bool(src.eq("")),
        "broadcast_dst_ratio": ratio_bool(dst.str.contains("ff:ff:ff:ff:ff:ff", case=False, regex=False)),
        "ra_marker_ratio": ratio_bool(dst.str.contains(r"\(RA\)", case=False, regex=True)),
        "bssid_marker_ratio": ratio_bool(
            src.str.contains(r"\(BSSID\)", case=False, regex=True) |
            dst.str.contains(r"\(BSSID\)", case=False, regex=True)),
        "ip_endpoint_ratio": ratio_bool(
            src.str.contains(r"^\d+\.\d+\.\d+\.\d+$", regex=True) |
            dst.str.contains(r"^\d+\.\d+\.\d+\.\d+$", regex=True)),
        "attack_packet_count": int((typ == "Attack").sum()),
        "attack_packet_ratio": ratio_bool(typ == "Attack"),
        "y_bin": int(y_bin), "y_attack_type": int(y_attack_type),
        "y_attack_type_name": dominant, "y_risk": int(y_risk), "y_risk_name": y_risk_name,
        "attack_scenario_meta": dominant_attack_label(normalize_string(sub["Attack Scenario"])),
        "packet_attack_type_mode_meta": dominant,
        "session_id": "", "split": "",
    }
    for p in PROTO_VALUES:
        cnt = int((proto == p).sum())
        key = re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_").lower()
        feat[f"proto_{key}_count"] = cnt
        feat[f"proto_{key}_ratio"] = float(cnt / len(sub))
    proto_seq = proto.tolist()
    n_trans = max(1, len(proto_seq) - 1)
    for a, b in [("802.11", "802.11"), ("802.11", "EAPOL"), ("EAPOL", "EAPOL"),
                 ("UDP", "UDP"), ("802.11", "UDP"), ("UDP", "ICMP")]:
        trans_cnt = sum(1 for i in range(len(proto_seq) - 1)
                        if proto_seq[i] == a and proto_seq[i + 1] == b)
        ka = re.sub(r"[^a-zA-Z0-9]+", "_", a).strip("_").lower()
        kb = re.sub(r"[^a-zA-Z0-9]+", "_", b).strip("_").lower()
        feat[f"proto_trans_{ka}_to_{kb}_count"] = int(trans_cnt)
        feat[f"proto_trans_{ka}_to_{kb}_ratio"] = float(trans_cnt / n_trans)
    info_lower = info.str.lower()
    for tok in INFO_TOKENS:
        cnt = int(info_lower.str.contains(tok, regex=False).sum())
        feat[f"info_tok_{tok}_count"] = cnt
        feat[f"info_tok_{tok}_ratio"] = float(cnt / len(sub))
    return feat


def build_windows_df(df: pd.DataFrame, window_size: int, step_size: int) -> pd.DataFrame:
    rows, wid = [], 0
    for start in range(0, len(df) - window_size + 1, step_size):
        rows.append(build_features_one_window(df.iloc[start:start + window_size], window_id=wid))
        wid += 1
    return pd.DataFrame(rows)


# ================================================================
#  5-Fold CV Data Splitting
# ================================================================
def _standardize_df(df):
    out = df.copy()
    out["row_id"] = out["window_id"].astype(np.int64)
    out["binary_label"] = out["y_bin"].astype(np.int64)
    out["label"] = out["y_attack_type_name"].astype(str)
    return out


def _get_feature_cols(win_df):
    drop_set = {
        "window_id", "row_id", "packet_id_start", "packet_id_end",
        "time_start", "time_end", "y_bin", "y_attack_type", "y_attack_type_name",
        "y_risk", "y_risk_name", "attack_scenario_meta", "packet_attack_type_mode_meta",
        "session_id", "split",
    }
    keep = []
    for c in win_df.columns:
        if c in drop_set or c in ALWAYS_DROP_EXACT:
            continue
        if any(c.startswith(p) for p in ALWAYS_DROP_PREFIX):
            continue
        if any(k in c.lower() for k in ALWAYS_DROP_CONTAINS):
            continue
        if not pd.api.types.is_numeric_dtype(win_df[c]):
            continue
        keep.append(c)
    return keep


def prepare_cv_folds(raw_df, window_size, n_folds=N_FOLDS):
    """Construct window data and generate n_folds stratified splits. Returns (folds, feature_cols).
    folds is a list of (train_df, val_df, test_df).
    """
    win_df = build_windows_df(raw_df, window_size=window_size, step_size=window_size)
    feature_cols = _get_feature_cols(win_df)

    # Stratify by y_attack_type to ensure consistent attack type distribution across folds
    y_strat = win_df["y_attack_type"].values.astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_indices = list(skf.split(np.zeros(len(win_df)), y_strat))

    folds = []
    for fold_i, (train_val_idx, test_idx) in enumerate(fold_indices):
        # Split val from train_val: use the last portion as validation
        # Use n_folds-1 folds for training, 1 fold for validation
        # Since we already have n_folds folds, test takes 1 fold, the remaining n_folds-1 folds are split
        # Simplified: train_val has n_folds-1 portions, use the last portion as validation
        val_size = len(train_val_idx) // (n_folds - 1)
        val_idx = train_val_idx[-val_size:]
        train_idx = train_val_idx[:-val_size]

        train_raw = _standardize_df(win_df.iloc[train_idx])
        val_raw = _standardize_df(win_df.iloc[val_idx])
        test_raw = _standardize_df(win_df.iloc[test_idx])

        # Fill missing values (use train median)
        med = train_raw[feature_cols].median(numeric_only=True).to_dict()
        for df in [train_raw, val_raw, test_raw]:
            df[feature_cols] = df[feature_cols].fillna(med)

        folds.append((train_raw, val_raw, test_raw))

    return folds, feature_cols


# ================================================================
#  Teacher Model
# ================================================================
def balanced_sample_weight(y):
    y = np.asarray(y).astype(np.int64)
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    w_map = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    return np.array([w_map[int(v)] for v in y], dtype=np.float32)


def build_xyw(df, feature_cols, target_col):
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)
    w = balanced_sample_weight(y)
    return X, y, w


def stratified_bootstrap(y, frac, rng):
    parts = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = max(1, int(round(len(idx) * frac)))
        parts.append(rng.choice(idx, size=n, replace=True))
    boot = np.concatenate(parts)
    rng.shuffle(boot)
    return boot


def make_hgb(task, seed_offset=0):
    cfgs = {
        "binary": (TEACHER_LR, TEACHER_MAX_ITER, TEACHER_MAX_LEAF, TEACHER_MIN_LEAF_BIN, TEACHER_L2),
        "risk": (TEACHER_RISK_LR, TEACHER_RISK_MAX_ITER, TEACHER_RISK_MAX_LEAF, TEACHER_RISK_MIN_LEAF, TEACHER_RISK_L2),
        "attack": (TEACHER_ATK_LR, TEACHER_ATK_MAX_ITER, TEACHER_ATK_MAX_LEAF, TEACHER_ATK_MIN_LEAF, TEACHER_ATK_L2),
    }
    lr, mi, ml, ms, l2 = cfgs[task]
    so = {"binary": 0, "risk": 37, "attack": 17}.get(task, 0)
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=lr, max_iter=mi,
        max_leaf_nodes=ml, min_samples_leaf=ms, l2_regularization=l2,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
        random_state=SEED + seed_offset + so,
    )


def _make_hgb_no_es(task, seed_offset=0):
    cfgs = {
        "risk": (TEACHER_RISK_LR, TEACHER_RISK_MAX_ITER, TEACHER_RISK_MAX_LEAF, TEACHER_RISK_MIN_LEAF, TEACHER_RISK_L2),
        "attack": (TEACHER_ATK_LR, TEACHER_ATK_MAX_ITER, TEACHER_ATK_MAX_LEAF, TEACHER_ATK_MIN_LEAF, TEACHER_ATK_L2),
    }
    lr, mi, ml, ms, l2 = cfgs[task]
    so = {"risk": 37, "attack": 17}.get(task, 0)
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=lr, max_iter=mi,
        max_leaf_nodes=ml, min_samples_leaf=ms, l2_regularization=l2,
        early_stopping=False, random_state=SEED + seed_offset + so,
    )


def train_teacher(train_df, feature_cols):
    X_bin, y_bin, w_bin = build_xyw(train_df, feature_cols, "binary_label")
    binary_models = []
    for i in range(TEACHER_ENSEMBLE_SIZE):
        rng = np.random.RandomState(SEED + 1009 * (i + 1))
        boot_idx = stratified_bootstrap(y_bin, TEACHER_BOOTSTRAP_FRAC, rng)
        m = make_hgb("binary", seed_offset=i)
        m.fit(X_bin[boot_idx], y_bin[boot_idx], sample_weight=w_bin[boot_idx])
        binary_models.append(m)

    X_risk, y_risk, w_risk = build_xyw(train_df, feature_cols, "y_risk")
    if np.min(np.bincount(y_risk)) < 2:
        risk_model = _make_hgb_no_es("risk")
    else:
        risk_model = make_hgb("risk")
    risk_model.fit(X_risk, y_risk, sample_weight=w_risk)

    sub = train_df.loc[train_df["binary_label"] == 1].copy()
    X_atk = sub[feature_cols].values.astype(np.float32)
    y_atk = sub["y_attack_type"].values.astype(np.int64)
    w_atk = balanced_sample_weight(y_atk)
    if np.min(np.bincount(y_atk)) < 2:
        attack_model = _make_hgb_no_es("attack")
    else:
        attack_model = make_hgb("attack")
    attack_model.fit(X_atk, y_atk, sample_weight=w_atk)
    return binary_models, risk_model, attack_model


def predict_binary_ensemble(models, X):
    probs = [m.predict_proba(X)[:, list(m.classes_).index(1)] for m in models]
    return np.mean(np.stack(probs, axis=0), axis=0)


def predict_proba_fixed(model, X, classes):
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
    mc = list(model.classes_)
    for j, c in enumerate(classes):
        if c in mc:
            out[:, j] = raw[:, mc.index(c)]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, np.clip(row_sum, 1e-12, None))


def teacher_predict_all(binary_models, risk_model, attack_model, df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    bin_prob = predict_binary_ensemble(binary_models, X)
    risk_prob = predict_proba_fixed(risk_model, X, [0, 1, 2])
    attack_prob = predict_proba_fixed(attack_model, X, [1, 2, 3])
    return pd.DataFrame({
        "row_id": df["row_id"].values.astype(np.int64),
        "teacher_prob_binary": bin_prob.astype(np.float32),
        "teacher_risk_prob_0": risk_prob[:, 0].astype(np.float32),
        "teacher_risk_prob_1": risk_prob[:, 1].astype(np.float32),
        "teacher_risk_prob_2": risk_prob[:, 2].astype(np.float32),
        "teacher_attack_prob_1": attack_prob[:, 0].astype(np.float32),
        "teacher_attack_prob_2": attack_prob[:, 1].astype(np.float32),
        "teacher_attack_prob_3": attack_prob[:, 2].astype(np.float32),
    })


# ================================================================
#  Student Model
# ================================================================
class StudentModel(nn.Module):
    def __init__(self, in_dim, h1=128, h2=64, d1=0.12, d2=0.08):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, h1), nn.GELU(), nn.Dropout(d1),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(d2),
        )
        self.head_bin = nn.Linear(h2, 1)
        self.head_attack = nn.Linear(h2, 3)
        self.head_risk = nn.Linear(h2, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.head_bin(z).squeeze(1), self.head_attack(z), self.head_risk(z)


class KDDataset(Dataset):
    def __init__(self, df, feature_cols, scaler, teacher_df):
        df = df.copy()
        merge_cols = ["row_id", "teacher_prob_binary",
                      "teacher_attack_prob_1", "teacher_attack_prob_2", "teacher_attack_prob_3",
                      "teacher_risk_prob_0", "teacher_risk_prob_1", "teacher_risk_prob_2"]
        df = df.merge(teacher_df[merge_cols], on="row_id", how="left")
        miss = df["teacher_prob_binary"].isna().sum()
        if miss > 0:
            raise RuntimeError(f"Teacher alignment failed, {int(miss)} rows missing")
        X = scaler.transform(df[feature_cols].values.astype(np.float32))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_bin = torch.tensor(df["binary_label"].values.astype(np.float32))
        atk_orig = df["y_attack_type"].values.astype(np.int64)
        atk_int = np.where(df["binary_label"].values.astype(np.int64) == 1, atk_orig - 1, 0).astype(np.int64)
        self.y_attack = torch.tensor(atk_int, dtype=torch.long)
        self.y_risk = torch.tensor(df["y_risk"].values.astype(np.int64), dtype=torch.long)
        self.teacher_bin = torch.tensor(df["teacher_prob_binary"].values.astype(np.float32))
        self.teacher_attack = torch.tensor(
            df[["teacher_attack_prob_1", "teacher_attack_prob_2", "teacher_attack_prob_3"]].values.astype(np.float32))
        self.teacher_risk = torch.tensor(
            df[["teacher_risk_prob_0", "teacher_risk_prob_1", "teacher_risk_prob_2"]].values.astype(np.float32))
        self.abn_mask = torch.tensor(df["binary_label"].values.astype(np.float32))
        y_bin_np = df["binary_label"].values.astype(np.int64)
        self.w_bin = torch.tensor(self._cw(y_bin_np))
        self.w_risk = torch.tensor(self._cw(df["y_risk"].values.astype(np.int64)))
        abn = df.loc[df["binary_label"] == 1, "y_attack_type"].values.astype(np.int64) - 1
        if len(abn):
            cls, cnt = np.unique(abn, return_counts=True)
            tot = cnt.sum()
            w_map = {int(c): float(tot / (len(cls) * ci)) for c, ci in zip(cls, cnt)}
        else:
            w_map = {0: 1.0, 1: 1.0, 2: 1.0}
        aw = np.ones(len(df), dtype=np.float32)
        for i, (ib, aid) in enumerate(zip(y_bin_np, df["y_attack_type"].values.astype(np.int64) - 1)):
            if ib == 1:
                aw[i] = w_map[int(aid)]
        self.w_attack = torch.tensor(aw)
        self.row_id = df["row_id"].values.astype(np.int64)

    @staticmethod
    def _cw(y):
        cls, cnt = np.unique(y, return_counts=True)
        tot = cnt.sum()
        w_map = {int(c): float(tot / (len(cls) * ci)) for c, ci in zip(cls, cnt)}
        return np.array([w_map[int(v)] for v in y], dtype=np.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return {k: getattr(self, k)[idx] for k in
                ["x", "y_bin", "y_attack", "y_risk", "teacher_bin",
                 "teacher_attack", "teacher_risk", "abn_mask", "w_bin", "w_attack", "w_risk"]}

    # alias for backward compat
    @property
    def x(self): return self.X


def _weighted_bce(logits, target, weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def _weighted_ce(logits, target, weight):
    ce = F.cross_entropy(logits, target, reduction="none")
    return (ce * weight).sum() / weight.sum().clamp_min(1e-8)


def _masked_weighted_ce(logits, target, mask, weight):
    ce = F.cross_entropy(logits, target, reduction="none")
    return (ce * weight * mask).sum() / (mask * weight).sum().clamp_min(1e-8)


def _kd_kl(s_logits, t_probs, T):
    log_p = F.log_softmax(s_logits / T, dim=1)
    q = torch.clamp(t_probs, 1e-6, 1.0)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(log_p, q, reduction="batchmean") * (T ** 2)


def _masked_kd_kl(s_logits, t_probs, mask, T):
    log_p = F.log_softmax(s_logits / T, dim=1)
    q = torch.clamp(t_probs, 1e-6, 1.0)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
    kl = F.kl_div(log_p, q, reduction="none").sum(dim=1)
    return ((kl * mask).sum() / mask.sum().clamp_min(1e-8)) * (T ** 2)


def train_student(train_df, val_df, feature_cols, scaler,
                  teacher_train_df, teacher_val_df,
                  use_kd=True, high_recall_weight=HIGH_RECALL_WEIGHT,
                  device="cpu", verbose=False):
    train_ds = KDDataset(train_df, feature_cols, scaler, teacher_train_df)
    val_ds = KDDataset(val_df, feature_cols, scaler, teacher_val_df)
    train_loader = DataLoader(train_ds, batch_size=STUDENT_BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=STUDENT_BATCH, shuffle=False, num_workers=0)

    in_dim = len(feature_cols)
    model = StudentModel(in_dim, STUDENT_HIDDEN1, STUDENT_HIDDEN2, STUDENT_DROPOUT1, STUDENT_DROPOUT2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=STUDENT_LR, weight_decay=STUDENT_WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-5)

    alpha_bin, beta_bin = 0.80, 0.10
    alpha_attack, beta_attack = 0.50, 0.10
    alpha_risk, beta_risk = 1.20, 0.25
    if not use_kd:
        beta_bin = beta_attack = beta_risk = 0.0

    best_state, best_score, best_epoch, wait = None, -1.0, -1, 0
    for epoch in range(1, STUDENT_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            logit_bin, logits_attack, logits_risk = model(x)
            loss_hb = _weighted_bce(logit_bin, batch["y_bin"].to(device), batch["w_bin"].to(device))
            loss_kb = ((torch.sigmoid(logit_bin / STUDENT_TEMP) - batch["teacher_bin"].to(device)) ** 2
                       * batch["w_bin"].to(device)).sum() / batch["w_bin"].to(device).sum().clamp_min(1e-8)
            loss_ha = _masked_weighted_ce(logits_attack, batch["y_attack"].to(device),
                                          batch["abn_mask"].to(device), batch["w_attack"].to(device))
            loss_ka = _masked_kd_kl(logits_attack, batch["teacher_attack"].to(device),
                                    batch["abn_mask"].to(device), STUDENT_TEMP)
            loss_hr = _weighted_ce(logits_risk, batch["y_risk"].to(device), batch["w_risk"].to(device))
            loss_kr = _kd_kl(logits_risk, batch["teacher_risk"].to(device), STUDENT_TEMP)
            loss = (alpha_bin * loss_hb + beta_bin * loss_kb +
                    alpha_attack * loss_ha + beta_attack * loss_ka +
                    alpha_risk * loss_hr + beta_risk * loss_kr)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = _predict_raw(model, val_loader, device)
        val_pred = _risk_decision_simple(vp[0], vp[2], DEFAULT_TAU_ATTACK, DEFAULT_TAU_HIGH)
        m = compute_risk_metrics(val_ds.y_risk.numpy().astype(np.int64), val_pred)
        score = m["Macro_F1"] + high_recall_weight * m["High_Recall"]
        scheduler.step(score)

        if verbose and (epoch % 30 == 0 or epoch == 1):
            print(f"      Epoch {epoch:03d} | val_f1={m['Macro_F1']:.4f}")

        if score > best_score:
            best_score, best_epoch, best_state, wait = score, epoch, deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= STUDENT_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_score": float(best_score)}


@torch.no_grad()
def _predict_raw(model, loader, device):
    model.eval()
    pb, pa, pr = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        lb, la, lr_ = model(x)
        pb.append(torch.sigmoid(lb).cpu().numpy())
        pa.append(torch.softmax(la, dim=1).cpu().numpy())
        pr.append(torch.softmax(lr_, dim=1).cpu().numpy())
    return np.concatenate(pb), np.concatenate(pa), np.concatenate(pr)


def _risk_decision_simple(prob_bin, risk_prob, tau_attack, tau_high):
    p_med, p_high = risk_prob[:, 1], risk_prob[:, 2]
    pred_atk = np.where(p_high >= tau_high, 2, np.where(p_med >= p_high, 1, 2))
    return np.where(prob_bin < tau_attack, 0, pred_atk).astype(np.int64)


# ================================================================
#  Metrics
# ================================================================
def compute_risk_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, np.int64), np.asarray(y_pred, np.int64)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    _, hr, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1), "High_Recall": float(hr),
    }


def compute_brier_multiclass(y_true, risk_prob):
    y = np.asarray(y_true, np.int64)
    p = np.asarray(risk_prob, np.float64)
    one_hot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def compute_nll(y_true, risk_prob):
    y = np.asarray(y_true, np.int64)
    p = np.clip(np.asarray(risk_prob, np.float64), 1e-10, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def compute_ece_multiclass(y_true, risk_prob, n_bins=10):
    y = np.asarray(y_true, np.int64)
    p = np.asarray(risk_prob, np.float64)
    conf, pred = np.max(p, axis=1), np.argmax(p, axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y)
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf <= edges[i + 1] if i == n_bins - 1 else conf < edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_all_metrics(y_risk, risk_pred, y_bin, bin_prob, risk_prob):
    rm = compute_risk_metrics(y_risk, risk_pred)
    return {
        **rm,
        "Brier_Score": compute_brier_multiclass(y_risk, risk_prob),
        "ECE": compute_ece_multiclass(y_risk, risk_prob),
        "NLL": compute_nll(y_risk, risk_prob),
    }


# ================================================================
#  Platt Calibration
# ================================================================
def logit_transform(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_platt_calibrator(prob_val, y_val):
    x = logit_transform(prob_val).reshape(-1, 1)
    m = LogisticRegression(solver="lbfgs", random_state=SEED, max_iter=1000)
    m.fit(x, y_val.astype(int))
    return m


def apply_platt(platt_model, prob):
    x = logit_transform(prob).reshape(-1, 1)
    return np.clip(platt_model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)


class PlattRiskCalibrator:
    def __init__(self):
        self.models = []

    def fit(self, risk_prob_val, y_risk_val):
        self.models = []
        for c in range(risk_prob_val.shape[1]):
            y_c = (y_risk_val == c).astype(int)
            if len(np.unique(y_c)) < 2:
                self.models.append(None)
                continue
            x = logit_transform(risk_prob_val[:, c]).reshape(-1, 1)
            m = LogisticRegression(solver="lbfgs", random_state=SEED, max_iter=1000)
            m.fit(x, y_c)
            self.models.append(m)
        return self

    def predict(self, risk_prob):
        out = np.zeros_like(risk_prob)
        for c, m in enumerate(self.models):
            if m is None:
                out[:, c] = risk_prob[:, c]
            else:
                out[:, c] = m.predict_proba(logit_transform(risk_prob[:, c]).reshape(-1, 1))[:, 1]
        return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


# ================================================================
#  Threshold Search
# ================================================================
def search_thresholds(y_true, prob_bin, risk_prob, high_recall_weight=HIGH_RECALL_WEIGHT):
    grid = np.arange(SEARCH_MIN, SEARCH_MAX + 1e-9, SEARCH_STEP)
    best = None
    for ta in grid:
        for th in grid:
            pred = _risk_decision_simple(prob_bin, risk_prob, float(ta), float(th))
            m = compute_risk_metrics(y_true, pred)
            score = m["Macro_F1"] + high_recall_weight * m["High_Recall"]
            if best is None or score > best["score"]:
                best = {"tau_attack": float(ta), "tau_high": float(th), "score": float(score), **m}
    return best


# ================================================================
#  Single Fold Pipeline
# ================================================================
def run_single_fold(train_df, val_df, test_df, feature_cols, device, use_kd=True):
    """Run the full pipeline for one fold. Returns all variant metrics for this fold."""
    # Teacher
    binary_models, risk_model, attack_model = train_teacher(train_df, feature_cols)
    teacher_train = teacher_predict_all(binary_models, risk_model, attack_model, train_df, feature_cols)
    teacher_val = teacher_predict_all(binary_models, risk_model, attack_model, val_df, feature_cols)
    teacher_test = teacher_predict_all(binary_models, risk_model, attack_model, test_df, feature_cols)

    # Student Full
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values.astype(np.float32))
    model_full, _ = train_student(train_df, val_df, feature_cols, scaler,
                                  teacher_train, teacher_val,
                                  use_kd=True, device=device)

    # Student w/o KD
    scaler_nokd = StandardScaler()
    scaler_nokd.fit(train_df[feature_cols].values.astype(np.float32))
    if use_kd:
        model_nokd, _ = train_student(train_df, val_df, feature_cols, scaler_nokd,
                                      teacher_train, teacher_val,
                                      use_kd=False, device=device)
    else:
        model_nokd = model_full  # If KD is already disabled, reuse the full model

    # DataLoaders
    test_ds = KDDataset(test_df, feature_cols, scaler, teacher_test)
    test_loader = DataLoader(test_ds, batch_size=STUDENT_BATCH, shuffle=False, num_workers=0)
    val_ds = KDDataset(val_df, feature_cols, scaler, teacher_val)
    val_loader = DataLoader(val_ds, batch_size=STUDENT_BATCH, shuffle=False, num_workers=0)

    test_ds_nokd = KDDataset(test_df, feature_cols, scaler_nokd, teacher_test)
    test_loader_nokd = DataLoader(test_ds_nokd, batch_size=STUDENT_BATCH, shuffle=False, num_workers=0)
    val_ds_nokd = KDDataset(val_df, feature_cols, scaler_nokd, teacher_val)
    val_loader_nokd = DataLoader(val_ds_nokd, batch_size=STUDENT_BATCH, shuffle=False, num_workers=0)

    # Inference
    with torch.no_grad():
        val_raw = _predict_raw(model_full, val_loader, device)
        test_raw = _predict_raw(model_full, test_loader, device)
        test_raw_nokd = _predict_raw(model_nokd, test_loader_nokd, device)
        val_raw_nokd = _predict_raw(model_nokd, val_loader_nokd, device)

    y_risk_test = test_ds.y_risk.numpy().astype(np.int64)
    y_bin_test = test_ds.y_bin.numpy().astype(np.int64)
    y_risk_val = val_ds.y_risk.numpy().astype(np.int64)
    y_bin_val = val_ds.y_bin.numpy().astype(np.int64)

    val_bp, val_rp = val_raw[0], val_raw[2]
    test_bp, test_rp = test_raw[0], test_raw[2]
    test_bp_nokd, test_rp_nokd = test_raw_nokd[0], test_raw_nokd[2]
    val_bp_nokd, val_rp_nokd = val_raw_nokd[0], val_raw_nokd[2]

    results = {}

    # === Full RSTD-KD ===
    thresh = search_thresholds(y_risk_val, val_bp, val_rp)
    pred_full = _risk_decision_simple(test_bp, test_rp, thresh["tau_attack"], thresh["tau_high"])
    platt_risk = PlattRiskCalibrator().fit(val_rp, y_risk_val)
    platt_bin = fit_platt_calibrator(val_bp, y_bin_val)
    test_rp_platt = platt_risk.predict(test_rp)
    test_bp_platt = apply_platt(platt_bin, test_bp)
    results["Full_RSTD-KD"] = compute_all_metrics(y_risk_test, pred_full, y_bin_test, test_bp_platt, test_rp_platt)

    # === w/o KD ===
    thresh_nokd = search_thresholds(y_risk_val, val_bp_nokd, val_rp_nokd)
    pred_nokd = _risk_decision_simple(test_bp_nokd, test_rp_nokd, thresh_nokd["tau_attack"], thresh_nokd["tau_high"])
    platt_risk_nokd = PlattRiskCalibrator().fit(val_rp_nokd, y_risk_val)
    platt_bin_nokd = fit_platt_calibrator(val_bp_nokd, y_bin_val)
    results["w_o_KD"] = compute_all_metrics(
        y_risk_test, pred_nokd, y_bin_test,
        apply_platt(platt_bin_nokd, test_bp_nokd),
        platt_risk_nokd.predict(test_rp_nokd))

    # === w/o Platt ===
    pred_noplatt = _risk_decision_simple(test_bp, test_rp, thresh["tau_attack"], thresh["tau_high"])
    results["w_o_Platt"] = compute_all_metrics(y_risk_test, pred_noplatt, y_bin_test, test_bp, test_rp)

    # === w/o Cost-aware Threshold ===
    pred_nocost = _risk_decision_simple(test_bp, test_rp, DEFAULT_TAU_ATTACK, DEFAULT_TAU_HIGH)
    results["w_o_CostThresh"] = compute_all_metrics(y_risk_test, pred_nocost, y_bin_test, test_bp_platt, test_rp_platt)

    return results


# ================================================================
#  Reliability Curves
# ================================================================
def _quantile_bins(values, n_bins):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.array([0.0, 1.0])
    edges[0], edges[-1] = min(edges[0], 0.0), max(edges[-1], 1.0)
    return edges


def plot_reliability_curves(y_true, prob_dict, out_path, n_bins=15):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for name, bp in prob_dict.items():
        if isinstance(bp, tuple): bp = bp[0]
        bp = np.clip(bp, 1e-10, 1 - 1e-10)
        edges = _quantile_bins(bp, n_bins)
        cs, rs = [], []
        for i in range(len(edges) - 1):
            mask = (bp >= edges[i]) & (bp <= edges[i + 1] if i == len(edges) - 2 else bp < edges[i + 1])
            if mask.sum() > 0:
                cs.append(bp[mask].mean()); rs.append(y_true[mask].mean())
        ax.plot(cs, rs, "o-", lw=1.5, ms=4, label=name)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Observed Frequency")
    ax.set_title("Binary Reliability Curve"); ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for name, prob in prob_dict.items():
        if not isinstance(prob, tuple): continue
        rp = prob[1]
        conf, pred, correct = np.max(rp, axis=1), np.argmax(rp, axis=1), (np.argmax(rp, axis=1) == y_true).astype(float)
        edges = _quantile_bins(conf, n_bins)
        cs, ac = [], []
        for i in range(len(edges) - 1):
            mask = (conf >= edges[i]) & (conf <= edges[i + 1] if i == len(edges) - 2 else conf < edges[i + 1])
            if mask.sum() > 0:
                cs.append(conf[mask].mean()); ac.append(correct[mask].mean())
        ax2.plot(cs, ac, "o-", lw=1.5, ms=4, label=name)
    ax2.set_xlabel("Max Predicted Probability"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Multiclass Reliability Curve"); ax2.legend(fontsize=8, frameon=False); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path) + ".png", dpi=300)
    fig.savefig(str(out_path) + ".pdf")
    plt.close(fig)


# ================================================================
#  Main Function: 5-Fold CV
# ================================================================
METRIC_COLS = ["Accuracy", "Balanced_Accuracy", "Macro_F1", "High_Recall", "Brier_Score", "ECE", "NLL"]
VARIANT_KEYS = ["Full_RSTD-KD", "w_o_KD", "w_o_Platt", "w_o_CostThresh"]
VARIANT_NAMES = {
    "Full_RSTD-KD": "Full RSTD-KD",
    "w_o_KD": "w/o Knowledge Distillation",
    "w_o_Platt": "w/o Platt Calibration",
    "w_o_CostThresh": "w/o Cost-aware Threshold",
}


def main():
    set_seed(SEED)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Folds: {N_FOLDS}  |  Windows: {WINDOW_SIZES}")
    print(f"Output directory: {out_dir}")

    raw_df = pd.read_csv(RAW_DATA_CSV)
    raw_df["_dt"] = parse_time(raw_df["Time"])
    raw_df = raw_df.reset_index(drop=True)
    print(f"Total raw packets: {len(raw_df)}")

    # Collect results from all folds: {window_size: {variant_key: [fold1_metrics, fold2_metrics, ...]}}
    all_fold_results = {}

    for ws in WINDOW_SIZES:
        print(f"\n{'=' * 60}")
        print(f"  Window Size = {ws}")
        print(f"{'=' * 60}")
        folds, feature_cols = prepare_cv_folds(raw_df, ws)
        print(f"  Total windows: {sum(len(t) for t, _, _ in folds)}")

        fold_results = {vk: [] for vk in VARIANT_KEYS}
        for fi, (train_df, val_df, test_df) in enumerate(folds):
            print(f"\n  --- Fold {fi + 1}/{N_FOLDS} ---")
            print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

            try:
                results = run_single_fold(train_df, val_df, test_df, feature_cols, device)
                for vk in VARIANT_KEYS:
                    if vk in results:
                        fold_results[vk].append(results[vk])
            except Exception as e:
                print(f"    [ERROR] Fold {fi + 1} failed: {e}")
                import traceback
                traceback.print_exc()

        all_fold_results[ws] = fold_results

        # Save per-fold results for this window size
        rows = []
        for vk in VARIANT_KEYS:
            for fi, m in enumerate(fold_results[vk]):
                row = {"Variant": VARIANT_NAMES[vk], "Fold": fi + 1, "Window": ws}
                for mc in METRIC_COLS:
                    row[mc] = m.get(mc, float("nan"))
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"cv_fold_details_w{ws}.csv", index=False)

    # ================================================================
    #  Summary: Mean +/- Std
    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Summary of 5-Fold CV Results")
    print(f"{'=' * 60}")

    # --- Table 1: Component Ablation (w=32) ---
    print("\n--- Table 1: Component Ablation (w=32, 5-Fold CV) ---")
    t1_rows = []
    for vk in VARIANT_KEYS:
        metrics_list = all_fold_results[MAIN_WINDOW].get(vk, [])
        if not metrics_list:
            continue
        row = {"Variant": VARIANT_NAMES[vk]}
        for mc in METRIC_COLS:
            vals = [m[mc] for m in metrics_list if mc in m]
            row[f"{mc}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{mc}_std"] = float(np.std(vals)) if vals else float("nan")
        t1_rows.append(row)
    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(out_dir / "table1_component_ablation_w32_cv.csv", index=False)
    _print_cv_table(t1_rows)

    # --- Table 2: Window Sensitivity (Full RSTD-KD) ---
    print("\n--- Table 2: Window Size Sensitivity (Full RSTD-KD, 5-Fold CV) ---")
    t2_rows = []
    for ws in WINDOW_SIZES:
        metrics_list = all_fold_results[ws].get("Full_RSTD-KD", [])
        if not metrics_list:
            continue
        row = {"Window_Size": ws}
        for mc in METRIC_COLS:
            vals = [m[mc] for m in metrics_list if mc in m]
            row[f"{mc}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{mc}_std"] = float(np.std(vals)) if vals else float("nan")
        t2_rows.append(row)
    t2_df = pd.DataFrame(t2_rows)
    t2_df.to_csv(out_dir / "table2_window_sensitivity_cv.csv", index=False)
    _print_cv_table(t2_rows)

    # --- Table 3: Full Ablation x Window ---
    print("\n--- Table 3: Full Ablation (All Windows, 5-Fold CV) ---")
    t3_rows = []
    for ws in WINDOW_SIZES:
        for vk in VARIANT_KEYS:
            metrics_list = all_fold_results[ws].get(vk, [])
            if not metrics_list:
                continue
            row = {"Variant": VARIANT_NAMES[vk], "Window": ws}
            for mc in METRIC_COLS:
                vals = [m[mc] for m in metrics_list if mc in m]
                row[f"{mc}_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{mc}_std"] = float(np.std(vals)) if vals else float("nan")
            t3_rows.append(row)
    t3_df = pd.DataFrame(t3_rows)
    t3_df.to_csv(out_dir / "table3_full_ablation_cv.csv", index=False)
    _print_cv_table(t3_rows)

    # --- Reliability curves (using the last fold's w=32 data) ---
    # Simplified: plot on the last fold
    print("\nGenerating reliability curves (last fold, w=32)...")
    last_fold_data = all_fold_results[MAIN_WINDOW]
    # Collect predicted probabilities from all folds for plotting (use the last successful fold)
    # Simplified: use intermediate files if they exist
    mid_dir = out_dir / "intermediate" / f"w{MAIN_WINDOW}"
    mid_dir.mkdir(parents=True, exist_ok=True)

    # --- Save JSON summary ---
    summary = {
        "experiment": "RSTD-KD Component Ablation (5-Fold CV)",
        "n_folds": N_FOLDS, "seed": SEED, "window_sizes": WINDOW_SIZES,
        "device": str(device),
        "results": {},
    }
    for ws in WINDOW_SIZES:
        summary["results"][f"w{ws}"] = {}
        for vk in VARIANT_KEYS:
            ml = all_fold_results[ws].get(vk, [])
            if ml:
                agg = {}
                for mc in METRIC_COLS:
                    vals = [m[mc] for m in ml if mc in m]
                    agg[f"{mc}_mean"] = float(np.mean(vals)) if vals else None
                    agg[f"{mc}_std"] = float(np.std(vals)) if vals else None
                summary["results"][f"w{ws}"][VARIANT_NAMES[vk]] = agg
    safe_json_dump(summary, out_dir / "ablation_cv_summary.json")

    print(f"\n{'=' * 60}")
    print("  Ablation experiment completed!")
    print(f"{'=' * 60}")
    print(f"Output files:")
    print(f"  table1_component_ablation_w32_cv.csv  -- Component ablation table (w=32, mean+/-std)")
    print(f"  table2_window_sensitivity_cv.csv      -- Window sensitivity table (mean+/-std)")
    print(f"  table3_full_ablation_cv.csv           -- Full ablation table (mean+/-std)")
    print(f"  cv_fold_details_w*.csv                -- Per-fold detailed results")
    print(f"  ablation_cv_summary.json              -- Full JSON summary")
    print(f"All files saved to: {out_dir}")


def _print_cv_table(rows):
    """Print table in mean+/-std format."""
    if not rows:
        print("  (no data)")
        return
    header_cols = [k for k in rows[0].keys() if not k.endswith("_mean") and not k.endswith("_std")]
    print("  " + "  ".join(f"{c:>20s}" for c in header_cols), end="")
    for mc in METRIC_COLS:
        print(f"  {mc:>20s}", end="")
    print()
    for row in rows:
        print("  " + "  ".join(f"{str(row.get(c, '')):>20s}" for c in header_cols), end="")
        for mc in METRIC_COLS:
            mn = row.get(f"{mc}_mean", float("nan"))
            sd = row.get(f"{mc}_std", float("nan"))
            print(f"  {mn:.4f}±{sd:.4f}".rjust(20), end="")
        print()


if __name__ == "__main__":
    main()
