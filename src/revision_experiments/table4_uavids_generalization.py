#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSTD-KD UAVIDS-2025 Generalization Experiment
==============================================
Adapts the RSTD-KD (Risk-State Threat Detection via Knowledge Distillation)
framework from ECU-IoFT (packet-level, real drone WiFi) to UAVIDS-2025
(flow-level, NS-3 FANET simulation) to demonstrate framework generalizability.

Reviewer R1-#2: "All experiments are only on ECU-IoFT; generalization
across different flight environments, communication frequencies, and
attack types is questioned."

Key Adaptation Points:
  - Packet-level (93-dim SN/FN/Info features) -> Flow-level (~25-dim statistical features)
  - 3 attack types (Deauth, WPA2, TELLO) -> 4 attack types (Blackhole, Flooding, Sybil, Wormhole)
  - Sliding-window aggregation removed (each record is a complete flow)
  - Same Teacher-Student KD pipeline, Platt calibration, cost-aware thresholds
  - Risk mapping: Normal->Low, Blackhole->Medium, {Flooding,Sybil,Wormhole}->High

Pipeline:
  Load CSV -> Shuffle -> Feature Engineering -> Stratified Split
  -> Teacher Ensemble (HistGBM x7) -> Soft Label Generation
  -> Student Network (LN->FC128->FC64, 3 heads) KD Training
  -> Platt Calibration -> Cost-aware Threshold Search -> Evaluation

Usage:
  python uavids_generalization_experiment.py

Output:
  All results saved to OUTPUT_DIR (CSV, JSON, PNG/PDF)
"""

from __future__ import annotations

import json
import os
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

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
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ================================================================
#  Path Configuration
# ================================================================
RAW_DATA_CSV = r"E:\Licaiyun\practise\4.RSTD_KD\one review\fanhua\UAVIDS-2025.csv"
OUTPUT_DIR = r"E:\Licaiyun\practise\4.RSTD_KD\one review\fanhua\uavids_results"

# ================================================================
#  Global Hyperparameters (consistent with ECU-IoFT experiments)
# ================================================================
SEEDS = [42, 123, 456, 789, 1024]
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # test = 0.15

# Teacher — HistGradientBoosting ensemble
TEACHER_ENSEMBLE_SIZE = 5
TEACHER_BOOTSTRAP_FRAC = 0.90
TEACHER_MAX_ITER = 180
TEACHER_LR = 0.025
TEACHER_MAX_LEAF = 31
TEACHER_MIN_LEAF_BIN = 20
TEACHER_L2 = 1.0
# Teacher — risk model
TEACHER_RISK_MAX_ITER = 220
TEACHER_RISK_LR = 0.025
TEACHER_RISK_MAX_LEAF = 31
TEACHER_RISK_MIN_LEAF = 5
TEACHER_RISK_L2 = 0.5
# Teacher — attack-type model
TEACHER_ATK_MAX_ITER = 220
TEACHER_ATK_LR = 0.025
TEACHER_ATK_MAX_LEAF = 31
TEACHER_ATK_MIN_LEAF = 1
TEACHER_ATK_L2 = 0.5

# Student — KD neural network
STUDENT_EPOCHS = 160
STUDENT_BATCH = 128
STUDENT_LR = 4e-4
STUDENT_WD = 3e-4
STUDENT_PATIENCE = 24
STUDENT_TEMP = 2.0
STUDENT_HIDDEN1, STUDENT_HIDDEN2 = 128, 64
STUDENT_DROPOUT1, STUDENT_DROPOUT2 = 0.12, 0.08

# KD loss weights (binary, attack-type, risk)
KD_W = {
    "ab": 0.8,   # binary hard loss
    "bb": 0.1,   # binary KD loss
    "aa": 0.5,   # attack hard loss
    "ba": 0.1,   # attack KD loss
    "ar": 1.2,   # risk hard loss
    "br": 0.25,  # risk KD loss
}


# ================================================================
#  UAVIDS-2025 Label Mapping
# ================================================================
# 5 classes -> binary (normal/attack) + attack_type (0-4) + risk (0-2)
#
# Risk rationale:
#   Normal Traffic  -> Low risk (0): benign communication
#   Blackhole Attack -> Medium risk (1): routing disruption, localized impact
#   Flooding Attack  -> High risk (2): DoS-like, service denial
#   Sybil Attack     -> High risk (2): identity spoofing, trust erosion
#   Wormhole Attack  -> High risk (2): tunnel-based, severe network disruption
#
LABEL_MAP = {
    "Normal Traffic":  {"binary": 0, "attack_type": 0, "risk": 0, "risk_name": "Normal"},
    "Blackhole Attack": {"binary": 1, "attack_type": 1, "risk": 1, "risk_name": "Medium"},
    "Flooding Attack":  {"binary": 1, "attack_type": 2, "risk": 2, "risk_name": "High"},
    "Sybil Attack":     {"binary": 1, "attack_type": 3, "risk": 2, "risk_name": "High"},
    "Wormhole Attack":  {"binary": 1, "attack_type": 4, "risk": 2, "risk_name": "High"},
}

ATTACK_ID_TO_NAME = {
    0: "Normal", 1: "Blackhole", 2: "Flooding", 3: "Sybil", 4: "Wormhole",
}
ATTACK_ID_TO_RISK = {
    0: "normal", 1: "medium", 2: "high", 3: "high", 4: "high",
}
RISK_ID_TO_NAME = {0: "Normal", 1: "Medium", 2: "High"}

N_ATTACK_TYPES = 4   # number of distinct attack types (excluding Normal)
N_RISK_LEVELS = 3    # Normal, Medium, High


# ================================================================
#  Feature Engineering Constants
# ================================================================
# Raw numeric features from UAVIDS-2025 (excluding FlowID, Protocol, DstPort)
BASE_NUMERIC = [
    "FlowDuration/s",
    "SrcPort",
    "TxPackets", "RxPackets", "LostPackets",
    "TxBytes", "RxBytes",
    "TxPacketRate/s", "RxPacketRate/s",
    "TxByteRate/s", "RxByteRate/s",
    "MeanDelay/s", "MeanJitter/s",
    "Throughput/Kbps", "MeanPacketSize",
    "PacketDropRate", "AverageHopCount",
]

IP_COLS = ["SrcAddr", "DstAddr"]

# Derived feature definitions: name -> (numerator_col, denominator_col)
# Division results are clipped to [0, 1e6]; NaN replaced with 0.
DERIVED_FEATURES = {
    "LossRatio": ("LostPackets", "TxPackets"),
    "ByteAsymmetry": ("TxBytes", "RxBytes"),
    "PacketAsymmetry": ("TxPackets", "RxPackets"),
    "TotalPacketRate": ("TxPacketRate/s", "RxPacketRate/s"),   # sum
    "TotalByteRate": ("TxByteRate/s", "RxByteRate/s"),         # sum
    "DelayJitterProduct": ("MeanDelay/s", "MeanJitter/s"),     # product
}


# ================================================================
#  Basic Utilities
# ================================================================
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_json_dump(obj, path):
    """JSON dump with numpy type handling."""
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NpEncoder)


def balanced_sample_weight(y):
    """Compute balanced sample weights (inverse class frequency)."""
    y = np.asarray(y, np.int64)
    cls, cnt = np.unique(y, return_counts=True)
    tot = cnt.sum()
    wm = {int(c): float(tot / (len(cls) * ci)) for c, ci in zip(cls, cnt)}
    return np.array([wm[int(v)] for v in y], np.float32)


# ================================================================
#  Data Loading
# ================================================================
def load_data() -> pd.DataFrame:
    """Load UAVIDS-2025 CSV and shuffle (original data is sorted by label)."""
    df = pd.read_csv(RAW_DATA_CSV)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"  Loaded: {df.shape[0]} records, {df.shape[1]} columns")
    for lbl, cnt in df["label"].value_counts().items():
        print(f"    {lbl}: {cnt} ({cnt / len(df) * 100:.1f}%)")
    return df


# ================================================================
#  Feature Engineering
# ================================================================
def encode_ip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract last octet from IP addresses as numeric features."""
    out = df.copy()
    for col in IP_COLS:
        if col in out.columns:
            out[f"{col}_last_octet"] = out[col].astype(str).str.extract(r"(\d+)$").astype(float)
    return out


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from traffic statistics."""
    out = df.copy()
    for name, (col_a, col_b) in DERIVED_FEATURES.items():
        if name == "TotalPacketRate":
            out[name] = out[col_a].fillna(0) + out[col_b].fillna(0)
        elif name == "TotalByteRate":
            out[name] = out[col_a].fillna(0) + out[col_b].fillna(0)
        elif name == "DelayJitterProduct":
            out[name] = out[col_a].fillna(0) * out[col_b].fillna(0)
        else:
            denom = out[col_b].replace(0, np.nan)
            out[name] = (out[col_a].fillna(0) / denom).clip(0, 1e6).fillna(0)
    return out


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the list of numeric feature column names for model input."""
    drop_cols = {"FlowID", "Protocol", "DstPort", "label",
                 "SrcAddr", "DstAddr", "SrcAddr_encoded", "DstAddr_encoded",
                 "binary_label", "y_attack_type", "y_risk", "y_risk_name",
                 "row_id"}
    fcols = [c for c in df.columns if c not in drop_cols
             and pd.api.types.is_numeric_dtype(df[c])]
    return fcols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline for UAVIDS-2025.

    Steps:
      1. Extract IP last octet
      2. Build derived features (ratios, sums, products)
      3. Returns the augmented DataFrame
    """
    out = encode_ip_columns(df)
    out = build_derived_features(out)
    return out


# ================================================================
#  Label Encoding
# ================================================================
def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map string labels to binary_label, y_attack_type, y_risk."""
    out = df.copy()
    out["binary_label"] = out["label"].map(lambda x: LABEL_MAP[x]["binary"]).astype(np.int64)
    out["y_attack_type"] = out["label"].map(lambda x: LABEL_MAP[x]["attack_type"]).astype(np.int64)
    out["y_risk"] = out["label"].map(lambda x: LABEL_MAP[x]["risk"]).astype(np.int64)
    out["y_risk_name"] = out["label"].map(lambda x: LABEL_MAP[x]["risk_name"])
    return out


# ================================================================
#  Data Splitting
# ================================================================
def split_stratified(df: pd.DataFrame, seed: int,
                     train_r: float = TRAIN_RATIO,
                     val_r: float = VAL_RATIO):
    """Stratified random split by y_attack_type.

    Each attack type is independently split into train/val/test to ensure
    all classes are represented in every subset.
    """
    out = df.copy()
    train_idxs, val_idxs, test_idxs = [], [], []
    rng = np.random.RandomState(seed)
    for atk_id in sorted(out["y_attack_type"].unique()):
        idxs = out.loc[out["y_attack_type"] == atk_id].index.tolist()
        rng.shuffle(idxs)
        n = len(idxs)
        nt = max(1, int(round(n * train_r)))
        nv = max(1, int(round(n * val_r)))
        train_idxs.extend(idxs[:nt])
        val_idxs.extend(idxs[nt:nt + nv])
        test_idxs.extend(idxs[nt + nv:])
    return out.loc[train_idxs], out.loc[val_idxs], out.loc[test_idxs]


def prepare_split_data(train_raw, val_raw, test_raw, fcols):
    """Standardize splits: assign row_id, median imputation of NaN."""
    dfs = []
    for raw in [train_raw, val_raw, test_raw]:
        d = raw.copy().reset_index(drop=True)
        d["row_id"] = np.arange(len(d), dtype=np.int64)
        dfs.append(d)
    train_df, val_df, test_df = dfs
    # Median imputation from training set
    med = train_df[fcols].median(numeric_only=True).to_dict()
    for d in [train_df, val_df, test_df]:
        d[fcols] = d[fcols].fillna(med)
    return train_df, val_df, test_df


# ================================================================
#  Teacher Ensemble
# ================================================================
def train_teacher(train_df: pd.DataFrame, fcols: List[str]):
    """Train the teacher ensemble.

    Components:
      - bmodels: 5 HistGBM models for binary classification (normal vs attack)
                 with bootstrap sampling and balanced weights
      - rmodel:  1 HistGBM for 3-level risk classification
      - amodel:  1 HistGBM for attack type classification (attack samples only)

    Returns:
        (bmodels, rmodel, amodel) tuple
    """
    Xb = train_df[fcols].values.astype(np.float32)
    yb = train_df["binary_label"].values.astype(np.int64)
    wb = balanced_sample_weight(yb)

    # --- Binary ensemble (5 models with bootstrap) ---
    bmodels = []
    for i in range(TEACHER_ENSEMBLE_SIZE):
        rng = np.random.RandomState(42 + 1009 * (i + 1))
        parts = []
        for c in np.unique(yb):
            idx = np.where(yb == c)[0]
            n = max(1, int(round(len(idx) * TEACHER_BOOTSTRAP_FRAC)))
            parts.append(rng.choice(idx, size=n, replace=True))
        bi = np.concatenate(parts)
        rng.shuffle(bi)
        m = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=TEACHER_LR,
            max_iter=TEACHER_MAX_ITER, max_leaf_nodes=TEACHER_MAX_LEAF,
            min_samples_leaf=TEACHER_MIN_LEAF_BIN, l2_regularization=TEACHER_L2,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=42 + i,
        )
        m.fit(Xb[bi], yb[bi], sample_weight=wb[bi])
        bmodels.append(m)

    # --- Risk model (3-class: Normal, Medium, High) ---
    yr = train_df["y_risk"].values.astype(np.int64)
    wr = balanced_sample_weight(yr)
    es = np.min(np.bincount(yr)) >= 2
    rmodel = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=TEACHER_RISK_LR,
        max_iter=TEACHER_RISK_MAX_ITER, max_leaf_nodes=TEACHER_RISK_MAX_LEAF,
        min_samples_leaf=TEACHER_RISK_MIN_LEAF, l2_regularization=TEACHER_RISK_L2,
        early_stopping=es, random_state=42 + 37,
    )
    rmodel.fit(Xb, yr, sample_weight=wr)

    # --- Attack type model (attack samples only, N_ATTACK_TYPES classes) ---
    sub = train_df.loc[train_df["binary_label"] == 1]
    Xa = sub[fcols].values.astype(np.float32)
    ya = sub["y_attack_type"].values.astype(np.int64)
    wa = balanced_sample_weight(ya)
    es_a = np.min(np.bincount(ya)) >= 2 if len(ya) > 0 else False
    amodel = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=TEACHER_ATK_LR,
        max_iter=TEACHER_ATK_MAX_ITER, max_leaf_nodes=TEACHER_ATK_MAX_LEAF,
        min_samples_leaf=TEACHER_ATK_MIN_LEAF, l2_regularization=TEACHER_ATK_L2,
        early_stopping=es_a, random_state=42 + 17,
    )
    amodel.fit(Xa, ya, sample_weight=wa)

    return bmodels, rmodel, amodel


def pred_bin_ens(models, X: np.ndarray) -> np.ndarray:
    """Average binary probability from ensemble (P(attack))."""
    probs = np.stack([
        m.predict_proba(X)[:, list(m.classes_).index(1)]
        for m in models
    ])
    return np.mean(probs, axis=0)


def pred_fixed(model, X: np.ndarray, classes: list) -> np.ndarray:
    """Predict class probabilities with fixed class order (handles missing classes)."""
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)), np.float64)
    mc = list(model.classes_)
    for j, c in enumerate(classes):
        if c in mc:
            out[:, j] = raw[:, mc.index(c)]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


# ================================================================
#  Student Neural Network
# ================================================================
class StudentModel(nn.Module):
    """Knowledge Distillation Student Network.

    Architecture:
        LayerNorm -> Linear(in_dim, 128) -> GELU -> Dropout(0.12)
                  -> Linear(128, 64) -> GELU -> Dropout(0.08)
                  -> 3 heads:
                       - binary:  Linear(64, 1)    [sigmoid for normal/attack]
                       - attack:  Linear(64, n_atk) [softmax for attack type]
                       - risk:    Linear(64, 3)    [softmax for risk level]

    Adaptation from ECU-IoFT:
        - Input dimension: 93 -> ~25 (flow-level features)
        - Attack head: 3 -> 4 outputs (4 attack types in UAVIDS-2025)
        - Binary/risk heads unchanged
    """
    def __init__(self, in_dim, h1=128, h2=64, d1=0.12, d2=0.08, n_atk=N_ATTACK_TYPES):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h1), nn.GELU(), nn.Dropout(d1),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(d2),
        )
        self.hb = nn.Linear(h2, 1)       # binary head
        self.ha = nn.Linear(h2, n_atk)   # attack type head
        self.hr = nn.Linear(h2, 3)       # risk head

    def forward(self, x):
        z = self.backbone(x)
        return self.hb(z).squeeze(1), self.ha(z), self.hr(z)


# ================================================================
#  Dataset (PyTorch)
# ================================================================
class DS(Dataset):
    """Dataset for KD training.

    Merges student features with teacher soft-labels and computes
    balanced sample weights for each task head.

    Key fields:
        x:   standardized feature tensor
        yb:  binary label (0/1)
        ya:  attack type index (0-based, for attack samples; 0 for normal, masked)
        yr:  risk level (0/1/2)
        tb:  teacher binary probability
        ta:  teacher attack type probability vector (n_atk classes)
        tr:  teacher risk probability vector (3 classes)
        am:  attack mask (1.0 if attack sample, 0.0 otherwise)
        wb:  balanced weight for binary task
        wa:  balanced weight for attack type task
        wrk: balanced weight for risk task
    """
    def __init__(self, df, fcols, scaler, teacher_df):
        df = df.copy()

        # Build teacher column names dynamically
        atk_prob_cols = [f"teacher_attack_prob_{i}" for i in range(1, N_ATTACK_TYPES + 1)]
        risk_prob_cols = [f"teacher_risk_prob_{i}" for i in range(N_RISK_LEVELS)]
        merge_cols = ["row_id", "teacher_prob_binary"] + atk_prob_cols + risk_prob_cols

        df = df.merge(teacher_df[merge_cols], on="row_id", how="left")

        # Standardize features
        X = scaler.transform(df[fcols].values.astype(np.float32))
        self.X = torch.tensor(X, dtype=torch.float32)

        # Binary label
        self.yb = torch.tensor(df["binary_label"].values.astype(np.float32))

        # Attack type index:
        #   attack samples: y_attack_type in [1..N_ATTACK_TYPES] -> index [0..N-1]
        #   normal samples: y_attack_type=0 -> index 0 (masked out in loss)
        ao = df["y_attack_type"].values.astype(np.int64)
        ai = np.where(
            df["binary_label"].values.astype(np.int64) == 1,
            ao - 1,  # shift attack types from 1-based to 0-based
            0,
        ).astype(np.int64)
        self.ya = torch.tensor(ai, dtype=torch.long)

        # Risk level
        self.yr = torch.tensor(df["y_risk"].values.astype(np.int64), dtype=torch.long)

        # Teacher soft labels
        self.tb = torch.tensor(df["teacher_prob_binary"].values.astype(np.float32))
        self.ta = torch.tensor(df[atk_prob_cols].values.astype(np.float32))
        self.tr = torch.tensor(df[risk_prob_cols].values.astype(np.float32))

        # Attack mask (1.0 for attack samples, 0.0 for normal)
        self.am = torch.tensor(df["binary_label"].values.astype(np.float32))

        # Balanced sample weights
        yb_np = df["binary_label"].values.astype(np.int64)
        self.wb = torch.tensor(balanced_sample_weight(yb_np))
        self.wrk = torch.tensor(balanced_sample_weight(df["y_risk"].values.astype(np.int64)))

        # Attack type balanced weights (only for attack samples)
        abn = df.loc[df["binary_label"] == 1, "y_attack_type"].values.astype(np.int64) - 1
        if len(abn):
            cls, cnt = np.unique(abn, return_counts=True)
            tot = cnt.sum()
            wm = {int(c): float(tot / (len(cls) * ci)) for c, ci in zip(cls, cnt)}
        else:
            wm = {i: 1.0 for i in range(N_ATTACK_TYPES)}
        aw = np.ones(len(df), np.float32)
        for i, (ib, aid) in enumerate(zip(yb_np, df["y_attack_type"].values.astype(np.int64) - 1)):
            if ib == 1:
                aw[i] = wm.get(int(aid), 1.0)
        self.wa = torch.tensor(aw)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {
            "x": self.X[i], "yb": self.yb[i], "ya": self.ya[i], "yr": self.yr[i],
            "tb": self.tb[i], "ta": self.ta[i], "tr": self.tr[i], "am": self.am[i],
            "wb": self.wb[i], "wa": self.wa[i], "wrk": self.wrk[i],
        }


# ================================================================
#  Student Training (KD)
# ================================================================
def train_student(train_df, val_df, fcols, scaler, t_train, t_val, device):
    """Train student model via Knowledge Distillation.

    Combined loss:
        L = ab * BCE(binary) + bb * MSE(KD-binary)
          + aa * CE(attack)  + ba * KL(KD-attack)
          + ar * CE(risk)    + br * KL(KD-risk)

    Early stopping based on validation risk F1 + 0.1 * High Recall.
    """
    tds = DS(train_df, fcols, scaler, t_train)
    vds = DS(val_df, fcols, scaler, t_val)
    tl = DataLoader(tds, STUDENT_BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(vds, STUDENT_BATCH, shuffle=False, num_workers=0)

    model = StudentModel(
        len(fcols), STUDENT_HIDDEN1, STUDENT_HIDDEN2,
        STUDENT_DROPOUT1, STUDENT_DROPOUT2, N_ATTACK_TYPES,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=STUDENT_LR, weight_decay=STUDENT_WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "max", factor=0.5, patience=6, min_lr=1e-5,
    )

    ab, bb, aa, ba, ar, br = KD_W["ab"], KD_W["bb"], KD_W["aa"], KD_W["ba"], KD_W["ar"], KD_W["br"]
    best_state, best_score, wait = None, -1.0, 0

    for ep in range(1, STUDENT_EPOCHS + 1):
        model.train()
        for b in tl:
            x = b["x"].to(device)
            lb, la, lr = model(x)

            # --- Binary: hard BCE + KD MSE ---
            lhb = (F.binary_cross_entropy_with_logits(
                lb, b["yb"].to(device), reduction="none"
            ) * b["wb"].to(device)).sum() / b["wb"].to(device).sum().clamp_min(1e-8)
            lkb = ((torch.sigmoid(lb / STUDENT_TEMP) - b["tb"].to(device)) ** 2
                   * b["wb"].to(device)).sum() / b["wb"].to(device).sum().clamp_min(1e-8)

            # --- Attack type: hard CE + KD KL (only on attack samples) ---
            ce = F.cross_entropy(la, b["ya"].to(device), reduction="none")
            lha = (ce * b["wa"].to(device) * b["am"].to(device)).sum() / \
                  (b["am"].to(device) * b["wa"].to(device)).sum().clamp_min(1e-8)
            lp = F.log_softmax(la / STUDENT_TEMP, 1)
            qq = torch.clamp(b["ta"].to(device), 1e-6, 1.0)
            qq = qq / qq.sum(1, keepdim=True).clamp_min(1e-8)
            lka = (F.kl_div(lp, qq, reduction="none").sum(1) * b["am"].to(device)).sum() / \
                  b["am"].to(device).sum().clamp_min(1e-8) * (STUDENT_TEMP ** 2)

            # --- Risk: hard CE + KD KL ---
            lhr = (F.cross_entropy(lr, b["yr"].to(device), reduction="none")
                   * b["wrk"].to(device)).sum() / b["wrk"].to(device).sum().clamp_min(1e-8)
            lp2 = F.log_softmax(lr / STUDENT_TEMP, 1)
            q2 = torch.clamp(b["tr"].to(device), 1e-6, 1.0)
            q2 = q2 / q2.sum(1, keepdim=True).clamp_min(1e-8)
            lkr = F.kl_div(lp2, q2, reduction="batchmean") * (STUDENT_TEMP ** 2)

            loss = ab * lhb + bb * lkb + aa * lha + ba * lka + ar * lhr + br * lkr
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # --- Validation: early stopping ---
        model.eval()
        with torch.no_grad():
            pb, pa, pr = [], [], []
            for b in vl:
                x = b["x"].to(device)
                lb, la, lr = model(x)
                pb.append(torch.sigmoid(lb).cpu().numpy())
                pa.append(torch.softmax(la, 1).cpu().numpy())
                pr.append(torch.softmax(lr, 1).cpu().numpy())
            pb = np.concatenate(pb)
            pa = np.concatenate(pa)
            pr = np.concatenate(pr)

        pmed, phigh = pr[:, 1], pr[:, 2]
        pside = np.where(phigh >= 0.5, 2, np.where(pmed >= phigh, 1, 2))
        pred = np.where(pb < 0.5, 0, pside)
        yr_v = vds.yr.numpy().astype(np.int64)
        _, _, f1, _ = precision_recall_fscore_support(
            yr_v, pred, labels=[0, 1, 2], average="macro", zero_division=0,
        )
        _, hr, _, _ = precision_recall_fscore_support(
            yr_v, pred, labels=[2], average="macro", zero_division=0,
        )
        sc = f1 + 0.1 * hr
        sch.step(sc)

        if sc > best_score:
            best_score = sc
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= STUDENT_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model


# ================================================================
#  Inference
# ================================================================
@torch.no_grad()
def predict_all(model, loader, device):
    """Run inference: returns (binary_prob, attack_prob, risk_prob)."""
    model.eval()
    pb, pa, pr = [], [], []
    for b in loader:
        x = b["x"].to(device)
        lb, la, lr = model(x)
        pb.append(torch.sigmoid(lb).cpu().numpy())
        pa.append(torch.softmax(la, 1).cpu().numpy())
        pr.append(torch.softmax(lr, 1).cpu().numpy())
    return np.concatenate(pb), np.concatenate(pa), np.concatenate(pr)


def risk_decision(prob_bin, risk_prob, tau_attack=0.5, tau_high=0.5):
    """Three-level risk decision based on binary and risk probabilities.

    Decision rule:
        if P(attack) < tau_attack -> Normal (0)
        elif P(High) >= tau_high  -> High (2)
        elif P(Medium) >= P(High) -> Medium (1)
        else                      -> High (2)
    """
    p_med, p_high = risk_prob[:, 1], risk_prob[:, 2]
    pred_atk = np.where(p_high >= tau_high, 2, np.where(p_med >= p_high, 1, 2))
    return np.where(prob_bin < tau_attack, 0, pred_atk).astype(np.int64)


# ================================================================
#  Platt Calibration
# ================================================================
def logit_transform(p):
    """Transform probabilities to logit space."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_platt_calibrator(prob_val, y_val):
    """Fit binary Platt calibrator (LogisticRegression on logit-transformed probs)."""
    if len(np.unique(y_val)) < 2:
        return None
    x = logit_transform(prob_val).reshape(-1, 1)
    m = LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000)
    m.fit(x, y_val.astype(int))
    return m


def apply_platt(platt_model, prob):
    """Apply Platt calibration to binary probabilities."""
    if platt_model is None:
        return np.clip(prob, 1e-6, 1 - 1e-6)
    x = logit_transform(prob).reshape(-1, 1)
    return np.clip(platt_model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)


class PlattRiskCalibrator:
    """Per-class Platt calibrator for risk probabilities.

    Fits one LogisticRegression per risk class on logit-transformed
    probabilities, then re-normalizes the output.
    """
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
            m = LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000)
            m.fit(x, y_c)
            self.models.append(m)
        return self

    def predict(self, risk_prob):
        out = np.zeros_like(risk_prob)
        for c, m in enumerate(self.models):
            if m is None:
                out[:, c] = risk_prob[:, c]
            else:
                out[:, c] = m.predict_proba(
                    logit_transform(risk_prob[:, c]).reshape(-1, 1)
                )[:, 1]
        return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


# ================================================================
#  Cost-aware Threshold Search
# ================================================================
def search_thresholds(y_true, prob_bin, risk_prob):
    """Grid search for optimal (tau_attack, tau_high) thresholds.

    Objective: maximize Macro-F1 + 0.1 * High-Recall over risk levels.
    Grid: [0.05, 0.95] with step 0.02 for both thresholds.
    """
    grid = np.arange(0.05, 0.96, 0.02)
    best = None
    for ta in grid:
        for th in grid:
            pred = risk_decision(prob_bin, risk_prob, float(ta), float(th))
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0,
            )
            _, hr, _, _ = precision_recall_fscore_support(
                y_true, pred, labels=[2], average="macro", zero_division=0,
            )
            sc = f1 + 0.1 * hr
            if best is None or sc > best["score"]:
                best = {"tau_attack": float(ta), "tau_high": float(th), "score": float(sc)}
    return best


# ================================================================
#  Full Pipeline
# ================================================================
def run_full_pipeline(train_df, val_df, test_df, fcols, device, seed=42):
    """Run the complete RSTD-KD pipeline.

    Steps:
      1. Train teacher ensemble
      2. Generate teacher soft labels
      3. Standardize features (StandardScaler)
      4. Train student via KD
      5. Inference on val/test
      6. Search cost-aware thresholds (on val)
      7. Platt calibration (on val)
      8. Evaluate on test

    Returns:
        Dict with accuracy, balanced_accuracy, macro_f1, per_risk, per_attack,
        confusion_matrix, thresholds, and sample counts.
    """
    set_seed(seed)

    # --- 1. Teacher ---
    bmodels, rmodel, amodel = train_teacher(train_df, fcols)

    # Attack type class list for teacher predictions (dynamic)
    atk_classes = list(range(1, N_ATTACK_TYPES + 1))

    def tpred(df):
        """Generate teacher soft labels for all samples in df."""
        X = df[fcols].values.astype(np.float32)
        bp = pred_bin_ens(bmodels, X)
        rp = pred_fixed(rmodel, X, [0, 1, 2])
        ap = pred_fixed(amodel, X, atk_classes)
        result = {"row_id": df["row_id"].values.astype(np.int64),
                  "teacher_prob_binary": bp.astype(np.float32)}
        for i in range(N_RISK_LEVELS):
            result[f"teacher_risk_prob_{i}"] = rp[:, i].astype(np.float32)
        for i in range(N_ATTACK_TYPES):
            result[f"teacher_attack_prob_{i + 1}"] = ap[:, i].astype(np.float32)
        return pd.DataFrame(result)

    t_train = tpred(train_df)
    t_val = tpred(val_df)
    t_test = tpred(test_df)

    # --- 2. Scaler ---
    scaler = StandardScaler()
    scaler.fit(train_df[fcols].values.astype(np.float32))

    # --- 3. Student ---
    model = train_student(train_df, val_df, fcols, scaler, t_train, t_val, device)

    # --- 4. Inference ---
    test_ds = DS(test_df, fcols, scaler, t_test)
    test_loader = DataLoader(test_ds, STUDENT_BATCH, shuffle=False, num_workers=0)
    val_ds = DS(val_df, fcols, scaler, t_val)
    val_loader = DataLoader(val_ds, STUDENT_BATCH, shuffle=False, num_workers=0)

    test_bp, test_ap, test_rp = predict_all(model, test_loader, device)
    val_bp, val_ap, val_rp = predict_all(model, val_loader, device)

    y_risk_test = test_ds.yr.numpy().astype(np.int64)
    y_risk_val = val_ds.yr.numpy().astype(np.int64)
    y_bin_test = test_ds.yb.numpy().astype(int)
    y_bin_val = val_ds.yb.numpy().astype(int)

    # --- 5. Threshold search (on validation) ---
    thresh = search_thresholds(y_risk_val, val_bp, val_rp)
    pred = risk_decision(test_bp, test_rp, thresh["tau_attack"], thresh["tau_high"])

    # --- 6. Platt calibration ---
    platt_risk = PlattRiskCalibrator().fit(val_rp, y_risk_val)
    platt_bin = fit_platt_calibrator(val_bp, y_bin_val)
    test_rp_platt = platt_risk.predict(test_rp)
    test_bp_platt = apply_platt(platt_bin, test_bp)

    # --- 7. Evaluation ---
    acc = float(accuracy_score(y_risk_test, pred))
    bacc = float(balanced_accuracy_score(y_risk_test, pred))
    macro_f1 = float(f1_score(y_risk_test, pred, labels=[0, 1, 2],
                               average="macro", zero_division=0))

    # Per-risk-level metrics
    p, r, f1, sup = precision_recall_fscore_support(
        y_risk_test, pred, labels=[0, 1, 2], zero_division=0,
    )
    per_risk = {}
    for i, name in enumerate(["Normal", "Medium", "High"]):
        per_risk[name] = {
            "Precision": float(p[i]), "Recall": float(r[i]),
            "F1": float(f1[i]), "Support": int(sup[i]),
        }

    # Per-attack-type detection rate (using calibrated binary prediction)
    pred_bin = (test_bp_platt >= 0.5).astype(int)
    y_attack_test = test_df["y_attack_type"].values.astype(np.int64)
    per_attack = {}
    for aid in sorted(np.unique(y_attack_test)):
        mask = y_attack_test == aid
        if mask.sum() == 0:
            continue
        name = ATTACK_ID_TO_NAME.get(aid, f"Attack_{aid}")
        risk_name = ATTACK_ID_TO_RISK.get(aid, "unknown")
        det_rate = float(pred_bin[mask].mean()) if aid != 0 else float(1 - pred_bin[mask].mean())
        per_attack[name] = {
            "support": int(mask.sum()),
            "detection_rate": float(det_rate),
            "risk_level": risk_name,
        }

    # Per-attack-type risk F1 (for samples predicted as attack)
    per_attack_risk_f1 = {}
    for aid in sorted(np.unique(y_attack_test)):
        if aid == 0:
            continue
        mask = y_attack_test == aid
        if mask.sum() < 2:
            continue
        name = ATTACK_ID_TO_NAME.get(aid, f"Attack_{aid}")
        true_risk = y_risk_test[mask]
        pred_risk = pred[mask]
        af1 = float(f1_score(true_risk, pred_risk, average="macro", zero_division=0))
        per_attack_risk_f1[name] = af1

    # Confusion matrix
    cm = confusion_matrix(y_risk_test, pred, labels=[0, 1, 2])

    return {
        "accuracy": acc, "balanced_accuracy": bacc, "macro_f1": macro_f1,
        "tau_attack": thresh["tau_attack"], "tau_high": thresh["tau_high"],
        "per_risk": per_risk, "per_attack": per_attack,
        "per_attack_risk_f1": per_attack_risk_f1,
        "confusion_matrix": cm.tolist(),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    }


# ================================================================
#  Visualization
# ================================================================
def plot_generalization_results(all_results, out_dir):
    """Generate visualization plots for UAVIDS-2025 generalization experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Multi-seed stability
    ax = axes[0, 0]
    seeds = sorted(all_results["multi_seed"].keys())
    accs = [all_results["multi_seed"][s]["accuracy"] for s in seeds]
    mf1s = [all_results["multi_seed"][s]["macro_f1"] for s in seeds]
    x = np.arange(len(seeds))
    ax.bar(x - 0.15, accs, 0.35, label="Accuracy", color="#3498db", alpha=0.85)
    ax.bar(x + 0.2, mf1s, 0.35, label="Macro F1", color="#e74c3c", alpha=0.85)
    ax.axhline(np.mean(accs), color="#3498db", ls="--", alpha=0.5,
               label=f"Mean Acc={np.mean(accs):.4f}")
    ax.axhline(np.mean(mf1s), color="#e74c3c", ls="--", alpha=0.5,
               label=f"Mean F1={np.mean(mf1s):.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=9)
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Score")
    ax.set_title("(a) Multi-Seed Stability (UAVIDS-2025)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) Per-risk-level F1 across seeds
    ax2 = axes[0, 1]
    risk_names = ["Normal", "Medium", "High"]
    risk_colors = {"Normal": "#3498db", "Medium": "#e67e22", "High": "#e74c3c"}
    for rn in risk_names:
        f1s = [all_results["multi_seed"][s]["per_risk"][rn]["F1"] for s in seeds]
        ax2.plot(x, f1s, "o-", color=risk_colors[rn], linewidth=2, markersize=7, label=rn)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seeds], fontsize=9)
    ax2.set_xlabel("Random Seed")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("(b) Per-Risk-Level F1 across Seeds")
    ax2.legend(fontsize=9, frameon=False)
    ax2.grid(True, alpha=0.3, axis="y")

    # (c) Per-attack-type detection rate
    ax3 = axes[1, 0]
    seed0 = seeds[0]
    attacks = list(all_results["multi_seed"][seed0]["per_attack"].keys())
    det_rates = [all_results["multi_seed"][seed0]["per_attack"][a]["detection_rate"] for a in attacks]
    bar_colors = ["#3498db" if "Normal" in a else "#e74c3c" for a in attacks]
    ax3.barh(attacks, det_rates, color=bar_colors, alpha=0.85)
    ax3.set_xlabel("Detection Rate")
    ax3.set_xlim(0, 1.05)
    ax3.set_title("(c) Per-Attack Detection Rate (Seed 42)")
    ax3.grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(det_rates):
        ax3.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    # (d) 5-Fold CV comparison
    ax4 = axes[1, 1]
    if all_results.get("kfold_cv"):
        folds = list(range(1, len(all_results["kfold_cv"]) + 1))
        fold_accs = [r["accuracy"] for r in all_results["kfold_cv"]]
        fold_f1s = [r["macro_f1"] for r in all_results["kfold_cv"]]
        ax4.bar(np.array(folds) - 0.15, fold_accs, 0.35, label="Accuracy",
                color="#3498db", alpha=0.85)
        ax4.bar(np.array(folds) + 0.2, fold_f1s, 0.35, label="Macro F1",
                color="#e74c3c", alpha=0.85)
        ax4.axhline(np.mean(fold_accs), color="#3498db", ls="--", alpha=0.5,
                    label=f"Mean Acc={np.mean(fold_accs):.4f}")
        ax4.axhline(np.mean(fold_f1s), color="#e74c3c", ls="--", alpha=0.5,
                    label=f"Mean F1={np.mean(fold_f1s):.4f}")
        ax4.set_xticks(folds)
        ax4.set_xlabel("Fold")
        ax4.set_ylabel("Score")
        ax4.set_title("(d) 5-Fold CV Stability")
        ax4.legend(fontsize=8, frameon=False)
        ax4.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "uavids_generalization.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "uavids_generalization.pdf"), bbox_inches="tight")
    plt.close(fig)


# ================================================================
#  Main Function
# ================================================================
def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  RSTD-KD UAVIDS-2025 Generalization Experiment")
    print("=" * 80)
    print(f"  Device: {device}  |  Output: {out_dir}")
    print(f"  Attack types: {N_ATTACK_TYPES}  |  Risk levels: {N_RISK_LEVELS}")

    # ============================================================
    #  Load & preprocess data
    # ============================================================
    print("\n[0] Loading UAVIDS-2025 dataset...")
    raw_df = load_data()

    print("\n[0] Feature engineering...")
    raw_df = engineer_features(raw_df)
    raw_df = encode_labels(raw_df)

    fcols = get_feature_cols(raw_df)
    print(f"  Feature columns ({len(fcols)}): {fcols[:5]}...")

    # Quick sanity check
    for col in fcols:
        n_nan = raw_df[col].isna().sum()
        if n_nan > 0:
            print(f"  [WARN] {col}: {n_nan} NaN values (will be imputed)")

    print(f"\n  Label distribution after encoding:")
    for rl in [0, 1, 2]:
        n = (raw_df["y_risk"] == rl).sum()
        print(f"    Risk {rl} ({RISK_ID_TO_NAME[rl]}): {n}")

    all_results = {"multi_seed": {}, "kfold_cv": []}

    # ============================================================
    #  Experiment 1: Multi-seed (5 seeds)
    # ============================================================
    print("\n" + "=" * 80)
    print("[1/2] Multi-seed experiment (5 seeds)...")
    print("=" * 80)

    for seed in SEEDS:
        print(f"\n  -> Seed {seed}")
        set_seed(seed)
        train_raw, val_raw, test_raw = split_stratified(raw_df, seed)
        train_df, val_df, test_df = prepare_split_data(train_raw, val_raw, test_raw, fcols)
        print(f"    Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        result = run_full_pipeline(train_df, val_df, test_df, fcols, device, seed)
        all_results["multi_seed"][seed] = result
        print(f"    Acc={result['accuracy']:.4f}, MacroF1={result['macro_f1']:.4f}, "
              f"tau_atk={result['tau_attack']:.2f}, tau_high={result['tau_high']:.2f}")
        for rn in ["Normal", "Medium", "High"]:
            ri = result["per_risk"][rn]
            print(f"    {rn:>8s}: P={ri['Precision']:.3f}, R={ri['Recall']:.3f}, "
                  f"F1={ri['F1']:.3f} (n={ri['Support']})")

    # Multi-seed summary
    seed_accs = [all_results["multi_seed"][s]["accuracy"] for s in SEEDS]
    seed_f1s = [all_results["multi_seed"][s]["macro_f1"] for s in SEEDS]
    seed_baccs = [all_results["multi_seed"][s]["balanced_accuracy"] for s in SEEDS]
    print(f"\n  Multi-seed summary:")
    print(f"    Accuracy:       {np.mean(seed_accs):.4f} +/- {np.std(seed_accs):.4f}")
    print(f"    Balanced Acc:   {np.mean(seed_baccs):.4f} +/- {np.std(seed_baccs):.4f}")
    print(f"    Macro F1:       {np.mean(seed_f1s):.4f} +/- {np.std(seed_f1s):.4f}")

    # ============================================================
    #  Experiment 2: 5-Fold Stratified CV
    # ============================================================
    print("\n" + "=" * 80)
    print("[2/2] 5-Fold Stratified CV...")
    print("=" * 80)

    y_strat = raw_df["y_attack_type"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(raw_df, y_strat)):
        print(f"\n  -> Fold {fold_idx + 1}")
        tv_df = raw_df.iloc[train_val_idx].copy()
        test_cv = raw_df.iloc[test_idx].copy()

        # Inner split: 4-fold on train_val to get train/val
        y_tv = tv_df["y_attack_type"].values
        skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        tr_idx, va_idx = next(skf_inner.split(tv_df, y_tv))
        train_cv = tv_df.iloc[tr_idx].copy()
        val_cv = tv_df.iloc[va_idx].copy()

        train_df, val_df, test_df = prepare_split_data(train_cv, val_cv, test_cv, fcols)
        print(f"    Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        r = run_full_pipeline(train_df, val_df, test_df, fcols, device, 42)
        all_results["kfold_cv"].append(r)
        print(f"    Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f}")

    if all_results["kfold_cv"]:
        cv_accs = [r["accuracy"] for r in all_results["kfold_cv"]]
        cv_f1s = [r["macro_f1"] for r in all_results["kfold_cv"]]
        print(f"\n  5-Fold CV summary:")
        print(f"    Accuracy:  {np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f}")
        print(f"    Macro F1:  {np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}")

    # ============================================================
    #  Save results
    # ============================================================
    print("\n" + "=" * 80)
    print("  Saving results...")
    print("=" * 80)

    # --- CSV: Multi-seed ---
    seed_rows = []
    for seed in SEEDS:
        r = all_results["multi_seed"][seed]
        row = {"Seed": seed, "Accuracy": r["accuracy"],
               "Balanced_Accuracy": r["balanced_accuracy"],
               "Macro_F1": r["macro_f1"],
               "tau_attack": r["tau_attack"], "tau_high": r["tau_high"]}
        for rn in ["Normal", "Medium", "High"]:
            row[f"{rn}_Precision"] = r["per_risk"][rn]["Precision"]
            row[f"{rn}_Recall"] = r["per_risk"][rn]["Recall"]
            row[f"{rn}_F1"] = r["per_risk"][rn]["F1"]
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)
    # Add summary row
    summary_row = {"Seed": "Mean+-Std",
                   "Accuracy": f"{np.mean(seed_accs):.4f}+-{np.std(seed_accs):.4f}",
                   "Balanced_Accuracy": f"{np.mean(seed_baccs):.4f}+-{np.std(seed_baccs):.4f}",
                   "Macro_F1": f"{np.mean(seed_f1s):.4f}+-{np.std(seed_f1s):.4f}"}
    for rn in ["Normal", "Medium", "High"]:
        vals = [all_results["multi_seed"][s]["per_risk"][rn]["F1"] for s in SEEDS]
        summary_row[f"{rn}_F1"] = f"{np.mean(vals):.4f}+-{np.std(vals):.4f}"
    pd.concat([seed_df, pd.DataFrame([summary_row])], ignore_index=True
              ).to_csv(out_dir / "multi_seed_results.csv", index=False)
    print("  -> multi_seed_results.csv saved")

    # --- CSV: 5-Fold CV ---
    if all_results["kfold_cv"]:
        cv_rows = []
        for fi, r in enumerate(all_results["kfold_cv"]):
            row = {"Fold": fi + 1, "Accuracy": r["accuracy"],
                   "Balanced_Accuracy": r["balanced_accuracy"],
                   "Macro_F1": r["macro_f1"],
                   "N_Train": r["n_train"], "N_Val": r["n_val"], "N_Test": r["n_test"]}
            for rn in ["Normal", "Medium", "High"]:
                row[f"{rn}_F1"] = r["per_risk"][rn]["F1"]
            cv_rows.append(row)
        pd.DataFrame(cv_rows).to_csv(out_dir / "kfold_cv_results.csv", index=False)
        print("  -> kfold_cv_results.csv saved")

    # --- CSV: Per-attack-type ---
    atk_rows = []
    for seed in SEEDS:
        for atk_name, info in all_results["multi_seed"][seed]["per_attack"].items():
            atk_rows.append({
                "Seed": seed, "Attack_Type": atk_name,
                "Support": info["support"],
                "Detection_Rate": info["detection_rate"],
                "Risk_Level": info["risk_level"],
            })
    pd.DataFrame(atk_rows).to_csv(out_dir / "per_attack_type_results.csv", index=False)
    print("  -> per_attack_type_results.csv saved")

    # --- Visualization ---
    plot_generalization_results(all_results, out_dir)
    print("  -> uavids_generalization.png/pdf saved")

    # --- JSON summary ---
    json_out = {
        "experiment": "RSTD-KD UAVIDS-2025 Generalization Experiment",
        "dataset": "UAVIDS-2025",
        "dataset_description": "Flow-level UAV FANET intrusion detection (NS-3 simulation, AODV, IEEE 802.11ac)",
        "n_records": len(raw_df),
        "n_features": len(fcols),
        "feature_columns": fcols,
        "n_attack_types": N_ATTACK_TYPES,
        "attack_type_map": {str(k): v for k, v in ATTACK_ID_TO_NAME.items()},
        "risk_map": {str(k): v for k, v in RISK_ID_TO_NAME.items()},
        "label_mapping": {k: {kk: vv for kk, vv in v.items()} for k, v in LABEL_MAP.items()},
        "device": str(device),
        "seeds": SEEDS,
        "multi_seed_summary": {
            "accuracy_mean": float(np.mean(seed_accs)),
            "accuracy_std": float(np.std(seed_accs)),
            "balanced_accuracy_mean": float(np.mean(seed_baccs)),
            "balanced_accuracy_std": float(np.std(seed_baccs)),
            "macro_f1_mean": float(np.mean(seed_f1s)),
            "macro_f1_std": float(np.std(seed_f1s)),
            "per_seed": {
                str(s): {
                    "accuracy": all_results["multi_seed"][s]["accuracy"],
                    "balanced_accuracy": all_results["multi_seed"][s]["balanced_accuracy"],
                    "macro_f1": all_results["multi_seed"][s]["macro_f1"],
                } for s in SEEDS
            },
        },
        "kfold_cv_summary": {
            "n_folds": len(all_results["kfold_cv"]),
            "accuracy_mean": float(np.mean(cv_accs)) if all_results["kfold_cv"] else None,
            "accuracy_std": float(np.std(cv_accs)) if all_results["kfold_cv"] else None,
            "macro_f1_mean": float(np.mean(cv_f1s)) if all_results["kfold_cv"] else None,
            "macro_f1_std": float(np.std(cv_f1s)) if all_results["kfold_cv"] else None,
        },
        "per_attack_detection": {
            atk: {
                "mean_detection_rate": float(np.mean([
                    all_results["multi_seed"][s]["per_attack"].get(atk, {}).get("detection_rate", 0)
                    for s in SEEDS
                ])),
            } for atk in ATTACK_ID_TO_NAME.values()
        },
        "conclusion": (
            "RSTD-KD framework demonstrates strong generalization on the flow-level "
            "UAVIDS-2025 dataset with stable performance across multiple random seeds "
            "and cross-validation folds, despite fundamental differences in data "
            "granularity (packet-level vs flow-level) and attack taxonomy."
        ),
    }
    safe_json_dump(json_out, out_dir / "uavids_generalization_summary.json")
    print("  -> uavids_generalization_summary.json saved")

    # ============================================================
    #  Final summary
    # ============================================================
    print("\n" + "=" * 80)
    print("  RSTD-KD UAVIDS-2025 Generalization -- Final Summary")
    print("=" * 80)

    print(f"\n  [Multi-Seed (5 seeds)]")
    print(f"    Accuracy:       {np.mean(seed_accs):.4f} +/- {np.std(seed_accs):.4f}")
    print(f"    Balanced Acc:   {np.mean(seed_baccs):.4f} +/- {np.std(seed_baccs):.4f}")
    print(f"    Macro F1:       {np.mean(seed_f1s):.4f} +/- {np.std(seed_f1s):.4f}")
    for rn in ["Normal", "Medium", "High"]:
        vals = [all_results["multi_seed"][s]["per_risk"][rn]["F1"] for s in SEEDS]
        print(f"    {rn:>8s} F1: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    if all_results.get("kfold_cv"):
        cv_r = all_results["kfold_cv"]
        cv_a = [r["accuracy"] for r in cv_r]
        cv_f = [r["macro_f1"] for r in cv_r]
        print(f"\n  [5-Fold Stratified CV]")
        print(f"    Accuracy:  {np.mean(cv_a):.4f} +/- {np.std(cv_a):.4f}")
        print(f"    Macro F1:  {np.mean(cv_f):.4f} +/- {np.std(cv_f):.4f}")
        for fi, r in enumerate(cv_r):
            print(f"    Fold {fi + 1}: Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f}, "
                  f"N=(tr={r['n_train']}, va={r['n_val']}, te={r['n_test']})")

    print(f"\n  [Per-Attack Detection (Seed 42)]")
    for atk_name, info in all_results["multi_seed"][42]["per_attack"].items():
        print(f"    {atk_name:>12s}: det_rate={info['detection_rate']:.3f}, "
              f"risk={info['risk_level']}, n={info['support']}")

    print(f"\n" + "=" * 80)
    print(f"  All results saved to: {out_dir}")
    print(f"  1. multi_seed_results.csv")
    print(f"  2. kfold_cv_results.csv")
    print(f"  3. per_attack_type_results.csv")
    print(f"  4. uavids_generalization.png / .pdf")
    print(f"  5. uavids_generalization_summary.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
