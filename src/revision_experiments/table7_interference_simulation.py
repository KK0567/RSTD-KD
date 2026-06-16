#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSTD-KD Physical-Layer Interference Simulation (Reviewer 1 #5)
Physical-Layer Interference Simulation

Reviewer comment: The entire study is based on offline dataset simulation, ignoring
real-time channel fading, environmental EMI noise, dynamic UAV topology changes, etc.

Experiment Design - Four dimensions simulating physical-layer interference in real deployment:
  1. Channel Fading           -> Feature amplitude attenuation alpha in [1.0, 0.5]
  2. EMI Noise                -> Additive white Gaussian noise SNR in [20dB, 0dB]
  3. Packet Loss              -> Random feature dropout rate in [0%, 30%]
  4. Packet Reorder           -> Feature-level distortion rate in [0%, 30%]

Principles:
  - Model trained on clean data, interference applied only on test set
  - Interference applied in StandardScaler-normalized feature space
  - Each interference level repeated N_REPEATS times, report mean and std
  - Channel fading: multiply by attenuation coefficient alpha, simulating Rayleigh/Rician fading
  - EMI noise: add N(0, sigma^2) noise, sigma^2=1/SNR_linear, simulating environmental EMI
  - Packet loss: randomly zero out features, simulating information loss from packet drops
  - Packet reorder: randomly replace with other sample values, simulating statistical bias from reordering

Usage:
  python physical_interference_simulation.py
"""

from __future__ import annotations

import json, os, random, re, warnings
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
    precision_recall_fscore_support, accuracy_score,
    balanced_accuracy_score, f1_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
#  Path Configuration
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_CSV = str(PROJECT_ROOT / "data" / "ECU-IoFT-Dataset.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "table7_interference")

# ================================================================
#  Global Hyperparameters
# ================================================================
SEED = 42
WINDOW_SIZE = 32
N_REPEATS = 5  # repeats per interference level

# Interference dimension gradients
FADING_ALPHAS  = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]          # channel attenuation (incl. thermal noise)
NOISE_SNR_DB   = [20.0, 15.0, 10.0, 5.0, 3.0, 0.0]       # SNR (dB)
LOSS_RATES     = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70] # packet loss rate (extended to 70%)
REORDER_RATES  = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70] # reorder rate (extended to 70%)

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
STUDENT_HIDDEN1, STUDENT_HIDDEN2 = 128, 64
STUDENT_DROPOUT1, STUDENT_DROPOUT2 = 0.12, 0.08

# Label Mapping
ATTACK_TYPE_MAP = {
    "No Attack": 0, "Wifi Deauthentication Attack": 1,
    "WPA2-PSK WIFI Cracking Attack": 2, "TELLO API Exploit": 3,
}
ATTACK_ID_TO_NAME = {0: "No Attack", 1: "WiFi Deauth", 2: "WPA2 Crack", 3: "TELLO API"}
ATTACK_ID_TO_RISK = {0: "normal", 1: "medium", 2: "high", 3: "high"}
RISK_MAP = {
    "No Attack": (0, "normal"), "Wifi Deauthentication Attack": (1, "medium"),
    "WPA2-PSK WIFI Cracking Attack": (2, "high"), "TELLO API Exploit": (2, "high"),
}
RISK_ID_TO_NAME = {0: "Normal", 1: "Medium", 2: "High"}

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
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def safe_json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def dominant_attack_label(labels):
    vc = labels.value_counts()
    if vc.empty: return "No Attack"
    if len(vc) == 1: return str(vc.index[0])
    items = sorted(vc.items(), key=lambda kv: (kv[1], kv[0] != "No Attack"), reverse=True)
    return str(items[0][0])


def ratio_bool(mask): return float(mask.mean()) if len(mask) > 0 else 0.0
def q(arr, v): return float(np.quantile(arr, v)) if arr.size > 0 else 0.0
def calc_entropy(values):
    vc = values.value_counts(normalize=True)
    if vc.empty: return 0.0
    p = vc.values.astype(np.float64)
    return float(-(p * np.log2(np.clip(p, 1e-12, 1.0))).sum())

def normalize_string(s): return s.fillna("").astype(str).str.strip()
def parse_time(s): return pd.to_datetime(s, errors="coerce", dayfirst=True)


# ================================================================
#  Data Construction (reusing feature engineering from stability_experiments.py)
# ================================================================
def build_features_one_window(sub, window_id):
    lengths = sub["Length"].astype(float).values
    info = normalize_string(sub["Info"])
    proto = normalize_string(sub["Protocol"])
    src = normalize_string(sub["Source"])
    dst = normalize_string(sub["Destination"])
    atk = normalize_string(sub["Type of Attack"])
    typ = normalize_string(sub["Type"])
    dt = sub["_dt"]

    dominant = dominant_attack_label(atk)
    y_attack_type = ATTACK_TYPE_MAP[dominant]
    y_bin = int(dominant != "No Attack")
    y_risk, y_risk_name = RISK_MAP[dominant]

    info_len = info.str.len().values.astype(float)
    t_ns = dt.astype("int64").to_numpy(dtype=np.int64)
    t_sec = t_ns.astype(np.float64) / 1e9
    ia = np.clip(np.diff(t_sec), 0.0, None) if len(t_sec) > 1 else np.array([], dtype=np.float64)
    dur = float((dt.iloc[-1] - dt.iloc[0]).total_seconds())
    pr = float(len(sub) / max(dur, 1e-6))
    if len(lengths) > 1:
        ld = np.diff(lengths)
        xi = np.arange(len(lengths), dtype=np.float64)
        ls = float(np.polyfit(xi, lengths.astype(np.float64), 1)[0])
    else:
        ld = np.array([], dtype=np.float64); ls = 0.0

    sne = src[src != ""].dropna(); dne = dst[dst != ""].dropna(); pne = proto[proto != ""].dropna()

    feat = {
        "window_id": int(window_id), "row_id": int(window_id),
        "packet_id_start": int(sub.index[0]), "packet_id_end": int(sub.index[-1]),
        "time_start": str(dt.iloc[0]), "time_end": str(dt.iloc[-1]),
        "window_packet_count": int(len(sub)), "duration_sec": dur, "packet_rate": pr,
        "inter_arrival_mean": float(ia.mean()) if ia.size else 0.0,
        "inter_arrival_std": float(ia.std(ddof=0)) if ia.size else 0.0,
        "inter_arrival_min": float(ia.min()) if ia.size else 0.0,
        "inter_arrival_max": float(ia.max()) if ia.size else 0.0,
        "inter_arrival_q25": q(ia, 0.25), "inter_arrival_q50": q(ia, 0.50), "inter_arrival_q75": q(ia, 0.75),
        "length_mean": float(lengths.mean()), "length_std": float(lengths.std(ddof=0)),
        "length_min": float(lengths.min()), "length_max": float(lengths.max()),
        "length_q25": q(lengths, 0.25), "length_q50": q(lengths, 0.50), "length_q75": q(lengths, 0.75),
        "length_first": float(lengths[0]) if lengths.size else 0.0,
        "length_last": float(lengths[-1]) if lengths.size else 0.0,
        "length_slope": ls,
        "length_diff_mean": float(ld.mean()) if ld.size else 0.0,
        "length_diff_std": float(ld.std(ddof=0)) if ld.size else 0.0,
        "length_absdiff_mean": float(np.abs(ld).mean()) if ld.size else 0.0,
        "length_absdiff_max": float(np.abs(ld).max()) if ld.size else 0.0,
        "info_len_mean": float(info_len.mean()), "info_len_std": float(info_len.std(ddof=0)),
        "info_len_max": float(info_len.max()),
        "src_unique": int(sne.nunique()), "dst_unique": int(dne.nunique()),
        "proto_unique": int(pne.nunique()),
        "src_entropy": calc_entropy(sne), "dst_entropy": calc_entropy(dne), "proto_entropy": calc_entropy(pne),
        "missing_source_ratio": ratio_bool(src.eq("")),
        "broadcast_dst_ratio": ratio_bool(dst.str.contains("ff:ff:ff:ff:ff:ff", case=False, regex=False)),
        "ra_marker_ratio": ratio_bool(dst.str.contains(r"\(RA\)", case=False, regex=True)),
        "bssid_marker_ratio": ratio_bool(
            src.str.contains(r"\(BSSID\)", case=False, regex=True) | dst.str.contains(r"\(BSSID\)", case=False, regex=True)),
        "ip_endpoint_ratio": ratio_bool(
            src.str.contains(r"^\d+\.\d+\.\d+\.\d+$", regex=True) | dst.str.contains(r"^\d+\.\d+\.\d+\.\d+$", regex=True)),
        "attack_packet_count": int((typ == "Attack").sum()),
        "attack_packet_ratio": ratio_bool(typ == "Attack"),
        "y_bin": int(y_bin), "y_attack_type": int(y_attack_type), "y_attack_type_name": dominant,
        "y_risk": int(y_risk), "y_risk_name": y_risk_name,
        "attack_scenario_meta": dominant_attack_label(normalize_string(sub["Attack Scenario"])),
        "packet_attack_type_mode_meta": dominant, "session_id": "", "split": "",
    }
    for p in PROTO_VALUES:
        cnt = int((proto == p).sum())
        key = re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_").lower()
        feat[f"proto_{key}_count"] = cnt; feat[f"proto_{key}_ratio"] = float(cnt / len(sub))
    pseq = proto.tolist(); nt = max(1, len(pseq) - 1)
    for a, b in [("802.11","802.11"),("802.11","EAPOL"),("EAPOL","EAPOL"),("UDP","UDP"),("802.11","UDP"),("UDP","ICMP")]:
        tc = sum(1 for i in range(len(pseq)-1) if pseq[i]==a and pseq[i+1]==b)
        ka = re.sub(r"[^a-zA-Z0-9]+","_",a).strip("_").lower()
        kb = re.sub(r"[^a-zA-Z0-9]+","_",b).strip("_").lower()
        feat[f"proto_trans_{ka}_to_{kb}_count"] = int(tc)
        feat[f"proto_trans_{ka}_to_{kb}_ratio"] = float(tc / nt)
    il = info.str.lower()
    for tok in INFO_TOKENS:
        cnt = int(il.str.contains(tok, regex=False).sum())
        feat[f"info_tok_{tok}_count"] = cnt; feat[f"info_tok_{tok}_ratio"] = float(cnt / len(sub))
    return feat


def build_windows(df, ws, ss):
    rows, wid = [], 0
    for s in range(0, len(df) - ws + 1, ss):
        rows.append(build_features_one_window(df.iloc[s:s+ws], wid)); wid += 1
    return pd.DataFrame(rows)


def get_feature_cols(win_df):
    drop_exact = {
        "window_id","row_id","packet_id_start","packet_id_end","time_start","time_end",
        "y_bin","y_attack_type","y_attack_type_name","y_risk","y_risk_name",
        "attack_scenario_meta","packet_attack_type_mode_meta","session_id","split",
    } | ALWAYS_DROP_EXACT
    fcols = [c for c in win_df.columns if c not in drop_exact
             and not any(c.startswith(p) for p in ALWAYS_DROP_PREFIX)
             and not any(k in c.lower() for k in ALWAYS_DROP_CONTAINS)]
    return [c for c in fcols if pd.api.types.is_numeric_dtype(win_df[c])]


def split_random(win_df, seed, train_r=0.7, val_r=0.15):
    out = win_df.copy()
    train_idxs, val_idxs, test_idxs = [], [], []
    rng = np.random.RandomState(seed)
    for aid in sorted(out["y_attack_type"].unique()):
        idxs = out.loc[out["y_attack_type"] == aid].index.tolist()
        rng.shuffle(idxs)
        n = len(idxs)
        nt = max(1, int(round(n * train_r)))
        nv = max(1, int(round(n * val_r)))
        nte = max(1, n - nt - nv)
        train_idxs.extend(idxs[:nt])
        val_idxs.extend(idxs[nt:nt+nv])
        test_idxs.extend(idxs[nt+nv:])
    return out.loc[train_idxs], out.loc[val_idxs], out.loc[test_idxs]


def std_split_df(df):
    o = df.copy()
    o["row_id"] = o["window_id"].astype(np.int64)
    o["binary_label"] = o["y_bin"].astype(np.int64)
    o["label"] = o["y_attack_type_name"].astype(str)
    return o


def prepare_split_data(train_raw, val_raw, test_raw, fcols):
    train_df = std_split_df(train_raw)
    val_df = std_split_df(val_raw)
    test_df = std_split_df(test_raw)
    med = train_df[fcols].median(numeric_only=True).to_dict()
    for d in [train_df, val_df, test_df]:
        d[fcols] = d[fcols].fillna(med)
    scaler = StandardScaler()
    scaler.fit(train_df[fcols].values.astype(np.float32))
    return train_df, val_df, test_df, scaler


# ================================================================
#  Teacher / Student
# ================================================================
def bsw(y):
    y = np.asarray(y, np.int64); cls, cnt = np.unique(y, return_counts=True); tot = cnt.sum()
    wm = {int(c): float(tot/(len(cls)*ci)) for c, ci in zip(cls, cnt)}
    return np.array([wm[int(v)] for v in y], np.float32)


def train_teacher(train_df, fcols):
    Xb = train_df[fcols].values.astype(np.float32)
    yb = train_df["binary_label"].values.astype(np.int64)
    wb = bsw(yb)
    bmodels = []
    for i in range(TEACHER_ENSEMBLE_SIZE):
        rng = np.random.RandomState(42 + 1009*(i+1))
        parts = []
        for c in np.unique(yb):
            idx = np.where(yb==c)[0]; n = max(1, int(round(len(idx)*TEACHER_BOOTSTRAP_FRAC)))
            parts.append(rng.choice(idx, size=n, replace=True))
        bi = np.concatenate(parts); rng.shuffle(bi)
        m = HistGradientBoostingClassifier(loss="log_loss", learning_rate=TEACHER_LR, max_iter=TEACHER_MAX_ITER,
            max_leaf_nodes=TEACHER_MAX_LEAF, min_samples_leaf=TEACHER_MIN_LEAF_BIN, l2_regularization=TEACHER_L2,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=42+i)
        m.fit(Xb[bi], yb[bi], sample_weight=wb[bi]); bmodels.append(m)

    Xr = train_df[fcols].values.astype(np.float32)
    yr = train_df["y_risk"].values.astype(np.int64)
    wr = bsw(yr)
    es = np.min(np.bincount(yr)) >= 2
    rmodel = HistGradientBoostingClassifier(loss="log_loss", learning_rate=TEACHER_RISK_LR, max_iter=TEACHER_RISK_MAX_ITER,
        max_leaf_nodes=TEACHER_RISK_MAX_LEAF, min_samples_leaf=TEACHER_RISK_MIN_LEAF, l2_regularization=TEACHER_RISK_L2,
        early_stopping=es, random_state=42+37)
    rmodel.fit(Xr, yr, sample_weight=wr)

    sub = train_df.loc[train_df["binary_label"]==1]
    Xa = sub[fcols].values.astype(np.float32); ya = sub["y_attack_type"].values.astype(np.int64); wa = bsw(ya)
    es_a = np.min(np.bincount(ya)) >= 2 if len(ya) > 0 else False
    amodel = HistGradientBoostingClassifier(loss="log_loss", learning_rate=TEACHER_ATK_LR, max_iter=TEACHER_ATK_MAX_ITER,
        max_leaf_nodes=TEACHER_ATK_MAX_LEAF, min_samples_leaf=TEACHER_ATK_MIN_LEAF, l2_regularization=TEACHER_ATK_L2,
        early_stopping=es_a, random_state=42+17)
    amodel.fit(Xa, ya, sample_weight=wa)
    return bmodels, rmodel, amodel


def pred_bin_ens(models, X):
    return np.mean(np.stack([m.predict_proba(X)[:, list(m.classes_).index(1)] for m in models]), axis=0)


def pred_fixed(model, X, classes):
    raw = model.predict_proba(X); out = np.zeros((X.shape[0], len(classes)), np.float64); mc = list(model.classes_)
    for j, c in enumerate(classes):
        if c in mc: out[:, j] = raw[:, mc.index(c)]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def make_teacher_preds(bmodels, rmodel, amodel, df, fcols):
    X = df[fcols].values.astype(np.float32)
    bp = pred_bin_ens(bmodels, X)
    rp = pred_fixed(rmodel, X, [0, 1, 2])
    ap = pred_fixed(amodel, X, [1, 2, 3])
    return pd.DataFrame({
        "row_id": df["row_id"].values.astype(np.int64),
        "teacher_prob_binary": bp.astype(np.float32),
        "teacher_risk_prob_0": rp[:,0].astype(np.float32), "teacher_risk_prob_1": rp[:,1].astype(np.float32),
        "teacher_risk_prob_2": rp[:,2].astype(np.float32),
        "teacher_attack_prob_1": ap[:,0].astype(np.float32), "teacher_attack_prob_2": ap[:,1].astype(np.float32),
        "teacher_attack_prob_3": ap[:,2].astype(np.float32),
    })


class StudentModel(nn.Module):
    def __init__(self, d, h1=128, h2=64, d1=0.12, d2=0.08):
        super().__init__()
        self.backbone = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,h1), nn.GELU(), nn.Dropout(d1),
                                       nn.Linear(h1,h2), nn.GELU(), nn.Dropout(d2))
        self.hb = nn.Linear(h2,1); self.ha = nn.Linear(h2,3); self.hr = nn.Linear(h2,3)
    def forward(self, x):
        z = self.backbone(x); return self.hb(z).squeeze(1), self.ha(z), self.hr(z)


class DS(Dataset):
    def __init__(self, df, fcols, scaler, tdf):
        df = df.copy()
        mc = ["row_id","teacher_prob_binary","teacher_attack_prob_1","teacher_attack_prob_2","teacher_attack_prob_3",
              "teacher_risk_prob_0","teacher_risk_prob_1","teacher_risk_prob_2"]
        df = df.merge(tdf[mc], on="row_id", how="left")
        X = scaler.transform(df[fcols].values.astype(np.float32))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.yb = torch.tensor(df["binary_label"].values.astype(np.float32))
        ao = df["y_attack_type"].values.astype(np.int64)
        ai = np.where(df["binary_label"].values.astype(np.int64)==1, ao-1, 0).astype(np.int64)
        self.ya = torch.tensor(ai, dtype=torch.long)
        self.yr = torch.tensor(df["y_risk"].values.astype(np.int64), dtype=torch.long)
        self.tb = torch.tensor(df["teacher_prob_binary"].values.astype(np.float32))
        self.ta = torch.tensor(df[["teacher_attack_prob_1","teacher_attack_prob_2","teacher_attack_prob_3"]].values.astype(np.float32))
        self.tr = torch.tensor(df[["teacher_risk_prob_0","teacher_risk_prob_1","teacher_risk_prob_2"]].values.astype(np.float32))
        self.am = torch.tensor(df["binary_label"].values.astype(np.float32))
        yb_np = df["binary_label"].values.astype(np.int64)
        self.wb = torch.tensor(bsw(yb_np))
        self.wrk = torch.tensor(bsw(df["y_risk"].values.astype(np.int64)))
        abn = df.loc[df["binary_label"]==1, "y_attack_type"].values.astype(np.int64)-1
        if len(abn):
            cls, cnt = np.unique(abn, return_counts=True); tot = cnt.sum()
            wm = {int(c): float(tot/(len(cls)*ci)) for c, ci in zip(cls, cnt)}
        else: wm = {0:1.,1:1.,2:1.}
        aw = np.ones(len(df), np.float32)
        for i,(ib,aid) in enumerate(zip(yb_np, df["y_attack_type"].values.astype(np.int64)-1)):
            if ib==1: aw[i] = wm[int(aid)]
        self.wa = torch.tensor(aw)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return {"x":self.X[i],"yb":self.yb[i],"ya":self.ya[i],"yr":self.yr[i],
                "tb":self.tb[i],"ta":self.ta[i],"tr":self.tr[i],"am":self.am[i],
                "wb":self.wb[i],"wa":self.wa[i],"wrk":self.wrk[i]}


def train_student(train_df, val_df, fcols, scaler, t_train, t_val, device):
    tds = DS(train_df, fcols, scaler, t_train); vds = DS(val_df, fcols, scaler, t_val)
    tl = DataLoader(tds, STUDENT_BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(vds, STUDENT_BATCH, shuffle=False, num_workers=0)
    model = StudentModel(len(fcols), STUDENT_HIDDEN1, STUDENT_HIDDEN2, STUDENT_DROPOUT1, STUDENT_DROPOUT2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=STUDENT_LR, weight_decay=STUDENT_WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=6, min_lr=1e-5)
    ab, bb, aa, ba, ar, br = 0.8, 0.1, 0.5, 0.1, 1.2, 0.25
    bs, bsc, w = None, -1., 0
    for ep in range(1, STUDENT_EPOCHS+1):
        model.train()
        for b in tl:
            x = b["x"].to(device)
            lb,la,lr = model(x)
            lhb = (F.binary_cross_entropy_with_logits(lb, b["yb"].to(device), reduction="none")*b["wb"].to(device)).sum()/b["wb"].to(device).sum().clamp_min(1e-8)
            lkb = ((torch.sigmoid(lb/STUDENT_TEMP)-b["tb"].to(device))**2*b["wb"].to(device)).sum()/b["wb"].to(device).sum().clamp_min(1e-8)
            ce = F.cross_entropy(la, b["ya"].to(device), reduction="none")
            lha = (ce*b["wa"].to(device)*b["am"].to(device)).sum()/(b["am"].to(device)*b["wa"].to(device)).sum().clamp_min(1e-8)
            lp = F.log_softmax(la/STUDENT_TEMP,1); qq = torch.clamp(b["ta"].to(device),1e-6,1.0); qq = qq/qq.sum(1,keepdim=True).clamp_min(1e-8)
            lka = (F.kl_div(lp,qq,reduction="none").sum(1)*b["am"].to(device)).sum()/b["am"].to(device).sum().clamp_min(1e-8)*(STUDENT_TEMP**2)
            lhr = (F.cross_entropy(lr,b["yr"].to(device),reduction="none")*b["wrk"].to(device)).sum()/b["wrk"].to(device).sum().clamp_min(1e-8)
            lp2 = F.log_softmax(lr/STUDENT_TEMP,1); q2 = torch.clamp(b["tr"].to(device),1e-6,1.0); q2 = q2/q2.sum(1,keepdim=True).clamp_min(1e-8)
            lkr = F.kl_div(lp2,q2,reduction="batchmean")*(STUDENT_TEMP**2)
            loss = ab*lhb + bb*lkb + aa*lha + ba*lka + ar*lhr + br*lkr
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad():
            pb,pa,pr = [],[],[]
            for b in vl:
                x = b["x"].to(device); lb,la,lr = model(x)
                pb.append(torch.sigmoid(lb).cpu().numpy()); pa.append(torch.softmax(la,1).cpu().numpy()); pr.append(torch.softmax(lr,1).cpu().numpy())
            pb,pa,pr = np.concatenate(pb), np.concatenate(pa), np.concatenate(pr)
        pmed, phigh = pr[:,1], pr[:,2]
        pside = np.where(phigh>=0.5, 2, np.where(pmed>=phigh, 1, 2))
        pred = np.where(pb<0.5, 0, pside)
        yr_v = vds.yr.numpy().astype(np.int64)
        _,_,f1,_ = precision_recall_fscore_support(yr_v,pred,labels=[0,1,2],average="macro",zero_division=0)
        _,hr,_,_ = precision_recall_fscore_support(yr_v,pred,labels=[2],average="macro",zero_division=0)
        sc = f1 + 0.1*hr; sch.step(sc)
        if sc > bsc: bsc, bs, w = sc, deepcopy(model.state_dict()), 0
        else: w += 1
        if w >= STUDENT_PATIENCE: break
    model.load_state_dict(bs); return model


# ================================================================
#  Inference and Decision
# ================================================================
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval(); pb,pa,pr = [],[],[]
    for b in loader:
        x = b["x"].to(device); lb,la,lr = model(x)
        pb.append(torch.sigmoid(lb).cpu().numpy()); pa.append(torch.softmax(la,1).cpu().numpy()); pr.append(torch.softmax(lr,1).cpu().numpy())
    return np.concatenate(pb), np.concatenate(pa), np.concatenate(pr)


def risk_decision(prob_bin, risk_prob, tau_attack=0.5, tau_high=0.5):
    p_med, p_high = risk_prob[:, 1], risk_prob[:, 2]
    pred_atk = np.where(p_high >= tau_high, 2, np.where(p_med >= p_high, 1, 2))
    return np.where(prob_bin < tau_attack, 0, pred_atk).astype(np.int64)


# ================================================================
#  Platt Calibration
# ================================================================
def logit_transform(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_platt_calibrator(prob_val, y_val):
    if len(np.unique(y_val)) < 2: return None
    x = logit_transform(prob_val).reshape(-1, 1)
    m = LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000)
    m.fit(x, y_val.astype(int))
    return m


def apply_platt(platt_model, prob):
    if platt_model is None: return np.clip(prob, 1e-6, 1 - 1e-6)
    x = logit_transform(prob).reshape(-1, 1)
    return np.clip(platt_model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)


class PlattRiskCalibrator:
    def __init__(self): self.models = []
    def fit(self, risk_prob_val, y_risk_val):
        self.models = []
        for c in range(risk_prob_val.shape[1]):
            y_c = (y_risk_val == c).astype(int)
            if len(np.unique(y_c)) < 2: self.models.append(None); continue
            x = logit_transform(risk_prob_val[:, c]).reshape(-1, 1)
            m = LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000)
            m.fit(x, y_c); self.models.append(m)
        return self
    def predict(self, risk_prob):
        out = np.zeros_like(risk_prob)
        for c, m in enumerate(self.models):
            if m is None: out[:, c] = risk_prob[:, c]
            else: out[:, c] = m.predict_proba(logit_transform(risk_prob[:, c]).reshape(-1, 1))[:, 1]
        return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def search_thresholds(y_true, prob_bin, risk_prob):
    grid = np.arange(0.05, 0.96, 0.02)
    best = None
    for ta in grid:
        for th in grid:
            pred = risk_decision(prob_bin, risk_prob, float(ta), float(th))
            _,_,f1,_ = precision_recall_fscore_support(y_true, pred, labels=[0,1,2], average="macro", zero_division=0)
            _,hr,_,_ = precision_recall_fscore_support(y_true, pred, labels=[2], average="macro", zero_division=0)
            sc = f1 + 0.1*hr
            if best is None or sc > best["score"]:
                best = {"tau_attack": float(ta), "tau_high": float(th), "score": float(sc)}
    return best


# ================================================================
#  Physical-Layer Interference Functions
# ================================================================
def apply_channel_fading(X_scaled, alpha, rng):
    """Channel fading: frequency-selective fading + thermal noise enhancement.

    Physical principle: wireless channel fading is frequency-selective; different frequency
    bands experience different fading depths. In feature space, we divide 92 features into
    5 groups (timing/length/info/network/proto), each with independent random attenuation

    (simulating frequency selectivity), while fading increases receiver thermal noise.
    Attenuation model:
      - Per-group attenuation: group_alpha_i ~ U(max(0, 2*alpha-1), 1)
      - Per-feature perturbation: feature_alpha ~ U(0.8, 1.2) * group_alpha

      - Thermal noise from fading: noise_std = (1 - alpha) * 0.5
    """
    n_samples, n_features = X_scaled.shape

    # This ensures LayerNorm cannot fully compensate (different group attenuations),
    # and noise increases with fading depth.
    # Define feature groups (based on feature construction order)
    group_boundaries = [0, 7, 20, 23, 29, n_features]

    result = X_scaled.copy()
    low = max(0.0, 2.0 * alpha - 1.0)

    for g in range(len(group_boundaries) - 1):
        g_start = group_boundaries[g]
        g_end = group_boundaries[g + 1]
        if g_start >= n_features:
            break

    # Group 0: timing (0-6), Group 1: length (7-19), Group 2: info (20-22)
        group_alpha = rng.uniform(low, 1.0)

        for j in range(g_start, min(g_end, n_features)):
    # Group 3: network (23-28), Group 4: protocol+other (29+)
            feature_perturb = rng.uniform(0.8, 1.2)
            total_alpha = min(1.0, group_alpha * feature_perturb)
            result[:, j] *= total_alpha

        # Group-level attenuation (frequency-selective)
    if alpha < 1.0:
        noise_std = (1.0 - alpha) * 0.5
        thermal_noise = rng.normal(0, noise_std, X_scaled.shape).astype(np.float32)
        result += thermal_noise

    return result


def apply_emi_noise(X_scaled, snr_db, rng):
            # Intra-group perturbation
    # Thermal noise enhancement from fading (deeper fading -> more noise)
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1.0 / snr_linear)
    noise = rng.normal(0, noise_std, X_scaled.shape).astype(np.float32)
    return X_scaled + noise


def apply_packet_loss(X_scaled, loss_rate, rng):
    """EMI noise: add white Gaussian noise, power determined by SNR(dB).
    sigma^2 = 1 / SNR_linear, signal power assumed = 1 (after normalization)."""
    """Packet loss: randomly zero out loss_rate fraction of features.
    return X_scaled * mask.astype(np.float32)


def apply_packet_reorder(X_scaled, reorder_rate, rng):
    """Packet reorder: randomly replace reorder_rate fraction of features with other sample values.
    Simulates window-level statistical bias from out-of-order packet arrival."""
    n_samples, n_features = X_scaled.shape
    result = X_scaled.copy()
    feat_mask = rng.random((n_samples, n_features)) < reorder_rate
    n_replace = int(feat_mask.sum())
    if n_replace > 0:
        random_rows = rng.randint(0, n_samples, size=n_replace)
        rows, cols = np.where(feat_mask)
        result[rows, cols] = X_scaled[random_rows, cols]
    return result


# ================================================================
#  Interference Evaluation (using pre-scaled features to avoid repeated scaler calls)
# ================================================================
class PreScaledDS(Dataset):
    """Directly use pre-scaled (possibly interfered) features without calling scaler again."""
    def __init__(self, X_scaled, df, tdf):
        mc = ["row_id","teacher_prob_binary","teacher_attack_prob_1","teacher_attack_prob_2","teacher_attack_prob_3",
              "teacher_risk_prob_0","teacher_risk_prob_1","teacher_risk_prob_2"]
        merged = df[["row_id"]].copy().merge(tdf[mc], on="row_id", how="left")
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.yb = torch.tensor(df["binary_label"].values.astype(np.float32))
        ao = df["y_attack_type"].values.astype(np.int64)
        ai = np.where(df["binary_label"].values.astype(np.int64)==1, ao-1, 0).astype(np.int64)
        self.ya = torch.tensor(ai, dtype=torch.long)
        self.yr = torch.tensor(df["y_risk"].values.astype(np.int64), dtype=torch.long)
        self.tb = torch.tensor(merged["teacher_prob_binary"].values.astype(np.float32))
        self.ta = torch.tensor(merged[["teacher_attack_prob_1","teacher_attack_prob_2","teacher_attack_prob_3"]].values.astype(np.float32))
        self.tr = torch.tensor(merged[["teacher_risk_prob_0","teacher_risk_prob_1","teacher_risk_prob_2"]].values.astype(np.float32))
        self.am = torch.tensor(df["binary_label"].values.astype(np.float32))
        yb_np = df["binary_label"].values.astype(np.int64)
        self.wb = torch.tensor(bsw(yb_np))
        self.wrk = torch.tensor(bsw(df["y_risk"].values.astype(np.int64)))
        abn = df.loc[df["binary_label"]==1, "y_attack_type"].values.astype(np.int64)-1
        if len(abn):
            cls, cnt = np.unique(abn, return_counts=True); tot = cnt.sum()
            wm = {int(c): float(tot/(len(cls)*ci)) for c, ci in zip(cls, cnt)}
        else: wm = {0:1.,1:1.,2:1.}
        aw = np.ones(len(df), np.float32)
        for i,(ib,aid) in enumerate(zip(yb_np, df["y_attack_type"].values.astype(np.int64)-1)):
            if ib==1: aw[i] = wm[int(aid)]
        self.wa = torch.tensor(aw)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return {"x":self.X[i],"yb":self.yb[i],"ya":self.ya[i],"yr":self.yr[i],
                "tb":self.tb[i],"ta":self.ta[i],"tr":self.tr[i],"am":self.am[i],
                "wb":self.wb[i],"wa":self.wa[i],"wrk":self.wrk[i]}


def evaluate_with_interference(model, test_std_df, fcols, scaler, t_test,
                                y_risk_test, y_bin_test, tau_attack, tau_high,
                                platt_bin, platt_risk, device,
                                interference_fn=None, seed_offset=0):
    """Apply interference on test set and evaluate performance.

    Key: first scaler.transform to get normalized features, then apply physical interference,
    then use PreScaledDS to pass interfered normalized features directly (skip 2nd scaler call).
    """
    # Standardize
    X_scaled = scaler.transform(test_std_df[fcols].values.astype(np.float32))

    # Apply interference (in normalized feature space)
    if interference_fn is not None:
        rng = np.random.RandomState(SEED + seed_offset + 1000)
        X_final = interference_fn(X_scaled, rng)
    else:
        X_final = X_scaled

    # Use PreScaledDS to avoid repeated scaler calls
    test_ds = PreScaledDS(X_final, test_std_df, t_test)
    test_loader = DataLoader(test_ds, STUDENT_BATCH, shuffle=False, num_workers=0)
    test_bp, test_ap, test_rp = predict_all(model, test_loader, device)

    # Platt calibration
    test_rp_platt = platt_risk.predict(test_rp)
    test_bp_platt = apply_platt(platt_bin, test_bp)

    # Risk decision
    pred = risk_decision(test_bp_platt, test_rp_platt, tau_attack, tau_high)

    # Main metrics
    acc = float(accuracy_score(y_risk_test, pred))
    bacc = float(balanced_accuracy_score(y_risk_test, pred))
    macro_f1 = float(f1_score(y_risk_test, pred, labels=[0,1,2], average="macro", zero_division=0))

    # Per-risk F1
    p, r, f1, sup = precision_recall_fscore_support(y_risk_test, pred, labels=[0,1,2], zero_division=0)
    per_risk_f1 = {"Normal": float(f1[0]), "Medium": float(f1[1]), "High": float(f1[2])}
    high_recall = float(r[2])

    # Binary detection rate
    pred_bin = (test_bp_platt >= 0.5).astype(int)
    atk_mask = y_bin_test == 1
    det_rate = float(pred_bin[atk_mask].mean()) if atk_mask.sum() > 0 else 0.0

    return {
        "accuracy": acc, "balanced_accuracy": bacc, "macro_f1": macro_f1,
        "high_recall": high_recall, "detection_rate": det_rate,
        "normal_f1": per_risk_f1["Normal"], "medium_f1": per_risk_f1["Medium"], "high_f1": per_risk_f1["High"],
    }


# ================================================================
#  Visualization
# ================================================================
def plot_degradation_curves(all_results, baseline, out_dir):
    """Plot 4-dimension performance degradation curves (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    configs = [
        ("channel_fading", "Channel Fading (Amplitude Attenuation)", "Attenuation Factor (alpha)", FADING_ALPHAS),
        ("emi_noise", "Electromagnetic Noise (Gaussian)", "SNR (dB)", NOISE_SNR_DB),
        ("topology_loss", "Topology Change (Packet Loss)", "Packet Loss Rate", LOSS_RATES),
        ("topology_reorder", "Topology Change (Packet Reordering)", "Reordering Rate", REORDER_RATES),
    ]

    for idx, (key, title, xlabel, levels) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        data = all_results[key]

        accs = [data[l]["accuracy_mean"] for l in levels]
        f1s  = [data[l]["macro_f1_mean"] for l in levels]
        hrs  = [data[l]["high_recall_mean"] for l in levels]
        acc_std = [data[l]["accuracy_std"] for l in levels]
        f1_std  = [data[l]["macro_f1_std"] for l in levels]

        x = list(range(len(levels)))
        ax.errorbar(x, accs, yerr=acc_std, fmt="o-", color="#3498db", linewidth=2,
                    markersize=8, capsize=4, label=f"Accuracy (base={baseline['accuracy']:.3f})")
        ax.errorbar(x, f1s, yerr=f1_std, fmt="s-", color="#e74c3c", linewidth=2,
                    markersize=8, capsize=4, label=f"Macro F1 (base={baseline['macro_f1']:.3f})")
        ax.plot(x, hrs, "^-", color="#2ecc71", linewidth=2, markersize=8,
                label=f"High Recall (base={baseline['high_recall']:.3f})")

        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in levels], fontsize=8, rotation=30)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, frameon=False, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.80, 1.02)

        # Annotate degradation percentage
        if len(levels) > 1:
            last_acc = accs[-1]
            deg_pct = (baseline["accuracy"] - last_acc) / baseline["accuracy"] * 100
            ax.annotate(f"Degradation: -{deg_pct:.1f}%",
                        xy=(len(levels)-1, last_acc), xytext=(len(levels)*0.5, last_acc - 0.03),
                        fontsize=8, color="#3498db", alpha=0.8,
                        arrowprops=dict(arrowstyle="->", color="#3498db", alpha=0.5))

    fig.suptitle("RSTD-KD Physical-Layer Interference Simulation", fontsize=14, fontweight="bold", y=0.99)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(str(Path(out_dir) / "interference_degradation.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "interference_degradation.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_robustness_heatmap(all_results, baseline, out_dir):
    """Plot robustness heatmap."""
    fig, ax = plt.subplots(figsize=(13, 5))

    dims = [
        ("channel_fading", "Channel\nFading", FADING_ALPHAS),
        ("emi_noise", "EMI\nNoise", NOISE_SNR_DB),
        ("topology_loss", "Packet\nLoss", LOSS_RATES),
        ("topology_reorder", "Packet\nReorder", REORDER_RATES),
    ]

    n_levels = max(len(d[2]) for d in dims)
    heatmap_data = np.ones((4, n_levels))

    for i, (key, _, levels) in enumerate(dims):
        data = all_results[key]
        for j, l in enumerate(levels):
            heatmap_data[i, j] = data[l]["accuracy_mean"] / baseline["accuracy"]

    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0.88, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Relative Accuracy (normalized)", shrink=0.8)

    ax.set_yticks(range(4))
    ax.set_yticklabels([d[1] for d in dims], fontsize=11)

    # X-axis labels (normalized to 0-100% severity)
    x_labels = [f"L{j+1}" for j in range(n_levels)]
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Interference Level (L1 = clean, increasing severity ->)", fontsize=11)

    for i in range(4):
        for j in range(n_levels):
            val = heatmap_data[i, j]
            text_color = "white" if val < 0.94 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                    color=text_color, fontweight="bold")

    ax.set_title("RSTD-KD Robustness Heatmap (Relative Accuracy)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "robustness_heatmap.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "robustness_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_summary_radar(all_results, baseline, out_dir):
    """Plot multi-dimension robustness comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dims_labels = ["Channel Fading\n(alpha=0.5)", "EMI Noise\n(SNR=0dB)",
                   "Packet Loss\n(rate=30%)", "Packet Reorder\n(rate=30%)"]
    dims_keys = ["channel_fading", "emi_noise", "topology_loss", "topology_reorder"]
    max_levels = [FADING_ALPHAS[-1], NOISE_SNR_DB[-1], LOSS_RATES[-1], REORDER_RATES[-1]]

    metrics = ["accuracy", "macro_f1", "high_recall"]
    metric_labels = ["Accuracy", "Macro F1", "High Recall"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    x = np.arange(len(dims_labels))
    width = 0.25

    for mi, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = []
        for key, ll in zip(dims_keys, max_levels):
            base_val = baseline[metric]
            worst_val = all_results[key][ll][f"{metric}_mean"]
            retention = worst_val / base_val * 100 if base_val > 0 else 100
            vals.append(retention)
        bars = ax.bar(x + mi * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Performance Retention (%)", fontsize=12)
    ax.set_xlabel("Interference Type (at maximum severity)", fontsize=12)
    ax.set_title("RSTD-KD Robustness: Performance Retention at Maximum Interference",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(dims_labels, fontsize=10)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(85, 105)
    ax.axhline(y=100, color="gray", ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "robustness_comparison.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "robustness_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)


# ================================================================
#  Main Function
# ================================================================
def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Output: {out_dir}")
    set_seed(SEED)

    # ============================================================
    #  [1/7] Load data & build window features
    # ============================================================
    print("\n[1/7] Loading data & building window features...")
    raw_df = pd.read_csv(RAW_DATA_CSV)
    raw_df["_dt"] = parse_time(raw_df["Time"])
    raw_df = raw_df.reset_index(drop=True)
    print(f"  Raw packets: {len(raw_df)}")

    win_df = build_windows(raw_df, WINDOW_SIZE, WINDOW_SIZE)
    fcols = get_feature_cols(win_df)
    print(f"  Windows: {len(win_df)}, Features: {len(fcols)}")

    # ============================================================
    #  [2/7] Data split & standardization
    # ============================================================
    print("\n[2/7] Data splitting (seed=42, Random Stratified)...")
    train_raw, val_raw, test_raw = split_random(win_df, SEED)
    train_df, val_df, test_df, scaler = prepare_split_data(train_raw, val_raw, test_raw, fcols)
    print(f"  Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # ============================================================
    #  [3/7] Train Teacher + Student (clean data)
    # ============================================================
    print("\n[3/7] Training Teacher model...")
    bmodels, rmodel, amodel = train_teacher(train_df, fcols)

    print("  Generating Teacher predictions...")
    t_train = make_teacher_preds(bmodels, rmodel, amodel, train_df, fcols)
    t_val   = make_teacher_preds(bmodels, rmodel, amodel, val_df, fcols)
    t_test  = make_teacher_preds(bmodels, rmodel, amodel, test_df, fcols)

    print("  Training Student model (KD)...")
    model = train_student(train_df, val_df, fcols, scaler, t_train, t_val, device)

    # ============================================================
    #  [4/7] Clean baseline evaluation + Platt calibration + threshold search
    # ============================================================
    print("\n[4/7] Clean baseline evaluation...")
    val_ds = DS(val_df, fcols, scaler, t_val)
    val_loader = DataLoader(val_ds, STUDENT_BATCH, shuffle=False, num_workers=0)
    test_ds = DS(test_df, fcols, scaler, t_test)
    test_loader = DataLoader(test_ds, STUDENT_BATCH, shuffle=False, num_workers=0)

    val_bp, val_ap, val_rp = predict_all(model, val_loader, device)
    test_bp, test_ap, test_rp = predict_all(model, test_loader, device)

    y_risk_val = val_ds.yr.numpy().astype(np.int64)
    y_bin_val = val_ds.yb.numpy().astype(int)
    y_risk_test = test_ds.yr.numpy().astype(np.int64)
    y_bin_test = test_ds.yb.numpy().astype(int)

    # Platt calibration
    platt_bin = fit_platt_calibrator(val_bp, y_bin_val)
    platt_risk = PlattRiskCalibrator().fit(val_rp, y_risk_val)

    # Platt-calibrated probabilities on validation set (threshold search must use same-scale probabilities)
    val_bp_cal = apply_platt(platt_bin, val_bp)
    val_rp_cal = platt_risk.predict(val_rp)

    # Threshold search (based on Platt-calibrated validation probabilities)
    thresh = search_thresholds(y_risk_val, val_bp_cal, val_rp_cal)
    tau_attack = thresh["tau_attack"]
    tau_high = thresh["tau_high"]
    print(f"  Optimal threshold: tau_attack={tau_attack:.2f}, tau_high={tau_high:.2f}")

    # Clean baseline
    baseline = evaluate_with_interference(
        model, test_df, fcols, scaler, t_test, y_risk_test, y_bin_test,
        tau_attack, tau_high, platt_bin, platt_risk, device, interference_fn=None)
    print(f"  Baseline: Acc={baseline['accuracy']:.4f}, MacroF1={baseline['macro_f1']:.4f}, "
          f"HighRecall={baseline['high_recall']:.3f}, DetRate={baseline['detection_rate']:.3f}")

    # ============================================================
    #  [5/7] Interference experiments
    # ============================================================
    all_results = {}

    # --- 5a: Channel Fading ---
    print("\n[5a/7] Channel Fading experiment...")
    fading_results = {}
    for alpha in FADING_ALPHAS:
        metrics_list = []
        for run in range(N_REPEATS):
            m = evaluate_with_interference(
                model, test_df, fcols, scaler, t_test, y_risk_test, y_bin_test,
                tau_attack, tau_high, platt_bin, platt_risk, device,
                interference_fn=lambda X, rng, a=alpha: apply_channel_fading(X, a, rng),
                seed_offset=run*10)
            metrics_list.append(m)
        agg = {}
        for k in metrics_list[0]:
            vals = [m[k] for m in metrics_list]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        fading_results[alpha] = agg
        deg = (baseline["accuracy"] - agg["accuracy_mean"]) / baseline["accuracy"] * 100
        print(f"    alpha={alpha:.1f}: Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
              f"MacroF1={agg['macro_f1_mean']:.4f}, Degradation={deg:.1f}%")
    all_results["channel_fading"] = fading_results

    # --- 5b: EMI Noise ---
    print("\n[5b/7] EMI Noise experiment...")
    noise_results = {}
    for snr in NOISE_SNR_DB:
        metrics_list = []
        for run in range(N_REPEATS):
            m = evaluate_with_interference(
                model, test_df, fcols, scaler, t_test, y_risk_test, y_bin_test,
                tau_attack, tau_high, platt_bin, platt_risk, device,
                interference_fn=lambda X, rng, s=snr: apply_emi_noise(X, s, rng),
                seed_offset=run*10+100)
            metrics_list.append(m)
        agg = {}
        for k in metrics_list[0]:
            vals = [m[k] for m in metrics_list]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        noise_results[snr] = agg
        deg = (baseline["accuracy"] - agg["accuracy_mean"]) / baseline["accuracy"] * 100
        print(f"    SNR={snr:>5.1f}dB: Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
              f"MacroF1={agg['macro_f1_mean']:.4f}, Degradation={deg:.1f}%")
    all_results["emi_noise"] = noise_results

    # --- 5c: Packet Loss ---
    print("\n[5c/7] Packet Loss experiment...")
    loss_results = {}
    for rate in LOSS_RATES:
        metrics_list = []
        for run in range(N_REPEATS):
            m = evaluate_with_interference(
                model, test_df, fcols, scaler, t_test, y_risk_test, y_bin_test,
                tau_attack, tau_high, platt_bin, platt_risk, device,
                interference_fn=lambda X, rng, r=rate: apply_packet_loss(X, r, rng),
                seed_offset=run*10+200)
            metrics_list.append(m)
        agg = {}
        for k in metrics_list[0]:
            vals = [m[k] for m in metrics_list]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        loss_results[rate] = agg
        deg = (baseline["accuracy"] - agg["accuracy_mean"]) / baseline["accuracy"] * 100
        print(f"    loss={rate:.2f}: Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
              f"MacroF1={agg['macro_f1_mean']:.4f}, Degradation={deg:.1f}%")
    all_results["topology_loss"] = loss_results

    # --- 5d: Packet Reordering ---
    print("\n[5d/7] Packet Reordering experiment...")
    reorder_results = {}
    for rate in REORDER_RATES:
        metrics_list = []
        for run in range(N_REPEATS):
            m = evaluate_with_interference(
                model, test_df, fcols, scaler, t_test, y_risk_test, y_bin_test,
                tau_attack, tau_high, platt_bin, platt_risk, device,
                interference_fn=lambda X, rng, r=rate: apply_packet_reorder(X, r, rng),
                seed_offset=run*10+300)
            metrics_list.append(m)
        agg = {}
        for k in metrics_list[0]:
            vals = [m[k] for m in metrics_list]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        reorder_results[rate] = agg
        deg = (baseline["accuracy"] - agg["accuracy_mean"]) / baseline["accuracy"] * 100
        print(f"    reorder={rate:.2f}: Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
              f"MacroF1={agg['macro_f1_mean']:.4f}, Degradation={deg:.1f}%")
    all_results["topology_reorder"] = reorder_results

    # ============================================================
    #  [6/7] Visualization
    # ============================================================
    print("\n[6/7] Generating figures...")
    plot_degradation_curves(all_results, baseline, out_dir)
    print("  -> interference_degradation.png/pdf")
    plot_robustness_heatmap(all_results, baseline, out_dir)
    print("  -> robustness_heatmap.png/pdf")
    plot_summary_radar(all_results, baseline, out_dir)
    print("  -> robustness_comparison.png/pdf")

    # ============================================================
    #  [7/7] Save results
    # ============================================================
    print("\n[7/7] Saving result files...")

    # --- CSV: one file per dimension ---
    dim_files = [
        ("channel_fading", FADING_ALPHAS, "Alpha"),
        ("emi_noise", NOISE_SNR_DB, "SNR_dB"),
        ("topology_loss", LOSS_RATES, "Loss_Rate"),
        ("topology_reorder", REORDER_RATES, "Reorder_Rate"),
    ]
    for dim_key, levels, col_name in dim_files:
        rows = []
        for l in levels:
            r = all_results[dim_key][l]
            row = {col_name: l}
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "high_recall",
                           "detection_rate", "normal_f1", "medium_f1", "high_f1"]:
                row[f"{metric}_mean"] = r[f"{metric}_mean"]
                row[f"{metric}_std"] = r[f"{metric}_std"]
            rows.append(row)
        fname = f"{dim_key}_results.csv"
        pd.DataFrame(rows).to_csv(out_dir / fname, index=False)
        print(f"  -> {fname}")

    # --- CSV: summary table (cross-dimension comparison) ---
    summary_rows = [
        {"Dimension": "Baseline", "Severity": "None",
         "Accuracy": baseline["accuracy"], "Macro_F1": baseline["macro_f1"],
         "High_Recall": baseline["high_recall"], "Detection_Rate": baseline["detection_rate"]},
    ]
    for dim_key, dim_name, levels, col_name in [
        ("channel_fading", "Channel Fading", FADING_ALPHAS, "alpha"),
        ("emi_noise", "EMI Noise", NOISE_SNR_DB, "SNR_dB"),
        ("topology_loss", "Packet Loss", LOSS_RATES, "rate"),
        ("topology_reorder", "Packet Reorder", REORDER_RATES, "rate"),
    ]:
        for l in levels:
            if l == levels[0] and l in [1.0, 20.0, 0.0]: continue  # skip baseline duplicates
            r = all_results[dim_key][l]
            summary_rows.append({
                "Dimension": dim_name, "Severity": str(l),
                "Accuracy": r["accuracy_mean"], "Macro_F1": r["macro_f1_mean"],
                "High_Recall": r["high_recall_mean"], "Detection_Rate": r["detection_rate_mean"],
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / "interference_summary.csv", index=False)
    print("  -> interference_summary.csv")

    # --- JSON complete summary ---
    json_out = {
        "experiment": "Physical-Layer Interference Simulation for RSTD-KD",
        "reviewer_comment": "Reviewer 1 #5: lack of real UAV field testing",
        "dataset": "ECU-IoFT",
        "window_size": WINDOW_SIZE, "seed": SEED, "n_repeats": N_REPEATS,
        "device": str(device),
        "baseline": baseline,
        "thresholds": {"tau_attack": tau_attack, "tau_high": tau_high},
        "interference_dimensions": {
            "channel_fading": {
                "description": "Channel fading simulated by amplitude attenuation",
                "physical_model": "Rayleigh/Rician fading -> signal amplitude reduction",
                "levels": {str(a): all_results["channel_fading"][a] for a in FADING_ALPHAS},
            },
            "emi_noise": {
                "description": "Electromagnetic interference simulated by additive Gaussian noise",
                "physical_model": "Environmental EMI -> additive white Gaussian noise at various SNR",
                "levels": {str(s): all_results["emi_noise"][s] for s in NOISE_SNR_DB},
            },
            "topology_loss": {
                "description": "Packet loss from dynamic topology simulated by feature dropout",
                "physical_model": "UAV movement -> packet loss -> information missing",
                "levels": {str(r): all_results["topology_loss"][r] for r in LOSS_RATES},
            },
            "topology_reorder": {
                "description": "Packet reordering from dynamic topology simulated by feature shuffling",
                "physical_model": "UAV movement -> packet reordering -> statistical deviation",
                "levels": {str(r): all_results["topology_reorder"][r] for r in REORDER_RATES},
            },
        },
        "conclusion": {
            "channel_fading": f"Accuracy remains >= {min(all_results['channel_fading'][a]['accuracy_mean'] for a in FADING_ALPHAS):.4f} "
                              f"even at alpha={min(FADING_ALPHAS)} (50% signal attenuation)",
            "emi_noise": f"Accuracy remains >= {min(all_results['emi_noise'][s]['accuracy_mean'] for s in NOISE_SNR_DB):.4f} "
                         f"even at SNR={min(NOISE_SNR_DB)}dB (extreme noise)",
            "topology_loss": f"Accuracy remains >= {min(all_results['topology_loss'][r]['accuracy_mean'] for r in LOSS_RATES):.4f} "
                             f"even at {max(LOSS_RATES)*100:.0f}% packet loss",
            "topology_reorder": f"Accuracy remains >= {min(all_results['topology_reorder'][r]['accuracy_mean'] for r in REORDER_RATES):.4f} "
                                f"even at {max(REORDER_RATES)*100:.0f}% reordering",
        },
    }
    safe_json_dump(json_out, out_dir / "interference_summary.json")
    print("  -> interference_summary.json")

    # ============================================================
    #  Final summary print
    # ============================================================
    print("\n" + "=" * 90)
    print("  Physical-Layer Interference Simulation -- Final Summary")
    print("=" * 90)

    print(f"\n  Baseline (clean):")
    print(f"    Accuracy:     {baseline['accuracy']:.4f}")
    print(f"    Macro F1:     {baseline['macro_f1']:.4f}")
    print(f"    High Recall:  {baseline['high_recall']:.3f}")
    print(f"    Detection:    {baseline['detection_rate']:.3f}")

    for dim_key, dim_name, levels, unit in [
        ("channel_fading", "Channel Fading", FADING_ALPHAS, "alpha"),
        ("emi_noise", "EMI Noise", NOISE_SNR_DB, "dB"),
        ("topology_loss", "Packet Loss", LOSS_RATES, "rate"),
        ("topology_reorder", "Packet Reorder", REORDER_RATES, "rate"),
    ]:
        print(f"\n  [{dim_name}]")
        print(f"    {'Severity':>12s}  {'Accuracy':>10s}  {'MacroF1':>10s}  {'HighRecall':>10s}  {'Deg%':>6s}")
        for l in levels:
            r = all_results[dim_key][l]
            deg = (baseline["accuracy"] - r["accuracy_mean"]) / baseline["accuracy"] * 100
            print(f"    {str(l):>12s}  {r['accuracy_mean']:>10.4f}  {r['macro_f1_mean']:>10.4f}  "
                  f"{r['high_recall_mean']:>10.3f}  {deg:>5.1f}%")

    # Overall robustness score
    print(f"\n  [Overall Robustness Score]")
    worst_accs = []
    for dim_key, levels in [("channel_fading", FADING_ALPHAS), ("emi_noise", NOISE_SNR_DB),
                            ("topology_loss", LOSS_RATES), ("topology_reorder", REORDER_RATES)]:
        worst = min(all_results[dim_key][l]["accuracy_mean"] for l in levels)
        worst_accs.append(worst)
    avg_retention = np.mean(worst_accs) / baseline["accuracy"] * 100
    print(f"    Average performance retention at max interference: {avg_retention:.1f}%")
    print(f"    Worst-case accuracy across all dimensions: {min(worst_accs):.4f}")

    print(f"\n  File list:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"    {f.name:40s}  ({f.stat().st_size / 1024:.1f} KB)")

    print(f"\n" + "=" * 90)
    print(f"  Experiment complete! All files saved in: {out_dir}")


if __name__ == "__main__":
    main()
