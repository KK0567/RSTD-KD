#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSTD-KD Task Cost Ratio Sensitivity Experiment
FP/FN Cost-Ratio Sensitivity Analysis

Reviewer Response (Reviewer 1 #4):
  Experiments with continuously varying FP/FN cost ratios are needed to show
  how the threshold dynamically drifts as task costs change.

Experiment Design:
  Cost Ratios: C_FP:C_FN = 5:1, 2:1, 1:1, 1:2, 1:5
  For each cost ratio, scan thresholds and find the optimal one.
  Report: optimal threshold, FPR, FNR, expected cost

  Simultaneously analyze:
  (1) Binary attack detection layer (binary tau_attack)
  (2) Three-level risk decision layer (tau_attack + tau_high)

Usage:
  Run directly (python fpfn_cost_sensitivity.py)
  All outputs are saved to OUTPUT_DIR
"""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
#  Path Configuration
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_CSV = str(PROJECT_ROOT / "data" / "ECU-IoFT-Dataset.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "table6_cost_ratio")

# ================================================================
#  Global Hyperparameters (consistent with ablation experiments)
# ================================================================
SEED = 42
WINDOW_SIZE = 32  # main window in the paper
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15

# Cost ratios (C_FP : C_FN)
COST_RATIOS = [
    (5, 1),   # high false-alarm cost -> model more conservative
    (2, 1),
    (1, 1),   # symmetric
    (1, 2),
    (1, 5),   # high miss cost -> model more aggressive
]

# Teacher / Student hyperparameters (consistent with ablation experiments)
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

STUDENT_EPOCHS = 160
STUDENT_BATCH = 128
STUDENT_LR = 4e-4
STUDENT_WD = 3e-4
STUDENT_PATIENCE = 24
STUDENT_TEMP = 2.0
STUDENT_HIDDEN1, STUDENT_HIDDEN2 = 128, 64
STUDENT_DROPOUT1, STUDENT_DROPOUT2 = 0.12, 0.08

# Label mapping (consistent with ablation experiments)
ATTACK_TYPE_MAP = {
    "No Attack": 0, "Wifi Deauthentication Attack": 1,
    "WPA2-PSK WIFI Cracking Attack": 2, "TELLO API Exploit": 3,
}
RISK_MAP = {
    "No Attack": (0, "normal"), "Wifi Deauthentication Attack": (1, "medium"),
    "WPA2-PSK WIFI Cracking Attack": (2, "high"), "TELLO API Exploit": (2, "high"),
}
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
#  Utility Functions (consistent with ablation experiments)
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


def _split_counts(n, tr, vr, ter):
    if n <= 0: return 0, 0, 0
    if n == 1: return 1, 0, 0
    if n == 2: return 1, 1, 0
    nt = max(1, int(round(n * tr))); nv = max(1, int(round(n * vr))); nte = max(1, n - nt - nv)
    while nt + nv + nte > n:
        if nt > 1: nt -= 1
        elif nv > 1: nv -= 1
        elif nte > 1: nte -= 1
        else: break
    while nt + nv + nte < n: nt += 1
    return nt, nv, nte


# ================================================================
#  Data Construction (identical to ablation experiments)
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


def assign_splits(win_df):
    from collections import Counter
    out = win_df.copy().sort_values("packet_id_start").reset_index(drop=True)
    out["split"] = ""; out["session_id"] = ""
    for aid, sub in out.groupby("y_attack_type", sort=True):
        aid = int(aid); idxs = sub.sort_values("packet_id_start").index.tolist(); n = len(idxs)
        nt, nv, nte = _split_counts(n, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        if aid == 3 and n >= 3:
            nt = max(nt, 1)
            while nt+nv+nte > n:
                if nte>1: nte-=1
                elif nv>1: nv-=1
                else: nt-=1
        nm = {0:"No Attack",1:"Wifi_Deauth",2:"WPA2_Crack",3:"TELLO"}.get(aid, f"atk{aid}")
        for i, idx in enumerate(idxs):
            if i < nt: sp = "train"
            elif i < nt+nv: sp = "val"
            else: sp = "test"
            out.loc[idx, "split"] = sp
            out.loc[idx, "session_id"] = f"atk{aid}_{nm}__win_{i:05d}"
    return out


def prepare_data(raw_df, ws=WINDOW_SIZE):
    win_df = build_windows(raw_df, ws, ws)
    out = win_df.copy().sort_values("packet_id_start").reset_index(drop=True)
    out["split"] = ""; out["session_id"] = ""
    for aid, sub in out.groupby("y_attack_type", sort=True):
        aid = int(aid); idxs = sub.sort_values("packet_id_start").index.tolist(); n = len(idxs)
        nt, nv, nte = _split_counts(n, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        if aid == 3 and n >= 3:
            nt = max(nt, 1)
            while nt+nv+nte > n:
                if nte>1: nte-=1
                elif nv>1: nv-=1
                else: nt-=1
        nm = {0:"No_Attack",1:"Wifi_Deauth",2:"WPA2_Crack",3:"TELLO"}.get(aid, f"atk{aid}")
        for i, idx in enumerate(idxs):
            if i < nt: sp = "train"
            elif i < nt+nv: sp = "val"
            else: sp = "test"
            out.loc[idx, "split"] = sp; out.loc[idx, "session_id"] = f"atk{aid}_{nm}__w{i:05d}"

    def std_df(df):
        o = df.copy(); o["row_id"] = o["window_id"].astype(np.int64)
        o["binary_label"] = o["y_bin"].astype(np.int64); o["label"] = o["y_attack_type_name"].astype(str)
        return o

    train_df = std_df(out.loc[out["split"]=="train"])
    val_df = std_df(out.loc[out["split"]=="val"])
    test_df = std_df(out.loc[out["split"]=="test"])

    fcols = [c for c in out.columns if c not in {
        "window_id","row_id","packet_id_start","packet_id_end","time_start","time_end",
        "y_bin","y_attack_type","y_attack_type_name","y_risk","y_risk_name",
        "attack_scenario_meta","packet_attack_type_mode_meta","session_id","split"}]
    keep = []
    for c in fcols:
        if c in ALWAYS_DROP_EXACT: continue
        if any(c.startswith(p) for p in ALWAYS_DROP_PREFIX): continue
        if any(k in c.lower() for k in ALWAYS_DROP_CONTAINS): continue
        if not pd.api.types.is_numeric_dtype(train_df[c]): continue
        keep.append(c)
    fcols = keep
    med = train_df[fcols].median(numeric_only=True).to_dict()
    for d in [train_df, val_df, test_df]: d[fcols] = d[fcols].fillna(med)
    return train_df, val_df, test_df, fcols


# ================================================================
#  Teacher / Student (consistent with ablation experiments)
# ================================================================
def bsw(y):
    y = np.asarray(y, np.int64); cls, cnt = np.unique(y, return_counts=True); tot = cnt.sum()
    wm = {int(c): float(tot/(len(cls)*ci)) for c, ci in zip(cls, cnt)}
    return np.array([wm[int(v)] for v in y], np.float32)


def train_teacher(train_df, fcols):
    Xb, yb, wb = train_df[fcols].values.astype(np.float32), train_df["binary_label"].values.astype(np.int64), bsw(train_df["binary_label"].values)
    bmodels = []
    for i in range(TEACHER_ENSEMBLE_SIZE):
        rng = np.random.RandomState(SEED + 1009*(i+1))
        parts = []
        for c in np.unique(yb):
            idx = np.where(yb==c)[0]; n = max(1, int(round(len(idx)*TEACHER_BOOTSTRAP_FRAC)))
            parts.append(rng.choice(idx, size=n, replace=True))
        bi = np.concatenate(parts); rng.shuffle(bi)
        m = HistGradientBoostingClassifier(loss="log_loss", learning_rate=TEACHER_LR, max_iter=TEACHER_MAX_ITER,
            max_leaf_nodes=TEACHER_MAX_LEAF, min_samples_leaf=TEACHER_MIN_LEAF_BIN, l2_regularization=TEACHER_L2,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=SEED+i)
        m.fit(Xb[bi], yb[bi], sample_weight=wb[bi]); bmodels.append(m)

    Xr, yr, wr = train_df[fcols].values.astype(np.float32), train_df["y_risk"].values.astype(np.int64), bsw(train_df["y_risk"].values)
    es = np.min(np.bincount(yr)) >= 2
    rmodel = HistGradientBoostingClassifier(loss="log_loss", learning_rate=TEACHER_RISK_LR, max_iter=TEACHER_RISK_MAX_ITER,
        max_leaf_nodes=TEACHER_RISK_MAX_LEAF, min_samples_leaf=TEACHER_RISK_MIN_LEAF, l2_regularization=TEACHER_RISK_L2,
        early_stopping=es, random_state=SEED+37)
    rmodel.fit(Xr, yr, sample_weight=wr)

    sub = train_df.loc[train_df["binary_label"]==1]
    Xa = sub[fcols].values.astype(np.float32); ya = sub["y_attack_type"].values.astype(np.int64); wa = bsw(ya)
    es_a = np.min(np.bincount(ya)) >= 2 if len(ya) > 0 else False
    amodel = HistGradientBoostingClassifier(loss="log_loss", learning_rate=TEACHER_ATK_LR, max_iter=TEACHER_ATK_MAX_ITER,
        max_leaf_nodes=TEACHER_ATK_MAX_LEAF, min_samples_leaf=TEACHER_ATK_MIN_LEAF, l2_regularization=TEACHER_ATK_L2,
        early_stopping=es_a, random_state=SEED+17)
    amodel.fit(Xa, ya, sample_weight=wa)
    return bmodels, rmodel, amodel


def pred_bin_ens(models, X):
    return np.mean(np.stack([m.predict_proba(X)[:, list(m.classes_).index(1)] for m in models]), axis=0)


def pred_fixed(model, X, classes):
    raw = model.predict_proba(X); out = np.zeros((X.shape[0], len(classes)), np.float64); mc = list(model.classes_)
    for j, c in enumerate(classes):
        if c in mc: out[:, j] = raw[:, mc.index(c)]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


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
        self.ya = torch.tensor(ai, dtype=torch.long); self.yr = torch.tensor(df["y_risk"].values.astype(np.int64), dtype=torch.long)
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
    bs, bsc, bep, w = None, -1., -1, 0
    for ep in range(1, STUDENT_EPOCHS+1):
        model.train()
        for b in tl:
            x = b["x"].to(device)
            lb, la, lr = model(x)
            lhb = (F.binary_cross_entropy_with_logits(lb, b["yb"].to(device), reduction="none")*b["wb"].to(device)).sum()/b["wb"].to(device).sum().clamp_min(1e-8)
            lkb = ((torch.sigmoid(lb/STUDENT_TEMP)-b["tb"].to(device))**2*b["wb"].to(device)).sum()/b["wb"].to(device).sum().clamp_min(1e-8)
            ce = F.cross_entropy(la, b["ya"].to(device), reduction="none")
            lha = (ce*b["wa"].to(device)*b["am"].to(device)).sum()/(b["am"].to(device)*b["wa"].to(device)).sum().clamp_min(1e-8)
            lp = F.log_softmax(la/STUDENT_TEMP,1); q = torch.clamp(b["ta"].to(device),1e-6,1.0); q = q/q.sum(1,keepdim=True).clamp_min(1e-8)
            lka = (F.kl_div(lp,q,reduction="none").sum(1)*b["am"].to(device)).sum()/b["am"].to(device).sum().clamp_min(1e-8)*(STUDENT_TEMP**2)
            lhr = (F.cross_entropy(lr,b["yr"].to(device),reduction="none")*b["wrk"].to(device)).sum()/b["wrk"].to(device).sum().clamp_min(1e-8)
            lp2 = F.log_softmax(lr/STUDENT_TEMP,1); q2 = torch.clamp(b["tr"].to(device),1e-6,1.0); q2 = q2/q2.sum(1,keepdim=True).clamp_min(1e-8)
            lkr = F.kl_div(lp2,q2,reduction="batchmean")*(STUDENT_TEMP**2)
            loss = ab*lhb+bb*lkb+aa*lha+ba*lka+ar*lhr+br*lkr
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
        yr = vds.yr.numpy().astype(np.int64)
        _,_,f1,_ = __import__("sklearn").metrics.precision_recall_fscore_support(yr,pred,labels=[0,1,2],average="macro",zero_division=0)
        _,hr,_,_ = __import__("sklearn").metrics.precision_recall_fscore_support(yr,pred,labels=[2],average="macro",zero_division=0)
        sc = f1 + 0.1*hr; sch.step(sc)
        if sc > bsc: bsc, bep, bs, w = sc, ep, deepcopy(model.state_dict()), 0
        else: w += 1
        if w >= STUDENT_PATIENCE: break

    model.load_state_dict(bs); return model


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval(); pb,pa,pr = [],[],[]
    for b in loader:
        x = b["x"].to(device); lb,la,lr = model(x)
        pb.append(torch.sigmoid(lb).cpu().numpy()); pa.append(torch.softmax(la,1).cpu().numpy()); pr.append(torch.softmax(lr,1).cpu().numpy())
    return np.concatenate(pb), np.concatenate(pa), np.concatenate(pr)


# ================================================================
#  Cost Sensitivity Analysis Core
# ================================================================
def binary_cost_scan(y_true, y_prob, c_fp, c_fn, thresholds=None):
    """Scan thresholds on binary probabilities, compute FPR/FNR/Expected Cost for each."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.005)
    y = np.asarray(y_true, int); p = np.asarray(y_prob, float)
    n = len(y)
    n_pos = max(1, int((y == 1).sum())); n_neg = max(1, int((y == 0).sum()))
    rows = []
    for th in thresholds:
        pred = (p >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        cost = c_fp * fpr + c_fn * fnr
        rows.append({"threshold": float(th), "fpr": float(fpr), "fnr": float(fnr),
                     "expected_cost": float(cost), "tp": int(tp), "fp": int(fp),
                     "tn": int(tn), "fn": int(fn)})
    return pd.DataFrame(rows)


def risk_cost_scan(y_risk, bin_prob, risk_prob, c_fn_medium, c_fn_high, c_fp,
                    tau_grid=None):
    """Scan tau_attack for risk decision (tau_high fixed at its optimal value),
    compute weighted misclassification cost."""
    if tau_grid is None:
        tau_grid = np.arange(0.01, 1.00, 0.01)
    yr = np.asarray(y_risk, int)
    bp = np.asarray(bin_prob, float)
    rp = np.asarray(risk_prob, float)
    n = len(yr)
    rows = []
    for ta in tau_grid:
        pmed, phigh = rp[:, 1], rp[:, 2]
        pside = np.where(phigh >= 0.3, 2, np.where(pmed >= phigh, 1, 2))
        pred = np.where(bp < ta, 0, pside)
        cm = confusion_matrix(yr, pred, labels=[0, 1, 2])
        # Cost matrix: rows=truth, cols=prediction
        # True high predicted as normal -> most severe (c_fn_high)
        # True normal predicted as high -> false alarm (c_fp)
        # True medium predicted as normal -> miss (c_fn_medium)
        fn_high = cm[2, 0] + cm[2, 1]  # true high predicted as 0 or 1
        fn_med = cm[1, 0]  # true medium predicted as 0
        fp = cm[0, 1] + cm[0, 2]  # true normal predicted as 1 or 2
        cost = (c_fn_high * fn_high + c_fn_medium * fn_med + c_fp * fp) / n
        rows.append({"tau_attack": float(ta), "fn_high": int(fn_high), "fn_medium": int(fn_med),
                     "fp_total": int(fp), "expected_cost": float(cost)})
    return pd.DataFrame(rows)


# ================================================================
#  Visualization
# ================================================================
def plot_cost_curves(scan_results, out_dir):
    """Plot cost-ratio sensitivity curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: Expected Cost vs Threshold (different cost ratios)
    ax = axes[0]
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]
    for i, ((cfp, cfn), df) in enumerate(scan_results.items()):
        ax.plot(df["threshold"], df["expected_cost"], linewidth=2, color=colors[i],
                label=f"$C_{{FP}}$:{cfp}  $C_{{FN}}$:{cfn}")
        best_idx = df["expected_cost"].idxmin()
        best_th = df.loc[best_idx, "threshold"]
        best_cost = df.loc[best_idx, "expected_cost"]
        ax.axvline(best_th, color=colors[i], linestyle="--", alpha=0.4, linewidth=1)
        ax.plot(best_th, best_cost, "o", color=colors[i], markersize=6)
    ax.set_xlabel("Attack Detection Threshold $\\tau$", fontsize=12)
    ax.set_ylabel("Expected Cost", fontsize=12)
    ax.set_title("(a) Expected Cost vs Threshold", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)

    # Middle: Optimal Threshold vs Cost Ratio
    ax2 = axes[1]
    ratios = []; opt_ths = []; opt_costs = []
    for (cfp, cfn), df in scan_results.items():
        ratios.append(cfp / cfn)
        best_idx = df["expected_cost"].idxmin()
        opt_ths.append(df.loc[best_idx, "threshold"])
        opt_costs.append(df.loc[best_idx, "expected_cost"])
    ax2.semilogx(ratios, opt_ths, "o-", color="#2c3e50", linewidth=2, markersize=8)
    for r, t in zip(ratios, opt_ths):
        ax2.annotate(f"{t:.2f}", (r, t), textcoords="offset points", xytext=(5, 10), fontsize=9)
    ax2.set_xlabel("$C_{FP} / C_{FN}$ Ratio", fontsize=12)
    ax2.set_ylabel("Optimal Threshold $\\tau^*$", fontsize=12)
    ax2.set_title("(b) Optimal Threshold Drift", fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Right: FPR vs FNR at optimal threshold
    ax3 = axes[2]
    fprs = []; fnrs = []
    labels = []
    for (cfp, cfn), df in scan_results.items():
        best_idx = df["expected_cost"].idxmin()
        fprs.append(df.loc[best_idx, "fpr"])
        fnrs.append(df.loc[best_idx, "fnr"])
        labels.append(f"{cfp}:{cfn}")
    x_pos = np.arange(len(labels))
    w = 0.35
    ax3.bar(x_pos - w/2, fprs, w, color="#3498db", label="FPR", alpha=0.85)
    ax3.bar(x_pos + w/2, fnrs, w, color="#e74c3c", label="FNR", alpha=0.85)
    ax3.set_xlabel("$C_{FP}:C_{FN}$ Cost Ratio", fontsize=12)
    ax3.set_ylabel("Rate", fontsize=12)
    ax3.set_title("(c) FPR / FNR at Optimal $\\tau^*$", fontsize=13)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.legend(fontsize=10, frameon=False)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "cost_sensitivity_curves.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "cost_sensitivity_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


# ================================================================
#  Main Function
# ================================================================
def main():
    set_seed(SEED)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Output: {out_dir}")

    # --- 1. Data Preparation ---
    print("\n[1/4] Loading data and building windows (w=32)...")
    raw_df = pd.read_csv(RAW_DATA_CSV)
    raw_df["_dt"] = parse_time(raw_df["Time"])
    raw_df = raw_df.reset_index(drop=True)
    train_df, val_df, test_df, fcols = prepare_data(raw_df)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Features: {len(fcols)}")

    # --- 2. Train Teacher + Student ---
    print("\n[2/4] Training Teacher + Student (Full RSTD-KD)...")
    bmodels, rmodel, amodel = train_teacher(train_df, fcols)

    def tpred(df):
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

    t_train = tpred(train_df); t_val = tpred(val_df); t_test = tpred(test_df)

    scaler = StandardScaler(); scaler.fit(train_df[fcols].values.astype(np.float32))
    model = train_student(train_df, val_df, fcols, scaler, t_train, t_val, device)

    # Get test predictions
    test_ds = DS(test_df, fcols, scaler, t_test)
    test_loader = DataLoader(test_ds, STUDENT_BATCH, shuffle=False, num_workers=0)
    val_ds = DS(val_df, fcols, scaler, t_val)
    val_loader = DataLoader(val_ds, STUDENT_BATCH, shuffle=False, num_workers=0)
    test_bp, test_ap, test_rp = predict_all(model, test_loader, device)
    val_bp, val_ap, val_rp = predict_all(model, val_loader, device)

    y_bin_test = test_ds.yb.numpy().astype(int)
    y_risk_test = test_ds.yr.numpy().astype(int)
    y_bin_val = val_ds.yb.numpy().astype(int)

    print(f"  Training complete. Test samples: {len(y_bin_test)}")

    # --- 3. Cost Sensitivity Analysis ---
    print("\n[3/4] Running FP/FN Cost-Ratio Sensitivity Analysis...")
    scan_results = {}
    summary_rows = []

    for c_fp, c_fn in COST_RATIOS:
        ratio_str = f"{c_fp}:{c_fn}"
        print(f"  -> C_FP:C_FN = {ratio_str}")

        # Binary threshold scan
        scan_df = binary_cost_scan(y_bin_test, test_bp, c_fp, c_fn)
        scan_results[(c_fp, c_fn)] = scan_df
        scan_df.to_csv(out_dir / f"cost_scan_{c_fp}v{c_fn}.csv", index=False)

        best_idx = scan_df["expected_cost"].idxmin()
        best = scan_df.loc[best_idx]

        summary_rows.append({
            "Cost_Ratio": ratio_str,
            "C_FP": c_fp, "C_FN": c_fn,
            "Optimal_Threshold": best["threshold"],
            "FPR": best["fpr"], "FNR": best["fnr"],
            "Expected_Cost": best["expected_cost"],
            "TP": int(best["tp"]), "FP": int(best["fp"]),
            "TN": int(best["tn"]), "FN": int(best["fn"]),
        })

    # Summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "cost_ratio_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("  Binary Attack Detection — Cost-Ratio Sensitivity Summary")
    print("=" * 80)
    print(f"  {'C_FP:C_FN':>10s}  {'Opt. τ':>8s}  {'FPR':>8s}  {'FNR':>8s}  {'Exp. Cost':>10s}")
    print("-" * 55)
    for _, r in summary_df.iterrows():
        print(f"  {r['Cost_Ratio']:>10s}  {r['Optimal_Threshold']:>8.3f}  "
              f"{r['FPR']:>8.4f}  {r['FNR']:>8.4f}  {r['Expected_Cost']:>10.4f}")

    # --- Risk-Level Cost Analysis ---
    print("\n  Risk-Level Cost Analysis (3-level risk)...")
    risk_rows = []
    # Risk-level cost: C_FN_high : C_FN_medium : C_FP
    risk_configs = [
        ("5:1:1", 1, 5, 1),    # high miss cost for high-risk
        ("2:1:1", 1, 2, 1),
        ("1:1:1", 1, 1, 1),    # symmetric
        ("1:1:2", 2, 1, 1),
        ("1:1:5", 5, 1, 1),    # high false alarm cost
    ]
    for name, c_fn_high, c_fn_med, c_fp in risk_configs:
        scan_r = risk_cost_scan(y_risk_test, test_bp, test_rp, c_fn_med, c_fn_high, c_fp)
        best_idx = scan_r["expected_cost"].idxmin()
        best = scan_r.loc[best_idx]
        risk_rows.append({
            "Cost_Config": name,
            "C_FN_high": c_fn_high, "C_FN_medium": c_fn_med, "C_FP": c_fp,
            "Optimal_tau_attack": best["tau_attack"],
            "FN_high": int(best["fn_high"]),
            "FN_medium": int(best["fn_medium"]),
            "FP_total": int(best["fp_total"]),
            "Expected_Cost": best["expected_cost"],
        })
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(out_dir / "risk_level_cost_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("  Risk-Level Decision — Cost Sensitivity Summary")
    print("=" * 80)
    print(f"  {'Config':>12s}  {'Opt. τ':>8s}  {'FN_high':>8s}  {'FN_med':>8s}  {'FP':>6s}  {'Exp. Cost':>10s}")
    print("-" * 65)
    for _, r in risk_df.iterrows():
        print(f"  {r['Cost_Config']:>12s}  {r['Optimal_tau_attack']:>8.3f}  "
              f"{r['FN_high']:>8d}  {r['FN_medium']:>8d}  {r['FP_total']:>6d}  "
              f"{r['Expected_Cost']:>10.4f}")

    # --- 4. Visualization ---
    print("\n[4/4] Generating figures...")
    plot_cost_curves(scan_results, out_dir)

    # Save prediction data
    pred_out = pd.DataFrame({
        "y_bin": y_bin_test, "y_risk": y_risk_test,
        "student_prob_binary": test_bp,
        "student_risk_prob_0": test_rp[:, 0], "student_risk_prob_1": test_rp[:, 1],
        "student_risk_prob_2": test_rp[:, 2],
    })
    pred_out.to_csv(out_dir / "student_test_predictions.csv", index=False)

    # Save JSON summary
    safe_json_dump({
        "experiment": "FP/FN Cost-Ratio Sensitivity Analysis for RSTD-KD",
        "window_size": WINDOW_SIZE, "seed": SEED, "device": str(device),
        "test_samples": len(y_bin_test),
        "attack_samples_test": int(y_bin_test.sum()),
        "normal_samples_test": int((y_bin_test == 0).sum()),
        "binary_cost_analysis": summary_rows,
        "risk_level_cost_analysis": risk_rows,
        "conclusion": {
            "when_fp_cost_increases": "optimal threshold rises, model becomes more conservative (fewer false alarms)",
            "when_fn_cost_increases": "optimal threshold drops, model becomes more aggressive (higher recall)",
        },
    }, out_dir / "cost_analysis_summary.json")

    print("\n" + "=" * 80)
    print("  Experiment complete!")
    print("=" * 80)
    print(f"  All files saved in: {out_dir}")
    print(f"  1. cost_ratio_summary.csv          — Binary cost sensitivity summary (core table)")
    print(f"  2. risk_level_cost_summary.csv     — 3-level risk cost analysis")
    print(f"  3. cost_scan_*.csv                 — Per-threshold scan for each cost ratio")
    print(f"  4. cost_sensitivity_curves.png/pdf — Cost sensitivity curves")
    print(f"  5. student_test_predictions.csv    — Student model test set predictions")
    print(f"  6. cost_analysis_summary.json      — Complete JSON summary")


if __name__ == "__main__":
    main()
