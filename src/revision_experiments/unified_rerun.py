#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified re-run: Tables 3, 6, 7 using the MAIN EXPERIMENT'S protocol.
  - Same model: StudentRiskCascade (96→48, 3-class attack head)
  - Same decision: cascade (binary≥0.5 → argmax attack → ATTACK_TO_RISK mapping)
  - Same data: attack-stratified split (matching output1 / Fig.4)
  - NO Platt calibration, NO threshold search

Usage:
  python unified_rerun.py --tables 3 6 7
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
#  Paths
# ================================================================
MAIN_EXP_DIR = Path(r"E:\Licaiyun\practise\4.RSTD_KD\风险识别-level划分")
DATA_PREP_SCRIPT = MAIN_EXP_DIR / "1.build_ecu_iotf_attack_risk_windows_run_time_level_fix3.py"
TEACHER_SCRIPT = MAIN_EXP_DIR / "2.teacher_risk_cascade_ecu_run_time_compatible.py"
STUDENT_SCRIPT = MAIN_EXP_DIR / "3.student_risk_cascade_ecu_run_time_compatible.py"
RAW_DATA_CSV = MAIN_EXP_DIR.parent / "Data" / "ECU-IoFT-Dataset.csv"
OUTPUT1_DIR = MAIN_EXP_DIR / "output1"  # pre-saved attack-stratified results
OUTPUT_BASE = MAIN_EXP_DIR  # for new outputs

REVISION_DIR = Path(r"E:\Licaiyun\论文投稿\4.RSTD-KD相关\论文修订\一审")
REVIEW_BASE = Path(r"E:\Licaiyun\practise\4.RSTD_KD\one review")

PYTHON = r"D:\TOOLS\anaconda3\envs\torch\python.exe"

# ================================================================
#  Constants (from main experiment)
# ================================================================
ATTACK_TO_RISK = {0: 1, 1: 2, 2: 2}  # internal attack id → risk level
ATTACK_INTERNAL_TO_ORIG = {0: 1, 1: 2, 2: 3}
RISK_CLASSES = [0, 1, 2]

SEED = 42
WINDOW_SIZES = [8, 16, 24, 32, 48, 64]

# ================================================================
#  Utility
# ================================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_cmd(cmd, desc=""):
    """Run a shell command and return exit code."""
    print(f"\n{'='*60}")
    print(f"  CMD: {desc}")
    print(f"  {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=False)
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
    return result.returncode


# ================================================================
#  Main experiment model (copied from 3.student_risk_cascade_ecu_run_time_compatible.py)
# ================================================================
ALWAYS_DROP_EXACT = {
    "y_bin", "y_attack_type", "y_attack_type_name", "y_risk", "y_risk_name",
    "attack_packet_count", "attack_packet_ratio",
    "window_id", "row_id", "binary_label", "label", "split", "session_id",
    "time_start", "time_end", "packet_id_start", "packet_id_end",
    "attack_scenario_meta", "packet_attack_type_mode_meta",
    "run_id", "run_source", "split_group_id", "group_id",
    "window_run_id", "window_run_count", "window_run_mixed", "window_label_rule",
    "window_no_attack_count", "window_deauth_count", "window_wpa2_count", "window_tello_count",
}
ALWAYS_DROP_PREFIX = ("cnt_",)
ALWAYS_DROP_CONTAINS = ("label", "target")


def sanitize_feature_cols(df, feature_cols):
    keep = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        if c in ALWAYS_DROP_EXACT:
            continue
        if any(c.startswith(p) for p in ALWAYS_DROP_PREFIX):
            continue
        lc = c.lower()
        if any(k in lc for k in ALWAYS_DROP_CONTAINS):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        keep.append(c)
    return keep


class StudentRiskCascade(nn.Module):
    """Exact same architecture as main experiment."""
    def __init__(self, in_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 96), nn.GELU(), nn.Dropout(0.18),
            nn.Linear(96, 48), nn.GELU(), nn.Dropout(0.12),
        )
        self.head_bin = nn.Linear(48, 1)
        self.head_attack = nn.Linear(48, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.head_bin(z).squeeze(1), self.head_attack(z)


class RiskCascadeDataset(Dataset):
    """Same dataset class as main experiment."""
    def __init__(self, df, feature_cols, scaler, teacher_df):
        df = df.copy()
        # Ensure teacher columns
        for c in ["teacher_attack_prob_1", "teacher_attack_prob_2", "teacher_attack_prob_3"]:
            if c not in df.columns:
                df[c] = 0.0
        merge_cols = ["row_id", "teacher_prob_binary",
                      "teacher_attack_prob_1", "teacher_attack_prob_2", "teacher_attack_prob_3"]
        if "teacher_prob_binary" not in df.columns:
            df = df.merge(teacher_df[merge_cols], on="row_id", how="left")

        X = scaler.transform(df[feature_cols].values.astype(np.float32))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_bin = torch.tensor(df["binary_label"].values.astype(np.float32))
        attack_orig = df["y_attack_type"].values.astype(np.int64)
        attack_internal = np.where(df["binary_label"].values.astype(np.int64) == 1,
                                   attack_orig - 1, 0).astype(np.int64)
        self.y_attack = torch.tensor(attack_internal, dtype=torch.long)
        self.y_risk = torch.tensor(df["y_risk"].values.astype(np.int64), dtype=torch.long)
        self.teacher_bin = torch.tensor(df["teacher_prob_binary"].values.astype(np.float32))
        self.teacher_attack = torch.tensor(
            df[["teacher_attack_prob_1", "teacher_attack_prob_2", "teacher_attack_prob_3"]].values.astype(np.float32))
        self.abn_mask = torch.tensor(df["binary_label"].values.astype(np.float32))

        y_bin_np = df["binary_label"].values.astype(np.int64)
        n_neg = max(1, int((y_bin_np == 0).sum()))
        n_pos = max(1, int((y_bin_np == 1).sum()))
        total = n_neg + n_pos
        w_neg = total / (2.0 * n_neg)
        w_pos = total / (2.0 * n_pos)
        self.w_bin = torch.tensor(np.where(y_bin_np == 1, w_pos, w_neg).astype(np.float32))

        abn = df.loc[df["binary_label"] == 1, "y_attack_type"].values.astype(np.int64) - 1
        classes, counts = np.unique(abn, return_counts=True)
        total_abn = counts.sum() if len(counts) else 1
        w_map = {int(c): float(total_abn / (len(classes) * cnt)) for c, cnt in zip(classes, counts)} if len(classes) else {}
        attack_w = np.ones(len(df), dtype=np.float32)
        for i, (is_abn, attack_id) in enumerate(zip(y_bin_np, df["y_attack_type"].values.astype(np.int64) - 1)):
            if is_abn == 1:
                attack_w[i] = w_map.get(int(attack_id), 1.0)
        self.w_attack = torch.tensor(attack_w, dtype=torch.float32)
        self.row_id = df["row_id"].values.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx], "y_bin": self.y_bin[idx], "y_attack": self.y_attack[idx],
            "y_risk": self.y_risk[idx], "teacher_bin": self.teacher_bin[idx],
            "teacher_attack": self.teacher_attack[idx], "abn_mask": self.abn_mask[idx],
            "w_bin": self.w_bin[idx], "w_attack": self.w_attack[idx],
        }


def cascade_predict(prob_bin, attack_probs, bin_threshold=0.5):
    """Main experiment's cascade decision logic."""
    pred_bin = (prob_bin >= bin_threshold).astype(np.int64)
    pred_attack_internal = attack_probs.argmax(axis=1)
    pred_risk = np.where(
        pred_bin == 0, 0,
        np.array([ATTACK_TO_RISK[int(v)] for v in pred_attack_internal], dtype=np.int64)
    )
    return pred_risk


def compute_metrics(y_true, y_pred):
    """Compute standard metrics."""
    acc = accuracy_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
    # Per-class metrics
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=RISK_CLASSES, average=None, zero_division=0)
    # Macro averages
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=RISK_CLASSES, average="macro", zero_division=0)
    hr = float(r_cls[2]) if len(r_cls) > 2 else 0.0
    return {
        "Accuracy": acc, "Balanced_Accuracy": ba, "Macro_F1": f1, "High_Recall": hr,
        "cm": confusion_matrix(y_true, y_pred, labels=RISK_CLASSES).tolist(),
    }


# ================================================================
#  Table 3: Window Sensitivity (attack-stratified, cascade decision)
# ================================================================
def run_table3():
    print("\n" + "="*70)
    print("  TABLE 3: Window Sensitivity (attack-stratified + cascade)")
    print("="*70)

    results = []

    # For w=32, use pre-saved predictions to verify 93.75%
    print("\n--- w=32: using pre-saved predictions from output1 ---")
    test_pred = pd.read_csv(OUTPUT1_DIR / "student_risk_cascade_v1" / "student_risk_preds_test.csv")
    y_risk = test_pred["y_risk"].values
    y_bin = test_pred["y_bin"].values
    prob_bin = test_pred["student_prob_binary"].values
    atk_probs = test_pred[["student_attack_prob_1", "student_attack_prob_2", "student_attack_prob_3"]].values
    pred_risk = cascade_predict(prob_bin, atk_probs, 0.5)
    m = compute_metrics(y_risk, pred_risk)
    m["Window_Size"] = 32
    m["N_Windows"] = 1702
    m["N_Test"] = len(y_risk)
    results.append(m)
    print(f"  w=32: Acc={m['Accuracy']:.4f}, MF1={m['Macro_F1']:.4f}, HighR={m['High_Recall']:.4f}")
    assert abs(m['Accuracy'] - 0.9375) < 0.001, "Expected 93.75%%, got %.2f%%" % (m['Accuracy']*100)
    print("  [OK] Verified: matches 93.75%%")

    # For other window sizes, run the full pipeline
    for ws in WINDOW_SIZES:
        if ws == 32:
            continue
        print(f"\n--- w={ws}: running full pipeline ---")
        data_dir = str(MAIN_EXP_DIR / f"Dataset/ecu_attack_risk_windows_v3_w{ws}")
        teacher_dir = str(MAIN_EXP_DIR / f"output1/teacher_risk_cascade_v1_w{ws}")
        student_dir = str(MAIN_EXP_DIR / f"output1/student_risk_cascade_v1_w{ws}")

        # Step 1: Data prep
        rc = run_cmd([PYTHON, str(DATA_PREP_SCRIPT),
                       "--input-csv", str(RAW_DATA_CSV),
                       "--output-dir", data_dir,
                       "--window-size", str(ws), "--step-size", str(ws),
                       "--split-strategy", "attack_stratified",
                       "--window-label-rule", "highest_risk",
                       "--keep-short-runs",
                       "--relaxed-split-check",
                       "--allow-missing-attack-type-in-val-test"],
                      desc=f"Data prep w={ws}")
        if rc != 0:
            print(f"  [SKIP] Data prep failed for w={ws}")
            continue

        # Step 2: Teacher training
        rc = run_cmd([PYTHON, str(TEACHER_SCRIPT),
                       "--data-dir", data_dir,
                       "--output-dir", teacher_dir],
                      desc=f"Teacher training w={ws}")
        if rc != 0:
            print(f"  [SKIP] Teacher training failed for w={ws}")
            continue

        # Step 3: Student training
        rc = run_cmd([PYTHON, str(STUDENT_SCRIPT),
                       "--data-dir", data_dir,
                       "--teacher-dir", teacher_dir,
                       "--output-dir", student_dir],
                      desc=f"Student training w={ws}")
        if rc != 0:
            print(f"  [SKIP] Student training failed for w={ws}")
            continue

        # Step 4: Load predictions and evaluate with cascade
        pred_path = Path(student_dir) / "student_risk_preds_test.csv"
        if not pred_path.exists():
            print(f"  [SKIP] No predictions for w={ws}")
            continue
        pred_df = pd.read_csv(pred_path)
        yr = pred_df["y_risk"].values
        pb = pred_df["student_prob_binary"].values
        ap = pred_df[["student_attack_prob_1", "student_attack_prob_2", "student_attack_prob_3"]].values
        pr = cascade_predict(pb, ap, 0.5)
        metrics = compute_metrics(yr, pr)

        # Read manifest for window count
        manifest_path = Path(data_dir) / "manifest.json"
        n_win = 0
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            n_win = manifest.get("n_windows", 0)

        metrics["Window_Size"] = ws
        metrics["N_Windows"] = n_win
        metrics["N_Test"] = len(yr)
        results.append(metrics)
        print(f"  w={ws}: N_win={n_win}, Acc={metrics['Accuracy']:.4f}, "
              f"MF1={metrics['Macro_F1']:.4f}, HighR={metrics['High_Recall']:.4f}")

    # Sort by window size
    results.sort(key=lambda x: x["Window_Size"])

    # Save
    out_csv = REVIEW_BASE / "xiaorong" / "table3_rerun_unified.csv"
    rows = []
    for r in results:
        rows.append({
            "Window_Size": r["Window_Size"], "N_Windows": r["N_Windows"],
            "N_Test": r["N_Test"],
            "Accuracy": r["Accuracy"], "Balanced_Accuracy": r["Balanced_Accuracy"],
            "Macro_F1": r["Macro_F1"], "High_Recall": r["High_Recall"],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # Print summary table
    print(f"\n{'='*70}")
    print("  TABLE 3 RESULTS (attack-stratified, cascade decision)")
    print(f"{'='*70}")
    print(f"{'w':>4s} {'N_Win':>6s} {'N_Test':>7s} {'Acc%':>8s} {'BA%':>8s} {'MF1%':>8s} {'HighR%':>8s}")
    for r in results:
        print(f"{r['Window_Size']:4d} {r['N_Windows']:6d} {r['N_Test']:7d} "
              f"{r['Accuracy']*100:8.2f} {r['Balanced_Accuracy']*100:8.2f} "
              f"{r['Macro_F1']*100:8.2f} {r['High_Recall']*100:8.2f}")

    return results


# ================================================================
#  Table 6: Cost-Ratio (attack-stratified, cascade decision)
# ================================================================
def run_table6():
    print("\n" + "="*70)
    print("  TABLE 6: Cost-Ratio Sensitivity (attack-stratified, cascade)")
    print("="*70)

    # Load pre-saved predictions from output1 (attack-stratified)
    val_pred = pd.read_csv(OUTPUT1_DIR / "student_risk_cascade_v1" / "student_risk_preds_val.csv")
    test_pred = pd.read_csv(OUTPUT1_DIR / "student_risk_cascade_v1" / "student_risk_preds_test.csv")

    y_bin_val = val_pred["y_bin"].values.astype(np.int64)
    y_risk_test = test_pred["y_risk"].values.astype(np.int64)
    y_bin_test = test_pred["y_bin"].values.astype(np.int64)

    prob_bin_val = val_pred["student_prob_binary"].values
    prob_bin_test = test_pred["student_prob_binary"].values
    atk_probs_test = test_pred[["student_attack_prob_1", "student_attack_prob_2", "student_attack_prob_3"]].values

    cost_ratios = [(5, 1), (2, 1), (1, 1), (1, 2), (1, 5)]
    SCAN_STEP = 0.005
    thresholds = np.arange(0.01, 1.0, SCAN_STEP)

    results = []
    for c_fp, c_fn in cost_ratios:
        # Find min-cost interval on validation set
        costs = []
        for tau in thresholds:
            pred = (prob_bin_val >= tau).astype(int)
            fp = ((pred == 1) & (y_bin_val == 0)).sum()
            fn = ((pred == 0) & (y_bin_val == 1)).sum()
            fpr = fp / max((y_bin_val == 0).sum(), 1)
            fnr = fn / max((y_bin_val == 1).sum(), 1)
            costs.append(c_fp * fpr + c_fn * fnr)
        costs = np.array(costs)
        min_cost = costs.min()
        min_mask = costs <= (min_cost + 1e-8)
        min_thrs = thresholds[min_mask]
        lower, upper = float(min_thrs[0]), float(min_thrs[-1])

        # Tie-breaking
        if c_fp > c_fn:
            tau = upper
        elif c_fn > c_fp:
            tau = lower
        else:
            tau = (lower + upper) / 2.0

        # Evaluate on test set using cascade with this tau
        pred_risk = cascade_predict(prob_bin_test, atk_probs_test, tau)
        m = compute_metrics(y_risk_test, pred_risk)

        # Compute expected cost on test set
        pred_bin = (prob_bin_test >= tau).astype(int)
        fp = ((pred_bin == 1) & (y_bin_test == 0)).sum()
        fn = ((pred_bin == 0) & (y_bin_test == 1)).sum()
        fpr = fp / max((y_bin_test == 0).sum(), 1)
        fnr = fn / max((y_bin_test == 1).sum(), 1)
        exp_cost = c_fp * fpr + c_fn * fnr

        result = {
            "C_FP": c_fp, "C_FN": c_fn, "Ratio": f"{c_fp}:{c_fn}",
            "Interval_Lower": round(lower, 3), "Interval_Upper": round(upper, 3),
            "Selected_tau": round(tau, 3),
            "FPR": round(fpr, 4), "FNR": round(fnr, 4),
            "Expected_Cost": round(exp_cost, 4),
            "Accuracy": round(m["Accuracy"], 4),
            "Macro_F1": round(m["Macro_F1"], 4),
            "High_Recall": round(m["High_Recall"], 4),
        }
        results.append(result)
        print(f"  {c_fp}:{c_fn:>2d}  tau=[{lower:.3f},{upper:.3f}]→{tau:.3f}  "
              f"FPR={fpr:.4f} FNR={fnr:.4f} Cost={exp_cost:.4f}  "
              f"Acc={m['Accuracy']:.4f} MF1={m['Macro_F1']:.4f} HighR={m['High_Recall']:.4f}")

    # Also show the main experiment's fixed tau=0.5 baseline
    pred_risk_05 = cascade_predict(prob_bin_test, atk_probs_test, 0.5)
    m05 = compute_metrics(y_risk_test, pred_risk_05)
    print(f"\n  [Reference] Main experiment fixed tau=0.500: "
          f"Acc={m05['Accuracy']:.4f}, MF1={m05['Macro_F1']:.4f}, HighR={m05['High_Recall']:.4f}")

    out_csv = REVIEW_BASE / "FPFN_cost" / "table6_rerun_unified.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    return results


# ================================================================
#  Table 7: Physical Interference Robustness (attack-stratified, cascade)
# ================================================================
def run_table7():
    print("\n" + "="*70)
    print("  TABLE 7: Physical Interference Robustness (attack-stratified, cascade)")
    print("="*70)

    # Import interference functions
    fangzhen_dir = REVIEW_BASE / "fangzhen"
    sys.path.insert(0, str(fangzhen_dir))
    from physical_interference_simulation import (
        apply_channel_fading, apply_emi_noise, apply_packet_loss, apply_packet_reorder,
    )

    # Load pre-saved data and model from output1 (attack-stratified)
    data_dir = MAIN_EXP_DIR / "Dataset" / "ecu_attack_risk_windows_v3"
    student_dir = OUTPUT1_DIR / "student_risk_cascade_v1"
    teacher_dir = OUTPUT1_DIR / "teacher_risk_cascade_v1"

    train_df = pd.read_csv(data_dir / "train_windows.csv")
    val_df = pd.read_csv(data_dir / "val_windows.csv")
    test_df = pd.read_csv(data_dir / "test_windows.csv")

    for df in [train_df, val_df, test_df]:
        df["row_id"] = df["window_id"].astype(np.int64)
        df["binary_label"] = df["y_bin"].astype(np.int64)
        df["label"] = df["y_attack_type_name"].astype(str)

    with open(teacher_dir / "preprocess_info.json", encoding="utf-8") as f:
        prep = json.load(f)
    fill_values = prep["fill_values"]

    # Load model checkpoint to get feature_cols
    device = torch.device("cpu")
    ckpt = torch.load(student_dir / "student_risk_best.pt", map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "feature_cols" in ckpt:
        feature_cols = ckpt["feature_cols"]
    else:
        feature_cols = sanitize_feature_cols(train_df, prep["feature_cols"])

    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].fillna(fill_values)

    # Load scaler
    with open(student_dir / "student_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load model
    model = StudentRiskCascade(len(feature_cols)).to(device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Get test data
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    y_risk_test = test_df["y_risk"].values.astype(np.int64)
    y_bin_test = test_df["y_bin"].values.astype(np.int64)

    N_REPEATS = 5
    FADING_ALPHAS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
    NOISE_SNR_DB = [20.0, 15.0, 10.0, 5.0, 3.0, 0.0]
    LOSS_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    REORDER_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

    def evaluate_perturbed(X_perturbed):
        """Run cascade on perturbed features."""
        X_t = torch.tensor(X_perturbed, dtype=torch.float32).to(device)
        with torch.no_grad():
            logit_bin, logits_atk = model(X_t)
            pb = torch.sigmoid(logit_bin).cpu().numpy()
            ap = torch.softmax(logits_atk, dim=1).cpu().numpy()
        pred_risk = cascade_predict(pb, ap, 0.5)
        return compute_metrics(y_risk_test, pred_risk)

    def run_dimension(dim_name, levels, interference_fn_factory):
        """Run one interference dimension."""
        print(f"\n  {dim_name}:")
        dim_results = []
        for level in levels:
            if level == levels[0] and (level == 1.0 or level == 20.0 or level == 0.0):
                # Baseline (no interference)
                m = evaluate_perturbed(X_test)
                accs = [m["Accuracy"]] * N_REPEATS
            else:
                accs = []
                for run in range(N_REPEATS):
                    rng = np.random.RandomState(SEED + run * 10 + hash(dim_name) % 100)
                    X_pert = interference_fn_factory(level)(X_test, rng)
                    m = evaluate_perturbed(X_pert)
                    accs.append(m["Accuracy"])
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            dim_results.append({
                "dimension": dim_name, "level": level,
                "accuracy_mean": mean_acc, "accuracy_std": std_acc,
            })
            print(f"    level={level}: Acc={mean_acc:.4f}±{std_acc:.4f}")
        return dim_results

    # Baseline
    baseline_m = evaluate_perturbed(X_test)
    baseline_acc = baseline_m["Accuracy"]
    print(f"\n  Baseline (clean): Acc={baseline_acc:.4f}, MF1={baseline_m['Macro_F1']:.4f}, "
          f"HighR={baseline_m['High_Recall']:.4f}")
    assert abs(baseline_acc - 0.9375) < 0.001, "Expected 93.75%%, got %.2f%%" % (baseline_acc*100)
    print("  [OK] Baseline matches 93.75%%")

    all_results = []

    # Channel fading
    all_results.extend(run_dimension("channel_fading", FADING_ALPHAS,
        lambda alpha: lambda X, rng: apply_channel_fading(X, alpha, rng)))

    # EMI noise
    all_results.extend(run_dimension("emi_noise", NOISE_SNR_DB,
        lambda snr: lambda X, rng: apply_emi_noise(X, snr, rng)))

    # Packet loss
    all_results.extend(run_dimension("packet_loss", LOSS_RATES,
        lambda rate: lambda X, rng: apply_packet_loss(X, rate, rng)))

    # Packet reorder
    all_results.extend(run_dimension("packet_reorder", REORDER_RATES,
        lambda rate: lambda X, rng: apply_packet_reorder(X, rate, rng)))

    # Save
    out_csv = fangzhen_dir / "table7_rerun_unified.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)

    # Extreme-level summary
    extreme = []
    for dim in ["channel_fading", "emi_noise", "packet_loss", "packet_reorder"]:
        dim_res = [r for r in all_results if r["dimension"] == dim]
        worst = min(dim_res, key=lambda r: r["accuracy_mean"])
        drop = baseline_acc - worst["accuracy_mean"]
        retention = worst["accuracy_mean"] / baseline_acc if baseline_acc > 0 else 0
        extreme.append({
            "Interference_Type": dim, "Extreme_Setting": str(worst["level"]),
            "Worst_Accuracy": worst["accuracy_mean"], "Accuracy_Drop": drop,
            "Retention": retention,
        })
    avg_ret = float(np.mean([r["Retention"] for r in extreme]))

    summary = {
        "experiment": "Table7 Robustness (attack-stratified, cascade)",
        "baseline_accuracy": baseline_acc,
        "baseline_metrics": baseline_m,
        "extreme_results": extreme,
        "average_retention": avg_ret,
        "all_results": all_results,
    }
    out_json = fangzhen_dir / "table7_rerun_unified_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*70}")
    print("  TABLE 7 SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline: {baseline_acc:.4f}")
    print(f"  {'Type':>20s} {'Worst':>14s} {'Acc':>8s} {'Drop':>8s} {'Retention':>10s}")
    for r in extreme:
        print(f"  {r['Interference_Type']:>20s} {r['Extreme_Setting']:>14s} "
              f"{r['Worst_Accuracy']:8.4f} {r['Accuracy_Drop']:8.4f} {r['Retention']:10.3f}")
    print(f"  {'Average Retention':>44s}: {avg_ret:.3f}")
    print(f"\n  Saved: {out_csv}")
    return all_results, extreme


# ================================================================
#  Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", type=int, default=[3, 6, 7],
                        help="Which tables to re-run (3, 6, 7)")
    args = parser.parse_args()

    set_seed()

    if 3 in args.tables:
        run_table3()

    if 6 in args.tables:
        run_table6()

    if 7 in args.tables:
        run_table7()

    print("\n\n" + "="*70)
    print("  ALL DONE")
    print("="*70)


if __name__ == "__main__":
    main()
