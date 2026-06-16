# -*- coding: utf-8 -*-
"""
Adaptive Window Experiment: Fixed vs Voting vs Confidence vs Oracle
===================================================================
All student models share the same 65-dim feature space (in_dim=65).
So we can directly feed the SAME w=32 test features to ALL models
and compare strategies.

Strategies:
  1. Fixed w=32 (baseline)
  2. Majority Voting (6 models)
  3. Confidence-weighted Voting (practical adaptive)
  4. Max-Confidence Selection (practical adaptive)
  5. Oracle (theoretical upper bound)
"""
import sys, os, json, pickle
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Paths ───────────────────────────────────────────────────────
BASE = str(PROJECT_ROOT / 'src' / 'main_experiment')
OUT1 = os.path.join(BASE, 'output1')
DS = os.path.join(BASE, 'Dataset')

WINDOW_SIZES = [8, 16, 24, 32, 48, 64]
BASE_W = 32

# ── Model Architecture (EXACTLY as in 3.student_risk_cascade_ecu_run_time_compatible.py) ──
class StudentRiskCascade(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),       # backbone.0
            nn.Linear(in_dim, 96),      # backbone.1
            nn.GELU(),                  # backbone.2
            nn.Dropout(0.18),           # backbone.3
            nn.Linear(96, 48),          # backbone.4
            nn.GELU(),                  # backbone.5
            nn.Dropout(0.12),           # backbone.6
        )
        self.head_bin = nn.Linear(48, 1)
        self.head_attack = nn.Linear(48, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.head_bin(z).squeeze(1), self.head_attack(z)

# ── Cascade Decision ────────────────────────────────────────────
ATTACK_TO_RISK = {0: 1, 1: 2, 2: 2}

def cascade_predict(prob_bin, attack_probs, threshold=0.5):
    pred_bin = (prob_bin >= threshold).astype(np.int64)
    pred_attack_internal = attack_probs.argmax(axis=1)
    pred_risk = np.where(
        pred_bin == 0, 0,
        np.array([ATTACK_TO_RISK[int(v)] for v in pred_attack_internal], dtype=np.int64)
    )
    return pred_risk

# ── Load Model ──────────────────────────────────────────────────
def load_model(w):
    suffix = '' if w == BASE_W else '_w' + str(w)
    model_dir = os.path.join(OUT1, 'student_risk_cascade_v1' + suffix)
    ckpt_path = os.path.join(model_dir, 'student_risk_best.pt')
    scaler_path = os.path.join(model_dir, 'student_scaler.pkl')

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    feat_cols = ckpt['feature_cols']
    in_dim = len(feat_cols)  # = 65, NOT w*65

    model = StudentRiskCascade(in_dim)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler, feat_cols

# ── Metrics ─────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y_true, y_pred)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    _, recall_per_class, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    mf1 = float(np.nanmean(f1_per_class))
    high_recall = float(recall_per_class[2]) if len(recall_per_class) > 2 else 0.0
    return acc, mf1, high_recall

# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ADAPTIVE WINDOW EXPERIMENT")
    print("=" * 70)

    # 1. Load w=32 test data
    test_folder = os.path.join(DS, 'ecu_attack_risk_windows_v3')
    test_df = pd.read_csv(os.path.join(test_folder, 'test_windows.csv'))
    N = len(test_df)
    print("\nTest set: " + str(N) + " windows (w=" + str(BASE_W) + ")")

    # 2. Load all models
    models = {}
    scalers = {}
    feat_cols_dict = {}
    for w in WINDOW_SIZES:
        model, scaler, feat_cols = load_model(w)
        models[w] = model
        scalers[w] = scaler
        feat_cols_dict[w] = feat_cols
        print("  w=" + str(w).rjust(2) + ": in_dim=" + str(len(feat_cols)))

    # Verify all use same features
    base_feats = feat_cols_dict[BASE_W]
    for w in WINDOW_SIZES:
        assert feat_cols_dict[w] == base_feats, "w=" + str(w) + " has different features!"
    print("  All models use same " + str(len(base_feats)) + " features [OK]")

    # 3. Prepare test features and labels
    y_true = test_df['y_risk'].values.astype(np.int64)
    X_raw = test_df[base_feats].values.astype(np.float32)
    print("  Features: " + str(X_raw.shape) + ", Labels dist: " +
          str(dict(zip(*np.unique(y_true, return_counts=True)))))

    # 4. Scale with w=32 scaler (all scalers should be equivalent)
    X_scaled = scalers[BASE_W].transform(X_raw).astype(np.float32)
    X_tensor = torch.tensor(X_scaled)

    # 5. Run all models on SAME test data
    print("\nRunning multi-model inference on " + str(N) + " shared samples...")
    all_preds = {}
    all_bin_probs = {}
    all_attack_probs = {}

    with torch.no_grad():
        for w in WINDOW_SIZES:
            model = models[w]
            logit_bin, logit_attack = model(X_tensor)
            prob_bin = torch.sigmoid(logit_bin).numpy()
            attack_probs = torch.softmax(logit_attack, dim=-1).numpy()
            pred_risk = cascade_predict(prob_bin, attack_probs, threshold=0.5)

            all_preds[w] = pred_risk
            all_bin_probs[w] = prob_bin
            all_attack_probs[w] = attack_probs

            acc, mf1, hr = compute_metrics(y_true, pred_risk)
            tag = " [BASELINE]" if w == BASE_W else ""
            print("  w=" + str(w).rjust(2) + ": Acc=" + f"{acc:.4f}" +
                  ", MF1=" + f"{mf1:.4f}" + ", HighR=" + f"{hr:.4f}" + tag)

    # 6. Adaptive Strategies
    print("\n" + "=" * 70)
    print("ADAPTIVE STRATEGIES")
    print("=" * 70)

    # ── S1: Fixed w=32 ──
    pred_fixed = all_preds[BASE_W]
    acc_f, mf1_f, hr_f = compute_metrics(y_true, pred_fixed)
    print("\n  [1] Fixed w=32:       Acc=" + f"{acc_f:.4f}, MF1={mf1_f:.4f}, HighR={hr_f:.4f}")

    # ── S2: Majority Voting ──
    vote_counts = np.zeros((N, 3))
    for w in WINDOW_SIZES:
        for i in range(N):
            vote_counts[i, all_preds[w][i]] += 1
    pred_vote = vote_counts.argmax(axis=1)
    acc_v, mf1_v, hr_v = compute_metrics(y_true, pred_vote)
    print("  [2] Majority Vote:    Acc=" + f"{acc_v:.4f}, MF1={mf1_v:.4f}, HighR={hr_v:.4f}")

    # ── S3: Confidence-weighted Voting ──
    conf_weighted_attack = np.zeros((N, 3))
    conf_weighted_bin = np.zeros(N)
    total_weight = np.zeros(N)
    for w in WINDOW_SIZES:
        bin_conf = np.abs(all_bin_probs[w] - 0.5) * 2.0
        attack_conf = all_attack_probs[w].max(axis=1)
        combined_conf = bin_conf * 0.5 + attack_conf * 0.5
        conf_weighted_attack += all_attack_probs[w] * combined_conf[:, None]
        conf_weighted_bin += all_bin_probs[w] * combined_conf
        total_weight += combined_conf
    conf_weighted_attack /= (total_weight[:, None] + 1e-10)
    conf_weighted_bin /= (total_weight + 1e-10)
    pred_conf = cascade_predict(conf_weighted_bin, conf_weighted_attack, threshold=0.5)
    acc_c, mf1_c, hr_c = compute_metrics(y_true, pred_conf)
    print("  [3] Conf-Weighted:    Acc=" + f"{acc_c:.4f}, MF1={mf1_c:.4f}, HighR={hr_c:.4f}")

    # ── S4: Max-Confidence Selection ──
    pred_maxconf = np.zeros(N, dtype=np.int64)
    for i in range(N):
        best_w = BASE_W
        best_conf = -1
        for w in WINDOW_SIZES:
            bin_conf = abs(all_bin_probs[w][i] - 0.5) * 2.0
            attack_conf = float(all_attack_probs[w][i].max())
            combined = bin_conf * 0.5 + attack_conf * 0.5
            if combined > best_conf:
                best_conf = combined
                best_w = w
        pred_maxconf[i] = all_preds[best_w][i]
    acc_mc, mf1_mc, hr_mc = compute_metrics(y_true, pred_maxconf)
    print("  [4] Max-Conf Select:  Acc=" + f"{acc_mc:.4f}, MF1={mf1_mc:.4f}, HighR={hr_mc:.4f}")

    # ── S5: Oracle (upper bound) ──
    pred_oracle = np.zeros(N, dtype=np.int64)
    for i in range(N):
        found = False
        for w in WINDOW_SIZES:
            if all_preds[w][i] == y_true[i]:
                pred_oracle[i] = y_true[i]
                found = True
                break
        if not found:
            pred_oracle[i] = all_preds[BASE_W][i]
    acc_o, mf1_o, hr_o = compute_metrics(y_true, pred_oracle)
    print("  [5] Oracle (UB):      Acc=" + f"{acc_o:.4f}, MF1={mf1_o:.4f}, HighR={hr_o:.4f}")

    # 7. Summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    hdr = "Strategy".ljust(24) + "Accuracy".rjust(10) + "Macro-F1".rjust(10) + "High-Recall".rjust(12) + "vs Fixed".rjust(10)
    print(hdr)
    print("-" * 68)

    results = [
        ('Fixed w=32 (baseline)', acc_f, mf1_f, hr_f),
        ('Majority Voting', acc_v, mf1_v, hr_v),
        ('Conf-Weighted Vote', acc_c, mf1_c, hr_c),
        ('Max-Conf Selection', acc_mc, mf1_mc, hr_mc),
        ('Oracle (upper bound)', acc_o, mf1_o, hr_o),
    ]
    for name, acc, mf1, hr in results:
        delta = acc - acc_f
        line = name.ljust(24) + f"{acc:.4f}".rjust(10) + f"{mf1:.4f}".rjust(10) + f"{hr:.4f}".rjust(12) + f"{delta:+.4f}".rjust(10)
        print(line)

    # 8. Error Analysis
    print("\n" + "=" * 70)
    print("ORACLE ERROR ANALYSIS")
    print("=" * 70)
    oracle_correct = pred_oracle == y_true
    fixed_correct = pred_fixed == y_true
    oracle_better = oracle_correct & ~fixed_correct
    both_wrong = ~oracle_correct & ~fixed_correct
    print("  Fixed correct:    " + str(int(fixed_correct.sum())) + "/" + str(N))
    print("  Oracle correct:   " + str(int(oracle_correct.sum())) + "/" + str(N))
    print("  Oracle fixes:     " + str(int(oracle_better.sum())) + " additional samples")
    print("  Irreducible err:  " + str(int(both_wrong.sum())) + " samples (all windows wrong)")
    print("  Oracle - Fixed:   " + f"{acc_o - acc_f:.2%}" + " accuracy improvement (ceiling)")

    # Per-window breakdown
    print("\n  Per-window accuracy on shared test set:")
    for w in WINDOW_SIZES:
        correct = int((all_preds[w] == y_true).sum())
        print("    w=" + str(w).rjust(2) + ": " + str(correct) + "/" + str(N) + " (" + f"{correct/N:.2%}" + ")")

    # 9. Save
    out_dir = str(PROJECT_ROOT / 'results' / 'adaptive_window')

    result_data = {'strategies': [], 'per_window': []}
    for name, acc, mf1, hr in results:
        result_data['strategies'].append(
            {'name': name, 'accuracy': round(acc, 4), 'macro_f1': round(mf1, 4), 'high_recall': round(hr, 4)})
    for w in WINDOW_SIZES:
        acc_w, mf1_w, hr_w = compute_metrics(y_true, all_preds[w])
        result_data['per_window'].append(
            {'window_size': w, 'accuracy': round(acc_w, 4), 'macro_f1': round(mf1_w, 4), 'high_recall': round(hr_w, 4)})
    result_data['oracle_analysis'] = {
        'fixed_correct': int(fixed_correct.sum()),
        'oracle_correct': int(oracle_correct.sum()),
        'oracle_fixes': int(oracle_better.sum()),
        'irreducible_errors': int(both_wrong.sum()),
    }

    out_json = os.path.join(out_dir, 'adaptive_window_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print("\n[OK] JSON: " + out_json)

    out_csv = os.path.join(out_dir, 'adaptive_window_comparison.csv')
    rows = []
    for name, acc, mf1, hr in results:
        rows.append({'Strategy': name, 'Accuracy': f'{acc:.4f}', 'Macro_F1': f'{mf1:.4f}', 'High_Recall': f'{hr:.4f}'})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("[OK] CSV: " + out_csv)


if __name__ == '__main__':
    main()
