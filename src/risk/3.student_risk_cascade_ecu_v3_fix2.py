#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ALWAYS_DROP_EXACT = {
    "y_bin", "y_attack_type", "y_attack_type_name", "y_risk", "y_risk_name",
    "attack_packet_count", "attack_packet_ratio",
    "window_id", "row_id", "binary_label", "label", "split", "session_id",
    "time_start", "time_end", "packet_id_start", "packet_id_end",
    "attack_scenario_meta", "packet_attack_type_mode_meta",
}
ALWAYS_DROP_PREFIX = ("cnt_",)
ALWAYS_DROP_CONTAINS = ("label", "target")
RISK_ID_TO_NAME = {0: "low", 1: "medium", 2: "high"}
ATTACK_TO_RISK = {0: 1, 1: 2, 2: 2}  # student attack head internal class ids 0/1/2 for original 1/2/3
ATTACK_INTERNAL_TO_ORIG = {0: 1, 1: 2, 2: 3}
ATTACK_NAME_MAP = {0: "No Attack", 1: "Wifi Deauthentication Attack", 2: "WPA2-PSK WIFI Cracking Attack", 3: "TELLO API Exploit"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_prepared_splits(data_dir: str):
    base = Path(data_dir)
    train_df = pd.read_csv(base / "train_windows.csv")
    val_df = pd.read_csv(base / "val_windows.csv")
    test_df = pd.read_csv(base / "test_windows.csv")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _ensure_required_columns(df, name)
    return _standardize_df(train_df), _standardize_df(val_df), _standardize_df(test_df)


def sanitize_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    keep: List[str] = []
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
    if not keep:
        raise ValueError("student 过滤后 feature_cols 为空")
    return keep


class RiskCascadeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], scaler: StandardScaler, teacher_df: pd.DataFrame):
        df = df.copy()
        merge_cols = [
            "row_id", "teacher_prob_binary",
            "teacher_attack_prob_1", "teacher_attack_prob_2", "teacher_attack_prob_3",
        ]
        df = df.merge(teacher_df[merge_cols], on="row_id", how="left")
        miss = df["teacher_prob_binary"].isna().sum()
        if miss > 0:
            raise RuntimeError(f"teacher 预测与 split 行对齐失败，缺失 {int(miss)} 行")

        X = scaler.transform(df[feature_cols].values.astype(np.float32))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_bin = torch.tensor(df["binary_label"].values.astype(np.float32), dtype=torch.float32)
        attack_orig = df["y_attack_type"].values.astype(np.int64)
        # normal 样本的 y_attack_type 通常为 0；不能直接减 1 变成 -1 后送入 CrossEntropy
        # 因此这里先把 abnormal 的 1/2/3 映射为内部 0/1/2，normal 统一放一个占位 0，后续再用 abn_mask 屏蔽
        attack_internal = np.where(df["binary_label"].values.astype(np.int64) == 1, attack_orig - 1, 0).astype(np.int64)
        self.y_attack = torch.tensor(attack_internal, dtype=torch.long)
        self.y_risk = torch.tensor(df["y_risk"].values.astype(np.int64), dtype=torch.long)
        self.teacher_bin = torch.tensor(df["teacher_prob_binary"].values.astype(np.float32), dtype=torch.float32)
        self.teacher_attack = torch.tensor(df[["teacher_attack_prob_1", "teacher_attack_prob_2", "teacher_attack_prob_3"]].values.astype(np.float32), dtype=torch.float32)
        self.abn_mask = torch.tensor(df["binary_label"].values.astype(np.float32), dtype=torch.float32)

        y_bin_np = df["binary_label"].values.astype(np.int64)
        n_neg = max(1, int((y_bin_np == 0).sum()))
        n_pos = max(1, int((y_bin_np == 1).sum()))
        total = n_neg + n_pos
        w_neg = total / (2.0 * n_neg)
        w_pos = total / (2.0 * n_pos)
        self.w_bin = torch.tensor(np.where(y_bin_np == 1, w_pos, w_neg).astype(np.float32), dtype=torch.float32)

        abn = df.loc[df["binary_label"] == 1, "y_attack_type"].values.astype(np.int64) - 1
        classes, counts = np.unique(abn, return_counts=True)
        total_abn = counts.sum() if len(counts) else 1
        w_map = {int(c): float(total_abn / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
        attack_w = np.ones(len(df), dtype=np.float32)
        for i, (is_abn, attack_id) in enumerate(zip(y_bin_np, df["y_attack_type"].values.astype(np.int64) - 1)):
            if is_abn == 1:
                attack_w[i] = w_map[int(attack_id)]
        self.w_attack = torch.tensor(attack_w, dtype=torch.float32)

        self.row_id = df["row_id"].values.astype(np.int64)
        self.window_id = df["window_id"].values.astype(np.int64)
        self.session_id = df["session_id"].astype(str).values if "session_id" in df.columns else np.array([""] * len(df))
        self.y_attack_orig = df["y_attack_type"].values.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return {
            "x": self.X[idx],
            "y_bin": self.y_bin[idx],
            "y_attack": self.y_attack[idx],
            "y_risk": self.y_risk[idx],
            "teacher_bin": self.teacher_bin[idx],
            "teacher_attack": self.teacher_attack[idx],
            "abn_mask": self.abn_mask[idx],
            "w_bin": self.w_bin[idx],
            "w_attack": self.w_attack[idx],
        }


class StudentRiskCascade(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 96),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(0.12),
        )
        self.head_bin = nn.Linear(48, 1)
        self.head_attack = nn.Linear(48, 3)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        return self.head_bin(z).squeeze(1), self.head_attack(z)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * weight
    return loss.sum() / weight.sum().clamp_min(1e-8)


def masked_weighted_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n_classes = logits.size(1)
    if torch.any((target < 0) | (target >= n_classes)):
        bad = target[(target < 0) | (target >= n_classes)][:10].detach().cpu().tolist()
        raise ValueError(f"attack target 越界，需在 [0, {n_classes-1}]，但发现 {bad}")
    ce = F.cross_entropy(logits, target, reduction="none")
    ce = ce * weight * mask
    return ce.sum() / (mask * weight).sum().clamp_min(1e-8)


def masked_attack_kd(student_logits: torch.Tensor, teacher_probs: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = torch.clamp(teacher_probs, 1e-6, 1.0)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if torch.isnan(q).any() or torch.isinf(q).any():
        raise ValueError("teacher attack probs 出现 NaN/Inf")
    kl = F.kl_div(log_p, q, reduction="none").sum(dim=1)
    kl = kl * mask
    return (kl.sum() / mask.sum().clamp_min(1e-8)) * (temperature ** 2)


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


def build_loader(df: pd.DataFrame, feature_cols: List[str], scaler: StandardScaler, teacher_df: pd.DataFrame, batch_size: int, shuffle: bool):
    ds = RiskCascadeDataset(df=df, feature_cols=feature_cols, scaler=scaler, teacher_df=teacher_df)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False), ds


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    agg = {"total": 0.0, "hard_bin": 0.0, "kd_bin": 0.0, "hard_attack": 0.0, "kd_attack": 0.0}
    steps = 0
    for batch in loader:
        x = batch["x"].to(device)
        y_bin = batch["y_bin"].to(device)
        y_attack = batch["y_attack"].to(device)
        teacher_bin = batch["teacher_bin"].to(device)
        teacher_attack = batch["teacher_attack"].to(device)
        abn_mask = batch["abn_mask"].to(device)
        w_bin = batch["w_bin"].to(device)
        w_attack = batch["w_attack"].to(device)

        logit_bin, logits_attack = model(x)
        loss_hard_bin = weighted_bce(logit_bin, y_bin, w_bin)
        loss_kd_bin = ((torch.sigmoid(logit_bin / args.temperature) - teacher_bin) ** 2 * w_bin).sum() / w_bin.sum().clamp_min(1e-8)
        loss_hard_attack = masked_weighted_ce(logits_attack, y_attack, abn_mask, w_attack)
        loss_kd_attack = masked_attack_kd(logits_attack, teacher_attack, abn_mask, args.temperature)
        loss = (
            args.alpha_bin * loss_hard_bin +
            args.beta_bin * loss_kd_bin +
            args.alpha_attack * loss_hard_attack +
            args.beta_attack * loss_kd_attack
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        agg["total"] += float(loss.item())
        agg["hard_bin"] += float(loss_hard_bin.item())
        agg["kd_bin"] += float(loss_kd_bin.item())
        agg["hard_attack"] += float(loss_hard_attack.item())
        agg["kd_attack"] += float(loss_kd_attack.item())
        steps += 1
    for k in agg:
        agg[k] /= max(1, steps)
    return agg


@torch.no_grad()
def predict_all(model, loader, device, bin_threshold: float = 0.5):
    model.eval()
    prob_bin_all, pred_attack_all = [], []
    attack_prob_all = []
    for batch in loader:
        x = batch["x"].to(device)
        logit_bin, logits_attack = model(x)
        prob_bin = torch.sigmoid(logit_bin)
        prob_attack = torch.softmax(logits_attack, dim=1)
        pred_attack = prob_attack.argmax(dim=1)
        prob_bin_all.append(prob_bin.cpu().numpy())
        attack_prob_all.append(prob_attack.cpu().numpy())
        pred_attack_all.append(pred_attack.cpu().numpy())
    prob_bin = np.concatenate(prob_bin_all)
    attack_prob = np.concatenate(attack_prob_all)
    pred_attack = np.concatenate(pred_attack_all)
    pred_bin = (prob_bin >= bin_threshold).astype(np.int64)
    pred_attack_orig = np.array([ATTACK_INTERNAL_TO_ORIG[int(v)] for v in pred_attack], dtype=np.int64)
    pred_attack_orig = np.where(pred_bin == 0, 0, pred_attack_orig)
    pred_risk = np.where(pred_bin == 0, 0, np.array([ATTACK_TO_RISK[int(v)] for v in pred_attack], dtype=np.int64))
    return prob_bin, pred_bin, attack_prob, pred_attack_orig, pred_risk


def eval_dataset(model, loader, ds: RiskCascadeDataset, device, bin_threshold: float):
    prob_bin, pred_bin, attack_prob, pred_attack, pred_risk = predict_all(model, loader, device, bin_threshold)
    binary_metrics = compute_binary_metrics(ds.y_bin.numpy().astype(np.int64), prob_bin, threshold=bin_threshold)
    risk_metrics = compute_multiclass_metrics(ds.y_risk.numpy().astype(np.int64), pred_risk)
    risk_metrics["labels"] = [0, 1, 2]
    risk_metrics["confusion_matrix"] = confusion_matrix(ds.y_risk.numpy().astype(np.int64), pred_risk, labels=[0,1,2]).tolist()
    return binary_metrics, risk_metrics, prob_bin, pred_bin, attack_prob, pred_attack, pred_risk


def main() -> None:
    parser = argparse.ArgumentParser(description="ECU-IoFT student cascade: binary + attack-type -> low/medium/high risk")
    parser.add_argument("--data-dir", type=str, default="Dataset_w25/ecu_attack_risk_windows_v3")
    parser.add_argument("--teacher-dir", type=str, default="output1/teacher_risk_cascade_v1")
    parser.add_argument("--output-dir", type=str, default="output1/student_risk_cascade_v1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-bin", type=float, default=0.90)
    parser.add_argument("--beta-bin", type=float, default=0.10)
    parser.add_argument("--alpha-attack", type=float, default=1.00)
    parser.add_argument("--beta-attack", type=float, default=0.20)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_df, val_df, test_df = load_prepared_splits(args.data_dir)
    with open(Path(args.teacher_dir) / "preprocess_info.json", "r", encoding="utf-8") as f:
        prep = json.load(f)
    feature_cols = sanitize_feature_cols(train_df, prep["feature_cols"])
    fill_values = prep["fill_values"]
    args.binary_threshold = float(prep.get("binary_threshold", args.binary_threshold))

    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].fillna(fill_values)

    teacher_train = pd.read_csv(Path(args.teacher_dir) / "teacher_risk_preds_train.csv")
    teacher_val = pd.read_csv(Path(args.teacher_dir) / "teacher_risk_preds_val.csv")
    teacher_test = pd.read_csv(Path(args.teacher_dir) / "teacher_risk_preds_test.csv")

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values.astype(np.float32))
    with open(Path(args.output_dir) / "student_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_loader, train_ds = build_loader(train_df, feature_cols, scaler, teacher_train, args.batch_size, shuffle=True)
    train_eval_loader, train_eval_ds = build_loader(train_df, feature_cols, scaler, teacher_train, args.batch_size, shuffle=False)
    val_loader, val_ds = build_loader(val_df, feature_cols, scaler, teacher_val, args.batch_size, shuffle=False)
    test_loader, test_ds = build_loader(test_df, feature_cols, scaler, teacher_test, args.batch_size, shuffle=False)

    model = StudentRiskCascade(len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5)

    best_state = None
    best_epoch = -1
    best_score = -1.0
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args)
        train_bin_m, train_risk_m, *_ = eval_dataset(model, train_eval_loader, train_eval_ds, device, args.binary_threshold)
        val_bin_m, val_risk_m, *_ = eval_dataset(model, val_loader, val_ds, device, args.binary_threshold)
        score = val_risk_m["f1_macro"]
        scheduler.step(score)

        history.append({
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"loss_{k}": v for k, v in train_loss.items()},
            **{f"train_binary_{k}": v for k, v in train_bin_m.items()},
            **{f"train_risk_{k}": v for k, v in train_risk_m.items() if k != "confusion_matrix"},
            **{f"val_binary_{k}": v for k, v in val_bin_m.items()},
            **{f"val_risk_{k}": v for k, v in val_risk_m.items() if k != "confusion_matrix"},
            "score": float(score),
        })

        print(
            f"[StudentRisk][Epoch {epoch:03d}] loss={train_loss['total']:.6f} "
            f"val_bin_bacc={val_bin_m['balanced_accuracy']:.4f} val_risk_f1={val_risk_m['f1_macro']:.4f} "
            f"val_risk_bacc={val_risk_m['balanced_accuracy']:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            print(f"[StudentRisk] Early stopping at epoch={epoch}")
            break

    if best_state is None:
        raise RuntimeError("未得到有效 best_state")

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "binary_threshold": args.binary_threshold,
    }, Path(args.output_dir) / "student_risk_best.pt")
    pd.DataFrame(history).to_csv(Path(args.output_dir) / "student_risk_history.csv", index=False)

    summary = {
        "task_design": {
            "student": "shared backbone + binary head + attack-type head",
            "final_risk": "if binary=normal -> low else map(pred_attack_type) to medium/high",
        },
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "binary": {},
        "final_risk": {},
    }

    for split_name, loader, ds in [("train", train_eval_loader, train_eval_ds), ("val", val_loader, val_ds), ("test", test_loader, test_ds)]:
        bin_m, risk_m, prob_bin, pred_bin, attack_prob, pred_attack, pred_risk = eval_dataset(model, loader, ds, device, args.binary_threshold)
        summary["binary"][split_name] = bin_m
        summary["final_risk"][split_name] = risk_m

        out = pd.DataFrame({
            "row_id": ds.row_id,
            "window_id": ds.window_id,
            "session_id": ds.session_id,
            "y_bin": ds.y_bin.numpy().astype(np.int64),
            "y_attack_type": ds.y_attack_orig,
            "y_attack_type_name": np.array([ATTACK_NAME_MAP.get(int(v), "Unknown") for v in ds.y_attack_orig]),
            "y_risk": ds.y_risk.numpy().astype(np.int64),
            "y_risk_name": np.array([RISK_ID_TO_NAME[int(v)] for v in ds.y_risk.numpy().astype(np.int64)]),
            "student_prob_binary": prob_bin.astype(np.float32),
            "student_pred_binary": pred_bin.astype(np.int64),
            "student_attack_prob_1": attack_prob[:, 0].astype(np.float32),
            "student_attack_prob_2": attack_prob[:, 1].astype(np.float32),
            "student_attack_prob_3": attack_prob[:, 2].astype(np.float32),
            "student_pred_attack_type": pred_attack.astype(np.int64),
            "student_pred_attack_type_name": np.array([ATTACK_NAME_MAP.get(int(v), "Unknown") for v in pred_attack]),
            "student_pred_risk": pred_risk.astype(np.int64),
            "student_pred_risk_name": np.array([RISK_ID_TO_NAME[int(v)] for v in pred_risk]),
        })
        out.to_csv(Path(args.output_dir) / f"student_risk_preds_{split_name}.csv", index=False)

    safe_json_dump(summary, Path(args.output_dir) / "student_risk_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
