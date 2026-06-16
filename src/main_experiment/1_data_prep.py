#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ECU-IoFT window-level data construction and model-compatible splitting script
(supports attack-stratified / run-level / time-level splits).

Core objectives:
1) Generate train_windows.csv / val_windows.csv / test_windows.csv compatible with
   the teacher/student training code.
2) Ensure all three splits (train/val/test) contain:
   - Normal samples: y_attack_type=0, y_risk=0
   - Low/medium-risk attack samples: y_attack_type=1, y_risk=1
   - High-risk attack samples: y_attack_type=2 or 3, y_risk=2
3) Additionally ensure attack types 1/2/3 required by the teacher attack-type model
   are all present in the training set; by default also require all four attack types
   0/1/2/3 to appear in train/val/test.

Notes:
- In the current teacher/student code, risk id=1 is named "medium"; if the paper
  refers to it as "low risk", clarify in the text that this class represents
  lower-severity / medium-low-risk attacks, while keeping id=1 unchanged in code.
- TELLO API Exploit has very few windows; under attack-stratified splitting, rare-class
  per-window assignment is used; under run/time-level splits, class absence or sample
  sparsity should be reported honestly in the paper.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ATTACK_TYPE_MAP = {
    "No Attack": 0,
    "Wifi Deauthentication Attack": 1,
    "WPA2-PSK WIFI Cracking Attack": 2,
    "TELLO API Exploit": 3,
}

ATTACK_ID_TO_NAME = {v: k for k, v in ATTACK_TYPE_MAP.items()}

# ID compatibility with teacher/student: 0=normal, 1=medium/low-risk attack, 2=high
RISK_MAP = {
    "No Attack": (0, "normal"),
    "Wifi Deauthentication Attack": (1, "medium"),
    "WPA2-PSK WIFI Cracking Attack": (2, "high"),
    "TELLO API Exploit": (2, "high"),
}

RISK_ID_TO_NAME = {0: "normal", 1: "medium", 2: "high"}
RISK_ALIAS_FOR_REPORT = {0: "normal", 1: "low_or_medium", 2: "high"}

ATTACK_PRIORITY = {
    "No Attack": 0,
    "Wifi Deauthentication Attack": 1,
    "WPA2-PSK WIFI Cracking Attack": 2,
    "TELLO API Exploit": 3,
}
RISK_PRIORITY = {name: RISK_MAP[name][0] for name in RISK_MAP}

INFO_TOKENS = [
    "ack", "deauthentication", "authentication", "eapol", "beacon",
    "probe", "request", "response", "data", "null", "qos", "udp",
    "icmp", "exploit", "api", "sae", "handshake",
]

PROTO_VALUES = ["802.11", "UDP", "EAPOL", "ICMP"]


# -----------------------------
# Utility Functions
# -----------------------------
def safe_json_dump(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_string(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def parse_time(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.isna().any():
        bad = s[dt.isna()].head(10).tolist()
        raise ValueError(f"Column Time contains unparseable values, count={int(dt.isna().sum())}, examples={bad}")
    return dt


def dominant_attack_label(labels: pd.Series) -> str:
    """Window label uses the dominant attack type; on tie, prefer attack over No Attack."""
    vc = labels.value_counts()
    if vc.empty:
        return "No Attack"
    if len(vc) == 1:
        return str(vc.index[0])
    items = sorted(vc.items(), key=lambda kv: (kv[1], kv[0] != "No Attack"), reverse=True)
    return str(items[0][0])


def select_window_attack_label(labels: pd.Series, label_rule: str = "dominant") -> str:
    """
    Select window-level attack label.

    dominant:
        Majority vote. Suitable for general classification but may swallow short attacks.
    any_attack:
        If any attack exists in the window, choose the dominant attack type among
        attack packets within the window.
    highest_risk:
        Select window label by risk level. Suitable for risk early warning:
        No Attack < Deauth < WPA2/TELLO. Within the same risk level, choose the
        more specific / higher-priority attack by ATTACK_PRIORITY.
    """
    vals = normalize_string(labels)
    vals = vals[vals != ""]
    if vals.empty:
        return "No Attack"

    unknown = sorted(set(vals.unique().tolist()) - set(ATTACK_TYPE_MAP.keys()))
    if unknown:
        raise ValueError(f"Unknown Type of Attack: {unknown[:10]}")

    if label_rule == "dominant":
        return dominant_attack_label(vals)

    attack_vals = vals[vals != "No Attack"]
    if attack_vals.empty:
        return "No Attack"

    if label_rule == "any_attack":
        return dominant_attack_label(attack_vals)

    if label_rule == "highest_risk":
        # First by risk level, then by attack priority; better for risk early warning,
        # preventing short attacks from being overridden by majority normal packets.
        return max(attack_vals.tolist(), key=lambda x: (RISK_PRIORITY.get(x, -1), ATTACK_PRIORITY.get(x, -1)))

    raise ValueError(f"Unknown window label rule: label_rule={label_rule}")


def ratio_bool(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.mean())


def q(arr: np.ndarray, v: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, v))


def calc_entropy(values: pd.Series) -> float:
    vc = values.value_counts(normalize=True)
    if vc.empty:
        return 0.0
    p = vc.values.astype(np.float64)
    return float(-(p * np.log2(np.clip(p, 1e-12, 1.0))).sum())


# -----------------------------
# Window Feature Construction
# -----------------------------
def build_features_one_window(sub: pd.DataFrame, window_id: int, label_rule: str = "dominant") -> Dict:
    lengths = sub["Length"].astype(float).values
    info = normalize_string(sub["Info"])
    proto = normalize_string(sub["Protocol"])
    src = normalize_string(sub["Source"])
    dst = normalize_string(sub["Destination"])
    atk = normalize_string(sub["Type of Attack"])
    typ = normalize_string(sub["Type"])
    dt = sub["_dt"]

    if "_raw_run_id" in sub.columns:
        run_ids = normalize_string(sub["_raw_run_id"])
    else:
        run_ids = pd.Series(["unknown"] * len(sub), index=sub.index)
    run_vc = run_ids.value_counts()
    window_run_id = str(run_vc.index[0]) if not run_vc.empty else "unknown"
    window_run_count = int(run_ids.nunique())
    window_run_mixed = int(window_run_count > 1)

    dominant = select_window_attack_label(atk, label_rule=label_rule)
    if dominant not in ATTACK_TYPE_MAP:
        raise ValueError(f"Unknown Type of Attack: {dominant}")

    y_attack_type = ATTACK_TYPE_MAP[dominant]
    y_bin = int(dominant != "No Attack")
    y_risk, y_risk_name = RISK_MAP[dominant]

    info_len = info.str.len().values.astype(float)
    src_nonempty = src[src != ""].dropna()
    dst_nonempty = dst[dst != ""].dropna()
    proto_nonempty = proto[proto != ""].dropna()

    feat = {
        "window_id": int(window_id),
        "row_id": int(window_id),
        "packet_id_start": int(sub.index[0]),
        "packet_id_end": int(sub.index[-1]),
        "time_start": str(dt.iloc[0]),
        "time_end": str(dt.iloc[-1]),
        # run/time-level splits require these window-level run audit fields.
        # window_run_id comes from the dominant value of raw packet-level _raw_run_id;
        # window_run_mixed=1 means this window spans multiple runs, should be avoided under strict run-level.
        "window_run_id": window_run_id,
        "window_run_count": int(window_run_count),
        "window_run_mixed": int(window_run_mixed),
        "window_packet_count": int(len(sub)),
        "duration_sec": float((dt.iloc[-1] - dt.iloc[0]).total_seconds()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "length_min": float(lengths.min()),
        "length_max": float(lengths.max()),
        "length_q25": q(lengths, 0.25),
        "length_q50": q(lengths, 0.50),
        "length_q75": q(lengths, 0.75),
        "info_len_mean": float(info_len.mean()),
        "info_len_std": float(info_len.std(ddof=0)),
        "info_len_max": float(info_len.max()),
        "src_unique": int(src_nonempty.nunique()),
        "dst_unique": int(dst_nonempty.nunique()),
        "proto_unique": int(proto_nonempty.nunique()),
        "src_entropy": calc_entropy(src_nonempty),
        "dst_entropy": calc_entropy(dst_nonempty),
        "proto_entropy": calc_entropy(proto_nonempty),
        "missing_source_ratio": ratio_bool(src.eq("")),
        "broadcast_dst_ratio": ratio_bool(dst.str.contains("ff:ff:ff:ff:ff:ff", case=False, regex=False)),
        "ra_marker_ratio": ratio_bool(dst.str.contains(r"\(RA\)", case=False, regex=True)),
        "bssid_marker_ratio": ratio_bool(
            src.str.contains(r"\(BSSID\)", case=False, regex=True)
            | dst.str.contains(r"\(BSSID\)", case=False, regex=True)
        ),
        "ip_endpoint_ratio": ratio_bool(
            src.str.contains(r"^\d+\.\d+\.\d+\.\d+$", regex=True)
            | dst.str.contains(r"^\d+\.\d+\.\d+\.\d+$", regex=True)
        ),
        # These two fields are dropped by teacher/student ALWAYS_DROP_EXACT; kept in the table for auditing only.
        "attack_packet_count": int((typ == "Attack").sum()),
        "attack_packet_ratio": ratio_bool(typ == "Attack"),
        "window_label_rule": label_rule,
        "window_no_attack_count": int((atk == "No Attack").sum()),
        "window_deauth_count": int((atk == "Wifi Deauthentication Attack").sum()),
        "window_wpa2_count": int((atk == "WPA2-PSK WIFI Cracking Attack").sum()),
        "window_tello_count": int((atk == "TELLO API Exploit").sum()),
        "y_bin": int(y_bin),
        "y_attack_type": int(y_attack_type),
        "y_attack_type_name": dominant,
        "y_risk": int(y_risk),
        "y_risk_name": y_risk_name,
        "attack_scenario_meta": dominant_attack_label(normalize_string(sub["Attack Scenario"])),
        "packet_attack_type_mode_meta": dominant,
        "session_id": "",
        "split": "",
    }

    for p in PROTO_VALUES:
        cnt = int((proto == p).sum())
        key = re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_").lower()
        feat[f"proto_{key}_count"] = cnt
        feat[f"proto_{key}_ratio"] = float(cnt / len(sub))

    info_lower = info.str.lower()
    for tok in INFO_TOKENS:
        mask = info_lower.str.contains(tok, regex=False)
        cnt = int(mask.sum())
        feat[f"info_tok_{tok}_count"] = cnt
        feat[f"info_tok_{tok}_ratio"] = float(cnt / len(sub))

    return feat


def build_raw_run_ids(raw: pd.DataFrame, run_col: str = "", time_gap_sec: float = 60.0) -> pd.DataFrame:
    """
    Construct raw packet-level run IDs.

    Prefer explicit run/session/capture columns if available; otherwise, automatically
    segment using consecutive runs of (Attack Scenario + Type of Attack) combined with
    a time-gap threshold. This automatic run ID is only a weak proxy and is not
    equivalent to real capture sessions.
    """
    out = raw.copy()
    if run_col and run_col in out.columns:
        out["_raw_run_id"] = normalize_string(out[run_col]).replace("", "unknown_run")
        return out

    scenario = normalize_string(out["Attack Scenario"]) if "Attack Scenario" in out.columns else pd.Series([""] * len(out), index=out.index)
    attack = normalize_string(out["Type of Attack"]) if "Type of Attack" in out.columns else pd.Series([""] * len(out), index=out.index)
    key = scenario + "||" + attack
    dt_gap = out["_dt"].diff().dt.total_seconds().fillna(0.0).astype(float)
    new_run = key.ne(key.shift(1)) | (dt_gap > float(time_gap_sec))
    run_no = new_run.cumsum().astype(int)
    out["_raw_run_id"] = "run_" + run_no.astype(str).str.zfill(4) + "__" + attack.str.replace(r"[^a-zA-Z0-9]+", "_", regex=True).str.strip("_")
    return out


def build_windows(df: pd.DataFrame, window_size: int, step_size: int, within_run: bool = False, keep_short_runs: bool = False, label_rule: str = "dominant") -> pd.DataFrame:
    rows: List[Dict] = []
    wid = 0

    if within_run:
        if "_raw_run_id" not in df.columns:
            raise ValueError("when within_run=True, build_raw_run_ids must be called first to generate _raw_run_id")
        iterable = [g for _, g in df.groupby("_raw_run_id", sort=False)]
    else:
        iterable = [df]

    for part in iterable:
        part = part.sort_index()
        if len(part) < window_size:
            if keep_short_runs and len(part) > 0:
                # Under strict run-level, discarding short attack runs would cause some risk classes
                # to disappear globally. Allow short runs to produce a variable-length window here;
                # features are still statistics, and the model can sense window length via window_packet_count.
                sub = part
                rows.append(build_features_one_window(sub, window_id=wid, label_rule=label_rule))
                wid += 1
            continue
        for start in range(0, len(part) - window_size + 1, step_size):
            sub = part.iloc[start:start + window_size]
            rows.append(build_features_one_window(sub, window_id=wid, label_rule=label_rule))
            wid += 1

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Windows are empty; reduce window_size, disable --window-within-run, or check run lengths after segmentation")
    return out


# -----------------------------
# Model-Compatible Split Logic
# -----------------------------
def _split_counts_for_class(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """
    Train/val/test counts within a single class.
    When n>=3, force all three to have at least 1, ensuring rare classes are not lost.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, n - n_train - n_val)

    # Shrink to total count n.
    while n_train + n_val + n_test > n:
        # Prefer deducting from train, since val/test must keep at least 1.
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    while n_train + n_val + n_test < n:
        n_train += 1

    return n_train, n_val, n_test


def assign_splits_model_compatible(
    win_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_train_tello: int = 1,
) -> pd.DataFrame:
    """
    Stratified split by y_attack_type.

    Why not split only by y_risk?
    - The teacher attack-type model outputs teacher_attack_prob_1/2/3;
    - If train is missing y_attack_type=3, the original teacher code will error on index(3);
    - Therefore attack types 1/2/3 must be present in the training set.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    out = win_df.copy().sort_values("packet_id_start").reset_index(drop=True)
    out["split"] = ""
    out["session_id"] = ""

    required_attack_types = [0, 1, 2, 3]
    available_attack_types = sorted(out["y_attack_type"].dropna().astype(int).unique().tolist())
    missing_global = [a for a in required_attack_types if a not in available_attack_types]
    if missing_global:
        raise ValueError(
            f"After windowing, global attack types {missing_global} are missing, "
            f"cannot satisfy the fixed 3-attack-head requirement of teacher/student. "
            f"Reduce --window-size or check raw labels. Available={available_attack_types}"
        )

    for attack_id, sub in out.groupby("y_attack_type", sort=True):
        attack_id = int(attack_id)
        idxs = sub.sort_values("packet_id_start").index.tolist()
        n = len(idxs)
        n_train, n_val, n_test = _split_counts_for_class(n, train_ratio, val_ratio, test_ratio)

        # TELLO is very rare; ensure the training set retains at least min_train_tello samples.
        if attack_id == 3 and n >= 3:
            n_train = max(n_train, min_train_tello)
            while n_train + n_val + n_test > n:
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    n_train -= 1

        name = ATTACK_ID_TO_NAME.get(attack_id, f"attack_{attack_id}")
        ordered = idxs
        for i, idx in enumerate(ordered):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            out.loc[idx, "split"] = split
            out.loc[idx, "session_id"] = f"atk{attack_id}_{name.replace(' ', '_')}__win_{i:05d}"

    if (out["split"] == "").any():
        raise RuntimeError("Some windows have unassigned splits")

    return out



def _assign_by_counts_to_indices(idxs: List[int], train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[int, str]:
    n = len(idxs)
    n_train, n_val, n_test = _split_counts_for_class(n, train_ratio, val_ratio, test_ratio)
    mapping: Dict[int, str] = {}
    for i, idx in enumerate(idxs):
        if i < n_train:
            mapping[idx] = "train"
        elif i < n_train + n_val:
            mapping[idx] = "val"
        else:
            mapping[idx] = "test"
    return mapping


def assign_splits_run_level(
    win_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_col: str = "window_run_id",
) -> pd.DataFrame:
    """
    Run-level split: all windows from the same run/session must go to the same split.
    To accommodate teacher/student training requirements, windows are first stratified
    by the dominant y_attack_type of each run, then assigned at the run level.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    if group_col not in win_df.columns:
        raise ValueError(f"run-level split requires column {group_col}; construct run IDs first")

    out = win_df.copy().sort_values(["packet_id_start", "window_id"]).reset_index(drop=True)
    out["split"] = ""
    out["session_id"] = out[group_col].astype(str) + "__win_" + out.groupby(group_col).cumcount().astype(str).str.zfill(5)

    group_rows = []
    for gid, sub in out.groupby(group_col, sort=False):
        # If a run spans multiple labels, the raw run construction is too coarse;
        # use the window majority label as the dominant type here.
        attack_mode = int(sub["y_attack_type"].mode().iloc[0])
        risk_mode = int(sub["y_risk"].mode().iloc[0])
        group_rows.append({
            "group_id": gid,
            "start": int(sub["packet_id_start"].min()),
            "n_windows": int(len(sub)),
            "attack_mode": attack_mode,
            "risk_mode": risk_mode,
        })
    groups = pd.DataFrame(group_rows).sort_values("start").reset_index(drop=True)

    group_split: Dict[str, str] = {}
    for attack_id, subg in groups.groupby("attack_mode", sort=True):
        idxs = subg.sort_values("start").index.tolist()
        idx_to_split = _assign_by_counts_to_indices(idxs, train_ratio, val_ratio, test_ratio)
        for gi, sp in idx_to_split.items():
            group_split[str(groups.loc[gi, "group_id"])] = sp

    out["split"] = out[group_col].astype(str).map(group_split).fillna("")
    if (out["split"] == "").any():
        raise RuntimeError("run-level split has windows with unassigned splits")
    return out


def assign_splits_time_level(
    win_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    """
    Time-level split: strictly sort by window start time / packet index;
    early segment for training, middle for validation, late for testing.
    This mode does not backfill classes; if a risk class only appears in the future
    test segment, the training set will be missing that class and the script will
    report the error accordingly.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    out = win_df.copy().sort_values(["time_start", "packet_id_start", "window_id"]).reset_index(drop=True)
    n = len(out)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("test set is empty after time-level split; adjust train/val/test ratios")

    out["split"] = "test"
    out.loc[:n_train - 1, "split"] = "train"
    out.loc[n_train:n_train + n_val - 1, "split"] = "val"

    if "window_run_id" in out.columns:
        out["session_id"] = out["window_run_id"].astype(str) + "__timewin_" + out.groupby("window_run_id").cumcount().astype(str).str.zfill(5)
    else:
        out["session_id"] = "timewin_" + out.index.astype(str).str.zfill(6)
    return out


def build_leakage_report(win_df: pd.DataFrame) -> Dict:
    report: Dict[str, Dict] = {}
    for split in ["train", "val", "test"]:
        sub = win_df.loc[win_df["split"] == split]
        report[split] = {
            "n_windows": int(len(sub)),
            "time_min": str(sub["time_start"].min()) if len(sub) else "",
            "time_max": str(sub["time_end"].max()) if len(sub) else "",
            "n_run_groups": int(sub["window_run_id"].nunique()) if "window_run_id" in sub.columns else 0,
            "mixed_run_windows": int(sub["window_run_mixed"].sum()) if "window_run_mixed" in sub.columns else 0,
        }
    if "window_run_id" in win_df.columns:
        sets = {sp: set(win_df.loc[win_df["split"] == sp, "window_run_id"].astype(str)) for sp in ["train", "val", "test"]}
        report["run_overlap"] = {
            "train_val": int(len(sets["train"] & sets["val"])),
            "train_test": int(len(sets["train"] & sets["test"])),
            "val_test": int(len(sets["val"] & sets["test"])),
        }
    return report


def build_split_report(win_df: pd.DataFrame) -> Dict:
    report: Dict[str, Dict] = {}
    for split in ["train", "val", "test"]:
        sub = win_df.loc[win_df["split"] == split]
        report[split] = {
            "n": int(len(sub)),
            "binary_counts": {str(k): int(v) for k, v in sub["y_bin"].value_counts().sort_index().to_dict().items()},
            "risk_id_counts": {str(k): int(v) for k, v in sub["y_risk"].value_counts().sort_index().to_dict().items()},
            "risk_name_counts": {str(k): int(v) for k, v in sub["y_risk_name"].value_counts().to_dict().items()},
            "risk_alias_counts": {
                RISK_ALIAS_FOR_REPORT[int(k)]: int(v)
                for k, v in sub["y_risk"].value_counts().sort_index().to_dict().items()
            },
            "attack_type_id_counts": {str(k): int(v) for k, v in sub["y_attack_type"].value_counts().sort_index().to_dict().items()},
            "attack_type_name_counts": {str(k): int(v) for k, v in sub["y_attack_type_name"].value_counts().to_dict().items()},
        }
    return report


def assert_model_compatible_splits(
    win_df: pd.DataFrame,
    require_all_attack_types_each_split: bool = True,
    require_all_risks_each_split: bool = True,
    require_train_attack_types_for_old_teacher: bool = False,
) -> None:
    """
    Split validation.

    Note: Under strict run-level / time-level, we cannot force train/val/test to all
    contain complete attack types, as that would break the "same run not crossing sets"
    or "temporal extrapolation" validation purpose.

    - When using old teacher/student, train needs at least attack types 1/2/3,
      otherwise teacher_attack_prob_1/2/3 will error.
    - When using run_time_compatible teacher/student, train may be missing some
      attack types; teacher will always pad teacher_attack_prob_1/2/3.
    """
    required_risks = {0, 1, 2}
    required_attack_types = {0, 1, 2, 3}

    problems: List[str] = []
    warnings: List[str] = []
    for split in ["train", "val", "test"]:
        sub = win_df.loc[win_df["split"] == split]
        risk_present = set(sub["y_risk"].astype(int).unique().tolist())
        attack_present = set(sub["y_attack_type"].astype(int).unique().tolist())

        miss_risk = sorted(required_risks - risk_present)
        if require_all_risks_each_split and miss_risk:
            problems.append(f"{split} missing risk class y_risk={miss_risk}, present={sorted(risk_present)}")
        elif miss_risk:
            warnings.append(f"{split} missing risk class y_risk={miss_risk}, present={sorted(risk_present)}")

        miss_attack = sorted(required_attack_types - attack_present)
        if require_all_attack_types_each_split and miss_attack:
            problems.append(f"{split} missing attack type y_attack_type={miss_attack}, present={sorted(attack_present)}")
        elif miss_attack:
            warnings.append(f"{split} missing attack type y_attack_type={miss_attack}, present={sorted(attack_present)}")

    train_attack_present = set(win_df.loc[win_df["split"] == "train", "y_attack_type"].astype(int).unique().tolist())
    miss_train_attack = sorted({1, 2, 3} - train_attack_present)
    if require_train_attack_types_for_old_teacher and miss_train_attack:
        problems.append(f"train missing attack types {miss_train_attack} required by old teacher attack model")
    elif miss_train_attack:
        warnings.append(
            f"train missing attack types {miss_train_attack}; can continue with run_time_compatible "
            f"teacher/student, but old teacher/student will error"
        )

    if warnings:
        print("[SplitCheck][WARNING] Strict splitting caused partial class absence:")
        for w in warnings:
            print(f"  - {w}")

    if problems:
        report = build_split_report(win_df)
        raise ValueError(
            "Data split does not satisfy current validation requirements:\n- "
            + "\n- ".join(problems)
            + "\nCurrent distribution:\n"
            + json.dumps(report, ensure_ascii=False, indent=2)
        )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build teacher/student compatible risk-stratified window data from ECU-IoFT raw packet-level CSV")
    parser.add_argument("--input-csv", type=str, default="../Data/ECU-IoFT-Dataset.csv")
    parser.add_argument("--output-dir", type=str, default="Dataset/ecu_attack_risk_windows_time_level")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--step-size", type=int, default=32, help="Should be equal to window-size to avoid overlapping window leakage")
    parser.add_argument(
        "--window-label-rule",
        type=str,
        default="highest_risk",
        choices=["dominant", "any_attack", "highest_risk"],
        help="Window label rule: dominant=majority vote; any_attack=use dominant attack type if any attack exists in window; highest_risk=label by highest risk, recommended for risk early warning.",
    )
    parser.add_argument(
        "--keep-short-runs",
        action="store_true", default=True,
        help="Under run/time-level, keep short runs shorter than window-size, generating one variable-length window to avoid discarding all short attack segments.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-train-tello", type=int, default=1)
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="time_level",
        choices=["attack_stratified", "run_level", "time_level"],
        help="attack_stratified=stratified by attack type; run_level=grouped by capture session / consecutive segment; time_level=split by temporal order.",
    )
    parser.add_argument("--run-col", type=str, default="", help="Column name for true run/session/capture in the raw CSV; leave empty to auto-construct if unavailable.")
    parser.add_argument("--run-gap-sec", type=float, default=60.0, help="Time gap threshold for segmenting consecutive packets when no true run column is available.")
    parser.add_argument(
        "--window-within-run",
        action="store_true",
        help="Generate windows only within the same run to avoid windows spanning two capture segments. Recommended for run/time-level; auto-enabled when run_level is selected.",
    )
    parser.add_argument(
        "--allow-missing-attack-type-in-val-test",
        action="store_true", default=True,
        help="Do not force val/test to simultaneously contain 0/1/2/3. Usually needed or kept at default auto-relax for run/time-level.",
    )
    parser.add_argument(
        "--relaxed-split-check",
        action="store_true", default=True,
        help="Do not force every split to contain y_risk=0/1/2; used when strict time-level naturally causes some risk classes to be absent.",
    )
    parser.add_argument(
        "--require-all-attack-types-each-split",
        action="store_true",
        help="Force every split to contain 0/1/2/3. Generally not recommended under strict run/time-level.",
    )
    parser.add_argument(
        "--use-old-teacher",
        action="store_true",
        help="If using old teacher/student, force training set to contain attack types 1/2/3; do not enable when using run_time_compatible version.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw = pd.read_csv(args.input_csv)
    raw["_dt"] = parse_time(raw["Time"])
    raw = raw.reset_index(drop=True)
    raw = build_raw_run_ids(raw, run_col=args.run_col, time_gap_sec=args.run_gap_sec)

    # run-level must avoid windows crossing runs; time-level also recommends avoiding
    # cross-segment windows to reduce label contamination.
    within_run = args.window_within_run or args.split_strategy in {"run_level", "time_level"}
    win_df = build_windows(
        raw,
        window_size=args.window_size,
        step_size=args.step_size,
        within_run=within_run,
        keep_short_runs=args.keep_short_runs,
        label_rule=args.window_label_rule,
    )

    if args.split_strategy == "attack_stratified":
        win_df = assign_splits_model_compatible(
            win_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_train_tello=args.min_train_tello,
        )
    elif args.split_strategy == "run_level":
        win_df = assign_splits_run_level(
            win_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            group_col="window_run_id",
        )
    elif args.split_strategy == "time_level":
        win_df = assign_splits_time_level(
            win_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    else:
        raise ValueError(f"Unknown split_strategy: {args.split_strategy}")

    require_attacks_each = args.require_all_attack_types_each_split or (
        args.split_strategy == "attack_stratified" and not args.allow_missing_attack_type_in_val_test
    )
    assert_model_compatible_splits(
        win_df,
        require_all_attack_types_each_split=require_attacks_each,
        require_all_risks_each_split=not args.relaxed_split_check,
        require_train_attack_types_for_old_teacher=args.use_old_teacher,
    )

    feature_cols = [
        c for c in win_df.columns
        if c not in {
            "window_id", "row_id", "packet_id_start", "packet_id_end",
            "time_start", "time_end", "window_run_id", "window_run_count", "window_run_mixed",
            "window_label_rule", "window_no_attack_count", "window_deauth_count", "window_wpa2_count", "window_tello_count",
            "y_bin", "y_attack_type", "y_attack_type_name",
            "y_risk", "y_risk_name", "attack_scenario_meta", "packet_attack_type_mode_meta",
            "session_id", "split",
        }
    ]

    out_dir = Path(args.output_dir)
    win_df.to_csv(out_dir / "window_features.csv", index=False)
    for split in ["train", "val", "test"]:
        win_df.loc[win_df["split"] == split].to_csv(out_dir / f"{split}_windows.csv", index=False)

    safe_json_dump(feature_cols, out_dir / "feature_columns.json")

    split_report = build_split_report(win_df)
    manifest = {
        "input_csv": args.input_csv,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "window_label_rule": args.window_label_rule,
        "keep_short_runs": bool(args.keep_short_runs),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "min_train_tello": args.min_train_tello,
        "split_strategy": args.split_strategy,
        "run_col": args.run_col,
        "run_gap_sec": args.run_gap_sec,
        "window_within_run": within_run,
        "attack_type_map": ATTACK_TYPE_MAP,
        "risk_map": {k: {"id": v[0], "name": v[1]} for k, v in RISK_MAP.items()},
        "risk_alias_for_report": RISK_ALIAS_FOR_REPORT,
        "n_windows": int(len(win_df)),
        "split_counts": win_df["split"].value_counts().to_dict(),
        "raw_attack_type_counts": raw["Type of Attack"].value_counts().to_dict() if "Type of Attack" in raw.columns else {},
        "window_attack_type_counts": win_df["y_attack_type_name"].value_counts().to_dict(),
        "window_risk_counts": win_df["y_risk_name"].value_counts().to_dict(),
        "attack_type_counts": win_df["y_attack_type_name"].value_counts().to_dict(),
        "risk_counts": win_df["y_risk_name"].value_counts().to_dict(),
        "split_report": split_report,
        "leakage_report": build_leakage_report(win_df),
        "compatibility_checks": {
            "each_split_has_y_risk_0_1_2": not args.relaxed_split_check,
            "train_has_attack_type_1_2_3_required_only_for_old_teacher": bool(args.use_old_teacher),
            "each_split_has_attack_type_0_1_2_3": require_attacks_each,
            "split_strategy": args.split_strategy,
            "compatible_teacher_student_expected": not args.use_old_teacher,
        },
        "notes": [
            "Default uses non-overlapping windows; step-size should equal window-size.",
            "attack_stratified is the most model-compatible standard split, but not strict time-level validation.",
            "run_level ensures the same run/window_run_id does not cross train/val/test; if runs are auto-constructed consecutive segments, their weak proxy nature should be noted in the paper.",
            "time_level uses past for training, future for validation/testing, without class backfill; natural class absence should be reported honestly.",
            "y_risk=1 is named medium in existing teacher/student; if the paper calls it low risk, explain it as medium-low risk / lower-severity attack.",
            "TELLO API Exploit has very few windows; generalization ability of this type should not be exaggerated under run/time-level.",
            "If window_attack_type_counts is missing y_attack_type=1, try --window-label-rule highest_risk and --keep-short-runs first.",
        ],
    }
    safe_json_dump(manifest, out_dir / "manifest.json")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
