# -*- coding: utf-8 -*-
"""
ECU-IoFT 清洗 + 窗口化脚本（run-level 改进版）

核心原则：
1) prepare 阶段只做清洗、packet->window 聚合、run/session 构造，不做最终实验划分。
2) 显式保留 run 级元信息（run_id / run_key），供后续 strict resplit 做“外层按 run、内层按 block”的硬约束划分。
3) session_id = run_key + block，因此同一 run 内的多个 block 可以作为样本单元，但最终 train/val/test 不应跨 run。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_PACKET_KEY = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
RISK_MAP = {
    "No Attack": 0,
    "Wifi Deauthentication Attack": 1,
    "WPA2-PSK WIFI Cracking Attack": 2,
    "TELLO API Exploit": 2,
}
RISK_NAME_MAP = {0: "safe", 1: "medium", 2: "high"}
ATTACK_NAME_TO_ID = {
    "No Attack": 0,
    "Wifi Deauthentication Attack": 1,
    "WPA2-PSK WIFI Cracking Attack": 2,
    "TELLO API Exploit": 3,
}
ATTACK_ID_TO_NAME = {v: k for k, v in ATTACK_NAME_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=str, default="../Data/ECU-IoFT-Dataset.csv")
    ap.add_argument("--output-dir", type=str, default="数据处理_w25")
    ap.add_argument("--window-size", type=int, default=25)
    ap.add_argument("--min-window-size", type=int, default=10)
    ap.add_argument("--drop-exact-duplicates", action="store_true")
    ap.add_argument("--run-gap-sec", type=float, default=5.0)
    ap.add_argument("--session-block-packets", type=int, default=250)
    ap.add_argument("--split-on-attack-change", action="store_true")
    ap.add_argument("--write-legacy-splits", action="store_true")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    return ap.parse_args()


def safe_mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["", "nan", "None", "NaN"]), c] = np.nan
    return df


def parse_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["parsed_time"] = pd.to_datetime(df["Time"], dayfirst=True, errors="coerce")
    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    type_norm = df["Type"].fillna("Unknown").astype(str).str.strip().str.lower()
    df["Type"] = np.where(type_norm.eq("attack"), "Attack", "Normal")

    toa = df["Type of Attack"].fillna("No Attack").astype(str).str.strip()
    toa = toa.replace({
        "No Attack ": "No Attack",
        "WPA2-PSK WIFI Cracking Attack ": "WPA2-PSK WIFI Cracking Attack",
        "Wifi Deauthentication Attack ": "Wifi Deauthentication Attack",
        "TELLO API Exploit ": "TELLO API Exploit",
    })
    df["Type of Attack"] = toa

    scen = df["Attack Scenario"].fillna("Unknown Scenario").astype(str).str.strip()
    df["Attack Scenario"] = scen

    df["y_bin_packet"] = (df["Type"] == "Attack").astype(int)
    df["risk_level_packet"] = df["Type of Attack"].map(RISK_MAP).fillna(0).astype(int)
    df["risk_name_packet"] = df["risk_level_packet"].map(RISK_NAME_MAP)
    df["y_attack_type_packet"] = df["Type of Attack"].map(ATTACK_NAME_TO_ID).fillna(0).astype(int)
    return df


def build_audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source_missing"] = df["Source"].isna().astype(int)
    df["destination_missing"] = df["Destination"].isna().astype(int)
    df["info_missing"] = df["Info"].isna().astype(int)
    df["protocol_missing"] = df["Protocol"].isna().astype(int)
    df["length_missing"] = df["Length"].isna().astype(int)

    df["is_duplicate_packet_any_label"] = df.duplicated(subset=BASE_PACKET_KEY, keep=False).astype(int)
    df["is_duplicate_packet_same_label"] = df.duplicated(subset=BASE_PACKET_KEY + ["Type of Attack"], keep=False).astype(int)

    nunique_by_key = (
        df.groupby(BASE_PACKET_KEY, dropna=False)["Type of Attack"]
        .nunique()
        .rename("n_unique_attack_labels")
        .reset_index()
    )
    conflict_keys = nunique_by_key[nunique_by_key["n_unique_attack_labels"] > 1][BASE_PACKET_KEY].copy()
    if len(conflict_keys) > 0:
        conflict_keys["is_conflict_duplicate"] = 1
        df = df.merge(conflict_keys, on=BASE_PACKET_KEY, how="left")
    else:
        df["is_conflict_duplicate"] = 0

    df["is_conflict_duplicate"] = df["is_conflict_duplicate"].fillna(0).astype(int)
    return df


def clean_packets(df: pd.DataFrame, drop_exact_duplicates: bool) -> pd.DataFrame:
    clean = df[df["is_conflict_duplicate"] == 0].copy()
    if drop_exact_duplicates:
        clean = clean.drop_duplicates(subset=BASE_PACKET_KEY + ["Type of Attack"], keep="first").copy()
    sort_cols = ["parsed_time"]
    if "ID" in clean.columns:
        sort_cols.append("ID")
    clean = clean.sort_values(sort_cols).reset_index(drop=True)
    return clean


def add_run_and_session_ids(
    df: pd.DataFrame,
    run_gap_sec: float = 5.0,
    session_block_packets: int = 250,
    split_on_attack_change: bool = False,
) -> pd.DataFrame:
    df = df.copy().sort_values(["parsed_time", "ID"]).reset_index(drop=True)

    scen = df["Attack Scenario"].fillna("unknown_scenario").astype(str)
    atk = df["Type of Attack"].fillna("No Attack").astype(str)
    dt = df["parsed_time"].diff().dt.total_seconds().fillna(1e9)

    new_run = scen.ne(scen.shift()) | (dt > float(run_gap_sec))
    if split_on_attack_change:
        new_run = new_run | atk.ne(atk.shift())

    df["run_id"] = new_run.cumsum().astype(int)
    df["run_packet_index"] = df.groupby("run_id").cumcount()
    df["session_block_index"] = (df["run_packet_index"] // int(session_block_packets)).astype(int)

    run_start = df.groupby("run_id")["parsed_time"].transform("min")
    run_start_str = run_start.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("unknown_time")
    scen_mode = df.groupby("run_id")["Attack Scenario"].transform(lambda s: s.mode(dropna=False).iloc[0] if len(s) else "unknown")

    df["run_key"] = (
        scen_mode.astype(str)
        + " | run_" + df["run_id"].astype(str)
        + " | start_" + run_start_str.astype(str)
    )
    df["session_id"] = df["run_key"] + " | blk_" + df["session_block_index"].astype(str)
    return df


def add_packet_level_parsed_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df["length_num"] = df["Length"].fillna(0).astype(float)

    src = df["Source"].fillna("").astype(str)
    dst = df["Destination"].fillna("").astype(str)
    proto = df["Protocol"].fillna("").astype(str).str.upper()
    info = df["Info"].fillna("").astype(str)

    df["proto_80211"] = (proto == "802.11".upper()).astype(int)
    df["proto_udp"] = (proto == "UDP").astype(int)
    df["proto_icmp"] = (proto == "ICMP").astype(int)
    df["proto_eapol"] = (proto == "EAPOL").astype(int)

    def contains(pat: str) -> pd.Series:
        return info.str.contains(pat, case=False, na=False, regex=True).astype(int)

    df["flag_ack"] = contains(r"Acknowledgement")
    df["flag_deauth"] = contains(r"Deauthentication")
    df["flag_qos_data"] = contains(r"QoS Data")
    df["flag_qos_null"] = contains(r"QoS Null")
    df["flag_beacon"] = contains(r"Beacon")
    df["flag_probe"] = contains(r"Probe")
    df["flag_request"] = contains(r"Request")
    df["flag_response"] = contains(r"Response")
    df["flag_port_unreachable"] = contains(r"Port unreachable")
    df["flag_tello_cmd_port"] = contains(r"\b8889\b") | contains(r"\b8890\b")

    mac_pat = r"^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$"
    ip_pat = r"^\d{1,3}(\.\d{1,3}){3}$"
    df["src_is_mac"] = src.str.match(mac_pat, na=False).astype(int)
    df["dst_is_mac"] = dst.str.match(mac_pat, na=False).astype(int)
    df["src_is_ip"] = src.str.match(ip_pat, na=False).astype(int)
    df["dst_is_ip"] = dst.str.match(ip_pat, na=False).astype(int)
    df["dst_is_broadcast"] = dst.str.contains(r"ff:ff:ff:ff:ff:ff", case=False, na=False).astype(int)

    df["sn_num"] = pd.to_numeric(info.str.extract(r"SN=(\d+)")[0], errors="coerce")
    df["fn_num"] = pd.to_numeric(info.str.extract(r"FN=(\d+)")[0], errors="coerce")
    df["udp_payload_len_info"] = pd.to_numeric(info.str.extract(r"Len=(\d+)")[0], errors="coerce")

    port_match = info.str.extract(r"(\d+)\s*>\s*(\d+)")
    df["src_port_info"] = pd.to_numeric(port_match[0], errors="coerce")
    df["dst_port_info"] = pd.to_numeric(port_match[1], errors="coerce")

    df["info_len_chars"] = info.str.len().fillna(0).astype(int)
    df["source_is_missing"] = src.eq("").astype(int)
    df["destination_is_missing"] = dst.eq("").astype(int)
    return df


def q_or_nan(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, q))


def numeric_summary(prefix: str, values: pd.Series) -> Dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q50": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_n_nonnull": 0.0,
        }
    std = float(np.std(valid, ddof=1)) if valid.size >= 2 else 0.0
    return {
        f"{prefix}_mean": float(np.mean(valid)),
        f"{prefix}_std": std,
        f"{prefix}_min": float(np.min(valid)),
        f"{prefix}_max": float(np.max(valid)),
        f"{prefix}_q25": q_or_nan(valid, 0.25),
        f"{prefix}_q50": q_or_nan(valid, 0.50),
        f"{prefix}_q75": q_or_nan(valid, 0.75),
        f"{prefix}_n_nonnull": float(valid.size),
    }


def ratio_summary(dfw: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    denom = max(len(dfw), 1)
    for c in cols:
        out[f"ratio_{c}"] = float(dfw[c].fillna(0).astype(float).sum() / denom)
    return out


def most_common_nonzero_attack(window_attack_ids: pd.Series) -> int:
    vals = window_attack_ids[window_attack_ids > 0]
    if len(vals) == 0:
        return 0
    return int(vals.value_counts().idxmax())


def dominant_risk(window_risk_ids: pd.Series) -> int:
    if len(window_risk_ids) == 0:
        return 0
    return int(np.max(window_risk_ids))


def blocked_split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> List[str]:
    if n <= 0:
        return []
    if n == 1:
        return ["train"]
    if n == 2:
        return ["train", "test"]
    if n == 3:
        return ["train", "val", "test"]

    n_train = int(math.floor(n * train_ratio))
    n_val = int(math.floor(n * val_ratio))
    n_test = n - n_train - n_val
    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    if len(splits) != n:
        raise RuntimeError("blocked_split_indices length mismatch")
    return splits


def window_one_session(sess_df: pd.DataFrame, window_size: int, min_window_size: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    sort_cols = ["parsed_time"]
    if "ID" in sess_df.columns:
        sort_cols.append("ID")
    sess_df = sess_df.sort_values(sort_cols).reset_index(drop=True)

    n = len(sess_df)
    start = 0
    window_idx = 0
    while start < n:
        end = min(start + window_size, n)
        if end - start < min_window_size and start != 0:
            break

        dfw = sess_df.iloc[start:end].copy()
        row: Dict[str, object] = {}
        row["run_id"] = int(dfw["run_id"].iloc[0])
        row["run_key"] = str(dfw["run_key"].iloc[0])
        row["session_id"] = str(dfw["session_id"].iloc[0])
        row["window_index_in_session"] = window_idx
        row["window_packet_count"] = int(len(dfw))
        row["packet_id_start"] = int(dfw["ID"].min()) if "ID" in dfw.columns else int(start)
        row["packet_id_end"] = int(dfw["ID"].max()) if "ID" in dfw.columns else int(end - 1)
        row["time_start"] = dfw["parsed_time"].min()
        row["time_end"] = dfw["parsed_time"].max()
        row["attack_scenario_meta"] = str(dfw["Attack Scenario"].mode(dropna=False).iloc[0])
        row["packet_attack_type_mode_meta"] = str(dfw["Type of Attack"].mode(dropna=False).iloc[0])

        attack_packet_count = int(dfw["y_bin_packet"].sum())
        row["attack_packet_count"] = attack_packet_count
        row["attack_packet_ratio"] = float(attack_packet_count / max(len(dfw), 1))
        row["y_bin"] = int(attack_packet_count > 0)

        dominant_attack_id = most_common_nonzero_attack(dfw["y_attack_type_packet"])
        row["y_attack_type"] = dominant_attack_id
        row["y_attack_type_name"] = ATTACK_ID_TO_NAME[dominant_attack_id]

        risk_level = dominant_risk(dfw["risk_level_packet"])
        row["y_risk"] = risk_level
        row["y_risk_name"] = RISK_NAME_MAP[risk_level]

        counts = dfw["Type of Attack"].value_counts(dropna=False).to_dict()
        for k in ATTACK_NAME_TO_ID.keys():
            safe_k = k.lower().replace("-", "_").replace(" ", "_")
            row[f"cnt_{safe_k}"] = int(counts.get(k, 0))

        row.update(numeric_summary("length", dfw["length_num"]))
        row.update(numeric_summary("sn", dfw["sn_num"]))
        row.update(numeric_summary("fn", dfw["fn_num"]))
        row.update(numeric_summary("udp_payload_len_info", dfw["udp_payload_len_info"]))
        row.update(numeric_summary("src_port_info", dfw["src_port_info"]))
        row.update(numeric_summary("dst_port_info", dfw["dst_port_info"]))
        row.update(numeric_summary("info_len_chars", dfw["info_len_chars"]))

        row.update(ratio_summary(dfw, cols=[
            "proto_80211", "proto_udp", "proto_icmp", "proto_eapol",
            "flag_ack", "flag_deauth", "flag_qos_data", "flag_qos_null",
            "flag_beacon", "flag_probe", "flag_request", "flag_response",
            "flag_port_unreachable", "flag_tello_cmd_port",
            "src_is_mac", "dst_is_mac", "src_is_ip", "dst_is_ip",
            "dst_is_broadcast", "source_is_missing", "destination_is_missing",
        ]))

        row["n_unique_source"] = int(dfw["Source"].fillna("NA").nunique(dropna=False))
        row["n_unique_destination"] = int(dfw["Destination"].fillna("NA").nunique(dropna=False))
        row["n_unique_protocol"] = int(dfw["Protocol"].fillna("NA").nunique(dropna=False))

        length_arr = pd.to_numeric(dfw["length_num"], errors="coerce").to_numpy(dtype=float)
        if len(length_arr) >= 2:
            diff_abs = np.abs(np.diff(length_arr))
            row["length_diff_abs_mean"] = float(np.nanmean(diff_abs))
            row["length_diff_abs_std"] = float(np.nanstd(diff_abs, ddof=1)) if len(length_arr) >= 3 else 0.0
        else:
            row["length_diff_abs_mean"] = np.nan
            row["length_diff_abs_std"] = np.nan

        sn_arr = pd.to_numeric(dfw["sn_num"], errors="coerce").to_numpy(dtype=float)
        sn_arr = sn_arr[~np.isnan(sn_arr)]
        if len(sn_arr) >= 2:
            sn_diff = np.diff(sn_arr)
            row["sn_diff_mean"] = float(np.mean(sn_diff))
            row["sn_diff_std"] = float(np.std(sn_diff, ddof=1)) if len(sn_diff) >= 2 else 0.0
            row["sn_diff_abs_mean"] = float(np.mean(np.abs(sn_diff)))
        else:
            row["sn_diff_mean"] = np.nan
            row["sn_diff_std"] = np.nan
            row["sn_diff_abs_mean"] = np.nan

        rows.append(row)
        start = end
        window_idx += 1
    return pd.DataFrame(rows)


def build_windows(clean_df: pd.DataFrame, window_size: int, min_window_size: int) -> pd.DataFrame:
    window_parts = []
    for _, sess_df in clean_df.groupby("session_id", sort=False):
        part = window_one_session(sess_df, window_size=window_size, min_window_size=min_window_size)
        if not part.empty:
            window_parts.append(part)
    if not window_parts:
        raise RuntimeError("No windows were generated. Check window_size/min_window_size.")
    windows = pd.concat(window_parts, axis=0, ignore_index=True)
    windows["window_id"] = np.arange(len(windows), dtype=int)
    return windows


def assign_legacy_blocked_splits(windows: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float) -> pd.DataFrame:
    windows = windows.copy().sort_values(["session_id", "window_index_in_session"]).reset_index(drop=True)
    split_col: List[str] = []
    for _, idx in windows.groupby("session_id", sort=False).groups.items():
        idx = list(idx)
        split_col.extend(blocked_split_indices(len(idx), train_ratio, val_ratio, test_ratio))
    if len(split_col) != len(windows):
        raise RuntimeError("Split assignment length mismatch")
    windows["split"] = split_col
    return windows


def get_feature_columns(windows: pd.DataFrame) -> List[str]:
    non_feature_cols = {
        "window_id", "run_id", "run_key", "session_id", "window_index_in_session",
        "packet_id_start", "packet_id_end", "time_start", "time_end",
        "attack_scenario_meta", "packet_attack_type_mode_meta",
        "y_bin", "y_attack_type", "y_attack_type_name", "y_risk", "y_risk_name", "split",
    }
    return [c for c in windows.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(windows[c])]


def dataset_overview(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "type_counts": df["Type"].value_counts(dropna=False).to_dict(),
        "attack_type_counts": df["Type of Attack"].value_counts(dropna=False).to_dict(),
        "attack_scenario_counts": df["Attack Scenario"].value_counts(dropna=False).to_dict(),
        "protocol_counts": df["Protocol"].value_counts(dropna=False).to_dict(),
        "time_counts": df["Time"].value_counts(dropna=False).head(20).to_dict(),
        "source_missing_ratio": float(df["Source"].isna().mean()),
        "destination_missing_ratio": float(df["Destination"].isna().mean()),
        "info_missing_ratio": float(df["Info"].isna().mean()),
    }


def save_outputs(raw_df: pd.DataFrame, audit_df: pd.DataFrame, clean_df: pd.DataFrame, windows_df: pd.DataFrame, out_dir: str, args: argparse.Namespace) -> None:
    safe_mkdir(out_dir)
    out_dir_p = Path(out_dir)

    audit_df.to_csv(out_dir_p / "audit_packets.csv", index=False, encoding="utf-8-sig")
    clean_df.to_csv(out_dir_p / "clean_packets.csv", index=False, encoding="utf-8-sig")
    windows_df.to_csv(out_dir_p / "window_features.csv", index=False, encoding="utf-8-sig")

    feature_cols = get_feature_columns(windows_df)
    with open(out_dir_p / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    run_overview = (
        windows_df.groupby("run_key", dropna=False)
        .agg(n_windows=("window_id", "count"), n_neg=("y_bin", lambda s: int((s.astype(int) == 0).sum())), n_pos=("y_bin", lambda s: int((s.astype(int) == 1).sum())))
        .reset_index()
    )

    manifest = {
        "raw_overview": dataset_overview(raw_df),
        "audit_stats": {
            "n_conflict_duplicate_packets": int(audit_df["is_conflict_duplicate"].sum()),
            "n_duplicate_packets_any_label": int(audit_df["is_duplicate_packet_any_label"].sum()),
            "n_duplicate_packets_same_label": int(audit_df["is_duplicate_packet_same_label"].sum()),
        },
        "clean_overview": {
            "n_rows_after_clean": int(len(clean_df)),
            "type_counts_after_clean": clean_df["Type"].value_counts(dropna=False).to_dict(),
            "attack_type_counts_after_clean": clean_df["Type of Attack"].value_counts(dropna=False).to_dict(),
            "n_runs": int(clean_df["run_key"].nunique()),
            "n_sessions": int(clean_df["session_id"].nunique()),
        },
        "window_overview": {
            "n_windows": int(len(windows_df)),
            "window_size_mean": float(windows_df["window_packet_count"].mean()),
            "y_bin_counts": windows_df["y_bin"].value_counts(dropna=False).sort_index().to_dict(),
            "y_attack_type_name_counts": windows_df["y_attack_type_name"].value_counts(dropna=False).to_dict(),
            "y_risk_name_counts": windows_df["y_risk_name"].value_counts(dropna=False).to_dict(),
            "n_unique_runs": int(windows_df["run_key"].nunique()),
            "n_unique_sessions": int(windows_df["session_id"].nunique()),
        },
        "run_overview": run_overview.to_dict(orient="records"),
        "session_build_config": {
            "run_gap_sec": float(args.run_gap_sec),
            "session_block_packets": int(args.session_block_packets),
            "split_on_attack_change": bool(args.split_on_attack_change),
        },
        "feature_columns_file": "feature_columns.json",
        "recommended_next_step": "Run session_strict_resplit_ecu_v3.py for run-level final train/val/test split.",
        "model_leakage_warning": [
            "Do NOT use: Type, Type of Attack, Attack Scenario, Time, run_id, run_key, session_id, time_start, time_end, packet_id_start, packet_id_end",
            "Use only numeric columns listed in feature_columns.json",
            "Do NOT treat prepare-stage legacy split as the final experimental split.",
        ],
    }

    if args.write_legacy_splits:
        ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
        if not np.isclose(ratio_sum, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0 when write_legacy_splits is enabled")
        legacy_df = assign_legacy_blocked_splits(windows_df, args.train_ratio, args.val_ratio, args.test_ratio)
        legacy_df.to_csv(out_dir_p / "window_features_legacy_split.csv", index=False, encoding="utf-8-sig")
        legacy_df[["window_id", "session_id", "split"]].to_csv(out_dir_p / "splits_legacy_blocked.csv", index=False, encoding="utf-8-sig")
        manifest["legacy_split_overview"] = {
            "split_counts": legacy_df["split"].value_counts(dropna=False).to_dict(),
            "warning": "legacy split is for compatibility/debug only; do not use for final reporting",
        }

    with open(out_dir_p / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    raw_df = pd.read_csv(args.input_csv)
    raw_df = normalize_columns(raw_df)
    raw_df = parse_time_column(raw_df)
    raw_df = normalize_labels(raw_df)
    if "ID" in raw_df.columns:
        raw_df["ID"] = pd.to_numeric(raw_df["ID"], errors="coerce")
    else:
        raw_df["ID"] = np.arange(1, len(raw_df) + 1)

    audit_df = build_audit_flags(raw_df)
    clean_df = clean_packets(audit_df, drop_exact_duplicates=args.drop_exact_duplicates)
    clean_df = add_run_and_session_ids(
        clean_df,
        run_gap_sec=args.run_gap_sec,
        session_block_packets=args.session_block_packets,
        split_on_attack_change=args.split_on_attack_change,
    )
    clean_df = add_packet_level_parsed_features(clean_df)

    windows_df = build_windows(clean_df, window_size=args.window_size, min_window_size=args.min_window_size)
    save_outputs(raw_df, audit_df, clean_df, windows_df, args.output_dir, args)

    print("=" * 80)
    print("Done.")
    print(f"input_csv             : {args.input_csv}")
    print(f"output_dir            : {args.output_dir}")
    print(f"window_size           : {args.window_size}")
    print(f"min_window_size       : {args.min_window_size}")
    print(f"run_gap_sec           : {args.run_gap_sec}")
    print(f"session_block_packets : {args.session_block_packets}")
    print(f"split_on_attack_change: {args.split_on_attack_change}")
    print(f"write_legacy_splits   : {args.write_legacy_splits}")
    print("-" * 80)
    print(f"raw rows              : {len(raw_df)}")
    print(f"clean rows            : {len(clean_df)}")
    print(f"n_runs                : {clean_df['run_key'].nunique()}")
    print(f"n_sessions            : {clean_df['session_id'].nunique()}")
    print(f"n_windows             : {len(windows_df)}")
    print(f"y_bin counts          : {windows_df['y_bin'].value_counts().sort_index().to_dict()}")
    print(f"risk counts           : {windows_df['y_risk_name'].value_counts().to_dict()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
