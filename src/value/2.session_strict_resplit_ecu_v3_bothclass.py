
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ECU-IoFT run-level strict resplit（both-class LORO 合法版）

核心原则：
1) 外层按 run_key 划分：同一 run 绝不跨 train / val / test。
2) 单次 strict split 继续使用 run-level 硬约束。
3) 生成 LORO folds 时，只生成“val 与 test 都同时含正负类”的合法 binary folds。
4) 单类 runs（例如全 normal / 全 abnormal）只允许进入 train，不允许作为 LORO 的 val/test。
"""

import argparse
import itertools
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ALWAYS_DROP_EXACT = {
    "y_bin", "y_attack_type", "y_attack_type_name", "y_risk", "y_risk_name",
    "attack_packet_count", "attack_packet_ratio",
    "window_id", "split", "row_id",
    "run_id", "run_key", "session_id", "window_index_in_session",
    "packet_id_start", "packet_id_end",
    "time_start", "time_end",
    "attack_scenario_meta", "packet_attack_type_mode_meta",
}
ALWAYS_DROP_PREFIXES = ("cnt_",)
SHORTCUT_LEVELS = {
    "none": set(),
    "light": {"ratio_flag_deauth", "ratio_flag_port_unreachable", "ratio_flag_tello_cmd_port", "ratio_proto_eapol"},
    "medium": {
        "ratio_flag_deauth", "ratio_flag_port_unreachable", "ratio_flag_tello_cmd_port", "ratio_proto_eapol",
        "ratio_flag_request", "ratio_flag_response",
        "udp_payload_len_info_mean", "udp_payload_len_info_std", "udp_payload_len_info_min", "udp_payload_len_info_max",
        "udp_payload_len_info_q25", "udp_payload_len_info_q50", "udp_payload_len_info_q75", "udp_payload_len_info_n_nonnull",
        "src_port_info_mean", "src_port_info_std", "src_port_info_min", "src_port_info_max",
        "src_port_info_q25", "src_port_info_q50", "src_port_info_q75", "src_port_info_n_nonnull",
        "dst_port_info_mean", "dst_port_info_std", "dst_port_info_min", "dst_port_info_max",
        "dst_port_info_q25", "dst_port_info_q50", "dst_port_info_q75", "dst_port_info_n_nonnull",
    },
    "strict": {
        "ratio_flag_deauth", "ratio_flag_port_unreachable", "ratio_flag_tello_cmd_port", "ratio_proto_eapol",
        "ratio_flag_request", "ratio_flag_response", "ratio_flag_probe", "ratio_flag_beacon",
        "ratio_flag_qos_data", "ratio_flag_qos_null", "ratio_proto_udp", "ratio_proto_icmp",
        "udp_payload_len_info_mean", "udp_payload_len_info_std", "udp_payload_len_info_min", "udp_payload_len_info_max",
        "udp_payload_len_info_q25", "udp_payload_len_info_q50", "udp_payload_len_info_q75", "udp_payload_len_info_n_nonnull",
        "src_port_info_mean", "src_port_info_std", "src_port_info_min", "src_port_info_max",
        "src_port_info_q25", "src_port_info_q50", "src_port_info_q75", "src_port_info_n_nonnull",
        "dst_port_info_mean", "dst_port_info_std", "dst_port_info_min", "dst_port_info_max",
        "dst_port_info_q25", "dst_port_info_q50", "dst_port_info_q75", "dst_port_info_n_nonnull",
    },
}


def safe_json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run-level hard-constrained split for ECU-IoFT window data")
    p.add_argument("--input-dir", type=str, default="数据处理_w25")
    p.add_argument("--output-dir", type=str, default="数据处理_w25_strict")
    p.add_argument("--window-csv", type=str, default="")
    p.add_argument("--shortcut-level", type=str, default="strict", choices=["none", "light", "medium", "strict"])
    p.add_argument("--val-runs", type=str, default="", help="Comma-separated run_key list for val. If empty, auto-search.")
    p.add_argument("--test-runs", type=str, default="", help="Comma-separated run_key list for test. If empty, auto-search.")
    p.add_argument("--n-val-runs", type=int, default=1)
    p.add_argument("--n-test-runs", type=int, default=1)
    p.add_argument("--target-train", type=float, default=0.60)
    p.add_argument("--target-val", type=float, default=0.20)
    p.add_argument("--target-test", type=float, default=0.20)
    p.add_argument("--min-train-pos", type=int, default=20)
    p.add_argument("--min-train-neg", type=int, default=20)
    p.add_argument("--min-val-pos", type=int, default=10)
    p.add_argument("--min-val-neg", type=int, default=10)
    p.add_argument("--min-test-pos", type=int, default=10)
    p.add_argument("--min-test-neg", type=int, default=10)
    p.add_argument("--max-search-combos", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-folds", default=True, action="store_true", help="Write legal both-class LORO folds")
    p.add_argument("--allow-single-class-loro", action="store_true", help="Debug only: allow illegal single-class LORO folds")
    return p.parse_args()


def load_window_df(input_dir: Path, window_csv: str) -> pd.DataFrame:
    path = Path(window_csv) if window_csv else input_dir / "window_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"window features file not found: {path}")
    df = pd.read_csv(path)
    if "run_key" not in df.columns:
        if "session_id" in df.columns:
            df = df.copy()
            df["run_key"] = df["session_id"].astype(str).str.replace(r"\s*\|\s*blk_\d+$", "", regex=True)
        else:
            raise ValueError("window_features.csv must contain run_key or session_id")
    if "y_bin" not in df.columns:
        raise ValueError("window_features.csv must contain y_bin")
    if "window_id" not in df.columns:
        df = df.copy()
        df["window_id"] = np.arange(len(df), dtype=np.int64)
    return df


def maybe_load_raw_feature_cols(input_dir: Path) -> Optional[List[str]]:
    path = input_dir / "feature_columns.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list):
        raise ValueError("feature_columns.json should be a list")
    return cols


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def sanitize_feature_cols(df: pd.DataFrame, raw_feature_cols: Optional[List[str]], shortcut_level: str) -> Tuple[List[str], Dict[str, List[str]]]:
    candidate_cols = raw_feature_cols[:] if raw_feature_cols is not None else [c for c in df.columns if is_numeric_series(df[c])]
    drop_exact = set(ALWAYS_DROP_EXACT)
    drop_shortcuts = set(SHORTCUT_LEVELS[shortcut_level])
    dropped = {"drop_exact": [], "drop_prefix": [], "drop_shortcut": [], "drop_non_numeric": [], "kept": []}

    feature_cols: List[str] = []
    col_order = list(df.columns)
    for c in candidate_cols:
        if c in drop_exact:
            dropped["drop_exact"].append(c)
            continue
        if any(c.startswith(p) for p in ALWAYS_DROP_PREFIXES):
            dropped["drop_prefix"].append(c)
            continue
        if c in drop_shortcuts:
            dropped["drop_shortcut"].append(c)
            continue
        if c not in df.columns:
            continue
        if not is_numeric_series(df[c]):
            dropped["drop_non_numeric"].append(c)
            continue
        feature_cols.append(c)

    feature_cols = sorted(set(feature_cols), key=lambda x: col_order.index(x))
    dropped["kept"] = feature_cols
    return feature_cols, dropped


def parse_list(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def has_min_counts(df: pd.DataFrame, min_pos: int, min_neg: int) -> bool:
    y = df["y_bin"].astype(int).values
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg >= min_neg and pos >= min_pos


def build_split_dfs_by_runs(df: pd.DataFrame, train_runs: Sequence[str], val_runs: Sequence[str], test_runs: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["run_key"].isin(train_runs)].copy()
    val_df = df[df["run_key"].isin(val_runs)].copy()
    test_df = df[df["run_key"].isin(test_runs)].copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return train_df, val_df, test_df


def split_overview(df: pd.DataFrame) -> Dict:
    y = df["y_bin"].astype(int).values
    attack_name_counts = df["y_attack_type_name"].astype(str).value_counts().to_dict() if "y_attack_type_name" in df.columns else {}
    return {
        "n_windows": int(len(df)),
        "n_neg": int((y == 0).sum()),
        "n_pos": int((y == 1).sum()),
        "attack_type_counts": {str(k): int(v) for k, v in attack_name_counts.items()},
        "run_counts": {str(k): int(v) for k, v in df["run_key"].astype(str).value_counts().to_dict().items()},
        "session_counts": {str(k): int(v) for k, v in df["session_id"].astype(str).value_counts().to_dict().items()},
    }


def run_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rk, g in df.groupby("run_key", sort=False):
        y = g["y_bin"].astype(int).values
        attack_types = sorted(set(g["y_attack_type_name"].astype(str).tolist())) if "y_attack_type_name" in g.columns else []
        rows.append({
            "run_key": rk,
            "n_windows": int(len(g)),
            "n_neg": int((y == 0).sum()),
            "n_pos": int((y == 1).sum()),
            "has_both_classes": bool(((y == 0).any()) and ((y == 1).any())),
            "attack_types": attack_types,
        })
    out = pd.DataFrame(rows)
    out["attack_type_count"] = out["attack_types"].apply(len)
    return out.sort_values(["has_both_classes", "n_windows"], ascending=[False, False]).reset_index(drop=True)


def eligible_both_class_runs(df: pd.DataFrame, min_pos: int, min_neg: int) -> List[str]:
    runs = []
    for rk, g in df.groupby("run_key", sort=False):
        if has_min_counts(g, min_pos=min_pos, min_neg=min_neg):
            runs.append(str(rk))
    return runs


def score_run_split(
    df: pd.DataFrame,
    train_runs: Sequence[str],
    val_runs: Sequence[str],
    test_runs: Sequence[str],
    target_ratios: Tuple[float, float, float],
) -> float:
    total_n = len(df)
    train_df, val_df, test_df = build_split_dfs_by_runs(df, train_runs, val_runs, test_runs)
    tr, vr, ter = len(train_df) / total_n, len(val_df) / total_n, len(test_df) / total_n
    tt, tv, tte = target_ratios

    score = 0.0
    score -= 18.0 * abs(tr - tt)
    score -= 15.0 * abs(vr - tv)
    score -= 15.0 * abs(ter - tte)
    score += 25.0 * tr

    if "y_attack_type_name" in train_df.columns:
        score += 4.0 * len(set(train_df["y_attack_type_name"].astype(str).tolist()) - {"No Attack"})
        score += 2.0 * len(set(val_df["y_attack_type_name"].astype(str).tolist()) - {"No Attack"})
        score += 2.0 * len(set(test_df["y_attack_type_name"].astype(str).tolist()) - {"No Attack"})
    return score


def auto_choose_runs(
    df: pd.DataFrame,
    target_ratios: Tuple[float, float, float],
    n_val_runs: int,
    n_test_runs: int,
    min_train_pos: int,
    min_train_neg: int,
    min_val_pos: int,
    min_val_neg: int,
    min_test_pos: int,
    min_test_neg: int,
    max_search_combos: int,
) -> Tuple[List[str], List[str], List[str], Dict]:
    runs = list(df["run_key"].drop_duplicates())
    if len(runs) < n_val_runs + n_test_runs + 1:
        raise RuntimeError(f"Not enough runs: total={len(runs)}, need at least {n_val_runs + n_test_runs + 1}")

    best_score = None
    best_info = None
    checked = 0

    for val_runs in itertools.combinations(runs, n_val_runs):
        remain = [r for r in runs if r not in set(val_runs)]
        for test_runs in itertools.combinations(remain, n_test_runs):
            checked += 1
            if checked > max_search_combos:
                break
            train_runs = [r for r in runs if r not in set(val_runs) | set(test_runs)]
            train_df, val_df, test_df = build_split_dfs_by_runs(df, train_runs, list(val_runs), list(test_runs))

            if not has_min_counts(train_df, min_train_pos, min_train_neg):
                continue
            if not has_min_counts(val_df, min_val_pos, min_val_neg):
                continue
            if not has_min_counts(test_df, min_test_pos, min_test_neg):
                continue

            score = score_run_split(df, train_runs, list(val_runs), list(test_runs), target_ratios)
            if best_score is None or score > best_score:
                best_score = score
                best_info = {
                    "train_runs": train_runs,
                    "val_runs": list(val_runs),
                    "test_runs": list(test_runs),
                    "score": float(score),
                    "checked_combinations": int(checked),
                }
        if checked > max_search_combos:
            break

    if best_info is None:
        raise RuntimeError(
            "No valid run-level split found under current hard constraints. "
            "Try lowering min_* thresholds or revising run construction in prepare stage."
        )
    return best_info["train_runs"], best_info["val_runs"], best_info["test_runs"], best_info


def slugify(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "|", "-", ":", "/"}:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:120]


def choose_val_run_for_loro(
    df: pd.DataFrame,
    test_run: str,
    min_val_pos: int,
    min_val_neg: int,
    candidate_runs: Optional[Sequence[str]] = None,
) -> str:
    remaining = [r for r in (candidate_runs or df["run_key"].drop_duplicates().tolist()) if r != test_run]
    candidates = []
    for rk in remaining:
        g = df[df["run_key"] == rk]
        if has_min_counts(g, min_val_pos, min_val_neg):
            candidates.append((rk, len(g)))
    if not candidates:
        raise RuntimeError(f"No valid val run found for test run: {test_run}")
    candidates = sorted(candidates, key=lambda x: (-x[1], str(x[0])))
    return candidates[0][0]


def write_loro_folds(
    df: pd.DataFrame,
    out_dir: Path,
    feature_cols: List[str],
    base_manifest: Dict,
    min_train_pos: int,
    min_train_neg: int,
    min_val_pos: int,
    min_val_neg: int,
    min_test_pos: int,
    min_test_neg: int,
    allow_single_class_loro: bool = False,
) -> None:
    folds_dir = out_dir / "loro_folds"
    if folds_dir.exists():
        shutil.rmtree(folds_dir)
    folds_dir.mkdir(parents=True, exist_ok=True)

    all_runs = list(df["run_key"].drop_duplicates())
    legal_test_runs = eligible_both_class_runs(df, min_pos=min_test_pos, min_neg=min_test_neg)
    legal_val_runs = eligible_both_class_runs(df, min_pos=min_val_pos, min_neg=min_val_neg)

    candidate_test_runs = all_runs if allow_single_class_loro else legal_test_runs
    skipped: List[Dict[str, object]] = []
    written = 0

    for test_run in candidate_test_runs:
        try:
            val_candidates = [r for r in (all_runs if allow_single_class_loro else legal_val_runs) if r != test_run]
            val_run = choose_val_run_for_loro(
                df, test_run,
                min_val_pos=min_val_pos,
                min_val_neg=min_val_neg,
                candidate_runs=val_candidates,
            )
        except RuntimeError as e:
            skipped.append({"test_run": test_run, "reason": str(e)})
            continue

        train_runs = [r for r in all_runs if r not in {test_run, val_run}]
        tr, va, te = build_split_dfs_by_runs(df, train_runs, [val_run], [test_run])

        invalid_reasons: List[str] = []
        if len(tr) == 0:
            invalid_reasons.append("empty_train")
        if not has_min_counts(tr, min_train_pos, min_train_neg):
            invalid_reasons.append("train_not_enough_pos_neg")
        if not has_min_counts(va, min_val_pos, min_val_neg):
            invalid_reasons.append("val_not_both_class_or_below_min")
        if not has_min_counts(te, min_test_pos, min_test_neg):
            invalid_reasons.append("test_not_both_class_or_below_min")

        if invalid_reasons and not allow_single_class_loro:
            skipped.append({"test_run": test_run, "val_run": val_run, "reason": ";".join(invalid_reasons)})
            continue

        fold_dir = folds_dir / f"test_{slugify(test_run)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tr.to_csv(fold_dir / "train_windows.csv", index=False)
        va.to_csv(fold_dir / "val_windows.csv", index=False)
        te.to_csv(fold_dir / "test_windows.csv", index=False)
        pd.concat([tr, va, te], ignore_index=True).to_csv(fold_dir / "window_features.csv", index=False)
        with open(fold_dir / "feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)

        fold_manifest = {
            **base_manifest,
            "fold_type": "LORO",
            "fold_legality": {
                "both_class_only": not allow_single_class_loro,
                "legal_binary_fold": len(invalid_reasons) == 0,
                "invalid_reasons": invalid_reasons,
            },
            "test_run": test_run,
            "val_run": val_run,
            "train_runs": train_runs,
            "split_overview": {
                "train": split_overview(tr),
                "val": split_overview(va),
                "test": split_overview(te),
            },
        }
        safe_json_dump(fold_manifest, fold_dir / "manifest.json")
        written += 1

    safe_json_dump(
        {
            "both_class_only": not allow_single_class_loro,
            "eligible_test_runs": legal_test_runs,
            "eligible_val_runs": legal_val_runs,
            "written_folds": written,
            "skipped_folds": skipped,
        },
        folds_dir / "_loro_index.json",
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_window_df(input_dir, args.window_csv)
    raw_feature_cols = maybe_load_raw_feature_cols(input_dir)
    feature_cols, drop_info = sanitize_feature_cols(df, raw_feature_cols, args.shortcut_level)
    if not feature_cols:
        raise RuntimeError("No usable feature columns left after sanitization.")

    rk_stats = run_stats(df)
    user_val = parse_list(args.val_runs)
    user_test = parse_list(args.test_runs)

    if user_val or user_test:
        if not user_val or not user_test:
            raise ValueError("Please provide both --val-runs and --test-runs, or leave both empty.")
        train_runs = [rk for rk in df["run_key"].drop_duplicates().tolist() if rk not in set(user_val + user_test)]
        train_df, val_df, test_df = build_split_dfs_by_runs(df, train_runs, user_val, user_test)
        if not has_min_counts(train_df, args.min_train_pos, args.min_train_neg):
            raise RuntimeError("Manual split invalid: train does not satisfy min_train_pos/min_train_neg")
        if not has_min_counts(val_df, args.min_val_pos, args.min_val_neg):
            raise RuntimeError("Manual split invalid: val does not satisfy min_val_pos/min_val_neg")
        if not has_min_counts(test_df, args.min_test_pos, args.min_test_neg):
            raise RuntimeError("Manual split invalid: test does not satisfy min_test_pos/min_test_neg")
        auto_info = {"mode": "manual"}
    else:
        train_runs, val_runs, test_runs, auto_info = auto_choose_runs(
            df,
            target_ratios=(args.target_train, args.target_val, args.target_test),
            n_val_runs=args.n_val_runs,
            n_test_runs=args.n_test_runs,
            min_train_pos=args.min_train_pos,
            min_train_neg=args.min_train_neg,
            min_val_pos=args.min_val_pos,
            min_val_neg=args.min_val_neg,
            min_test_pos=args.min_test_pos,
            min_test_neg=args.min_test_neg,
            max_search_combos=args.max_search_combos,
        )
        auto_info["mode"] = "auto"
        user_val, user_test = val_runs, test_runs

    train_df, val_df, test_df = build_split_dfs_by_runs(df, train_runs, user_val, user_test)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    train_df.to_csv(output_dir / "train_windows.csv", index=False)
    val_df.to_csv(output_dir / "val_windows.csv", index=False)
    test_df.to_csv(output_dir / "test_windows.csv", index=False)
    full_df.to_csv(output_dir / "window_features.csv", index=False)
    full_df[["window_id", "run_key", "session_id", "split"]].to_csv(output_dir / "splits_run_strict.csv", index=False)

    with open(output_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    legal_loro_test_runs = eligible_both_class_runs(df, min_pos=args.min_test_pos, min_neg=args.min_test_neg)
    legal_loro_val_runs = eligible_both_class_runs(df, min_pos=args.min_val_pos, min_neg=args.min_val_neg)

    manifest = {
        "input_dir": str(input_dir),
        "shortcut_level": args.shortcut_level,
        "n_total_windows": int(len(df)),
        "n_runs": int(df["run_key"].nunique()),
        "n_sessions": int(df["session_id"].nunique()),
        "feature_cols_before": raw_feature_cols if raw_feature_cols is not None else "derived_from_numeric_columns",
        "n_feature_cols_after": int(len(feature_cols)),
        "feature_drop_info": drop_info,
        "run_stats": rk_stats.to_dict(orient="records"),
        "split_choice": auto_info,
        "split_runs": {
            "train_runs": list(train_runs),
            "val_runs": list(user_val),
            "test_runs": list(user_test),
        },
        "split_constraints": {
            "n_val_runs": int(args.n_val_runs),
            "n_test_runs": int(args.n_test_runs),
            "min_train_pos": int(args.min_train_pos),
            "min_train_neg": int(args.min_train_neg),
            "min_val_pos": int(args.min_val_pos),
            "min_val_neg": int(args.min_val_neg),
            "min_test_pos": int(args.min_test_pos),
            "min_test_neg": int(args.min_test_neg),
        },
        "split_overview": {
            "train": split_overview(train_df),
            "val": split_overview(val_df),
            "test": split_overview(test_df),
        },
        "loro_eligibility": {
            "both_class_only": (not args.allow_single_class_loro),
            "eligible_test_runs": legal_loro_test_runs,
            "eligible_val_runs": legal_loro_val_runs,
            "excluded_single_class_or_small_runs": [rk for rk in df["run_key"].drop_duplicates().tolist() if rk not in legal_loro_test_runs],
        },
        "warnings": [
            "This is a hard-constrained run-level split: the same run never appears across train/val/test.",
            "LORO folds are generated only from legal both-class runs unless --allow-single-class-loro is explicitly enabled.",
            "Single-class runs remain available for training but are excluded from binary val/test LORO folds.",
        ],
    }
    safe_json_dump(manifest, output_dir / "manifest.json")

    if args.write_folds:
        write_loro_folds(
            df,
            output_dir,
            feature_cols,
            manifest,
            min_train_pos=args.min_train_pos,
            min_train_neg=args.min_train_neg,
            min_val_pos=args.min_val_pos,
            min_val_neg=args.min_val_neg,
            min_test_pos=args.min_test_pos,
            min_test_neg=args.min_test_neg,
            allow_single_class_loro=args.allow_single_class_loro,
        )

    print(json.dumps({
        "output_dir": str(output_dir),
        "n_feature_cols_after": len(feature_cols),
        "split_runs": manifest["split_runs"],
        "loro_eligibility": manifest["loro_eligibility"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
