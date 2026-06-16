#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ECU-IoFT window 数据构建与模型兼容划分脚本。

核心目标：
1) 生成与 teacher/student 代码兼容的 train_windows.csv / val_windows.csv / test_windows.csv。
2) 保证 train/val/test 三个 split 都包含：
   - 正常样本 y_attack_type=0, y_risk=0
   - 低/中风险攻击样本 y_attack_type=1, y_risk=1
   - 高风险攻击样本 y_attack_type=2 或 3, y_risk=2
3) 进一步保证 teacher 攻击类型模型需要的 1/2/3 三个攻击类型在训练集中都存在；
   默认也要求 0/1/2/3 四种攻击类型在 train/val/test 中都出现。

注意：
- 当前 teacher/student 代码中风险 id=1 被命名为 medium；如果论文表述为“低风险”，
  建议在文字中说明该类为较低危害攻击/中低风险攻击，代码层面仍保持 id=1 不变。
- TELLO API Exploit 窗口数很少，默认采用 rare 类逐窗口分配，以满足 teacher/student 的硬性类别要求。
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

# 与 teacher/student 保持 id 兼容：0=normal, 1=medium/low-risk attack, 2=high
RISK_MAP = {
    "No Attack": (0, "normal"),
    "Wifi Deauthentication Attack": (1, "medium"),
    "WPA2-PSK WIFI Cracking Attack": (2, "high"),
    "TELLO API Exploit": (2, "high"),
}

RISK_ID_TO_NAME = {0: "normal", 1: "medium", 2: "high"}
RISK_ALIAS_FOR_REPORT = {0: "normal", 1: "low_or_medium", 2: "high"}

INFO_TOKENS = [
    "ack", "deauthentication", "authentication", "eapol", "beacon",
    "probe", "request", "response", "data", "null", "qos", "udp",
    "icmp", "exploit", "api", "sae", "handshake",
]

PROTO_VALUES = ["802.11", "UDP", "EAPOL", "ICMP"]


# -----------------------------
# 基础工具
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
        raise ValueError(f"Time 列存在无法解析的值，数量={int(dt.isna().sum())}，示例={bad}")
    return dt


def dominant_attack_label(labels: pd.Series) -> str:
    """窗口标签采用主导攻击类型；同票时优先攻击而不是 No Attack。"""
    vc = labels.value_counts()
    if vc.empty:
        return "No Attack"
    if len(vc) == 1:
        return str(vc.index[0])
    items = sorted(vc.items(), key=lambda kv: (kv[1], kv[0] != "No Attack"), reverse=True)
    return str(items[0][0])


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
# 窗口特征构建
# -----------------------------
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
        raise ValueError(f"未知 Type of Attack: {dominant}")

    y_attack_type = ATTACK_TYPE_MAP[dominant]
    y_bin = int(dominant != "No Attack")
    y_risk, y_risk_name = RISK_MAP[dominant]

    info_len = info.str.len().values.astype(float)

    # -----------------------------
    # 增强特征 1：窗口内时间动态
    # 这些特征不使用标签，只刻画包间时间间隔和包速率，适合区分突发攻击与正常通信。
    # -----------------------------
    t_ns = dt.astype("int64").to_numpy(dtype=np.int64)
    t_sec = t_ns.astype(np.float64) / 1e9
    if len(t_sec) > 1:
        inter_arrival = np.diff(t_sec)
        inter_arrival = np.clip(inter_arrival, 0.0, None)
    else:
        inter_arrival = np.array([], dtype=np.float64)

    duration_raw = float((dt.iloc[-1] - dt.iloc[0]).total_seconds())
    packet_rate = float(len(sub) / max(duration_raw, 1e-6))

    # -----------------------------
    # 增强特征 2：长度序列变化
    # 攻击窗口常体现为包长突变、重复控制帧或握手帧聚集。
    # -----------------------------
    if len(lengths) > 1:
        length_diff = np.diff(lengths)
        x_idx = np.arange(len(lengths), dtype=np.float64)
        length_slope = float(np.polyfit(x_idx, lengths.astype(np.float64), 1)[0])
    else:
        length_diff = np.array([], dtype=np.float64)
        length_slope = 0.0

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
        "window_packet_count": int(len(sub)),
        "duration_sec": duration_raw,
        "packet_rate": packet_rate,
        "inter_arrival_mean": float(inter_arrival.mean()) if inter_arrival.size else 0.0,
        "inter_arrival_std": float(inter_arrival.std(ddof=0)) if inter_arrival.size else 0.0,
        "inter_arrival_min": float(inter_arrival.min()) if inter_arrival.size else 0.0,
        "inter_arrival_max": float(inter_arrival.max()) if inter_arrival.size else 0.0,
        "inter_arrival_q25": q(inter_arrival, 0.25),
        "inter_arrival_q50": q(inter_arrival, 0.50),
        "inter_arrival_q75": q(inter_arrival, 0.75),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "length_min": float(lengths.min()),
        "length_max": float(lengths.max()),
        "length_q25": q(lengths, 0.25),
        "length_q50": q(lengths, 0.50),
        "length_q75": q(lengths, 0.75),
        "length_first": float(lengths[0]) if lengths.size else 0.0,
        "length_last": float(lengths[-1]) if lengths.size else 0.0,
        "length_slope": length_slope,
        "length_diff_mean": float(length_diff.mean()) if length_diff.size else 0.0,
        "length_diff_std": float(length_diff.std(ddof=0)) if length_diff.size else 0.0,
        "length_absdiff_mean": float(np.abs(length_diff).mean()) if length_diff.size else 0.0,
        "length_absdiff_max": float(np.abs(length_diff).max()) if length_diff.size else 0.0,
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
        # 这两个字段会被 teacher/student 的 ALWAYS_DROP_EXACT 丢弃，保留在表中仅用于审计。
        "attack_packet_count": int((typ == "Attack").sum()),
        "attack_packet_ratio": ratio_bool(typ == "Attack"),
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

    # -----------------------------
    # 增强特征 3：协议转移统计
    # 仅使用窗口内协议序列，不使用任何标签。
    # -----------------------------
    proto_seq = proto.tolist()
    n_trans = max(1, len(proto_seq) - 1)
    for a, b in [("802.11", "802.11"), ("802.11", "EAPOL"), ("EAPOL", "EAPOL"), ("UDP", "UDP"), ("802.11", "UDP"), ("UDP", "ICMP")]:
        trans_cnt = 0
        for i in range(len(proto_seq) - 1):
            if proto_seq[i] == a and proto_seq[i + 1] == b:
                trans_cnt += 1
        ka = re.sub(r"[^a-zA-Z0-9]+", "_", a).strip("_").lower()
        kb = re.sub(r"[^a-zA-Z0-9]+", "_", b).strip("_").lower()
        feat[f"proto_trans_{ka}_to_{kb}_count"] = int(trans_cnt)
        feat[f"proto_trans_{ka}_to_{kb}_ratio"] = float(trans_cnt / n_trans)

    info_lower = info.str.lower()
    for tok in INFO_TOKENS:
        mask = info_lower.str.contains(tok, regex=False)
        cnt = int(mask.sum())
        feat[f"info_tok_{tok}_count"] = cnt
        feat[f"info_tok_{tok}_ratio"] = float(cnt / len(sub))

    return feat


def build_windows(df: pd.DataFrame, window_size: int, step_size: int) -> pd.DataFrame:
    rows: List[Dict] = []
    wid = 0
    for start in range(0, len(df) - window_size + 1, step_size):
        sub = df.iloc[start:start + window_size]
        rows.append(build_features_one_window(sub, window_id=wid))
        wid += 1
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("窗口为空，请减小 window_size")
    return out


# -----------------------------
# 模型兼容划分逻辑
# -----------------------------
def _split_counts_for_class(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """
    单个类别内部的 train/val/test 数量。
    n>=3 时强制三者均至少 1，保证 rare 类也不会消失。
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

    # 收缩到总数 n。
    while n_train + n_val + n_test > n:
        # 优先从训练集扣，因为 val/test 至少要保留 1 个。
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
    按 y_attack_type 分层切分。

    为什么不用只按 y_risk 切分？
    - teacher 的攻击类型模型会输出 teacher_attack_prob_1/2/3；
    - 如果 train 缺少 y_attack_type=3，原 teacher 代码会因为 index(3) 报错；
    - 因此必须优先保证攻击类型 1/2/3 在训练集中存在。
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
            f"窗口化之后全局缺少攻击类型 {missing_global}，无法满足 teacher/student 固定三攻击头要求。"
            f"请减小 --window-size 或检查原始标签。当前可用={available_attack_types}"
        )

    for attack_id, sub in out.groupby("y_attack_type", sort=True):
        attack_id = int(attack_id)
        idxs = sub.sort_values("packet_id_start").index.tolist()
        n = len(idxs)
        n_train, n_val, n_test = _split_counts_for_class(n, train_ratio, val_ratio, test_ratio)

        # TELLO 特别少，确保训练集至少保留 min_train_tello 个。
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
        raise RuntimeError("存在未分配 split 的窗口")

    return out


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


def assert_model_compatible_splits(win_df: pd.DataFrame, require_all_attack_types_each_split: bool = True) -> None:
    """
    硬性校验：不满足就直接报错，不生成有隐患的数据集。
    """
    required_risks = {0, 1, 2}
    required_attack_types = {0, 1, 2, 3}

    problems: List[str] = []
    for split in ["train", "val", "test"]:
        sub = win_df.loc[win_df["split"] == split]
        risk_present = set(sub["y_risk"].astype(int).unique().tolist())
        attack_present = set(sub["y_attack_type"].astype(int).unique().tolist())

        miss_risk = sorted(required_risks - risk_present)
        if miss_risk:
            problems.append(f"{split} 缺少风险类别 y_risk={miss_risk}，当前={sorted(risk_present)}")

        if require_all_attack_types_each_split:
            miss_attack = sorted(required_attack_types - attack_present)
            if miss_attack:
                problems.append(f"{split} 缺少攻击类型 y_attack_type={miss_attack}，当前={sorted(attack_present)}")

    # teacher 训练阶段至少必须包含 1/2/3，否则 teacher_attack_prob_3 等会出错。
    train_attack_present = set(win_df.loc[win_df["split"] == "train", "y_attack_type"].astype(int).unique().tolist())
    miss_train_attack = sorted({1, 2, 3} - train_attack_present)
    if miss_train_attack:
        problems.append(f"train 缺少 teacher 攻击模型需要的攻击类型 {miss_train_attack}")

    if problems:
        report = build_split_report(win_df)
        raise ValueError(
            "数据划分不满足 teacher/student 要求：\n- "
            + "\n- ".join(problems)
            + "\n当前分布：\n"
            + json.dumps(report, ensure_ascii=False, indent=2)
        )


# -----------------------------
# 主程序
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="从 ECU-IoFT 原始包级 CSV 构建 teacher/student 兼容的风险分层窗口数据")
    parser.add_argument("--input-csv", type=str, default="../Data/ECU-IoFT-Dataset.csv")
    parser.add_argument("--output-dir", type=str, default="Dataset/ecu_attack_risk_windows_v3")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--step-size", type=int, default=32, help="建议与 window-size 相同，避免重叠窗口泄露")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-train-tello", type=int, default=1)
    parser.add_argument(
        "--allow-missing-attack-type-in-val-test",
        action="store_true",
        help="默认要求 train/val/test 都有 0/1/2/3；打开后只强制风险类别完整、train 攻击类型完整。",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw = pd.read_csv(args.input_csv)
    raw["_dt"] = parse_time(raw["Time"])
    raw = raw.reset_index(drop=True)

    win_df = build_windows(raw, window_size=args.window_size, step_size=args.step_size)
    win_df = assign_splits_model_compatible(
        win_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_train_tello=args.min_train_tello,
    )

    assert_model_compatible_splits(
        win_df,
        require_all_attack_types_each_split=not args.allow_missing_attack_type_in_val_test,
    )

    feature_cols = [
        c for c in win_df.columns
        if c not in {
            "window_id", "row_id", "packet_id_start", "packet_id_end",
            "time_start", "time_end", "y_bin", "y_attack_type", "y_attack_type_name",
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
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "min_train_tello": args.min_train_tello,
        "attack_type_map": ATTACK_TYPE_MAP,
        "risk_map": {k: {"id": v[0], "name": v[1]} for k, v in RISK_MAP.items()},
        "risk_alias_for_report": RISK_ALIAS_FOR_REPORT,
        "n_windows": int(len(win_df)),
        "split_counts": win_df["split"].value_counts().to_dict(),
        "attack_type_counts": win_df["y_attack_type_name"].value_counts().to_dict(),
        "risk_counts": win_df["y_risk_name"].value_counts().to_dict(),
        "split_report": split_report,
        "compatibility_checks": {
            "each_split_has_y_risk_0_1_2": True,
            "train_has_attack_type_1_2_3_for_teacher": True,
            "each_split_has_attack_type_0_1_2_3": not args.allow_missing_attack_type_in_val_test,
        },
        "notes": [
            "默认采用非重叠窗口，step-size 建议等于 window-size。",
            "划分优先按 y_attack_type 分层，而不是只按 y_risk 分层，因为 teacher 代码固定读取 teacher_attack_prob_1/2/3。",
            "y_risk=1 在现有 teacher/student 中命名为 medium；如论文写低风险，请将其解释为中低风险/较低危害攻击。",
            "TELLO API Exploit 窗口数极少，当前划分只保证模型流程可运行，不宜夸大 TELLO 类型泛化能力。",
        ],
    }
    safe_json_dump(manifest, out_dir / "manifest.json")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
