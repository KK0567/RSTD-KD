# RSTD-KD: Risk Stratification and Task-Cost-Aware Decision via Knowledge Distillation

Official code and data repository for the paper **"RSTD-KD: A Task-Cost-Aware Approach for UAV Communication Risk Warning via Knowledge Distillation"** submitted to CMC-Computers, Materials & Continua.

## Overview

RSTD-KD is a lightweight risk-state recognition framework for UAV communication security. It uses knowledge distillation to transfer a teacher ensemble's knowledge into a compact student network, and applies Platt calibration with task-cost-aware threshold selection for three-level risk-state decision (Normal / Medium / High).

### Key Features

- **Teacher-Student Knowledge Distillation**: 5-model HistGBM teacher ensemble → compact MLP student (96→48 backbone)
- **Cascade Risk Decision**: Binary detection → attack-type classification → risk-state mapping
- **Platt Probability Calibration**: Reduces Brier Score from 0.0762 to 0.0113
- **Task-Cost-Aware Threshold Selection**: Minimizes expected cost under FP/FN ratio constraints
- **Cross-Dataset Portability**: Validated on both ECU-IoFT (packet-level WiFi) and UAVIDS-2025 (flow-level FANET)

## Project Structure

```
RSTD-KD/
├── src/
│   ├── main_experiment/
│   │   ├── 1_data_prep.py                  # Window generation + attack-stratified split
│   │   ├── 2_teacher_train.py              # Teacher ensemble training
│   │   └── 3_student_train.py              # Student KD training + cascade evaluation
│   └── revision_experiments/
│       ├── table2_ablation_study.py        # KD component ablation
│       ├── table4_uavids_generalization.py # UAVIDS-2025 portability
│       ├── table5_mh_separability.py       # Medium/High separability diagnostic
│       ├── table6_cost_sensitivity.py      # FP/FN cost-ratio analysis
│       ├── table7_interference_simulation.py # Physical interference robustness
│       ├── adaptive_window_experiment.py   # Multi-window adaptive strategies
│       └── unified_rerun.py               # Unified re-run (Tables 3, 6, 7)
├── data/
│   ├── uavids_2025/
│   │   ├── UAVIDS-2025.csv                # External dataset (122,171 flow records)
│   │   └── exploration.json               # Dataset statistics
│   └── processed/
│       ├── attack_stratified_w8/           # Window datasets for w ∈ {8,16,24,32,48,64}
│       ├── attack_stratified_w16/
│       ├── attack_stratified_w24/
│       ├── attack_stratified_w32/          # Default window size
│       ├── attack_stratified_w48/
│       └── attack_stratified_w64/
├── results/
│   ├── table3_window_sensitivity/         # Window size sensitivity (6 sizes)
│   ├── table4_uavids_portability/         # UAVIDS-2025 5-seed + 5-fold CV
│   ├── table5_mh_separability/            # M/H separability (6 KD configs)
│   ├── table6_cost_ratio/                 # Cost-ratio sensitivity (5 ratios)
│   ├── table7_interference/               # Physical interference (4 types)
│   └── adaptive_window/                   # Adaptive window strategies
├── models/
│   └── attack_stratified/
│       └── student_w{8,16,24,32,48,64}/   # Pre-trained student models
├── figures/                               # PDF figures
├── requirements.txt
└── README.md
```

## Datasets

### ECU-IoFT (Primary)
- **Source**: Real Tello drone WiFi 802.11 captures
- **Records**: 54,492 packet-level records, 93 features
- **Attacks**: WiFi Deauthentication (→Medium), WPA2-PSK Cracking (→High), Tello API Exploit (→High)
- **Processing**: Sliding window (w=32, step=32) → 1,702 windows → attack-stratified 70/15/15 split

### UAVIDS-2025 (External Validation)
- **Source**: NS-3 FANET simulation (AODV, IEEE 802.11ac)
- **Records**: 122,171 flow-level records, 25 features
- **Attacks**: Blackhole (→Medium), Flooding (→High), Sybil (→High), Wormhole (→High)
- **Processing**: Per-flow features → 70/15/15 stratified split (5 seeds + 5-fold CV)

## Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Main Experiment (ECU-IoFT, w=32)
```bash
# Step 1: Generate windowed dataset
python src/main_experiment/1_data_prep.py

# Step 2: Train teacher ensemble
python src/main_experiment/2_teacher_train.py

# Step 3: Train student via KD
python src/main_experiment/3_student_train.py
```

### 3. Revision Experiments
```bash
# Window sensitivity (Table 3)
python src/revision_experiments/unified_rerun.py

# UAVIDS-2025 portability (Table 4)
python src/revision_experiments/table4_uavids_generalization.py

# M/H separability (Table 5)
python src/revision_experiments/table5_mh_separability.py

# Cost-ratio sensitivity (Table 6)
python src/revision_experiments/table6_cost_sensitivity.py

# Physical interference (Table 7)
python src/revision_experiments/table7_interference_simulation.py

# Adaptive window analysis
python src/revision_experiments/adaptive_window_experiment.py
```

## Key Results

| Experiment | Metric | Result |
|---|---|---|
| Main (ECU-IoFT, w=32) | Accuracy / Macro-F1 / High-Recall | 93.75% / 94.65% / 100.00% |
| Window Sensitivity | Accuracy range (w∈{8,64}) | 93.02%–94.71% |
| UAVIDS-2025 (5-seed) | Accuracy / Macro-F1 | 94.92% / 94.38% |
| UAVIDS-2025 (5-fold CV) | Accuracy / Balanced Acc | 94.84% / 93.32% |
| M/H Separability | Direct sep. accuracy (all configs) | 1.000 |
| Cost Ratio (1:1) | Selected threshold τ | 0.50 |
| Channel Fading (α=0.1) | Accuracy retention | 90.5% |
| EMI Noise (all SNR) | Accuracy retention | 100.0% |
| Packet Loss (70%) | Accuracy retention | 94.6% |
| Packet Reorder (70%) | Accuracy retention | 62.9% |

## Model Architecture

**StudentRiskCascade**: LayerNorm → Linear(65, 96) → GELU → Dropout(0.18) → Linear(96, 48) → GELU → Dropout(0.12) → Binary Head(48, 1) + Attack Head(48, 3)

**Cascade Decision**: binary_prob ≥ 0.5 → argmax(attack_probs) → ATTACK_TO_RISK mapping

## Citation

If you use this code or data, please cite:
```
@article{rstd_kd_2026,
  title={RSTD-KD: A Task-Cost-Aware Approach for UAV Communication Risk Warning via Knowledge Distillation},
  journal={CMC-Computers, Materials \& Continua},
  year={2026}
}
```

## License

This project is provided for academic research purposes.
