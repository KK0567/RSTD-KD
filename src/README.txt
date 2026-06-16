# RSTD-KD

RSTD-KD is a task-cost-aware framework for UAV communication risk warning via knowledge distillation. The framework is designed to support mission-sensitive UAV communication security scenarios, where different tasks may have different tolerances for false alarms and missed attacks.

Instead of relying on a fixed anomaly decision threshold, RSTD-KD calibrates the predicted anomaly probability and selects task-specific thresholds according to the relative cost of false positives and false negatives. This design enables flexible risk warning under different UAV mission requirements.

## Overview

RSTD-KD is built around three core components:

1. **Knowledge distillation-based risk modeling**
   A compact risk warning model is trained to learn task-relevant decision behavior from a stronger model, improving the effectiveness of UAV communication risk prediction.

2. **Probability calibration**
   The predicted anomaly probabilities are calibrated to improve the reliability of downstream threshold selection and risk warning decisions.

3. **Task-cost-aware threshold decision-making**
   Different UAV missions are assigned different false-positive and false-negative costs. The final warning threshold is selected according to the task-specific cost setting, enabling flexible decision-making for different operational scenarios.

## Supported Dataset

The repository contains experimental materials for UAV communication risk warning based on the ECU-IoFT dataset.

The processed data are organized as window-level UAV communication behavior samples. Each sample contains communication behavior features and the corresponding security label.

## Task Settings

RSTD-KD considers different UAV mission scenarios with different risk preferences:

* **False-positive-sensitive scenario**: false alarms are assigned a higher cost, and the decision threshold is adjusted to reduce unnecessary warnings.
* **False-negative-sensitive scenario**: missed attacks are assigned a higher cost, and the decision threshold is adjusted to improve attack recall.
* **Balanced scenario**: false positives and false negatives are treated with comparable importance.

## Environment

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
Python
PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib
```

## Usage

The source code is provided in the `src/` directory, and the processed dataset is placed in the `Data/` directory.

The general experimental workflow includes:

1. training the risk warning model;
2. calibrating predicted anomaly probabilities;
3. selecting task-specific decision thresholds;
4. evaluating classification performance and task-specific warning cost.
