# ThresholdDetectionModel

## Overview
`ThresholdDetectionModel` is a rule-based anomaly detection model.  
It is designed to report the proportion of anomalies in time-series turbine data by comparing against user-defined thresholds.  
This model requires no training and is useful for quick checks and alerting.

## Key Features
- **No training required** — purely rule-based detection
- **Fast and interpretable**
- **Simple CLI interface**

## Workflow
1. **Load data**.
2. **Specify a target column**.
3. **Provide lower and upper thresholds**.
4. The model flags any rows where the target value is outside the threshold range. 

## Data Requirements
- A **DataFrame** containing a numeric target column that represents the monitored metric. 
- Available target columns are:
```
"wind_speed", "power", "pitch1_moto_tmp", "pitch2_moto_tmp", "pitch3_moto_tmp", "environment_tmp", "int_tmp"
```
- This model only supports 'predict' mode as it requires no machine learning model for training.

## Usage

### Install Dependencies
```
pip install -r requirements.txt
```

### Run Detection
Example:
```bash
python3 run.py   --model ThresholdDetectionModel   --mode predict   --target power   --left_threshold 300.0   --right_threshold 500.0   --debug
```

### CLI Arguments
| Argument | Description |
|-----------|-------------|
| `--model` | Model name `ThresholdDetectionModel` |
| `--mode` | Must be `predict` (training not supported) |
| `--target` | Column name to evaluate (e.g., power) |
| `--left_threshold` | Minimum acceptable value |
| `--right_threshold` | Maximum acceptable value |
| `--debug` | Debug |

## Output
- **Returns:** When ran in 'predict' mode, the model returns the ratio of records that fall into each classification category, based on the provided threshold values.

| Label | Meaning         | Condition                                        |
| ----- | --------------- | ----------------------------------------------- |
| `0`   | Normal          | Value within `[left_threshold, right_threshold]` |
| `1`   | Below Threshold | Value < `left_threshold`                         |
| `2`   | Above Threshold | Value > `right_threshold`                        |

- **Example:**  
```
          wind_speed
label                
0            0.82
1            0.10
2            0.08
```
