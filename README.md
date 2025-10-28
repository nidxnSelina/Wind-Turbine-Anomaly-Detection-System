# Wind Turbine Time-Series Anomaly Detection System

This project provides an anomaly detection framework for wind turbine time-series data sourced from **Kaggle**. 
It supports both rule-based and machine-learning–based approaches to detect abnormal behavior in turbine performance metrics such as power and pitch temperatures.

---

## System Components

The system includes two main models:

### 1. `ThresholdDetectionModel`
A **rule-based model** for fast, interpretable anomaly detection using user-defined thresholds.

**Highlights**
- No training required  
- Lightweight and fast  
- Works on a single numeric feature (e.g., `power`)  
- Reports the proportion of values falling below, within, or above thresholds  

**Usage Example**
```bash
python3 run.py --model ThresholdDetectionModel --mode predict   --target power --left_threshold 300.0 --right_threshold 500.0 --debug
```

---

### 2. `AnomalyForecastModel`
A **supervised Random Forest–based model** that learns from historical data to detect anomalies automatically.

**Highlights**
- Uses `RandomForestClassifier` for binary classification (`is_anomaly` label derived)
- Automatically handles class imbalance with `RandomUnderSampler`
- Scales features using `StandardScaler`
- Splits training/testing automatically and saves model artifacts
- Supports both `train` and `predict` modes

**Usage Example**
```bash
python3 run.py --model AnomalyForecastModel --mode train --debug
python3 run.py --model AnomalyForecastModel --mode predict --debug
```

---

## Data Requirements

Both models operate on structured turbine datasets containing:
- `time` (datetime index or column)
- Numeric features such as  
  `wind_speed`, `power`, `pitch1_moto_tmp`, `pitch2_moto_tmp`, `pitch3_moto_tmp`, `environment_tmp`, `int_tmp`

Training mode additionally expects:
- `anomaly_data/observations.csv`
- `positive_windows.csv`, `negative_windows.csv` (for labeling)

---

## Installation & Execution

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train or predict:
```bash
python3 run.py --model AnomalyForecastModel --mode train --debug
python3 run.py --model ThresholdDetectionModel --mode predict --target power --left_threshold 300 --right_threshold 500
```

---

## Outputs
Outputs are saved in `checkpoints/` folder.
- **AnomalyForecastModel:**  
  JSON or DataFrame of time-indexed anomaly flags (`True`/`False`).

- **ThresholdDetectionModel:**  
  Summary table of proportions for below, normal, and above-threshold data.

---

## Author
Developed by **Selina Wang**  
