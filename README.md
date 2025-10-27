# Wind Turbine Anomaly & Threshold Detection

A small, production-leaning toolkit for anomaly detection on wind-turbine (or similar time-series) data.  
It currently provides:

- **AnomalyForecastModel** — a supervised, classical ML pipeline (RandomForest) with scaling and class-imbalance handling.
- **ThresholdDetectionModel** — a lightweight, training-free detector using simple numeric thresholds.

## Features

- End-to-end CLI: train and predict with `run.py`
- Balanced training via `RandomUnderSampler`
- Train/test split with held-out evaluation
- Consistent scaling using `StandardScaler` fit on training data only
- Predict on external datasets without retraining
- Minimal interface via `argparse.Namespace`

## Project Structure (simplified)

```
.
├─ run.py
├─ src/
│  ├─ cmd_cli.py
│  ├─ utils.py
│  └─ model/
│     ├─ AnomalyForecastModel/
│     │  ├─ anomaly_data/
│     │  └─ model.py
│     └─ ThresholdDetectionModel/
│        └─ model.py
└─ test_data/
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

### Train

```bash
python3 run.py --model AnomalyForecastModel --mode train --debug
```

### Predict

```bash
python3 run.py --model AnomalyForecastModel --mode predict --freq 15min --debug
```

### Threshold Detection

```bash
python3 run.py --model ThresholdDetectionModel --mode predict --target wind_speed --left_threshold 3.0 --right_threshold 25.0 --debug
```

## .gitignore

```gitignore
__pycache__/
*.pyc
checkpoints/
src/model/AnomalyForecastModel/anomaly_data/
*.csv
*.pkl
```

## License

MIT
