# AnomalyForecastModel

## Overview
`AnomalyForecastModel` is a supervised anomaly detection model built around a **RandomForestClassifier**.  
It automates the process of detecting abnormal behavior in time-series data, such as wind turbine telemetry.  
The model includes end-to-end handling of data preparation, class balancing, feature scaling, and prediction.

## Key Features
- **Supervised learning** with scikit-learn's `RandomForestClassifier`
- **Automatic class balancing** using `RandomUnderSampler`
- **StandardScaler normalization** to ensure consistent feature scaling
- **Train/test split** for evaluation during training
- **Reusable trained model** saved via `joblib`
- **Easy CLI interface** via `run.py`

## Workflow
1. **Load and preprocess** training or prediction data.
2. **Split** into train/test subsets when training.
3. **Balance** the training data to handle class imbalance.
4. **Scale** the features using `StandardScaler`.
5. **Train** the Random Forest model.
6. **Predict** anomalies on new datasets.

## Data Requirements
### Training Mode
- Requires a labeled dataset containing:
  - Feature columns (sensor readings, etc.)
  - A binary label column named **`is_anomaly`** (0 = normal, 1 = anomaly)

### Prediction Mode
- Requires only the feature columns.
- Can use an external dataset, typically located under `test_data/`.

## CLI Usage

### Train the model
```bash
python3 run.py --model AnomalyForecastModel --mode train --debug
```

### Predict anomalies
```bash
python3 run.py --model AnomalyForecastModel --mode predict --freq 15min --debug
```

## Code Example
```python
import pandas as pd
from argparse import Namespace
from src.model.AnomalyForecastModel.model import AnomalyForecastModel

df = pd.read_csv("test_data/sample.csv")
args = Namespace(flag="predict", freq="15min")

model = AnomalyForecastModel(flag=args.flag, freq=args.freq, dataset=df)
predictions = model.run(args)
```

## Model Details
- **Split ratio:** 80% training / 20% testing
- **Sampler:** `RandomUnderSampler(random_state=42)`
- **Scaler:** `StandardScaler()`
- **Estimator:** `RandomForestClassifier(random_state=42)`
- **Metrics:** Confusion matrix and accuracy printed during debug mode

## Output
- **Training:** Saves fitted model (`.pkl` or `.joblib`) and logs metrics
- **Prediction:** Outputs a DataFrame or list of predicted anomaly flags

## Example Output
```
[INFO] Training completed: Accuracy = 0.87
[INFO] Confusion Matrix:
[[900   30]
 [ 50  120]]
```

## Folder Structure
```
src/model/AnomalyForecastModel/
├─ anomaly_data/            # Training data folder
├─ model.py                 # Model implementation
├─ __init__.py
└─ README.md                # This file
```

## Notes
- Make sure your dataset has consistent column names.
- Use `--debug` mode to print detailed logs.
- Use `.pkl` or `.joblib` to save trained models for later use.

## License
MIT
