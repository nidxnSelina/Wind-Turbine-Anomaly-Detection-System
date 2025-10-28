# AnomalyForecastModel

## Overview
`AnomalyForecastModel` is a supervised anomaly prediction model built around a **RandomForestClassifier**.  
It automates the process of predicting abnormal behavior in time-series turbine data.
The model includes end-to-end handling of data preparation, class balancing, feature scaling, and prediction.
During training, the model automatically labels samples as normal or anomalous by referencing window definition files 
(positive_windows.csv and negative_windows.csv) against the main observation dataset.
During prediction, the trained model classifies unseen time-series data as either normal or anomalous.

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
- This model supports two modes:
    - 'train'   — builds a classifier from internally stored labeled time windows.
    - 'predict' — loads the saved scaler + classifier and scores incoming data.

### Training Mode
- Requires a dataset (unlabeled) containing the following feature columns the model would be trained on: 
`wind_speed`, `power`, `pitch1_moto_tmp`, `pitch2_moto_tmp`, `pitch3_moto_tmp`, `environment_tmp`, `int_tmp`

- Expects a folder named 'anomaly_data' next to this file containing:
    - observations.csv        (raw time series with a 'time' column)
    - positive_windows.csv    (columns: 'startTime', 'endTime')
    - negative_windows.csv    (columns: 'startTime', 'endTime')

- The model checks whether the timestamp falls into any of the defined windows:
    - If time ∈ negative_windows.csv → is_anomaly = 1
    - If time ∈ positive_windows.csv → is_anomaly = 0

The derived label `is_anomaly` is the target and the labeled dataset is then used to train the Random Forest classifier.

### Prediction Mode
- Expects `dataset` to contain:
    - The feature columns that the model is trained on
    - A DatetimeIndex column OR A time column named 'time' (will be parsed to datetime and set as index).

- In debug, it uses an external dataset located under `test_data/` in `.xls` format.

### Artifacts
- './checkpoints/scaler.pkl'      : StandardScaler fit on training data.
- './checkpoints/classifier.pkl'  : Trained RandomForestClassifier.

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python3 run.py --model AnomalyForecastModel --mode train --debug
```

### Predict anomalies
```bash
python3 run.py --model AnomalyForecastModel --mode predict --debug
```

### CLI Arguments
| Argument | Description |
|-----------|-------------|
| `--model` | Model name `AnomalyForecastModel` |
| `--mode` | `predict` or `train` |
| `--debug` | Debug |

## Output
All outputs are stored under `checkpoints\`.
- **Training:** 
    - fitted model and scalar (`.pkl` or `.joblib`)
    - (prints a confusion matrix using the training observations)
- **Prediction:** 
    - a time-indexed DataFrame indicating whether each observation is classified as an anomaly (True) or normal (False).

## Example Output
Train mode
```
[INFO:fit:161] Confusion matrix:
[[69944    97]
 [    3  4775]]
```

Prediction mode
```
                         is_anomaly
time                                
2023-09-01 00:00:00         False
2023-09-01 00:10:00         False
2023-09-01 00:20:00         False
2023-09-01 00:30:00          True
2023-09-01 00:40:00          True
2023-09-01 00:50:00         False
2023-09-01 01:00:00         False
```