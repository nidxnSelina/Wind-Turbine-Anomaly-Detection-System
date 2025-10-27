# ThresholdDetectionModel

## Overview
`ThresholdDetectionModel` is a lightweight, rule-based anomaly detection model.  
It flags anomalies in time-series data by comparing sensor readings against user-defined thresholds.  
This model requires no training and is useful for quick checks, diagnostics, and interpretable alerting.

## Key Features
- **No training required** — purely rule-based detection
- **Fast and interpretable**
- **Simple CLI interface**
- **Works directly with raw or preprocessed numeric data**
- **Easy integration** into pipelines or APIs

## Workflow
1. **Load data** (e.g., turbine telemetry).
2. **Specify a target column** (e.g., wind_speed).
3. **Provide lower and upper thresholds**.
4. The model flags any rows where the target value is outside the threshold range.

## Data Requirements
- A **DataFrame** containing a numeric column that represents the monitored metric (e.g., wind speed, temperature, power).

Example columns:
```
time, wind_speed, generator_speed, power, temperature
```

## CLI Usage

### Run prediction
```bash
python3 run.py   --model ThresholdDetectionModel   --mode predict   --target wind_speed   --left_threshold 3.0   --right_threshold 25.0   --debug
```

### CLI Arguments
| Argument | Description |
|-----------|-------------|
| `--target` | Column name to evaluate (e.g., wind_speed) |
| `--left_threshold` | Minimum acceptable value (optional) |
| `--right_threshold` | Maximum acceptable value (optional) |
| `--mode` | Must be `predict` (training not supported) |
| `--debug` | Prints detailed runtime information |

## Code Example
```python
import pandas as pd
from argparse import Namespace
from src.model.ThresholdDetectionModel.model import ThresholdDetectionModel

df = pd.read_csv("test_data/telemetry.csv")
args = Namespace(mode="predict")

model = ThresholdDetectionModel(data=df, target=["wind_speed"])
model.left_threshold = 3.0
model.right_threshold = 25.0

flags = model.run(args)
print(flags.head())
```

## Output
- **Returns:** Boolean flags for each row indicating whether it is an anomaly.
- **Example:**  
  `True` → anomaly detected (value out of bounds)  
  `False` → normal behavior

Example snippet:
```
0    False
1    False
2     True
3    False
4     True
Name: is_anomaly, dtype: bool
```

## Notes
- You can set either threshold independently to create one-sided rules (e.g., only `--left_threshold`).
- The model can be integrated into a FastAPI or Flask backend for real-time monitoring.
- In `debug` mode, it prints the total anomaly rate and sample flagged values.

## Folder Structure
```
src/model/ThresholdDetectionModel/
├─ model.py
├─ __init__.py
└─ README.md                # This file
```

## Example Use Cases
- Detect turbine overspeed or low wind conditions.
- Monitor generator or blade temperatures.
- Alert when power output drops below operational limits.

## License
MIT
