import os.path
from argparse import Namespace
import joblib
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


class AnomalyForecastModel():
    """
    Random-Forest based binary anomaly forecaster.

    This model supports two modes:
      • 'train'   — builds a classifier from internally stored labeled time windows.
      • 'predict' — loads the saved scaler + classifier and scores incoming data.

    Data requirements
    -----------------
    Training mode:
        - Expects a folder named 'anomaly_data' next to this file containing:
            observations.csv        (raw time series with a 'time' column)
            positive_windows.csv    (columns: 'startTime', 'endTime')
            negative_windows.csv    (columns: 'startTime', 'endTime')
        - The classifier is trained on `feature_names` and a derived label `is_anomaly`.

    Prediction mode:
        - Expects `dataset` to contain either:
            • A DatetimeIndex, OR
            • A time column named 'time' (will be parsed to datetime and set as index).
        - Only the columns listed in `feature_names` are used for scoring.

    Persisted artifacts
    -------------------
    - './checkpoints/scaler.pkl'      : StandardScaler fit on training data.
    - './checkpoints/classifier.pkl'  : Trained RandomForestClassifier.
    """


    def __init__(self, flag: str, freq, dataset: pd.DataFrame):
        """
        Initialize the model.

        Args:
            flag (str): 'train' or 'predict' mode.
            freq: Resample frequency for prediction (e.g., '15min').
            dataset (pd.DataFrame): Input time-series DataFrame used for prediction.

        Notes:
            - In training mode, the model loads internal labeled windows and raw data from the 'anomaly_data' folder.
            - In prediction mode, it uses the external dataset.
        """
        assert flag in ['train', 'predict'], "Mode must be either 'train' or 'predict'"

        self.data = dataset
        self.flag = flag
        self.freq = freq

        # Load training data and labeled windows
        if self.flag == 'train':
            current_dir = os.path.split(os.path.realpath(__file__))[0] + "/anomaly_data"
            data_path = os.path.join(current_dir, "observations.csv")
            positive_windows = os.path.join(current_dir, "positive_windows.csv")
            negative_windows = os.path.join(current_dir, "negative_windows.csv")

            self.normal_data = pd.read_csv(negative_windows)
            self.fail_data = pd.read_csv(positive_windows)
            self.data = pd.read_csv(data_path)

        # Initialize the feature set used for model training and prediction
        self.feature_names = [
            'wind_speed', 'power', 'pitch1_moto_tmp', 'pitch2_moto_tmp','pitch3_moto_tmp', 'environment_tmp', 'int_tmp']

        # Initialize the classifier
        self.classifier = RandomForestClassifier(
            max_depth=146,
            n_estimators=2500,
            max_leaf_nodes=2500,
            oob_score=True,
            random_state=30,
            n_jobs=-1
        )

    @logger.catch()
    def run(self, args: Namespace):
        target = None
        if self.flag == 'train':
            self.fit(args)
        elif self.flag == 'predict':
            target = self.forecast(args)
        return target

    def train(self, args: Namespace):
        raise NotImplementedError("Machine Learning Model please use fit method.")

    def fit(self, args: Namespace):
        # time columns in windows are 'startTime' and 'endTime'
        # data has 'time'
        self.data.time = pd.to_datetime(self.data.time)
        self.normal_data = self.normal_data.apply(pd.to_datetime)
        self.fail_data = self.fail_data.apply(pd.to_datetime)

        # create target variable is_anomaly
        self.data['is_anomaly'] = None
        for _, row in tqdm(self.normal_data.iterrows()):
            normal = (row['startTime'] < self.data['time']) & (row['endTime'] > self.data["time"])
            self.data.loc[normal, 'is_anomaly'] = False
        for _, row in tqdm(self.fail_data.iterrows()):
            fail = (row['startTime'] <= self.data['time']) & (row['endTime'] >= self.data["time"])
            self.data.loc[fail, 'is_anomaly'] = True

        self.data = self.data.dropna()
        self.data = self.data.drop('time', axis=1)
        self.data = self.data.astype({'is_anomaly': 'bool'})
        self.data = self.data.loc[:, self.feature_names + ['is_anomaly']]

        # 1) Keep DataFrame/Series (preserves names + index)
        X = self.data[self.feature_names]
        y = self.data["is_anomaly"]

        # 2) Train/test split (stratify to keep class ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3) Under-sample training set (returns arrays)
        under = RandomUnderSampler(random_state=42)
        X_train_under, y_train_under = under.fit_resample(X_train, y_train)

        # 4) Convert back to pandas objects (RangeIndex is fine)
        X_train_under = pd.DataFrame(X_train_under, columns=self.feature_names)
        y_train_under = pd.Series(y_train_under, name="is_anomaly")

        # 5) Scale with one scaler fit on training data ONLY
        from sklearn.preprocessing import StandardScaler
        std_scaler = StandardScaler()

        X_train_under_scaled = pd.DataFrame(
            std_scaler.fit_transform(X_train_under),
            columns=self.feature_names,
            index=X_train_under.index,
        )

        # Ensure X_test is a DataFrame with same columns/order (it already is if step 1 kept DF)
        X_test = X_test[self.feature_names]

        X_test_scaled = pd.DataFrame(
            std_scaler.transform(X_test),
            columns=self.feature_names,
            index=X_test.index,
        )


        self.classifier.fit(X_train_under_scaled, y_train_under)
        prediction = self.classifier.predict(X_test_scaled)
        cm = confusion_matrix(y_test, prediction)
        logger.info(f'Confusion matrix:\n{cm}')

        # save model + scaler
        os.makedirs('./checkpoints', exist_ok=True)
        joblib.dump(std_scaler, './checkpoints/scaler.pkl')
        joblib.dump(self.classifier, './checkpoints/classifier.pkl')
        logger.info('Anomaly Forecast Model dump successfully')

    def forecast(self, args: Namespace) -> pd.DataFrame:
        # load artifacts
        self.classifier = joblib.load('./checkpoints/classifier.pkl')
        std_scaler = joblib.load('./checkpoints/scaler.pkl')

        if not isinstance(self.data.index, pd.DatetimeIndex):
            col = 'time'
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                self.data = self.data.set_index(col)
            if not isinstance(self.data.index, pd.DatetimeIndex):
                raise TypeError("Prediction data must have a DatetimeIndex or a time column.")


        # resample incoming data then scale + predict
        X = self.data.resample(self.freq).bfill().dropna()
        X_scaled = pd.DataFrame(
            std_scaler.transform(X),
            index=X.index,
            columns=self.feature_names,        # ✅ column names preserved again
        )
        y = self.classifier.predict(X_scaled)
        return pd.DataFrame(y, index=X.index, columns=['is_anomaly'])
