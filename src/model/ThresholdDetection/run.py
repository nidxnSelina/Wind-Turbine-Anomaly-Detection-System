from typing import List
from loguru import logger
from argparse import Namespace
import pandas as pd
from pandas import DataFrame


class ThresholdDetectionModel():
    """
    A simple threshold-based detection model

    This model classifies each record into categories based on 
    user-defined numeric thresholds for a given target variable. 
    It is useful for detecting abnormal or extreme values without training a
    machine learning model.
    """

    def __init__(self, data: DataFrame, target: List[str]):
        """
        Initialize the model.

        Args:
            data (DataFrame): The input dataset containing at least one target column.
            target (List[str]): A list with the name(s) of target column(s) to evaluate.
                Only the first element of this list is used.
        """
        self.dataset = data
        self.target = target[0]

    @logger.catch()
    def run(self, args: Namespace):
        """
        Execute the threshold detection pipeline.

        Args:
            args (Namespace): Command-line arguments or configuration parameters.

        Returns:
            DataFrame: A DataFrame summarizing the proportion of records
            falling into each threshold label category.
        """
        target = self.forecast(args)
        return target

    def preprocess(self, dataset: DataFrame, target: str) -> DataFrame:
        """
        Extract and prepare the target column from the dataset.

        Args:
            dataset (DataFrame): The raw dataset.
            target (str): The name of the target column to isolate.

        Returns:
            DataFrame: A subset DataFrame containing only the target column.
        """
        dataset = dataset[[target]]
        return dataset

    def train(self, args: Namespace):
        """Placeholder method for training."""
        raise NotImplementedError("This model doesn't have train method")

    def fit(self, args):
        """Placeholder method for fitting."""
        raise NotImplementedError("This model doesn't have fit method")

    def forecast(self, args: Namespace) -> pd.DataFrame:
        """
        Perform threshold-based classification.

        Args:
            args (Namespace): Argument namespace with the following attributes:
                - left_threshold (float): Lower threshold value.
                - right_threshold (float): Upper threshold value.

        Returns:
            DataFrame: A DataFrame showing the proportion of samples in each label category:
                - 0: within thresholds (normal range)
                - 1: below left_threshold
                - 2: above right_threshold

        Raises:
            ValueError: If both threshold values are None.
        """
        self.check_threshold_args(args)
        self.dataset = self.preprocess(self.dataset, self.target)
        lt = args.left_threshold
        rt = args.right_threshold

        dataset = self.dataset.copy()

        if rt is not None:
            dataset.loc[dataset[self.target] > rt, 'label'] = 2
        if lt is not None:
            dataset.loc[dataset[self.target] < lt, 'label'] = 1
        if lt is None and rt is None:
            error_message = f"Invalid thresholds: min={lt}, max={rt}"
            self.logger.error(error_message)
            raise ValueError(error_message)

        dataset.fillna(0, inplace=True)
        dataset = dataset.astype({'label': int})

        label_ratio = dataset.groupby('label').count() / dataset.shape[0]

        return label_ratio

    @staticmethod
    def check_threshold_args(args: Namespace):
        """
        Validate that at least one threshold argument is provided.

        Args:
            args (Namespace): The argument namespace to check.

        Raises:
            AssertionError: If both left_threshold and right_threshold are None.
        """
        assert args.left_threshold is not None or args.right_threshold is not None
