import argparse
from loguru import logger
from src.cmd_cli import client
from src.utils import init_logger

def get_args():
    """
    Parse command-line arguments.

    This function sets up an argument parser for selecting models, execution modes,
    debugging options, etc.

    Returns:
        argparse.Namespace: Object containing parsed arguments with the following fields:
            - model (str): The model to run. Choices are ["AnomalyForecastModel", "ThresholdDetectionModel"].
            - mode (str): Operation mode. Choices are ["train", "predict"].
            - debug (bool): Enables debug mode for testing.
            - target (List[str]): A list of target variables for the ThresholdDetectionModel. Only the first is used.
            - left_threshold (float): Minimum threshold value for ThresholdDetectionModel.
            - right_threshold (float): Maximum threshold value for ThresholdDetectionModel.

    """
    p = argparse.ArgumentParser()

    p.add_argument("--model", choices=["AnomalyForecastModel", "ThresholdDetectionModel"], required=True)
    p.add_argument("--mode", choices=["train", "predict"], required=True)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--target", nargs="+", default=["wind_speed", "power", "pitch1_moto_tmp", "pitch2_moto_tmp", "pitch3_moto_tmp", "environment_tmp", "int_tmp"])
    p.add_argument('--left_threshold', action='store', type=float, default=None)
    p.add_argument('--right_threshold', action='store', type=float, default=None)

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    init_logger(logger)
    client(args)
