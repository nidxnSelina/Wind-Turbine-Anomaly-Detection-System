import os
from datetime import datetime
from pandas import Timedelta, DataFrame
from .model.AnomalyForecastModel.run import AnomalyForecastModel
from .model.ThresholdDetection.run import ThresholdDetectionModel
from .utils import load_xls
import ast

default_freq = Timedelta(minutes=10)


def client(args):
    """
    Main client function to run the specified model with either training or prediction mode.
    
    This is the main entry point for executing models from the command line.

    Args:
        args (argparse.Namespace): Parsed command-line arguments that must include:
            - model (str): Model name ('ThresholdDetectionModel' or 'AnomalyForecastModel').
            - mode (str): Operation mode ('train' or 'predict').
            - debug (bool): Whether to enable debug mode (loads test data and prints logs).
            - target (List[str]): Target variable(s)

    Raises:
        RuntimeError: If the specified model name is invalid or prediction output is None.

    """
    if args.model is None:
        exit(0)

    # Load test data to debug
    if args.debug:
        cols_str = "['time','wind_speed', 'power', 'pitch1_moto_tmp', 'pitch2_moto_tmp','pitch3_moto_tmp', 'environment_tmp', 'int_tmp']"
        required_cols = ast.literal_eval(cols_str)

        data = load_xls(val_data_path="test_data", required_english_cols=required_cols)

        # Formatting
        args.target = [col.replace('/', 'p') for col in args.target]
    else:
        data = DataFrame()

    # Instantiate the model
    if args.model == 'ThresholdDetectionModel':
        model = ThresholdDetectionModel(data, target=args.target)
    elif args.model == 'AnomalyForecastModel':
        model = AnomalyForecastModel(dataset=data, flag=args.mode, freq=default_freq)
    else:
        raise RuntimeError(f'There is no model named {args.model}.')

    # Run the model based on mode
    if args.mode == 'train':
        model.run(args)
    elif args.mode == 'predict':
        pred_y = model.run(args)
        if pred_y is None:
            raise RuntimeError("Prediction returned None; check logs above.")

        # Save output to a JSON file
        os.makedirs("checkpoints", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("checkpoints", f"prediction_{timestamp}.json")
        pred_y.to_json(output_path, orient='split', date_format='iso', force_ascii=False)

        print(f"Predictions saved to: {output_path}")
