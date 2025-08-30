from src.data_preprocessing import load_spam_data, preprocess_data
from src.feature_engineering import encode_labels, encode_features
from src.model_training import train_nb
from src.evaluation import evaluate_model

import argparse
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split


def train_pipeline(data: pd.DataFrame, **kwargs) -> dict:
    """
    Runs spam detection training pipeline.

    Parameters:
    data (DataFrame): The input dataset containing text messages and labels.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    # Step 1: Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Step 2: Encode features and labels
    X = encode_features(preprocessed_data)
    y = encode_labels(preprocessed_data)

    # Step 3: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Step 4: Train the model
    model = train_nb(X_train, y_train)

    # Step 5: Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    return metrics


def train_task() -> dict:
    """Handsfree version, created to verify Airflow image works"""
    data = load_spam_data()
    return train_pipeline(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spam detection training pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the input dataset",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="models/model.pkl",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="reports/metrics.txt",
        help="Path to save the evaluation metrics",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    for path in [args.model_output, args.metrics_output]:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        data = pd.read_csv(args.data_path)
    else:
        data = load_spam_data()
    metrics = train_pipeline(data, **vars(args))
    print(metrics)
