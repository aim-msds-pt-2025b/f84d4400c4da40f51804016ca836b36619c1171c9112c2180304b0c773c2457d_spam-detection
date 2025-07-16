from src.data_preprocessing import preprocess_data
from src.feature_engineering import encode_labels, encode_features
from src.model_training import train_nb
from src.evaluation import evaluate_model

import pandas as pd
from sklearn.model_selection import train_test_split


def train_pipeline(data: pd.DataFrame) -> dict:
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train the model
    model = train_nb(X_train, y_train)

    # Step 5: Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    return metrics
