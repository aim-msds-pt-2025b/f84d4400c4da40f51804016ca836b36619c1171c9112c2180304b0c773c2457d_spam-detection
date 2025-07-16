from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluates the performance of the trained model on the test dataset.

    Parameters:
    model: The trained model to evaluate.
    X_test (np.ndarray): The feature matrix for the test set.
    y_test (np.ndarray): The true labels for the test set.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # we don't need weights because its binary classification

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
