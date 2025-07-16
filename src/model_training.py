import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB


def train_nb(
    X: np.ndarray, y: np.ndarray, alpha: float = 0.2, **kwargs
) -> MultinomialNB:
    """
    Trains multinomial Naive Bayes classifier on provided features and labels.

    Parameters:
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target labels.

    Returns:
    MultinomialNB: trained Naive Bayes classifier.
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X, y)

    save_model(model, filename=kwargs.get("filename", "models/model.pkl"))
    return model


def save_model(model, filename: str = "models/model.pkl") -> None:
    """
    Saves the trained model to a file.

    Parameters:
    model (MultinomialNB): The trained Naive Bayes model.
    filename (str): The name of the file to save the model.
    """
    with open(filename, "wb") as file:
        pickle.dump(model, file, protocol=5)
