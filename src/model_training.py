import numpy as np
from sklearn.naive_bayes import MultinomialNB


def train_nb(X: np.ndarray, y: np.ndarray, alpha: float = 0.2) -> MultinomialNB:
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
    return model
