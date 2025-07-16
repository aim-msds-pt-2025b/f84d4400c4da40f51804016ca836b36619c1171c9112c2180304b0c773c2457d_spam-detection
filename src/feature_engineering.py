import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def encode_labels(data: pd.DataFrame) -> np.ndarray:
    """
    Encodes the target labels in the dataset.

    Parameters:
    data (DataFrame): The input dataset containing the target labels.

    Returns:
    np.ndarray: Encoded target labels.
    """
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(data.target)


def encode_features(data: pd.DataFrame) -> scipy.sparse.csr.csr_matrix:
    """
    Encodes the text features in the dataset using TF-IDF.

    Parameters:
    data (DataFrame): The input dataset containing the text features.

    Returns:
    scipy.sparse.csr.csr_matrix: TF-IDF encoded text features.
    """
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(data.message)
