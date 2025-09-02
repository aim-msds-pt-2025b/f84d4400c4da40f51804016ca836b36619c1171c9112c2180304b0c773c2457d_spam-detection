import numpy as np
import pandas as pd
import pickle
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


def encode_features(data: pd.DataFrame, **kwargs) -> scipy.sparse.csr.csr_matrix:
    """
    Encodes the text features in the dataset using TF-IDF.

    Parameters:
    data (DataFrame): The input dataset containing the text features.

    Returns:
    scipy.sparse.csr.csr_matrix: TF-IDF encoded text features.
    """
    output_path = kwargs.pop("output_path", "outputs/tfidf.pkl")
    # using pop instead of get so other args can be passed e.g. to vectorizer eventually

    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(data.message)
    with open(output_path, "wb") as file:
        pickle.dump(tfidf, file, protocol=5)
    return features


def encode_dataset(data_path: dict) -> dict:
    """
    Encodes the dataset by applying label encoding and feature encoding.

    Parameters:
    data_path (dict): The input dataset containing the text features.

    Returns:
    dict: The encoded dataset with labels and features.
        {"labels": np.ndarray, "features": scipy.sparse.csr.csr_matrix}
    """
    data = pd.read_csv(data_path.get("processed", "data/processed/cleaned_spam.csv"))
    encoded_labels = encode_labels(data)
    encoded_features = encode_features(data)

    embedded = {"labels": encoded_labels, "features": encoded_features}
    path = "outputs/embedded.pkl"
    with open(path, "wb") as file:
        pickle.dump(embedded, file)
    return {"embedded": path}
