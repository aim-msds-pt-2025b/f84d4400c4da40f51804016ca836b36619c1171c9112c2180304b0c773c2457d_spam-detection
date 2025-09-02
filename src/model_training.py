import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

models = {
    "naivebayes": MultinomialNB,
    "randomforest": RandomForestClassifier,
    "svc": SVC,
}


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


def train_model(data_paths: dict, model_args: dict) -> dict:
    """
    Train model given parameters.

    Parameters:
    data_paths (dict): Paths to the training data files.
    model_args (dict): Arguments for the model constructor.
        type (str): model class name, looked up in models
        model_path (str): Path to save the trained model.

    Returns:
    dict: Path to trained model files.
    """
    embedded_path = data_paths.get("embedded", "outputs/embedded.pkl")
    with open(embedded_path, "rb") as embedded_file:
        embedded = pickle.load(embedded_file)

    X = embedded["features"]
    y = embedded["labels"]

    model_str = model_args.pop("type")
    model_class = models.get(model_str)
    if not model_class:
        raise ValueError(f"Unknown model: {model_str}")

    model_path = model_args.pop("model_path", "models/model.pkl")
    model = model_class(**model_args)
    model.fit(X, y)

    save_model(model, filename=model_path)
    return {"model_path": model_path}


def split_data(data_paths: dict, split: float = 0.2) -> dict:
    """
    Splits the data into training and validation sets.

    Parameters:
    data_path (dict): Paths to the training data files.
    split (float): test set proportion

    Returns:
    dict: Paths to the split data files.
    """
    embedded_path = data_paths.get("embedded", "outputs/embedded.pkl")
    with open(embedded_path, "rb") as embedded_file:
        embedded = pickle.load(embedded_file)

    X = embedded["features"]
    y = embedded["labels"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split)

    with open("outputs/train.pkl", "wb") as train_file:
        pickle.dump({"features": X_train, "labels": y_train}, train_file)

    with open("outputs/val.pkl", "wb") as val_file:
        pickle.dump({"features": X_val, "labels": y_val}, val_file)

    return {"train": "outputs/train.pkl", "val": "outputs/val.pkl"}
