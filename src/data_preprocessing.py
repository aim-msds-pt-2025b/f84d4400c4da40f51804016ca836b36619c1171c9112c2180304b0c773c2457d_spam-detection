import pandas as pd
from nltk.corpus import stopwords
import string  # builtin string module

stopword_list = stopwords.words("english")


def load_spam_data(**kwargs) -> dict:
    """
    Loads and dumps spam data to standard pandas CSV format

    Parameters:
    file_path (str): The path to the CSV file containing the spam data.

    Returns:
    dict: {"processed": /path/to/processed/data.csv}.
    """
    # TODO: make this configurable via kwargs
    data = pd.read_csv("data/raw/spam.csv", encoding="latin-1")
    data = data.dropna(how="any", axis=1)
    data.columns = ["target", "message"]

    loaded_path = "data/processed/spam.csv"
    data.to_csv(loaded_path, index=False)
    return {"loaded": loaded_path}


def clean_message(text: str) -> str:
    """
    Cleans the input text message by removing punctuation,
    converting to lowercase, and removing stopwords.

    Parameters:
    text (str): The input text message.

    Returns:
    str: The cleaned text message.
    """
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = [word for word in text.split() if word not in stopword_list]
    text = " ".join(text)
    return text


def preprocess_data(data: dict) -> dict:
    """
    cleaning text messages in input dataset.

    Parameters:
    data (dict): {'loaded': /path/to/input/data}
        Expects loadable with plain `pd.read_csv`

    Returns:
    dict: preprocessed dataset with cleaned text messages.
    """
    data = pd.read_csv(data["loaded"])

    data.loc[:, "message"] = data["message"].apply(clean_message)

    processed_path = "data/processed/cleaned_spam.csv"
    data.to_csv(processed_path, index=False)
    return {"processed": processed_path}
