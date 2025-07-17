import pandas as pd
from nltk.corpus import stopwords
import string  # builtin string module

stopword_list = stopwords.words("english")


def load_spam_data() -> pd.DataFrame:
    """
    Loads the spam dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the spam data.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv("data/raw/spam.csv", encoding="latin-1")


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


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input dataset by cleaning the text messages.

    Parameters:
    data (DataFrame): The input dataset containing the text messages.

    Returns:
    DataFrame: The preprocessed dataset with cleaned text messages.
    """
    data = data.dropna(how="any", axis=1)
    data.columns = ["target", "message"]
    data.loc[:, "message"] = data["message"].apply(clean_message)
    # TODO: Dump preprocessed data to a file, needs to adapt to file name
    return data
