import pandas as pd
from nltk.corpus import stopwords
import string  # builtin string module

stopword_list = stopwords.words("english")


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
    data["message"] = data["message"].apply(clean_message)
    return data
