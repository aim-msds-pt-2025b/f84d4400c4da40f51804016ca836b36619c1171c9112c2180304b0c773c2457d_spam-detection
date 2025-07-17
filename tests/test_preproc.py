import pandas as pd
from src.data_preprocessing import clean_message, load_spam_data


def test_load_spam_data():
    data = load_spam_data()
    assert isinstance(data, pd.DataFrame)
    assert "v1" in data.columns
    assert "v2" in data.columns
    assert data.v1.unique().tolist() == ["ham", "spam"]
    assert data.iloc[1, 1] == "Ok lar... Joking wif u oni..."


def test_clean_message():
    assert clean_message("Hello, World!") == "hello world"
    assert clean_message("Buy now!") == "buy"
    assert clean_message("Spam message") == "spam message"
    assert clean_message("This is a test message.") == "test message"
    assert (
        clean_message("The quick brown fox jumps over the lazy dog.")
        == "quick brown fox jumps lazy dog"
    )
