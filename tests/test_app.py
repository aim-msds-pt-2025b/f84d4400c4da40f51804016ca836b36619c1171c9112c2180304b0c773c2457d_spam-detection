import pytest
from src.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_status(client):
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert response.get_json() == "API running"


def test_predict_ham(client, mocker):
    # Mock pipeline and clean_message for predictable output
    mocker.patch("src.app.clean_message", return_value="hello")
    mock_pipeline = mocker.patch("src.app.pipeline")
    mock_pipeline.predict.return_value = [0]
    response = client.post("/api/v1/predict", data={"message": "hello"})
    assert response.status_code == 200
    assert response.get_json() == "ham"


def test_predict_spam(client, mocker):
    mocker.patch("src.app.clean_message", return_value="buy now")
    mock_pipeline = mocker.patch("src.app.pipeline")
    mock_pipeline.predict.return_value = [1]
    response = client.post("/api/v1/predict", data={"message": "buy now"})
    assert response.status_code == 200
    assert response.get_json() == "spam"


def test_predict_no_message(client):
    response = client.post("/api/v1/predict", data={})
    assert response.status_code == 200
    assert "no message specified" in response.get_data(as_text=True)
