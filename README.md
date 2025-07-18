# Spam Detection

This repository explores improving the [spam-detection](https://github.com/young-daniel/spam-detection) repository for MLOps and production workflows. 

The original repository explores contains deployment for a simple app with a model built for classifying spam messages.

I've opted to leave the output of the notebook in because I've also done parameter tuning and model selection as well. The results are presented in there. 

The data can be obtained from Kaggle [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), but is also released by UCI [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). 

### Folder Structure

Here's the folder structure we'll target:

```
spam-detection
├── uv.lock
├── pyproject.toml
├── README.md
├── data
│   ├── raw
│   │   └── spam.csv
│   └── processed
├── models
│   └── trained.pkl
├── notebooks
│   └── spam-detection.ipynb
├── outputs
├── src
│   ├── __init__.py
│   ├── preprocessing.py
...
├── tests
│   ├── test_app.py
...
```

I've added the dataset to the repo since its a small file. I've separated the data into raw and processed folders. I think this is a bit of overkill for this project because of there being only one small file (as opposed to many files), but nevertheless this should make it easier to identify which file is relevant. Future changes should include writing the cleaned/processed dataset with more granularity/control.

I've also added a tests directory for the tests. There is an option to have the tests as subfolders under `src` instead, but I chose tests separate from application code instead. 

### Environment Setup

To set up the `uv` environment, I did the ff:
* `uv init`
* `uv add numpy pandas scikit-learn seaborn nltk requests`
Since I've committed the `uv.lock` file, users can simply clone the repository and then run `uv sync`. We can also dump the package requirements in the `pip` format by running `uv export --format requirements-txt > requirements.txt`. I think `uv sync` is more the better choice though. 

### Pre-Commit Hooks

I've chosen to use Ruff for both linting and formatting. It's built in Rust by the authors of the `uv` project and shares 99% drop-in compatibility with Black, with some exceptions I don't have strong feelings about. This means I can use a single tool in place of two (a linter + a formatter). On the whole, this should improve code quality. 

I've also added a `gitleaks` pre-commit check for ensuring that secrets aren't committed to the repository. This is a very minor change and I don't do that in general, so it's not really a constraint on my end, but is still meaningful at least. 

I'm not quite ready to add the static type checker `mypy` though. I think that'll require too many code changes in the short run. Perhaps another time. 

### Tests

I've added tests using the `pytest` framework. So far, that includes tests for the app (REST  API) and for the data preprocessing (cleaning messages and loading data). I'm not sure how one would write tests for the model training itself though. 

### Reflection

The project structure and the tests felt a bit sticky: I ended up having to set a change to the pythonpath to include the root directory (`spam-detection` or such) of the repository when running the tests (see the pytest config in `pyproject.toml`). I think I need to study the typical pytest structure a bit more and allow that to inform the choices made for project structure. I'm also not sure what the contents of some of the folders are meant to be, possibly because this project is a bit too simple to require more complexity. 

I've used linters and formatters in some ways before, although Ruff seems to change the game enough that it can be used as a pre-commit now. I've also attempted to create a Ruff action that runs automatically within Github whenever a PR is created. That's new to me, so I'm still figuring out how to get things to work the way I want. 