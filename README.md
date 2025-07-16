# Spam Detection

This repository explores improving the [spam-detection](https://github.com/young-daniel/spam-detection) repository for MLOps and production workflows. 

The original repository explores contains deployment for a simple app with a model built for classifying SMS messages as spam. 

The data can be obtained from Kaggle [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), but is also released by UCI [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). 

### Pre-Commit Hooks

I've chosen to use Ruff for both linting and formatting. It's built in Rust by the authors of the `uv` project and shares 99% drop-in compatibility with Black, with some exceptions I don't have strong feelings about. This means I can use a single tool in place of two (a linter + a formatter). On the whole, this should improve code quality. 

I've also added a `gitleaks` pre-commit check for ensuring that secrets aren't committed to the repository. This is a very minor change and I don't do that in general, so it's not really a constraint on my end, but is still meaningful at least. 

I'm not quite ready to add the static type checker `mypy` though. I think that'll require too many code changes in the short run. Perhaps another time. 

