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

I've addded a deployments directory to organize the Dockerfiles and deployments. An alternate approach would be to name the Dockerfiles differently, e.g. `inferencing.Dockerfile`. This approach keeps things a little cleaner, we can indicate the root directory as the build directory in place of identifying the specific Dockerfile to build with `-f`.  

### Environment Setup

To set up the `uv` environment, I did the ff:
* `uv init`
* `uv add numpy pandas scikit-learn seaborn nltk requests`
Since I've committed the `uv.lock` file, users can simply clone the repository and then run `uv sync`. We can also dump the package requirements in the `pip` format by running `uv export --format requirements-txt > requirements.txt`. I think `uv sync` is more the better choice though. 

### Pre-Commit Hooks

I've chosen to use Ruff for both linting and formatting. It's built in Rust by the authors of the `uv` project and shares 99% drop-in compatibility with Black, with some exceptions I don't have strong feelings about. This means I can use a single tool in place of two (a linter + a formatter). On the whole, this should improve code quality. 

I've also added a `gitleaks` pre-commit check for ensuring that secrets aren't committed to the repository. This is a very minor change and I don't do that in general, so it's not really a constraint on my end, but is still meaningful at least. 

I'm not quite ready to add the static type checker `mypy` though. I think that'll require too many code changes in the short run. Perhaps another time. 

I've added hadolint for Dockerfiles and yamllint for YAML files as pre-commits. For yamllint I'v relaxed the 80 character line limit to 120 because the default docker-compose YAML for Airflow uses more than 80 characters in several lines. Together, these two linters will help ensure code quality by highlighting common issues and encouraging best practices.

### Docker Setup

I have previously had an installation of Docker on my systems, but installing Docker is relatively simple now on both Linux and Windows. 

On Linux, 
* `apt-get install docker-desktop` 

On Windows:
* We start by installing Windows Subsystem Linux (WSL).
  * From a terminal (such as PowerShell or through a wrapper like Windows Terminal), do `wsl --install`. 
  * This step may fail the first time around, if so, it'll be because its enabling Windows features and installing dependencies. You'll need to reboot your PC and run the command again to complete the install.
* `winget install docker.dockerdesktop`

I made an initial Docker image that runs the pipeline as built before, with changes to the pipeline to follow. I've followed the examples provided in `uv` documentationi for how to build the Docker images so as to minimize the changes. Instead of installing `uv` directly and then syncing to the `uv.lock` file, we copy the `uv` bbinaries from `uv`'s official distroless image releases (with the version pinned to the one we're using). Similarly, we do not copy the `uv.lock` file into the container, choosing instead to mount the `uv.lock` and install dependencies from it wihtout copying it into the Docker image itself. Note also the use of `--locked` with the `uv sync` command, which tells `uv` not to make any changes to the set of packages to be installed, improving reproducibility. Our approach should also minimiize the changes made between versions, improving Docker build times and reducing size requirements. 

We should note that this image requires the `data/`, `models/` and `outputs/` directories to be mounted. We expect changes to this requirement as changes are made to the pipeline.

### Airflow

Currently the Airflow setup is mostly a smoke test to validate that the Airflow image I've created can actually run the task (has all the dependencies, etc). Running the DAG as it is now can be done by running `docker compose up -d` from the project's root level. I wasn't able to get `uv` to work with the Airflow environment, I think a little bit more work is required to get there, possibly by altering the Python version to match the one in the Airflow Docker image. I tried using the `--inexact` flag to prevent `uv` from uninstalling packages that weren't in the `pyproject.toml` spec,  but couldn't get this argument to be recognized by `uv` in the Docker build environment. 

More work is required to refactor the pipeline in such a way as to not require passing Python objects in memory between tasks/phases--the implementation choices made previously do not fit well with Airflow's DAG model. I think the goal is to have tasks write artifacts to the disk and return paths pointing at those artifacts for use by other tasks. This means more I/O overhead but will allow tasks to be separated cleanly. 

#### Airflow 3.x

Migrating to Airflow 3.x was painful. To be fair, this guide [here](https://airflow.apache.org/docs/apache-airflow/stable/installation/upgrading_to_airflow3.html) provided some information, but it really was not enough. Trying to figure out why the previous docker-compose configuration wasn't working required parsing through logs matching commands to outputs, only to realize the `airflow version` command was somehow also invoking `airflow db migrate` and the user creation process (when yet another poorly documented setting was set correctly). 

### Inferencing

There are two options for inferencing: the CLI and the deployment. The CLI expects a file path as an input, with each line corresponding to a message. The script will print the outputs unless a path is passed to the output option. Use `-h` to view a help message. The deployment creates a server with a REST API interface using `flask`. I've not rebuilt the image, but the old one can be found [here](https://hub.docker.com/r/doyoung04/spam-detection/). Creating a container with this Docker image will start the server automatically. 

#### REST API Docs
- URL
    /api/v1/predict
- Method: POST
- Data Params
    - message=[str]
- Success Response
    Code: 200
    Content: `"ham\n"`
- Sample Call
    `curl -X POST -d "message=I'm on the way home, be there in fifteen." http://127.0.0.1:5000/api/v1/predict`

### Tests

I've added tests using the `pytest` framework. So far, that includes tests for the app (REST  API) and for the data preprocessing (cleaning messages and loading data). I'm not sure how one would write tests for the model training itself though. 

### Reflection

The project structure and the tests felt a bit sticky: I ended up having to set a change to the pythonpath to include the root directory (`spam-detection` or such) of the repository when running the tests (see the pytest config in `pyproject.toml`). I think I need to study the typical pytest structure a bit more and allow that to inform the choices made for project structure. I'm also not sure what the contents of some of the folders are meant to be, possibly because this project is a bit too simple to require more complexity. 

I've used linters and formatters in some ways before, although Ruff seems to change the game enough that it can be used as a pre-commit now. I've also attempted to create a Ruff action that runs automatically within Github whenever a PR is created. That's new to me, so I'm still figuring out how to get things to work the way I want. 