# ShiftLab

ShiftLab is a machine-learning experimentation platform with a FastAPI backend and a bundled single-page frontend. It brings algorithm benchmarking, dataset uploads, experiment tracking, and distribution-shift reliability analysis into one workspace.

## Features

- **81 algorithms** across classification, regression, clustering, dimensionality reduction, anomaly detection, ensemble methods, and pure-NumPy neural networks.
- **Leakage-aware preprocessing:** imputers and scalers are fitted on the training split.
- **Task-aware evaluation:** metrics include accuracy, F1, AUC, MAE, RMSE, R², MAPE, silhouette score, Davies–Bouldin score, and anomaly-detection metrics.
- **Experiment tracking:** saves experiment records as JSON under `experiments/`.
- **CSV dataset uploads:** upload a dataset, choose or infer the target column, and select or infer the task.
- **Arena comparisons:** train multiple algorithms on a shared split and compare their results.
- **Reliability Lab:** includes MMD² covariate-shift detection, domain-classifier two-sample tests, chi-squared label-shift tests, importance weighting, Kish effective sample size, and performance-under-shift sweeps.
- **Bundled frontend:** FastAPI serves the interface directly; no separate Node build step is required.

## Technology

- Python
- FastAPI
- scikit-learn and NumPy-based algorithms
- Vanilla JavaScript frontend
- pytest

## Requirements

- Python 3.11 or later
- Dependencies listed in `backend/requirements.txt`

Install dependencies from the repository root:

```powershell
python -m pip install -r backend/requirements.txt
```

## Run Locally

From the repository root, start the API:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
```

Open the application:

- UI: http://127.0.0.1:8080
- Interactive API documentation: http://127.0.0.1:8080/docs

The host and port can be changed to suit your local environment.

## Run Tests

```powershell
python -m pytest -q
```

## Project Structure

```text
backend/
  main.py
  config.py
  algorithms/
  core/
  routers/
  arena/
  reliability/
  tests/
frontend/
experiments/
data/
docs/
```

## CSV Upload

A CSV file should contain a header row. Select the target column and task type (classification or regression) in the upload workflow, or use the application's inference option.

Example:

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
6.0,3.0,4.8,1.8,virginica
```

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `SHIFTLAB_SEED` | `42` | Global random seed |
| `SHIFTLAB_N_JOBS` | `1` | Worker count for estimators supporting `n_jobs` |

CSV upload size is limited by `MAX_UPLOAD_MB` in `backend/config.py`.

## Current Limitations

- Prediction is exploratory: the application retrains on the full dataset before predicting supplied rows. Model persistence is not yet implemented.
- Training is synchronous, so long-running experiments can block a request.
- The implementation intentionally does not use `HistGradientBoosting` or `HistGradientBoostingRegressor`.

## Author

**Darshan A R**  
[GitHub Profile](https://github.com/darsh11712)
