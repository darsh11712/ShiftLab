# ShiftLab

ShiftLab is a research-grade machine-learning algorithm laboratory. It exposes
81 real algorithms through a FastAPI backend and a bundled single-page
frontend. Every score is produced by a real fit on real data — there are no
mocked metrics and no fabricated results.

## Features

- **81 algorithms**
  - 20 supervised classifiers
  - 18 supervised regressors
  - 11 clustering algorithms
  - 10 dimensionality reducers
  - 7 anomaly detectors
  - 10 ensemble methods
  - 5 pure-NumPy neural networks
- **Leakage-free preprocessing** — imputers and scalers are fit on the training
  split only.
- **Task-aware metrics** — accuracy/F1/AUC, MAE/RMSE/R²/MAPE, silhouette/
  Davies–Bouldin, and anomaly precision/recall/F1.
- **Experiment persistence** — runs are stored as JSON under `experiments/`.
- **CSV uploads** — upload a CSV, choose or infer the target/task, then train on
  it like any built-in dataset.
- **Arena** — train several algorithms on one identical split and rank them by
  the task's primary metric. The leaderboard ranks saved experiments.
- **Reliability Lab** — MMD² covariate-shift detection, domain-classifier
  two-sample tests, label-shift chi-squared tests, importance weights with
  Kish effective sample size, and performance-under-shift sweeps.
- **Bundled frontend** — no Node build step required; FastAPI serves it at `/`.

## Requirements

- Python 3.11+ (verified on Python 3.14.5)
- Dependencies from `backend/requirements.txt`

```powershell
python -m pip install -r backend/requirements.txt
```

## Run

From the repository root:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Then open <http://localhost:8000>.

In this workspace the working interpreter is:

```powershell
C:\Users\admin\AppData\Local\Python\bin\python.exe -m uvicorn backend.main:app --reload --port 8000
```

Interactive API docs are available at <http://localhost:8000/docs>.

## Test

```powershell
python -m pytest -q
```

The suite covers every algorithm family, the API, uploaded datasets, arena
comparisons, and reliability statistics.

## Project layout

```
backend/
  main.py                 FastAPI application + frontend mount
  config.py               paths, seed, worker count
  algorithms/             all registered algorithms
  core/                   registry, datasets, preprocessing, metrics, experiments
  routers/                algorithms, datasets, training, experiments, arena, reliability
  arena/                  comparison + leaderboard logic
  reliability/            distribution-shift statistics
  tests/                  pytest suite
frontend/                 vanilla-JS SPA served by FastAPI
experiments/              saved experiment JSON records
data/
  builtin/                reserved for bundled data files
  user_uploads/           uploaded CSVs + metadata sidecars
docs/                     API and architecture notes
```

## CSV upload format

The CSV must contain a header row. Feature columns may be numeric or textual
(text is label-encoded automatically). The target column is specified at upload
time or inferred as the last column. Task type is `classification` or
`regression`; when omitted it is inferred from the target's cardinality.

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
6.0,3.0,4.8,1.8,virginica
```

Upload with the UI or:

```powershell
curl.exe -X POST http://localhost:8000/api/datasets/upload `
  -F "file=@my_data.csv" `
  -F "target_column=species" `
  -F "task_type=classification"
```

After upload, the filename (e.g. `my_data.csv`) is the dataset name used by
`/api/train`, `/api/arena/compare`, and the reliability endpoints.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `SHIFTLAB_SEED` | `42` | Global random seed |
| `SHIFTLAB_N_JOBS` | `1` | Worker count for estimators that support `n_jobs`. Single-process is the safe default; set `-1` to opt into parallelism. |

Upload size is limited to 50 MB (`MAX_UPLOAD_MB` in `backend/config.py`).

## Notes

- The platform intentionally never uses `HistGradientBoosting` /
  `HistGradientBoostingRegressor`.
- `predict` is exploratory: it retrains on the full dataset before predicting
  the supplied rows. Model persistence is not yet implemented.
- Training is synchronous; long runs block the request. A job queue would be
  the next scaling step.
