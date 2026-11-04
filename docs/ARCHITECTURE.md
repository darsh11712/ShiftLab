# ShiftLab Architecture

## Layers

```
frontend/  (static SPA)
    │ fetch JSON
    ▼
backend/routers/         HTTP boundary: validation + error mapping
    │
    ├── backend/core/    registry, datasets, preprocessing, metrics, experiments
    ├── backend/algorithms/
    ├── backend/arena/   comparison + leaderboard
    └── backend/reliability/  distribution-shift statistics
```

## Algorithm registry

`backend/core/registry.py` walks `backend.algorithms` with `pkgutil` and imports
every module. Importing a module executes `@AlgorithmRegistry.register` on each
`BaseAlgorithm` subclass.

- Slug: `f"{family}.{task_type}.{ClassName}".lower()`.
- `AlgorithmRegistry.list_all()` returns lightweight catalog entries.
- `AlgorithmRegistry.get(slug)` returns the class.
- `AlgorithmRegistry.instantiate(slug, random_state=..., **hyperparams)` builds
  an instance.

`BaseAlgorithm` defines the contract:

- `_build_model()` — construct the underlying estimator
- `fit()` / `_fit_impl()` — fit and time the operation
- `predict()` / `_predict_impl()`
- `predict_proba()` — optional
- `score(X, y)` — task-appropriate real metrics
- `explain()` — static metadata for the library UI
- `_n_jobs()` — resolves the worker count from hyperparameters / config

## Data flow (training)

1. `load_dataset(name)` resolves a built-in loader or an uploaded CSV.
2. `split_dataset(...)` performs a stratified/reproducible split.
3. `build_preprocessor(...)` builds an imputer + scaler pipeline.
4. `fit_transform(pp, X_train, X_test)` fits only on train.
5. The algorithm is instantiated and fit (unsupervised families fit on `X`).
6. `score(X_test, y_test)` computes metrics.
7. Optionally `save_experiment(...)` writes JSON under `experiments/`.

## Uploaded datasets

`backend/core/dataset_loader.py`:

- Files live in `data/user_uploads/`.
- Each CSV gets a `<name>.csv.meta.json` sidecar containing target column,
  task type, dimensions, and description.
- `register_user_dataset()` validates the CSV by loading/encoding it before
  writing metadata.
- `list_datasets()` merges built-ins and uploads; `load_dataset()` resolves both.
- Path traversal is blocked by requiring `os.path.basename(name) == name`.

## Arena

`backend/arena/comparison.py`:

- `run_comparison()` loads data once, preprocesses once, then fits each
  requested algorithm on the identical split.
- Results are ranked by `accuracy` or `r2`. Failures are captured per entry.
- `build_leaderboard()` reads saved experiments, resolves each algorithm's task
  from the registry, and ranks by the task's primary metric. Unknown slugs
  (e.g. synthetic test records) are skipped.

## Reliability

`backend/reliability/distribution_shift.py` contains pure functions:

- `mmd_squared` — unbiased RBF-kernel MMD² with median-heuristic bandwidth.
- `classifier_two_sample_test` — logistic-regression domain classifier with CV AUC.
- `label_shift_analysis` — chi-squared test + total-variation distance.
- `importance_weights` — classifier or KDE density-ratio weights, clipped and
  normalized, with Kish effective sample size.
- `apply_covariate_shift` + `performance_under_shift` — synthetic shift sweeps.

`backend/routers/reliability.py` validates inputs, loads/splits data, trains a
real model where needed, and returns the raw statistics.

## Frontend

`frontend/` is a dependency-free SPA (`index.html`, `styles.css`, `app.js`).
FastAPI mounts it at `/` with `StaticFiles(html=True)` **after** all API routes,
so `/api/*` keeps priority. Pages: Overview, Algorithms, Datasets, Train,
Experiments, Arena, Reliability Lab.

## Stability notes

- Estimators default to `n_jobs=1` (`SHIFTLAB_N_JOBS` overrides). The joblib/loky
  backend can terminate workers on some Python/OS combinations; this default
  keeps ensemble training reliable.
- The pure-NumPy MLP and autoencoder use stable sigmoid/softmax implementations
  and global-norm gradient clipping to prevent overflow.
- XGBoost's objective is resolved from the training labels before the estimator
  is built, so multiclass runs use `multi:softprob` with `num_class`.
