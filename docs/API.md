# ShiftLab API

Base URL: `http://localhost:8000`

Interactive docs: `/docs`

## Health

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Liveness + version |

## Algorithms

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/algorithms` | List all algorithms |
| GET | `/api/algorithms?family=neural` | Filter by family |
| GET | `/api/algorithms?task_type=regression` | Filter by task |
| GET | `/api/algorithms?q=ridge` | Search names/families/tags |
| GET | `/api/algorithms/{slug}` | Full metadata (description, math, hyperparameters, strengths, weaknesses) |

Slug format: `family.task_type.ClassName` lowercased, e.g.
`supervised.classification.logisticregressionalgo`.

## Datasets

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/datasets` | List built-in + uploaded datasets |
| GET | `/api/datasets/{name}` | Dataset metadata |
| POST | `/api/datasets/upload` | Upload and register a CSV (multipart) |

Upload form fields:

- `file` (required) — `.csv`
- `target_column` (optional) — inferred as the last column when omitted
- `task_type` (optional) — `classification` or `regression`; inferred when omitted
- `description` (optional)

Response includes dataset metadata plus `size_bytes` and the stored `path`.

## Training

### `POST /api/train`

```json
{
  "algorithm_slug": "supervised.classification.logisticregressionalgo",
  "dataset_name": "iris",
  "test_size": 0.2,
  "random_state": 42,
  "hyperparams": {},
  "scaler": "standard",
  "imputer": "mean",
  "save": true
}
```

Returns real held-out metrics, training time, inference time, resolved
hyperparameters, task type, and the saved experiment ID (or `null`).

### `POST /api/predict`

```json
{
  "algorithm_slug": "supervised.classification.logisticregressionalgo",
  "dataset_name": "iris",
  "hyperparams": {},
  "scaler": "standard",
  "imputer": "mean",
  "X": [[5.1, 3.5, 1.4, 0.2]]
}
```

Trains on the full dataset, then predicts the supplied rows. Returns
`predictions` and, when supported, `probabilities`.

## Experiments

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/experiments` | List saved runs |
| GET | `/api/experiments/{id}` | Full experiment record |

## Arena

### `POST /api/arena/compare`

```json
{
  "algorithm_slugs": [
    "supervised.classification.logisticregressionalgo",
    "supervised.classification.randomforestclassifieralgo"
  ],
  "dataset_name": "iris",
  "test_size": 0.2,
  "random_state": 42,
  "hyperparams_by_slug": {
    "supervised.classification.randomforestclassifieralgo": {"n_estimators": 100}
  },
  "scaler": "standard",
  "imputer": "mean",
  "save": false
}
```

Every algorithm is trained on the same split and preprocessing pipeline.
Results are ranked by `accuracy` (classification) or `r2` (regression).
Individual algorithm failures appear as `error` entries rather than aborting
the comparison.

### `GET /api/arena/leaderboard`

Query: `dataset_name`, `task_type`, `limit` (1–500).

Ranks saved experiments by their task's primary metric. Experiments whose
algorithm is no longer registered are ignored.

## Reliability Lab

All endpoints use real statistics on the requested dataset.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/reliability/shift-types` | Synthetic shift generator metadata |
| POST | `/api/reliability/covariate-shift` | MMD² + domain-classifier AUC |
| POST | `/api/reliability/label-shift` | Chi-squared test + TV distance |
| POST | `/api/reliability/importance-weights` | Density-ratio weights + ESS |
| POST | `/api/reliability/performance-under-shift` | Metric degradation sweep |

### Covariate shift

```json
{"dataset_name": "iris", "test_size": 0.25, "sigma": null}
```

### Label shift

```json
{"dataset_name": "iris", "test_size": 0.25}
```

Classification datasets only.

### Importance weights

```json
{"dataset_name": "iris", "test_size": 0.25, "method": "classifier", "clip_percentile": 95}
```

`method` is `classifier` or `kde`.

### Performance under shift

```json
{
  "dataset_name": "iris",
  "algorithm_slug": "supervised.classification.logisticregressionalgo",
  "hyperparams": {},
  "test_size": 0.2,
  "shift_type": "gaussian",
  "shift_magnitudes": [0.0, 0.25, 0.5, 1.0, 2.0]
}
```

`shift_type`: `gaussian`, `mean_shift`, or `scale_shift`. Magnitudes must be
non-negative, sorted ascending, and at most 20 values. Magnitude `0` is the
unshifted baseline.

## Errors

| Status | Meaning |
| --- | --- |
| 400 | Invalid request, task mismatch, malformed CSV, unsupported dataset/task |
| 404 | Unknown algorithm, dataset, or experiment |
| 413 | Upload exceeds the size limit |
| 500 | Training/scoring/analysis failure (detail included) |
