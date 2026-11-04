"""
/api/datasets — list built-in datasets and upload CSVs.
"""
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.core.dataset_loader import (
    list_datasets, load_dataset, register_user_dataset, USER_UPLOAD_DIR,
)
from backend.config import MAX_UPLOAD_MB

router = APIRouter()


@router.get("")
def get_datasets():
    return list_datasets()


@router.get("/{name}")
def get_dataset_info(name: str):
    try:
        ds = load_dataset(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "name": ds.name,
        "task_type": ds.task_type,
        "n_samples": ds.X.shape[0],
        "n_features": len(ds.feature_names),
        "feature_names": ds.feature_names,
        "target_name": ds.target_name,
        "description": ds.description,
        "n_classes": ds.n_classes,
        "class_names": ds.class_names,
    }


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    task_type: str | None = Form(None),
    description: str | None = Form(None),
):
    """
    Upload a CSV and register it as a trainable dataset.

    `target_column` and `task_type` are optional; when omitted they are
    inferred (last column; numeric/many-unique → regression, else
    classification). The file is validated immediately and a metadata
    sidecar is written so `/api/train` can use it by filename.
    """
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    if task_type not in (None, "", "classification", "regression"):
        raise HTTPException(
            status_code=400,
            detail="task_type must be 'classification' or 'regression'.",
        )

    os.makedirs(USER_UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(USER_UPLOAD_DIR, filename)
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(8192):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                os.remove(dest)
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")
            f.write(chunk)

    try:
        meta = register_user_dataset(
            filename,
            target_column=target_column or None,
            task_type=task_type or None,
            description=description or None,
        )
    except ValueError as exc:
        # Invalid upload should not leave an unusable file behind.
        for path in (dest, dest + ".meta.json"):
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        for path in (dest, dest + ".meta.json"):
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    return {**meta, "size_bytes": size, "path": dest}
