"""
/api/experiments — list, load saved experiments.
"""
from fastapi import APIRouter, HTTPException
from backend.core.experiment import list_experiments, load_experiment

router = APIRouter()


@router.get("")
def get_experiments():
    return list_experiments()


@router.get("/{exp_id}")
def get_experiment(exp_id: str):
    try:
        return load_experiment(exp_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
