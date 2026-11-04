"""
/api/arena — algorithm comparisons and experiment leaderboards.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import RANDOM_STATE
from backend.arena.comparison import run_comparison, build_leaderboard

router = APIRouter()


class CompareRequest(BaseModel):
    algorithm_slugs: list[str] = Field(min_length=1, max_length=30)
    dataset_name: str
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = Field(default=RANDOM_STATE)
    hyperparams_by_slug: dict[str, dict[str, Any]] = Field(default_factory=dict)
    scaler: str = Field(default="standard")
    imputer: str = Field(default="mean")
    save: bool = Field(default=False)


@router.post("/compare")
def compare(req: CompareRequest):
    """Train a set of algorithms on one identical split and rank them."""
    try:
        return run_comparison(
            algorithm_slugs=req.algorithm_slugs,
            dataset_name=req.dataset_name,
            test_size=req.test_size,
            random_state=req.random_state,
            hyperparams_by_slug=req.hyperparams_by_slug,
            scaler=req.scaler,
            imputer=req.imputer,
            save=req.save,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/leaderboard")
def leaderboard(
    dataset_name: str | None = Query(None, description="Filter by dataset"),
    task_type: str | None = Query(None, description="Filter by task type"),
    limit: int = Query(50, ge=1, le=500),
):
    """Rank saved experiments by task-appropriate primary metric."""
    return build_leaderboard(
        dataset_name=dataset_name,
        task_type=task_type,
        limit=limit,
    )
