"""
/api/algorithms — list, search, get metadata.
"""
from fastapi import APIRouter, HTTPException, Query
from backend.core.registry import AlgorithmRegistry

router = APIRouter()


@router.get("")
def list_algorithms(
    family: str | None = Query(None, description="Filter by family"),
    task_type: str | None = Query(None, description="Filter by task type"),
    q: str | None = Query(None, description="Search query"),
):
    AlgorithmRegistry.load_all()
    if q:
        return AlgorithmRegistry.search(q)
    if family:
        return AlgorithmRegistry.list_by_family(family)
    if task_type:
        return AlgorithmRegistry.list_by_task(task_type)
    return AlgorithmRegistry.list_all()


@router.get("/{slug:path}")
def get_algorithm(slug: str):
    AlgorithmRegistry.load_all()
    try:
        klass = AlgorithmRegistry.get(slug)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return klass.explain()
