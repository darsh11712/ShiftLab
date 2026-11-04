"""
ShiftLab FastAPI application.
All routes return real ML results — no mock data.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import (
    algorithms, datasets, training, experiments, reliability, arena,
)

app = FastAPI(
    title="ShiftLab API",
    description="Research-grade Machine Learning Algorithm Laboratory",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(algorithms.router, prefix="/api/algorithms", tags=["algorithms"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api", tags=["training"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(reliability.router, prefix="/api/reliability", tags=["reliability"])
app.include_router(arena.router, prefix="/api/arena", tags=["arena"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


# Serve the bundled single-page frontend (if present). API routes are declared
# before this mount, so they keep priority.
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
