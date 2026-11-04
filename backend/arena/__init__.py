"""ShiftLab Arena — algorithm comparison and leaderboard utilities."""

from backend.arena.comparison import (
    PRIMARY_METRIC,
    build_leaderboard,
    primary_metric_for,
    run_comparison,
)

__all__ = [
    "PRIMARY_METRIC",
    "build_leaderboard",
    "primary_metric_for",
    "run_comparison",
]
