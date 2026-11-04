"""
ShiftLab global configuration.
All randomness in the platform flows through RANDOM_STATE.
"""
import os

RANDOM_STATE: int = int(os.environ.get("SHIFTLAB_SEED", 42))
# Parallelism default. The joblib/loky backend can terminate workers on some
# Python/OS combinations (notably Python 3.14 on Windows), so single-process is
# the safe default. Override with SHIFTLAB_N_JOBS=-1 to opt into parallelism.
N_JOBS: int = int(os.environ.get("SHIFTLAB_N_JOBS", 1))
VERSION: str = "0.1.0"
EXPERIMENTS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "experiments")
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_UPLOAD_ROWS: int = 500_000
MAX_UPLOAD_MB: int = 50
