"""Project paths and shared constants for Week 6."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
PREPROCESSED_DATA_DIR: Path = DATA_DIR / "preprocessed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
METADATA_DIR: Path = RESULTS_DIR / "metadata"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
TABLES_DIR: Path = RESULTS_DIR / "tables"
EXAMPLES_DIR: Path = RESULTS_DIR / "examples"

SEED: int = 42
