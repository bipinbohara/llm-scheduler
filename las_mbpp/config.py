from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_GATE_CHECKPOINT = PROJECT_ROOT / "artifacts" / "gate" / "best_model_state.pt"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
