from __future__ import annotations

from pathlib import Path


INTEGRATION_MODE_FRONTLOADED = "frontloaded_handoff"
INTEGRATION_MODE_CHOICES = (INTEGRATION_MODE_FRONTLOADED,)

FEEDBACK_RENDER_MODE_DIRECT = "AmbiModel_direct"
FEEDBACK_RENDER_MODE_CHOICES = (FEEDBACK_RENDER_MODE_DIRECT,)

MODULE_A_CONTEXTUALIZATION_LLM = "llm_detection"
MODULE_A_CONTEXTUALIZATION_MODE_CHOICES = (
    MODULE_A_CONTEXTUALIZATION_LLM,
)

# ADD: Opt-in detection-side COLUMN/VALUE aggregation.  ``None`` remains the
# default so established Module A runs keep their original slice construction.
COL_LIT_AGGREGATION_MODES = (
    "prune_combine",
)

MODULE_A_COLUMN_DESCRIP_EMPTY = "empty"
MODULE_A_COLUMN_DESCRIP_CHOICES = (MODULE_A_COLUMN_DESCRIP_EMPTY,)

# ADD: Opt-in source selector for Module A COLUMN candidates. The default
# preserves the established offline column-graph retrieval path exactly.
MODULE_A_COLUMN_RETRIEVAL_SOURCE_COLUMN_GROUP = "column_group"
MODULE_A_COLUMN_RETRIEVAL_SOURCE_REALTIME_REASONING = "realtime_reasoning"
MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES = (
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_COLUMN_GROUP,
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_REALTIME_REASONING,
)


DDL_MODE_PLAIN = "Sphinteract_plain"
DDL_MODE_CHOICES = (DDL_MODE_PLAIN,)

DEFAULT_INTEGRATION_MODE = INTEGRATION_MODE_FRONTLOADED
DEFAULT_FEEDBACK_RENDER_MODE = FEEDBACK_RENDER_MODE_DIRECT
DEFAULT_MODULE_A_CONTEXTUALIZATION_MODE = MODULE_A_CONTEXTUALIZATION_LLM
DEFAULT_MODULE_A_COLUMN_DESCRIP = MODULE_A_COLUMN_DESCRIP_EMPTY
DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE = MODULE_A_COLUMN_RETRIEVAL_SOURCE_COLUMN_GROUP
DEFAULT_DDL_MODE = DDL_MODE_PLAIN
DEFAULT_MODULE_A_DB_ROOT = ""


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "PROJECT_DIRECTORY_INDEX.md").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root from module_a_base/config.py")


def default_results_root() -> Path:
    return Path.cwd() / "results"


def should_run_module_a_turn(turn: int, *, integration_mode: str) -> bool:
    if integration_mode != INTEGRATION_MODE_FRONTLOADED:
        raise ValueError(f"Unsupported integration_mode: {integration_mode}")
    return turn == 0


def should_run_sphinteract_turn(turn: int, *, integration_mode: str) -> bool:
    if integration_mode != INTEGRATION_MODE_FRONTLOADED:
        raise ValueError(f"Unsupported integration_mode: {integration_mode}")
    return turn > 0


def should_stop_after_turn(turn: int, *, integration_mode: str) -> bool:
    if integration_mode != INTEGRATION_MODE_FRONTLOADED:
        raise ValueError(f"Unsupported integration_mode: {integration_mode}")
    return False
