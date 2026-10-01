"""
Stage artifacts for checkpoint/resume.
"""
import json, os, time
from pathlib import Path


STAGE_ARTIFACTS_DIR = "stage_artifacts"
LATEST_STAGE_FILE = "-latest-stage.json"

STAGE_ORDER = [
    "seed_generation",
    "seed_evaluation",
    "module_a_retrieval",
    "module_a_feedback_generation",
    "cq_generation",       # per round, index 0
    "feedback_generation", # per round, index 1
    "sql_regeneration",    # per round, index 2
    "sql_evaluation",      # per round, index 3 (includes fix_invalid)
]


def stage_dir(run_dir: str, stage: str, round_idx: int = 0) -> Path:
    """Return the stage artifact directory path."""
    if round_idx > 0 and stage in (
        "module_a_feedback_generation",
        "module_a_retrieval",
        "module_a_realtime_column_retrieval",
        "cq_generation",
        "feedback_generation",
        "sql_regeneration",
        "sql_evaluation",
    ):
        return Path(run_dir) / STAGE_ARTIFACTS_DIR / f"round_{round_idx:02d}" / stage
    return Path(run_dir) / STAGE_ARTIFACTS_DIR / stage


def stage_key(stage: str, round_idx: int = 0) -> str:
    """Unique key for a stage (used in latest-stage tracking)."""
    if round_idx > 0 and stage in (
        "module_a_feedback_generation",
        "module_a_retrieval",
        "module_a_realtime_column_retrieval",
        "cq_generation",
        "feedback_generation",
        "sql_regeneration",
        "sql_evaluation",
    ):
        return f"round_{round_idx:02d}/{stage}"
    return stage


def write_stage_json(run_dir: str, stage: str, round_idx: int, data: dict):
    """Write .stage.json for a completed stage."""
    d = stage_dir(run_dir, stage, round_idx)
    os.makedirs(d, exist_ok=True)
    path = d / f"{stage}.stage.json"
    data["stage"] = stage
    data["round"] = round_idx
    data["updated_at"] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # Update latest-stage pointer
    update_latest_stage(run_dir, stage, round_idx, data.get("status", "ok"))
    if isinstance(data.get("timing"), dict):
        record = {
            "record_type": "stage_timing",
            "stage": stage,
            "turn": round_idx,
            **data["timing"],
            "status": data.get("status", "ok"),
        }
        if "usage_summary" in data:
            record["usage_summary"] = data["usage_summary"]
        if "substage_usage_summary" in data:
            record["substage_usage_summary"] = data["substage_usage_summary"]
        write_stage_timing_jsonl(run_dir, record)


def write_stage_timing_jsonl(run_dir: str, record: dict) -> None:
    from .telemetry import append_jsonl, metrics_path

    append_jsonl(metrics_path(run_dir, "stage_timings.jsonl"), record)


def load_stage_json(run_dir: str, stage: str, round_idx: int = 0) -> dict:
    """Load .stage.json, return None if missing."""
    path = stage_dir(run_dir, stage, round_idx) / f"{stage}.stage.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def is_stage_complete(run_dir: str, stage: str, round_idx: int = 0) -> bool:
    """Check if a stage was completed successfully."""
    data = load_stage_json(run_dir, stage, round_idx)
    return data is not None and data.get("status") == "ok"


def write_response_json(run_dir: str, stage: str, round_idx: int, requests: list, responses: list):
    """Write .response.json with serialized requests and responses."""
    d = stage_dir(run_dir, stage, round_idx)
    os.makedirs(d, exist_ok=True)
    path = d / f"{stage}.response.json"

    req_list = []
    for r in requests:
        req_list.append({
            "request_id": r.request_id,
            "question_id": r.question_id,
            "internal_index": r.internal_index,
                "stage": r.stage,
                "turn": r.turn,
                "stop_thinking": r.stop_thinking,
                "metadata": getattr(r, "metadata", {}) or {},
            })

    resp_list = []
    for r in responses:
        resp_list.append({
            "request_id": r.request_id,
            "question_id": r.question_id,
            "internal_index": r.internal_index,
            "stage": getattr(r, "stage", stage),
            "turn": getattr(r, "turn", round_idx),
            "ok": r.ok,
            "content": r.content[:2000] if r.content else "",
            "perplexity": r.perplexity,
            "reasoning": r.reasoning[:500] if r.reasoning else "",
            "usage": r.usage,
            "timing": getattr(r, "timing", {}) or {},
            "error": r.error,
            "request_log_path": r.request_log_path,
        })

    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"stage": stage, "round": round_idx,
                   "requests": req_list, "responses": resp_list}, f, ensure_ascii=False, indent=2)


def load_response_json(run_dir: str, stage: str, round_idx: int = 0) -> dict:
    """Load .response.json, return None if missing."""
    path = stage_dir(run_dir, stage, round_idx) / f"{stage}.response.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def update_latest_stage(run_dir: str, stage: str, round_idx: int, status: str):
    """Update -latest-stage.json quick pointer."""
    path = Path(run_dir) / LATEST_STAGE_FILE
    data = {
        "stage": stage,
        "round": round_idx,
        "status": status,
        "updated_at": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_latest_stage(run_dir: str) -> dict:
    """Load -latest-stage.json, return None if missing."""
    path = Path(run_dir) / LATEST_STAGE_FILE
    if path.exists():
        return json.loads(path.read_text())
    return None
