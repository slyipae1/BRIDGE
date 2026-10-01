from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def monotonic_s() -> float:
    return time.perf_counter()


def elapsed_s(start: float, end: float | None = None) -> float:
    finish = monotonic_s() if end is None else end
    return round(max(0.0, finish - start), 6)


def metrics_path(run_dir: str | Path, filename: str) -> Path:
    path = Path(run_dir) / "metrics" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: str | Path, record: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_usage(responses: Iterable[object]) -> dict:
    items = list(responses or [])
    summary = {
        "request_count": len(items),
        "ok_count": 0,
        "failed_count": 0,
        "missing_usage_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for response in items:
        if bool(getattr(response, "ok", False)):
            summary["ok_count"] += 1
        else:
            summary["failed_count"] += 1
        usage = getattr(response, "usage", None) or {}
        if not usage:
            summary["missing_usage_count"] += 1
        summary["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        summary["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        summary["total_tokens"] += int(usage.get("total_tokens") or 0)
    return summary


class Timer:
    def __init__(self, label: str):
        self.label = label
        self.started_at = iso_now()
        self._start = monotonic_s()

    def finish(self, extra: dict | None = None) -> dict:
        record = {
            "label": self.label,
            "started_at": self.started_at,
            "completed_at": iso_now(),
            "elapsed_s": elapsed_s(self._start),
        }
        if extra:
            record.update(extra)
        return record


def stage_timing(timer: Timer, extra: dict | None = None) -> dict:
    record = timer.finish(extra)
    record.pop("label", None)
    return record


def append_cpu_step(run_dir: str | Path, record: dict) -> None:
    append_jsonl(
        metrics_path(run_dir, "cpu_steps.jsonl"),
        {"record_type": "cpu_step", **record},
    )


def append_round_timing(run_dir: str | Path, record: dict) -> None:
    append_jsonl(
        metrics_path(run_dir, "round_timings.jsonl"),
        {"record_type": "round_timing", **record},
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _empty_usage_bucket() -> dict:
    return {
        "request_count": 0,
        "ok_count": 0,
        "failed_count": 0,
        "missing_usage_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _add_llm_usage(bucket: dict, record: dict) -> None:
    usage = record.get("usage") or {}
    bucket["request_count"] += 1
    if record.get("status") == "succeeded":
        bucket["ok_count"] += 1
    else:
        bucket["failed_count"] += 1
    if not usage:
        bucket["missing_usage_count"] += 1
    bucket["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    bucket["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    bucket["total_tokens"] += int(usage.get("total_tokens") or 0)


def _usage_group(groups: dict, key: str) -> dict:
    return groups.setdefault(str(key), _empty_usage_bucket())


def build_cost_summary(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"
    llm_records = _read_jsonl(metrics_dir / "llm_calls.jsonl")
    cpu_records = _read_jsonl(metrics_dir / "cpu_steps.jsonl")
    stage_records = _read_jsonl(metrics_dir / "stage_timings.jsonl")

    usage_by_stage: dict[str, dict] = {}
    usage_by_model: dict[str, dict] = {}
    usage_by_turn: dict[str, dict] = {}
    total_llm_api_elapsed_s = 0.0
    total_llm_total_task_elapsed_s = 0.0

    for record in llm_records:
        stage_key = record.get("stage_artifact") or record.get("stage") or "unknown"
        model_key = record.get("model") or "unknown"
        turn_key = record.get("turn", "unknown")
        _add_llm_usage(_usage_group(usage_by_stage, stage_key), record)
        _add_llm_usage(_usage_group(usage_by_model, model_key), record)
        _add_llm_usage(_usage_group(usage_by_turn, turn_key), record)
        timing = record.get("timing") or {}
        total_llm_api_elapsed_s += float(timing.get("api_elapsed_s") or 0.0)
        total_llm_total_task_elapsed_s += float(timing.get("total_task_elapsed_s") or 0.0)

    total_stage_elapsed_s = sum(float(record.get("elapsed_s") or 0.0) for record in stage_records)
    cpu_elapsed_by_step: dict[str, float] = {}
    for record in cpu_records:
        step = str(record.get("step") or "unknown")
        cpu_elapsed_by_step[step] = cpu_elapsed_by_step.get(step, 0.0) + float(
            record.get("elapsed_s") or 0.0
        )
    cpu_elapsed_by_step = {
        step: round(elapsed, 6)
        for step, elapsed in sorted(cpu_elapsed_by_step.items())
    }
    total_sql_parsing_elapsed_s = sum(
        float(record.get("elapsed_s") or 0.0)
        for record in cpu_records
        if record.get("step") == "sql_parsing"
    )
    total_sql_evaluation_elapsed_s = sum(
        float(record.get("elapsed_s") or 0.0)
        for record in cpu_records
        if record.get("step") == "sql_evaluation"
    )
    total_module_a_direct_render_elapsed_s = sum(
        float(record.get("elapsed_s") or 0.0)
        for record in cpu_records
        if record.get("step") == "module_a_direct_render"
    )

    output_path = metrics_path(run_dir, "cost_summary.json")
    summary = {
        "run_dir": str(run_dir),
        "generated_at": iso_now(),
        "cost_summary_path": str(output_path),
        "llm_usage_by_stage": usage_by_stage,
        "llm_usage_by_model": usage_by_model,
        "llm_usage_by_turn": usage_by_turn,
        "timing_summary": {
            "total_stage_elapsed_s": round(total_stage_elapsed_s, 6),
            "total_llm_api_elapsed_s": round(total_llm_api_elapsed_s, 6),
            "total_llm_total_task_elapsed_s": round(total_llm_total_task_elapsed_s, 6),
            "total_sql_parsing_elapsed_s": round(total_sql_parsing_elapsed_s, 6),
            "total_sql_evaluation_elapsed_s": round(total_sql_evaluation_elapsed_s, 6),
            "total_module_a_direct_render_elapsed_s": round(total_module_a_direct_render_elapsed_s, 6),
            "cpu_elapsed_by_step": cpu_elapsed_by_step,
        },
        "pricing": {
            "status": "not_applied",
            "reason": "token usage is recorded; model price table is intentionally external",
        },
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
