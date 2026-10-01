from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from ..batch_client import BatchRequest
from ..output_formatter import flush_question_json
from ..stage_artifacts import is_stage_complete, stage_dir, write_response_json, write_stage_json
from ..vendor_pipeline0612.live_retrieval import retrieve_sql_full
from ..telemetry import Timer, append_cpu_step, append_jsonl, stage_timing
from ..module_a_base.retrieval_ablation import (
    DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
    validate_retrieval_ablation_disable_channel,
)
from ..module_a_base.config import (
    DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES,
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_COLUMN_GROUP,
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_REALTIME_REASONING,
)
from ..module_a_base.ddl_enricher import enrich_schema_ddl
from ..module_a_base.realtime_column_retrieval import (
    REALTIME_COLUMN_RETRIEVAL_PROMPT_VERSION,
    build_realtime_column_retrieval_prompt,
    load_schema_column_map,
    parse_realtime_column_retrieval_response,
)


_REALTIME_COLUMN_RETRIEVAL_STAGE = "module_a_realtime_column_retrieval"
_REALTIME_CHECKPOINT_BATCH_SIZE = 64


def _active_questions(questions):
    return {qid: qs for qid, qs in questions.items() if qs.get("status") == "active"}


def _current_sql_for_question(qs: dict[str, Any]) -> str:
    if qs.get("_last_sql"):
        return qs.get("_last_sql", "")
    sql_log = qs.get("sql_log", [])
    if sql_log:
        return sql_log[-1][2]
    query_set = qs.get("query_set", set())
    return next(iter(query_set), "")


def _payload_path(run_dir: str, turn: int) -> Path:
    return stage_dir(run_dir, "module_a_retrieval", turn) / "module_a_retrieval_payload.json"


def _load_completed_payload(run_dir: str, turn: int) -> dict[str, Any] | None:
    path = _payload_path(run_dir, turn)
    if not is_stage_complete(run_dir, "module_a_retrieval", turn) or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _attach_payload_to_questions(questions: dict[int, dict[str, Any]], payload: dict[str, Any]) -> None:
    for qid_text, record in payload.get("questions", {}).items():
        try:
            qid = int(qid_text)
        except (TypeError, ValueError):
            continue
        target_key = qid if qid in questions else str(qid) if str(qid) in questions else None
        if target_key is not None:
            questions[target_key]["module_a_retrieval"] = record


def _write_payload(run_dir: str, turn: int, payload: dict[str, Any]) -> Path:
    path = _payload_path(run_dir, turn)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _realtime_paths(run_dir: str, turn: int) -> dict[str, Path]:
    directory = stage_dir(run_dir, _REALTIME_COLUMN_RETRIEVAL_STAGE, turn)
    return {
        "directory": directory,
        "manifest": directory / "realtime_column_retrieval_manifest.jsonl",
        "results": directory / "realtime_column_retrieval_results.jsonl",
    }


def _write_jsonl_snapshot(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _realtime_result_key(question_id: int, anchor_column: str, input_hash: str) -> tuple[int, str, str]:
    return int(question_id), str(anchor_column).casefold(), str(input_hash)


def _successful_realtime_results(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    results: dict[tuple[int, str, str], dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if record.get("status") != "succeeded" or not isinstance(record.get("parsed_entry"), dict):
            continue
        try:
            key = _realtime_result_key(
                int(record.get("question_id")),
                str(record.get("anchor_column") or ""),
                str(record.get("input_hash") or ""),
            )
        except (TypeError, ValueError):
            continue
        results[key] = record
    return results


def _parsed_column_anchors(record: dict[str, Any]) -> list[str]:
    parse_meta = (record.get("db_retrieval_sql_full") or {}).get("sql_parse_meta") or {}
    columns_dict = parse_meta.get("columns_dict") if isinstance(parse_meta, dict) else {}
    anchors: list[str] = []
    seen: set[str] = set()
    for table_name, columns in (columns_dict or {}).items():
        for column_name in columns or []:
            anchor = f"{str(table_name)}.{str(column_name)}"
            key = anchor.casefold()
            if anchor and key not in seen:
                seen.add(key)
                anchors.append(anchor)
    return anchors


def _realtime_input_hash(
    *,
    question: str,
    evidence: str,
    target_column: str,
    schema_ddl: str,
) -> str:
    payload = {
        "prompt_version": REALTIME_COLUMN_RETRIEVAL_PROMPT_VERSION,
        "question": question,
        "evidence": evidence,
        "target_column": target_column,
        "schema_ddl": schema_ddl,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summarize_realtime_results(results: dict[tuple[int, str, str], dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "request_count": len(results),
        "ok_count": len(results),
        "failed_count": 0,
        "missing_usage_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for record in results.values():
        usage = record.get("usage") or {}
        if not usage:
            summary["missing_usage_count"] += 1
        summary["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        summary["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        summary["total_tokens"] += int(usage.get("total_tokens") or 0)
    return summary


def _run_realtime_column_retrieval(
    *,
    run_dir: str,
    questions: dict[int, dict[str, Any]],
    payload: dict[str, Any],
    turn: int,
    batch_client: Any,
) -> dict[str, Any]:
    """Materialize COLUMN entries from resumable batched schema reasoning."""
    if batch_client is None:
        raise ValueError("realtime COLUMN retrieval requires an OnlineBatchClient")

    paths = _realtime_paths(run_dir, turn)
    successful_results = _successful_realtime_results(paths["results"])
    schema_maps: dict[str, dict[str, str]] = {}
    manifests: list[dict[str, Any]] = []
    pending: list[tuple[BatchRequest, dict[str, str]]] = []
    expected_keys: list[tuple[int, str, str]] = []

    for qid, qs in _active_questions(questions).items():
        record = payload["questions"].get(str(int(qid)))
        if not record:
            continue
        anchors = _parsed_column_anchors(record)
        if not anchors:
            continue
        db_path = str(qs.get("db_file") or "")
        if db_path not in schema_maps:
            schema_maps[db_path] = load_schema_column_map(db_path)
        schema_ddl = enrich_schema_ddl(
            qs.get("dbschema", ""),
            additional_info_by_column={},
            ddl_mode="Sphinteract_plain",
            subset_only=False,
            current_sql="",
            dbelement_options=[],
        )
        question = str(qs.get("question") or "")
        evidence = str(qs.get("evidence") or "")
        for anchor_column in anchors:
            input_hash = _realtime_input_hash(
                question=question,
                evidence=evidence,
                target_column=anchor_column,
                schema_ddl=schema_ddl,
            )
            key = _realtime_result_key(int(qid), anchor_column, input_hash)
            expected_keys.append(key)
            prompt = build_realtime_column_retrieval_prompt(
                question=question,
                evidence=evidence,
                target_column=anchor_column,
                schema_ddl=schema_ddl,
            )
            metadata = {
                "realtime_column_retrieval": True,
                "prompt_version": REALTIME_COLUMN_RETRIEVAL_PROMPT_VERSION,
                "target_column": anchor_column,
                "input_hash": input_hash,
                "schema_mode": "Sphinteract_plain",
                "evidence_included": bool(evidence),
            }
            manifests.append(
                {
                    "question_id": int(qid),
                    "db_id": qs.get("db_id", ""),
                    "anchor_column": anchor_column,
                    "input_hash": input_hash,
                    "prompt_version": REALTIME_COLUMN_RETRIEVAL_PROMPT_VERSION,
                    "prompt": prompt,
                    "metadata": metadata,
                }
            )
            if key in successful_results:
                continue
            pending.append(
                (
                    BatchRequest(
                        request_id=uuid.uuid4().hex,
                        question_id=int(qid),
                        internal_index=qs.get("internal_index", 0),
                        prompt=prompt,
                        stage=_REALTIME_COLUMN_RETRIEVAL_STAGE,
                        turn=turn,
                        stop_thinking=True,
                        temperature=0.0,
                        max_tokens=4096,
                        metadata=metadata,
                    ),
                    {"anchor_column": anchor_column, "input_hash": input_hash, "db_path": db_path},
                )
            )

    _write_jsonl_snapshot(paths["manifest"], manifests)
    executed_requests: list[BatchRequest] = []
    executed_responses: list[Any] = []
    failed_requests: list[dict[str, Any]] = []

    for start in range(0, len(pending), _REALTIME_CHECKPOINT_BATCH_SIZE):
        chunk = pending[start : start + _REALTIME_CHECKPOINT_BATCH_SIZE]
        requests = [item[0] for item in chunk]
        responses = batch_client.batch_generate(requests)
        executed_requests.extend(requests)
        executed_responses.extend(responses)
        for request, context, response in zip(requests, [item[1] for item in chunk], responses):
            result = {
                "request_id": request.request_id,
                "question_id": int(request.question_id),
                "anchor_column": context["anchor_column"],
                "input_hash": context["input_hash"],
                "prompt_version": REALTIME_COLUMN_RETRIEVAL_PROMPT_VERSION,
                "request_log_path": response.request_log_path,
                "usage": response.usage,
                "timing": response.timing,
                "raw_content": response.content,
            }
            if not response.ok:
                result.update({"status": "failed", "error": response.error})
                failed_requests.append(result)
                append_jsonl(paths["results"], result)
                continue
            try:
                entry, parse_audit = parse_realtime_column_retrieval_response(
                    response.content,
                    target_column=context["anchor_column"],
                    schema_column_map=schema_maps[context["db_path"]],
                )
            except Exception as exc:
                result.update({"status": "failed", "error": f"{exc.__class__.__name__}: {exc}"})
                failed_requests.append(result)
                append_jsonl(paths["results"], result)
                continue
            result.update(
                {
                    "status": "succeeded",
                    "parsed_entry": entry,
                    "parse_audit": parse_audit,
                }
            )
            successful_results[_realtime_result_key(
                request.question_id,
                context["anchor_column"],
                context["input_hash"],
            )] = result
            append_jsonl(paths["results"], result)

    if executed_requests:
        write_response_json(
            run_dir,
            _REALTIME_COLUMN_RETRIEVAL_STAGE,
            turn,
            executed_requests,
            executed_responses,
        )
    missing_keys = [key for key in expected_keys if key not in successful_results]
    if failed_requests or missing_keys:
        raise RuntimeError(
            "realtime COLUMN retrieval did not complete; resumable per-anchor results were saved "
            f"({len(failed_requests)} failed, {len(missing_keys)} missing)"
        )

    entries_by_qid: dict[int, list[dict[str, Any]]] = {}
    audits_by_qid: dict[int, list[dict[str, Any]]] = {}
    for qid, anchor_column, input_hash in expected_keys:
        result = successful_results[(qid, anchor_column, input_hash)]
        entries_by_qid.setdefault(qid, []).append(result["parsed_entry"])
        audits_by_qid.setdefault(qid, []).append(
            {
                "anchor_column": anchor_column,
                "input_hash": input_hash,
                "request_id": result.get("request_id", ""),
                "request_log_path": result.get("request_log_path", ""),
                "parse_audit": result.get("parse_audit", {}),
            }
        )

    for qid, qs in _active_questions(questions).items():
        record = payload["questions"].get(str(int(qid)))
        if not record:
            continue
        retrieval_payload = record.setdefault("db_retrieval_sql_full", {})
        value_entries = [
            entry
            for entry in (retrieval_payload.get("dbelement_options") or [])
            if str(entry.get("type", "")).upper() != "COLUMN"
        ]
        realtime_entries = entries_by_qid.get(int(qid), [])
        retrieval_payload["dbelement_options"] = [*realtime_entries, *value_entries]
        # NOTE: This field is not consumed by downstream Module A stages. A
        # full plain schema remains coherent with realtime candidates and avoids
        # retaining the partial schema from before replacement.
        retrieval_payload["schema_string"] = str(qs.get("dbschema") or "")
        retrieval_payload["schema_string_scope"] = "full_sphinteract_plain"
        retrieval_payload["column_retrieval_source"] = "realtime_reasoning"
        retrieval_payload["realtime_column_retrieval"] = {
            "prompt_version": REALTIME_COLUMN_RETRIEVAL_PROMPT_VERSION,
            "anchor_count": len(realtime_entries),
            "anchors": audits_by_qid.get(int(qid), []),
        }
        retrieval_ablation = retrieval_payload.get("retrieval_ablation") or {}
        channel_meta = (retrieval_ablation.get("channels") or {}).get("column")
        if isinstance(channel_meta, dict):
            channel_meta["candidate_generation_executed"] = True
            channel_meta["candidate_generation_executed_by"] = _REALTIME_COLUMN_RETRIEVAL_STAGE
        active_counts = retrieval_ablation.setdefault("active_dbelement_entry_counts", {})
        active_counts["COLUMN"] = len(realtime_entries)
        active_counts["VALUE"] = len(value_entries)
        retrieval_payload["retrieval_ablation"] = retrieval_ablation
        qs["module_a_retrieval"] = record
        qs.setdefault("events", []).append(
            {
                "event_type": _REALTIME_COLUMN_RETRIEVAL_STAGE,
                "turn": turn,
                "anchor_count": len(realtime_entries),
                "source": "realtime_reasoning",
            }
        )

    return {
        "stage": _REALTIME_COLUMN_RETRIEVAL_STAGE,
        "prompt_version": REALTIME_COLUMN_RETRIEVAL_PROMPT_VERSION,
        "manifest_path": str(paths["manifest"]),
        "results_path": str(paths["results"]),
        "request_count": len(expected_keys),
        "new_request_count": len(executed_requests),
        "resumed_request_count": len(expected_keys) - len(executed_requests),
        "usage_summary": _summarize_realtime_results(
            {
                key: successful_results[key]
                for key in expected_keys
            }
        ),
    }


def _count_columns(columns_dict: dict[str, list[str]] | None) -> int:
    return sum(len(columns or []) for columns in (columns_dict or {}).values())


def _append_sql_parse_cpu_metric(
    run_dir: str,
    *,
    turn: int,
    qid: int,
    db_id: str,
    sql_parse_meta: dict[str, Any] | None,
    status: str = "ok",
    fallback_elapsed_s: float | None = None,
    error: str = "",
) -> None:
    meta = sql_parse_meta or {}
    timing = meta.get("timing") or {}
    elapsed = timing.get("total_parse_elapsed_s")
    if elapsed is None:
        elapsed = fallback_elapsed_s if fallback_elapsed_s is not None else 0.0
    append_cpu_step(
        run_dir,
        {
            "step": "sql_parsing",
            "stage": "module_a_retrieval",
            "turn": turn,
            "question_id": int(qid),
            "db_id": db_id,
            "status": status,
            "elapsed_s": float(elapsed or 0.0),
            "metadata": {
                "source": meta.get("source", "none"),
                "tables_count": len(meta.get("tables") or []),
                "columns_count": _count_columns(meta.get("columns_dict") or {}),
                "literal_count": int(meta.get("literal_count") or 0),
                "llm_fallback_used": bool(timing.get("llm_fallback_used", False)),
                "column_parse_elapsed_s": float(timing.get("column_parse_elapsed_s") or 0.0),
                "literal_parse_elapsed_s": float(timing.get("literal_parse_elapsed_s") or 0.0),
                "llm_fallback_elapsed_s": float(timing.get("llm_fallback_elapsed_s") or 0.0),
                "error": error,
            },
        },
    )


def run_module_a_retrieval(
    run_dir,
    questions,
    turn,
    *,
    db_root_path,
    db_mode="dev",
    lsh_top_n=20,
    enable_llm_fallback=True,
    column_group_version="manual",
    column_group_artifact_root=None,
    retrieval_ablation_disable_channel=DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
    column_retrieval_source=DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
    batch_client=None,
    debug=False,
):
    """Run or resume live DB-element retrieval for Module A."""
    retrieval_ablation_disable_channel = validate_retrieval_ablation_disable_channel(
        retrieval_ablation_disable_channel
    )
    if column_retrieval_source not in MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES:
        raise ValueError(
            "column_retrieval_source must be one of "
            f"{MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES}, got {column_retrieval_source!r}"
        )
    if column_retrieval_source == MODULE_A_COLUMN_RETRIEVAL_SOURCE_REALTIME_REASONING:
        if retrieval_ablation_disable_channel == "column":
            raise ValueError("realtime COLUMN retrieval is incompatible with --retrieval-ablation-disable-channel column")
    completed_payload = _load_completed_payload(run_dir, turn)
    if completed_payload is not None:
        if debug:
            print(f"[MODA-RET] round_{turn}/module_a_retrieval already complete, loading payload.")
        _attach_payload_to_questions(questions, completed_payload)
        return _active_questions(questions)

    timer = Timer("module_a_retrieval")
    active = _active_questions(questions)
    payload: dict[str, Any] = {
        "meta": {
            "stage": "module_a_retrieval",
            "source": "live_pipeline0612_dbelement_options_v0",
            "retrieval_anchor": "current_full_sql",
            "lsh_top_n": int(lsh_top_n),
            "column_retrieval_source": column_retrieval_source,
            "retrieval_ablation_disable_channel": retrieval_ablation_disable_channel,
            "column_group_version": column_group_version,
            "column_group_artifact_root": str(column_group_artifact_root) if column_group_artifact_root else None,
            "db_mode": db_mode,
            "total_questions": len(active),
            "completed_questions": 0,
            "failed_questions": 0,
        },
        "questions": {},
    }

    for qid, qs in active.items():
        current_sql = _current_sql_for_question(qs)
        question_timer = Timer("sql_parsing")
        try:
            record = retrieve_sql_full(
                question_id=int(qid),
                db_id=qs.get("db_id", ""),
                question=qs.get("question", ""),
                evidence=qs.get("evidence", ""),
                current_sql=current_sql,
                db_root_path=db_root_path,
                db_mode=db_mode,
                lsh_top_n=lsh_top_n,
                enable_llm_fallback=enable_llm_fallback,
                column_group_version=column_group_version,
                column_group_artifact_root=str(column_group_artifact_root) if column_group_artifact_root else None,
                retrieval_ablation_disable_channel=retrieval_ablation_disable_channel,
                column_retrieval_source=column_retrieval_source,
            )
            sql_parse_meta = record.get("db_retrieval_sql_full", {}).get("sql_parse_meta") or {}
            _append_sql_parse_cpu_metric(
                run_dir,
                turn=turn,
                qid=int(qid),
                db_id=qs.get("db_id", ""),
                sql_parse_meta=sql_parse_meta,
            )
            qs["module_a_retrieval"] = record
            qs.setdefault("events", []).append(
                {
                    "event_type": "module_a_retrieval",
                    "turn": turn,
                    "status": "ok",
                    "source": "live_pipeline0612_dbelement_options_v0",
                    "dbelement_option_count": len(
                        record.get("db_retrieval_sql_full", {}).get("dbelement_options", []) or []
                    ),
                }
            )
            payload["questions"][str(int(qid))] = record
            payload["meta"]["completed_questions"] += 1
        except Exception as exc:
            failure_timing = stage_timing(question_timer)
            _append_sql_parse_cpu_metric(
                run_dir,
                turn=turn,
                qid=int(qid),
                db_id=qs.get("db_id", ""),
                sql_parse_meta=None,
                status="failed",
                fallback_elapsed_s=failure_timing.get("elapsed_s", 0.0),
                error=str(exc),
            )
            qs.setdefault("events", []).append(
                {
                    "event_type": "module_a_retrieval",
                    "turn": turn,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            payload["questions"][str(int(qid))] = {
                "question_id": int(qid),
                "db_id": qs.get("db_id", ""),
                "question": qs.get("question", ""),
                "evidence": qs.get("evidence", ""),
                "base_sql": current_sql,
                "db_retrieval_sql_full": {
                    "source_scope": "question_sql",
                    "source_group_id": None,
                    "source_period_id": None,
                    "dbelement_options": [],
                    "schema_string": "",
                },
                "error": str(exc),
            }
            payload["meta"]["failed_questions"] += 1
        finally:
            flush_question_json(run_dir, qid, qs)

    realtime_summary = None
    if column_retrieval_source == MODULE_A_COLUMN_RETRIEVAL_SOURCE_REALTIME_REASONING:
        realtime_summary = _run_realtime_column_retrieval(
            run_dir=run_dir,
            questions=questions,
            payload=payload,
            turn=turn,
            batch_client=batch_client,
        )
        payload["meta"]["realtime_column_retrieval"] = realtime_summary

    payload_path = _write_payload(run_dir, turn, payload)
    write_stage_json(
        run_dir,
        "module_a_retrieval",
        turn,
        {
            "status": "ok",
            "payload_path": str(payload_path),
            "completed_questions": payload["meta"]["completed_questions"],
            "failed_questions": payload["meta"]["failed_questions"],
            "realtime_column_retrieval": realtime_summary,
            "timing": stage_timing(timer),
        },
    )
    return active
