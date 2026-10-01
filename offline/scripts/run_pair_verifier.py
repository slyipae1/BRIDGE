#!/usr/bin/env python3
"""Run or dry-run a resumable evidence-free BRIDGE pair-verifier queue.

The queue ordering is operational metadata only. Every model request uses the
schema-only final-verifier projection, which deliberately excludes candidate
channel membership and scores.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError


SOURCE_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE_ROOT))

from auto_construct_graph.final_pair_verifier_v02 import (  # noqa: E402
    build_final_pair_verifier_messages,
    decision_to_lean_edge,
    extract_json_object,
    load_final_pair_verifier_prompt,
    normalize_final_verifier_payload,
    validate_final_verifier_payload,
)
from auto_construct_graph.llm_client import (  # noqa: E402
    OpenAICompatibleChatClient,
    OpenAIResponseDecodeError,
    build_chat_request_payload,
    extract_choice_message,
)
from auto_construct_graph.proposal_aggregation import write_jsonl_records  # noqa: E402


SCHEMA_ONLY_CONTEXT_MODE = "schema_only_no_channel_evidence"
TAXONOMY_PLACEHOLDER = "{{TAXONOMY_GUIDANCE}}"
DEFAULT_PROMPT = SOURCE_CODE_ROOT / "prompts" / "pair_verifier.txt"
DEFAULT_TAXONOMY = SOURCE_CODE_ROOT / "prompts" / "taxonomy_guidance.txt"


class _APIResponseError(RuntimeError):
    """A syntactically valid HTTP response that explicitly reports an API error."""


def _digest_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compose_prompt(prompt_path: Path, taxonomy_path: Path) -> tuple[str, dict[str, Any]]:
    base = load_final_pair_verifier_prompt(prompt_path)
    taxonomy = load_final_pair_verifier_prompt(taxonomy_path)
    guidance = (
        "Taxonomy guidance:\n\n"
        f"{taxonomy}\n\n"
        "Use the taxonomy only as reasoning lenses for the ctx/spec analysis. It does not override supplied "
        "schema evidence and is not itself proof that an edge exists."
    )
    composed = base.replace(TAXONOMY_PLACEHOLDER, guidance) if TAXONOMY_PLACEHOLDER in base else f"{base}\n\n{guidance}"
    return composed, {
        "verifier_mode": "taxonomy_guided",
        "base_prompt_path": str(prompt_path),
        "base_prompt_sha256": _text_digest(base),
        "taxonomy_prompt_path": str(taxonomy_path) if taxonomy else None,
        "taxonomy_prompt_sha256": _text_digest(taxonomy) if taxonomy else None,
        "composed_prompt_sha256": _text_digest(composed),
    }


def _is_paid(base_url: str, model: str) -> bool:
    lowered = f"{base_url} {model}".casefold()
    return "dashscope" in lowered or "aliyuncs" in lowered or "deepseek" in lowered or "qwen3.7" in lowered


def _terminal_by_candidate(decision_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    terminal: dict[str, dict[str, Any]] = {}
    for row in decision_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if row.get("status") == "OK" and candidate_id:
            terminal[candidate_id] = row
    return terminal


def _seen_by_candidate(decision_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the latest recorded attempt for each candidate, regardless of status."""
    seen: dict[str, dict[str, Any]] = {}
    for row in decision_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            seen[candidate_id] = row
    return seen


def _is_api_error_row(row: dict[str, Any]) -> bool:
    error_class = str(row.get("error_class") or "")
    if error_class.startswith("API_"):
        return True
    if error_class:
        return False
    # Backward-compatible classification for ledgers written before error_class.
    error = str(row.get("error") or "").casefold()
    return any(token in error for token in ("timeout", "urlerror", "http error", "connection refused", "connection reset", "quota"))


def _skipped_by_candidate(
    decision_rows: list[dict[str, Any]], *, resume_policy: str,
) -> dict[str, dict[str, Any]]:
    terminal = _terminal_by_candidate(decision_rows)
    if resume_policy == "ok_only":
        return terminal
    seen = _seen_by_candidate(decision_rows)
    if resume_policy == "any_record":
        return seen
    if resume_policy != "primary_sweep":
        raise ValueError(f"unsupported resume policy: {resume_policy}")
    skipped = dict(terminal)
    for candidate_id, row in seen.items():
        if candidate_id not in terminal and not _is_api_error_row(row):
            skipped[candidate_id] = row
    return skipped


def _classify_call_error(exc: Exception, *, response_received: bool, response_content: str | None) -> str:
    if isinstance(exc, OpenAIResponseDecodeError):
        return "API_RESPONSE_PROTOCOL_ERROR"
    if isinstance(exc, _APIResponseError):
        return "API_RESPONSE_ERROR"
    if not response_received:
        if isinstance(exc, HTTPError):
            return "API_HTTP_ERROR"
        if isinstance(exc, (URLError, TimeoutError, ConnectionError)):
            return "API_TRANSPORT_ERROR"
        return "API_CLIENT_ERROR"
    if response_content is not None and isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "PARSER_ERROR"
    return "API_RESPONSE_PROTOCOL_ERROR"


def _error_response_text(exc: Exception) -> str | None:
    if isinstance(exc, OpenAIResponseDecodeError):
        return exc.response_text
    if not isinstance(exc, HTTPError):
        return None
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _update_api_error_streak(
    state_path: Path,
    decision_rows: list[dict[str, Any]],
    *,
    batch_dir: Path,
) -> dict[str, Any]:
    """Persist a global streak of checkpoints whose every call hit an API error."""
    previous: dict[str, Any] = _load_json(state_path) if state_path.exists() else {}
    full_api_error_batch = bool(decision_rows) and all(_is_api_error_row(row) for row in decision_rows)
    count = int(previous.get("consecutive_full_api_error_batches") or 0) + 1 if full_api_error_batch else 0
    payload = {
        "consecutive_full_api_error_batches": count,
        "last_batch_was_full_api_error": full_api_error_batch,
        "last_batch_dir": str(batch_dir),
        "last_batch_size": len(decision_rows),
        "last_batch_api_error_count": sum(1 for row in decision_rows if _is_api_error_row(row)),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(state_path)
    return payload


def _request_payload(
    bundle: dict[str, Any], *, model: str, prompt: str, max_tokens: int, temperature: float,
    disable_thinking: bool, thinking_control: str,
) -> dict[str, Any]:
    return build_chat_request_payload(
        model=model,
        messages=build_final_pair_verifier_messages(prompt, bundle, bundle_context_mode=SCHEMA_ONLY_CONTEXT_MODE),
        max_tokens=max_tokens,
        temperature=temperature,
        disable_thinking=disable_thinking,
        thinking_control=thinking_control,
    )


def _call_one(
    queue_row: dict[str, Any], bundle: dict[str, Any], *, base_url: str, api_key: str, timeout: int,
    request_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    candidate_id = str(queue_row["candidate_id"])
    base = {
        "candidate_id": candidate_id,
        "pair_key": list(queue_row["pair_key"]),
        "queue_position": queue_row["queue_position"],
    }
    raw: dict[str, Any] = {
        **base,
        "status": "ERROR",
        "raw_response": None,
        "response_content": None,
        "error_class": None,
        "error": None,
    }
    decision: dict[str, Any] = {
        **base,
        "status": "ERROR",
        "parsed_payload": None,
        "validation_errors": [],
        "error_class": None,
        "error": None,
    }
    response_received = False
    response_content: str | None = None
    try:
        response = OpenAICompatibleChatClient(base_url=base_url, api_key=api_key, timeout=timeout).chat(request_payload)
        response_received = True
        raw["raw_response"] = response
        if isinstance(response, dict) and response.get("error") is not None:
            raise _APIResponseError(json.dumps(response["error"], ensure_ascii=False))
        message = extract_choice_message(response)
        content = str(message.get("content") or "")
        response_content = content
        choices = response.get("choices") or [{}]
        raw.update(
            {
                "response_content": content,
                "finish_reason": choices[0].get("finish_reason") if isinstance(choices[0], dict) else None,
                "usage": response.get("usage") or {},
                "reasoning_content_present": bool(message.get("reasoning_content") or message.get("reasoning")),
            }
        )
        parsed = normalize_final_verifier_payload(extract_json_object(content), bundle)
        validation_errors = validate_final_verifier_payload(parsed, bundle)
        raw["status"] = "OK"
        decision.update(
            {
                "status": "OK" if not validation_errors else "VALIDATION_ERROR",
                "parsed_payload": parsed,
                "validation_errors": validation_errors,
            }
        )
    except Exception as exc:
        error_class = _classify_call_error(exc, response_received=response_received, response_content=response_content)
        raw["error_class"] = error_class
        raw["error"] = f"{type(exc).__name__}: {exc}"
        if raw["raw_response"] is None:
            raw["raw_response"] = _error_response_text(exc)
        decision["error"] = raw["error"]
        decision["error_class"] = error_class
    elapsed = round(time.time() - started, 6)
    raw["elapsed_seconds"] = elapsed
    decision["elapsed_seconds"] = elapsed
    return raw, decision


def _write_cumulative(output_dir: Path, decision_rows: list[dict[str, Any]]) -> dict[str, int]:
    latest = {str(row.get("candidate_id")): row for row in decision_rows}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate_id, row in sorted(latest.items()):
        if row.get("status") != "OK":
            continue
        parsed = row.get("parsed_payload") or {}
        if parsed.get("decision") == "ADD_EDGE":
            edge = decision_to_lean_edge(parsed)
            edge["candidate_id"] = candidate_id
            accepted.append(edge)
        elif parsed.get("decision") == "REJECT":
            rejected.append({
                "candidate_id": candidate_id, "pair_key": row.get("pair_key"), "col1": parsed.get("col1"),
                "col2": parsed.get("col2"), "reason_summary": parsed.get("reason_summary"), "confidence": parsed.get("confidence"),
            })
    cumulative = output_dir / "cumulative"
    write_jsonl_records(cumulative / "verifier_decisions.jsonl", [latest[key] for key in sorted(latest)])
    write_jsonl_records(cumulative / "accepted_edges.jsonl", accepted)
    write_jsonl_records(cumulative / "rejected_pairs.jsonl", rejected)
    return {"terminal_ok_count": len(accepted) + len(rejected), "accepted_count": len(accepted), "rejected_count": len(rejected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--bundles-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--taxonomy-prompt", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME") or os.getenv("MODEL") or "")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL") or "")
    parser.add_argument("--api-key", default=os.getenv("API_KEY") or "local-token")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--api-error-streak-state", type=Path, default=None)
    parser.add_argument(
        "--api-error-stop-consecutive-full-batches",
        type=int,
        default=0,
        help="Stop only after this many consecutive checkpoints where every call is an API error; 0 disables the guard.",
    )
    parser.add_argument("--run-limit", type=int, default=None)
    parser.add_argument(
        "--resume-policy",
        choices=["ok_only", "any_record", "primary_sweep"],
        default="ok_only",
        help="Skip only successful decisions, or skip every candidate with an existing attempt record.",
    )
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thinking-control", choices=["local_qwen", "dashscope", "both", "none"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=10)
    parser.add_argument("--allow-paid-api", action="store_true")
    args = parser.parse_args()
    if not args.model or not args.base_url:
        raise ValueError("--model and --base-url (or MODEL_NAME/MODEL and BASE_URL) are required")
    if args.max_tokens <= 0 or args.max_concurrency <= 0 or args.batch_size <= 0:
        raise ValueError("max tokens, concurrency, and batch size must be positive")
    if args.api_error_stop_consecutive_full_batches < 0:
        raise ValueError("api-error-stop-consecutive-full-batches must be non-negative")
    if args.api_error_stop_consecutive_full_batches and args.api_error_streak_state is None:
        raise ValueError("--api-error-streak-state is required when the consecutive full API-error guard is enabled")
    if _is_paid(args.base_url, args.model) and not args.dry_run and not args.allow_paid_api:
        raise ValueError("paid API execution requires --allow-paid-api after explicit user approval")

    queue = _load_jsonl(args.queue_path)
    bundle_path = args.bundles_path or args.queue_path.with_name("candidate_bundles.jsonl")
    bundles = _load_jsonl(bundle_path)
    bundles_by_id = {str(bundle.get("candidate_id")): bundle for bundle in bundles}
    if len(bundles_by_id) != len(bundles):
        raise ValueError("candidate bundles have duplicate candidate IDs")
    if {str(row.get("candidate_id")) for row in queue} != set(bundles_by_id):
        raise ValueError("queue candidate IDs must exactly match the candidate-bundle universe")
    prompt, prompt_metadata = _compose_prompt(args.prompt, args.taxonomy_prompt)
    contract = {
        "queue_sha256": _digest_json(queue), "bundle_sha256": _digest_json(bundles),
        "bundle_context_mode": SCHEMA_ONLY_CONTEXT_MODE, "model": args.model, "base_url": args.base_url.rstrip("/"),
        "max_tokens": args.max_tokens, "temperature": args.temperature, "disable_thinking": bool(args.disable_thinking),
        "thinking_control": args.thinking_control, **prompt_metadata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.exists() and _load_json(contract_path) != contract:
        raise ValueError("existing verifier ledger has a different immutable queue/prompt/model contract")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ledger_path = args.output_dir / "ledger" / "verifier_decisions.jsonl"
    raw_path = args.output_dir / "ledger" / "raw_responses.jsonl"
    existing = _load_jsonl(ledger_path)
    terminal = _terminal_by_candidate(existing)
    skipped = _skipped_by_candidate(existing, resume_policy=args.resume_policy)
    pending = [
        row for row in queue
        if str(row["candidate_id"]) not in skipped
    ]
    if args.run_limit is not None:
        pending = pending[:args.run_limit]
    if args.dry_run:
        preview = pending[:args.dry_run_limit]
        request_rows = []
        for row in preview:
            bundle = bundles_by_id[str(row["candidate_id"])]
            payload = _request_payload(bundle, model=args.model, prompt=prompt, max_tokens=args.max_tokens,
                                       temperature=args.temperature, disable_thinking=args.disable_thinking,
                                       thinking_control=args.thinking_control)
            if "proposals_by_channel" in payload["messages"][1]["content"] or "schema_embedding_score" in payload["messages"][1]["content"]:
                raise AssertionError("operational channel metadata leaked into an evidence-free verifier request")
            request_rows.append({"candidate_id": row["candidate_id"], "queue_position": row["queue_position"], "request_payload": payload})
        write_jsonl_records(args.output_dir / "dry_run_request_preview.jsonl", request_rows)
        print(json.dumps({
            "dry_run": True,
            "queue_count": len(queue),
            "already_ok": len(terminal),
            "already_skipped": len(skipped),
            "resume_policy": args.resume_policy,
            "pending": len(pending),
            "preview_count": len(preview),
        }))
        return

    existing_batch_dirs = sorted(path for path in (args.output_dir / "batches").glob("batch_*") if path.is_dir())
    batch_index = len(existing_batch_dirs)
    processed = 0
    stopped_after_consecutive_full_api_error_batches = False
    api_error_streak: dict[str, Any] | None = None
    while pending:
        batch = pending[:args.batch_size]
        pending = pending[args.batch_size:]
        batch_index += 1
        batch_dir = args.output_dir / "batches" / f"batch_{batch_index:05d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        request_rows = [
            {"candidate_id": row["candidate_id"], "queue_position": row["queue_position"], "request_payload": _request_payload(
                bundles_by_id[str(row["candidate_id"])], model=args.model, prompt=prompt, max_tokens=args.max_tokens,
                temperature=args.temperature, disable_thinking=args.disable_thinking, thinking_control=args.thinking_control,
            )}
            for row in batch
        ]
        write_jsonl_records(batch_dir / "request_payloads.jsonl", request_rows)
        request_by_id = {str(row["candidate_id"]): row["request_payload"] for row in request_rows}
        def invoke(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            return _call_one(row, bundles_by_id[str(row["candidate_id"])], base_url=args.base_url, api_key=args.api_key,
                             timeout=args.timeout, request_payload=request_by_id[str(row["candidate_id"])])
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(batch))) as executor:
            results = list(executor.map(invoke, batch))
        raw_rows, decision_rows = zip(*results) if results else ([], [])
        write_jsonl_records(batch_dir / "raw_responses.jsonl", raw_rows)
        write_jsonl_records(batch_dir / "verifier_decisions.jsonl", decision_rows)
        for raw, decision in results:
            _append_jsonl(raw_path, raw)
            _append_jsonl(ledger_path, decision)
        processed += len(results)
        if args.api_error_stop_consecutive_full_batches:
            api_error_streak = _update_api_error_streak(
                args.api_error_streak_state,
                list(decision_rows),
                batch_dir=batch_dir,
            )
            if int(api_error_streak["consecutive_full_api_error_batches"]) >= args.api_error_stop_consecutive_full_batches:
                stopped_after_consecutive_full_api_error_batches = True
                break
    all_decisions = _load_jsonl(ledger_path)
    summary = _write_cumulative(args.output_dir, all_decisions)
    terminal_after = _terminal_by_candidate(all_decisions)
    skipped_after = _skipped_by_candidate(all_decisions, resume_policy=args.resume_policy)
    pending_after_run = sum(
        1
        for row in queue
        if str(row["candidate_id"]) not in skipped_after
    )
    summary.update({
        "processed_this_invocation": processed,
        "queue_count": len(queue),
        "resume_policy": args.resume_policy,
        "pending_after_run": pending_after_run,
        "api_error_streak": api_error_streak,
        "stopped_after_consecutive_full_api_error_batches": stopped_after_consecutive_full_api_error_batches,
    })
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if stopped_after_consecutive_full_api_error_batches:
        raise SystemExit("stopped after consecutive checkpoints fully failed with API errors")


if __name__ == "__main__":
    main()
