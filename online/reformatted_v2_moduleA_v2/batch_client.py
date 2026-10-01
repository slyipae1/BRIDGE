"""
Online batch client for concurrent async OpenAI-compatible API calls.
Same design as reformatted/vllm_batch_client.py but self-contained.
"""
import os, json, time, uuid, asyncio
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .telemetry import append_jsonl, elapsed_s, iso_now, metrics_path, monotonic_s


@dataclass
class BatchRequest:
    request_id: str
    question_id: int
    internal_index: int
    prompt: str
    stage: str
    turn: int
    stop_thinking: bool
    temperature: float = 0.0
    max_tokens: int = 16384
    metadata: dict = field(default_factory=dict)


@dataclass
class BatchResponse:
    request_id: str
    question_id: int
    internal_index: int
    stage: str
    turn: int
    ok: bool
    content: str
    perplexity: float
    reasoning: str = ''
    usage: Optional[dict] = None
    error: str = ''
    request_log_path: str = ''
    timing: dict = field(default_factory=dict)
    raw_response: object = None


REQUEST_STAGE_TO_ARTIFACT = {
    "seed": "seed_generation",
    "sql_fix_seed": "seed_evaluation",
    "module_a_detection": "module_a_detection",
    "module_a_feedback_generation": "module_a_feedback_generation",
    "cq_gen": "cq_generation",
    "feedback": "feedback_generation",
    "sql_gen": "sql_regeneration",
    "sql_fix": "sql_evaluation",
}


def _env_flag(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _resolve_provider_profile(value: str | None) -> str:
    profile = str(value or "generic").strip().casefold()
    if profile not in {"generic", "dashscope_batch"}:
        raise ValueError(f"Unsupported OpenAI-compatible provider profile: {profile!r}")
    return profile


def _resolve_timeout_seconds(value: object, *, default: float = 600.0) -> float:
    try:
        timeout = float(value if value not in (None, "") else default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM API timeout must be positive, got {value!r}") from exc
    if timeout <= 0:
        raise ValueError(f"LLM API timeout must be positive, got {timeout!r}")
    return timeout


def _write_request_log(request_id, payload):
    log_dir = os.environ.get("LLM_REQUEST_LOG_DIR")
    if not log_dir:
        return None
    path = Path(log_dir) / f"{request_id}.json"
    os.makedirs(log_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'request_id': request_id, **payload}, f, ensure_ascii=False, indent=2)
    return str(path)


def _stage_artifact_name(stage: str) -> str:
    return REQUEST_STAGE_TO_ARTIFACT.get(stage, stage)


def _run_dir_from_request_log_dir(log_dir: str | None) -> Optional[Path]:
    if not log_dir:
        return None
    path = Path(log_dir).expanduser()
    if path.name == "llm_requests":
        return path.parent
    return path.parent


def _usage_to_dict(response) -> Optional[dict]:
    try:
        usage = response.usage
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    except Exception:
        return None


def _write_llm_call_metric(log_dir: str | None, payload: dict) -> None:
    run_dir = _run_dir_from_request_log_dir(log_dir)
    if run_dir is None:
        return
    append_jsonl(metrics_path(run_dir, "llm_calls.jsonl"), payload)


def compute_perplexity_from_response(response) -> float:
    """exp(-mean(logprob)) from OpenAI-format response."""
    try:
        logprobs = response.choices[0].logprobs.content
        lp = [t.logprob for t in logprobs]
        return float(np.exp(-np.mean(lp)))
    except Exception:
        return 0.0


def extract_reasoning(response) -> str:
    try:
        msg = response.choices[0].message
        return getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_content', None) or ''
    except Exception:
        return ''


def build_thinking_suppressed_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You are a helpful assistant. Output ONLY the answer with no explanation, "
            "no thinking, no chain-of-thought, no markdown, no code block markers, "
            "no backticks, no extra commentary. If the answer is SQL, return exactly one "
            "plain-text SQLite SELECT query and nothing else."
        )},
        {"role": "user", "content": prompt},
    ]


def build_normal_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


class OnlineBatchClient:
    """Async concurrent LLM calls to an OpenAI-compatible endpoint."""

    def __init__(self, model_name: str, base_url: str, api_key: str,
                 temperature: float = 0.0, max_tokens: int = 16384,
                 concurrency: int = 8, provider_profile: str | None = None,
                 api_timeout_s: float | None = None):
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._concurrency = concurrency
        self._provider_profile = _resolve_provider_profile(provider_profile)
        self._api_timeout_s = _resolve_timeout_seconds(api_timeout_s)

    def batch_generate(self, requests: list[BatchRequest]) -> list[BatchResponse]:
        if not requests:
            return []
        return asyncio.run(self._async_batch_generate(requests))

    async def _async_batch_generate(self, requests: list[BatchRequest]) -> list[BatchResponse]:
        semaphore = asyncio.Semaphore(self._concurrency)
        force_disable = _env_flag("FORCE_DISABLE_THINKING", False)

        async def _call_one(req: BatchRequest) -> BatchResponse:
            request_id = req.request_id
            stop_thinking = True if force_disable else req.stop_thinking
            task_started_m = monotonic_s()
            task_created_at = iso_now()

            _write_request_log(request_id, {
                'status': 'queued', 'created_at': task_created_at,
                'model': self._model_name, 'stage': req.stage, 'turn': req.turn,
                'stop_thinking': stop_thinking, 'question_id': req.question_id,
                'internal_index': req.internal_index, 'prompt': req.prompt,
                'metadata': req.metadata,
                'timing': {'task_created_at': task_created_at},
            })

            async with semaphore:
                semaphore_acquired_m = monotonic_s()
                semaphore_acquired_at = iso_now()
                from openai import AsyncOpenAI
                client_setup_start_m = monotonic_s()
                client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url, max_retries=0)
                try:
                    messages = build_normal_messages(req.prompt)
                    kwargs = dict(
                        model=self._model_name, messages=messages,
                        temperature=req.temperature or self._default_temperature,
                        max_tokens=req.max_tokens or self._default_max_tokens,
                        timeout=self._api_timeout_s,
                    )
                    if self._provider_profile == "generic":
                        kwargs["logprobs"] = True
                    if stop_thinking:
                        kwargs['messages'] = build_thinking_suppressed_messages(req.prompt)
                        if self._provider_profile == "dashscope_batch":
                            # DashScope requires this non-standard parameter at
                            # the Chat Completions body level, not under vLLM's
                            # chat_template_kwargs wrapper.
                            kwargs['extra_body'] = {"enable_thinking": False}
                        else:
                            kwargs['extra_body'] = {"chat_template_kwargs": {"enable_thinking": False}}

                    api_request_started_m = monotonic_s()
                    api_request_started_at = iso_now()
                    resp = await client.chat.completions.create(**kwargs)
                    api_response_received_m = monotonic_s()
                    api_response_received_at = iso_now()
                    content = (resp.choices[0].message.content or '').strip()
                    perplexity = compute_perplexity_from_response(resp)
                    reasoning = extract_reasoning(resp)
                    usage = _usage_to_dict(resp)
                    completed_m = monotonic_s()
                    completed_at = iso_now()
                    timing = {
                        "task_created_at": task_created_at,
                        "semaphore_acquired_at": semaphore_acquired_at,
                        "api_request_started_at": api_request_started_at,
                        "api_response_received_at": api_response_received_at,
                        "completed_at": completed_at,
                        "semaphore_wait_s": elapsed_s(task_started_m, semaphore_acquired_m),
                        "client_setup_s": elapsed_s(client_setup_start_m, api_request_started_m),
                        "api_elapsed_s": elapsed_s(api_request_started_m, api_response_received_m),
                        "postprocess_s": elapsed_s(api_response_received_m, completed_m),
                        "total_task_elapsed_s": elapsed_s(task_started_m, completed_m),
                    }

                    log_path = _write_request_log(request_id, {
                        'status': 'succeeded', 'model': self._model_name,
                        'base_url': self._base_url,
                        'stage': req.stage, 'turn': req.turn,
                        'stop_thinking': stop_thinking, 'question_id': req.question_id,
                        'internal_index': req.internal_index, 'prompt': req.prompt,
                        'metadata': req.metadata, 'content': content,
                        'content_preview': content[:500], 'reasoning': reasoning,
                        'perplexity': perplexity, 'usage': usage,
                        'created_at': task_created_at, 'completed_at': completed_at,
                        'timing': timing,
                    })
                    _write_llm_call_metric(os.environ.get("LLM_REQUEST_LOG_DIR"), {
                        "record_type": "llm_call",
                        "request_id": request_id,
                        "question_id": req.question_id,
                        "internal_index": req.internal_index,
                        "stage": req.stage,
                        "stage_artifact": _stage_artifact_name(req.stage),
                        "turn": req.turn,
                        "model": self._model_name,
                        "base_url": self._base_url,
                        "status": "succeeded",
                        "stop_thinking": stop_thinking,
                        "prompt_chars": len(req.prompt or ""),
                        "completion_chars": len(content or ""),
                        "usage": usage,
                        "timing": timing,
                        "request_log_path": log_path or "",
                        "error": "",
                    })
                    return BatchResponse(
                        request_id=request_id, question_id=req.question_id,
                        internal_index=req.internal_index, stage=req.stage, turn=req.turn,
                        ok=True, content=content, perplexity=perplexity,
                        reasoning=reasoning, usage=usage, request_log_path=log_path or '',
                        timing=timing,
                    )
                except Exception as exc:
                    completed_m = monotonic_s()
                    completed_at = iso_now()
                    err = f"{exc.__class__.__name__}: {exc}"
                    timing = {
                        "task_created_at": task_created_at,
                        "semaphore_acquired_at": semaphore_acquired_at,
                        "api_request_started_at": locals().get("api_request_started_at"),
                        "api_response_received_at": locals().get("api_response_received_at"),
                        "completed_at": completed_at,
                        "semaphore_wait_s": elapsed_s(task_started_m, semaphore_acquired_m),
                        "client_setup_s": elapsed_s(client_setup_start_m, locals().get("api_request_started_m", completed_m)),
                        "api_elapsed_s": (
                            elapsed_s(api_request_started_m, locals().get("api_response_received_m", completed_m))
                            if "api_request_started_m" in locals()
                            else 0.0
                        ),
                        "postprocess_s": 0.0,
                        "total_task_elapsed_s": elapsed_s(task_started_m, completed_m),
                    }
                    log_path = _write_request_log(request_id, {
                        'status': 'failed', 'model': self._model_name,
                        'base_url': self._base_url,
                        'stage': req.stage, 'turn': req.turn,
                        'stop_thinking': stop_thinking, 'question_id': req.question_id,
                        'internal_index': req.internal_index, 'prompt': req.prompt,
                        'metadata': req.metadata, 'error': err,
                        'created_at': task_created_at, 'completed_at': completed_at,
                        'timing': timing,
                    })
                    _write_llm_call_metric(os.environ.get("LLM_REQUEST_LOG_DIR"), {
                        "record_type": "llm_call",
                        "request_id": request_id,
                        "question_id": req.question_id,
                        "internal_index": req.internal_index,
                        "stage": req.stage,
                        "stage_artifact": _stage_artifact_name(req.stage),
                        "turn": req.turn,
                        "model": self._model_name,
                        "base_url": self._base_url,
                        "status": "failed",
                        "stop_thinking": stop_thinking,
                        "prompt_chars": len(req.prompt or ""),
                        "completion_chars": 0,
                        "usage": None,
                        "timing": timing,
                        "request_log_path": log_path or "",
                        "error": err,
                    })
                    return BatchResponse(
                        request_id=request_id, question_id=req.question_id,
                        internal_index=req.internal_index, stage=req.stage, turn=req.turn,
                        ok=False, content='', perplexity=0.0, error=err,
                        request_log_path=log_path or '', timing=timing,
                    )
                finally:
                    await client.close()

        tasks = [_call_one(req) for req in requests]
        return await asyncio.gather(*tasks)
