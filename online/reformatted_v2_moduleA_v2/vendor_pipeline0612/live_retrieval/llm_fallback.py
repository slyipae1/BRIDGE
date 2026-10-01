from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple

def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def _build_chat_payload(prompt: str) -> dict:
    _, _, model_name = _chat_config()
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    enable_thinking = _resolve_enable_thinking()
    if enable_thinking is not None:
        provider_profile = str(os.getenv("SYSTEM_LLM_PROVIDER_PROFILE", "generic")).strip().casefold()
        if provider_profile == "dashscope_batch":
            payload["enable_thinking"] = enable_thinking
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return payload


def _resolve_enable_thinking() -> bool:
    """Parser fallback always disables thinking to keep JSON responses strict."""
    # MOD: Parser fallback is a JSON extraction step. Letting global pipeline
    # thinking flags override this made Qwen return non-JSON reasoning content.
    return False


def _chat_config() -> tuple[str, str, str]:
    base_url = (
        os.getenv("MODULE_A_LLM_FALLBACK_BASE_URL")
        or os.getenv("MODULE_A_LIVE_API_BASE")
        or os.getenv("CHAT_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://127.0.0.1:3005/v1"
    )
    api_key = (
        os.getenv("MODULE_A_LLM_FALLBACK_API_KEY")
        or os.getenv("MODULE_A_LIVE_API_KEY")
        or os.getenv("CHAT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "local-token"
    )
    model_name = (
        os.getenv("MODULE_A_LLM_FALLBACK_MODEL_NAME")
        or os.getenv("MODULE_A_LIVE_MODEL_NAME")
        or os.getenv("CHAT_MODEL_NAME")
        or os.getenv("OPENAI_MODEL_NAME")
        or "Qwen3.5-27B"
    )
    return base_url, api_key, model_name


def _extract_message_text(data: dict) -> str:
    message = data.get("choices", [{}])[0].get("message", {}) or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        reasoning = message.get(key)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
    return ""


def _is_private_endpoint(base_url: str) -> bool:
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1"} or hostname.endswith(".local"):
        return True
    return hostname.startswith("10.") or hostname.startswith("192.168.") or any(
        hostname.startswith(f"172.{i}.") for i in range(16, 32)
    )


def _urlopen_private_safe(request: urllib.request.Request, base_url: str):
    if _is_private_endpoint(base_url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=_api_timeout_seconds())
    return urllib.request.urlopen(request, timeout=_api_timeout_seconds())


def _api_timeout_seconds() -> float:
    raw = os.getenv("SYSTEM_LLM_API_TIMEOUT_SECONDS", "180")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid SYSTEM_LLM_API_TIMEOUT_SECONDS: {raw!r}") from exc
    if timeout <= 0:
        raise ValueError("SYSTEM_LLM_API_TIMEOUT_SECONDS must be positive")
    return timeout


def _chat_completion(prompt: str) -> str:
    base_url, api_key, _ = _chat_config()
    endpoint = base_url.rstrip("/") + "/chat/completions"

    payload = _build_chat_payload(prompt)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with _urlopen_private_safe(request, base_url) as response:
        data = json.loads(response.read().decode("utf-8"))
    return _extract_message_text(data)


def extract_elements_with_llm(
    sql: str,
    db_id: str,
) -> Optional[Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]]:
    """Optional OpenAI-compatible parser fallback, matching pipeline_0612 output shape."""
    del db_id
    prompt = (
        "You are a SQL expert. Given the SQL query below, identify all database "
        "table columns and literal values used in the query.\n\n"
        f"SQL: {sql}\n\n"
        "Output ONLY valid JSON with this exact structure:\n"
        "```json\n"
        "{\n"
        '  "columns": [{"table": "table_name", "column": "column_name"}, ...],\n'
        '  "values": [{"table": "table_name", "column": "column_name", "value": "literal_value"}, ...]\n'
        "}\n"
        "```\n\n"
        "Rules:\n"
        "- Infer table names from column references if not explicit in the SQL.\n"
        "- Include ALL columns and literal values appearing in the SQL.\n"
        "- Use actual table.column names from the database schema, not aliases.\n"
        "- If a table name cannot be determined, use 'unknown' as the table name."
    )
    try:
        start = time.perf_counter()
        content = _chat_completion(prompt)
        logging.info("LLM fallback extraction took %.2fs", time.perf_counter() - start)
        data = json.loads(_strip_json_fence(content))
    except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError, KeyError) as exc:
        logging.warning("LLM fallback extraction failed: %s", exc)
        return None

    columns_dict: Dict[str, List[str]] = {}
    for col_item in data.get("columns", []):
        table_name = str(col_item.get("table", "unknown") or "unknown").strip()
        column_name = str(col_item.get("column", "") or "").strip()
        if column_name:
            columns_dict.setdefault(table_name, [])
            if column_name not in columns_dict[table_name]:
                columns_dict[table_name].append(column_name)

    literals: Dict[str, Dict[str, List[str]]] = {}
    for val_item in data.get("values", []):
        table_name = str(val_item.get("table", "unknown") or "unknown").strip()
        column_name = str(val_item.get("column", "") or "").strip()
        value = str(val_item.get("value", "") or "").strip()
        if column_name and value:
            literals.setdefault(table_name, {}).setdefault(column_name, [])
            if value not in literals[table_name][column_name]:
                literals[table_name][column_name].append(value)

    return columns_dict, literals
