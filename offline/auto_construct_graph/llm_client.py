from __future__ import annotations

import json
from typing import Any
from urllib import request
from urllib.parse import urlparse


class OpenAIResponseDecodeError(ValueError):
    """An HTTP response body that could not be decoded as OpenAI-compatible JSON."""

    def __init__(self, response_text: str, original_error: json.JSONDecodeError) -> None:
        super().__init__(f"invalid JSON response body: {original_error}")
        self.response_text = response_text
        self.original_error = original_error


def is_private_base_url(base_url: str) -> bool:
    host = (urlparse(str(base_url)).hostname or "").casefold()
    if host in {"localhost", "127.0.0.1"} or host.endswith(".local"):
        return True
    parts = host.split(".")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        return False
    first, second = int(parts[0]), int(parts[1])
    return first == 10 or first == 192 and second == 168 or first == 172 and 16 <= second <= 31


def build_chat_request_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    disable_thinking: bool,
    thinking_control: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        if thinking_control == "local_qwen":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif thinking_control == "dashscope":
            payload["enable_thinking"] = False
        elif thinking_control == "both":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["enable_thinking"] = False
        elif thinking_control == "none":
            pass
        else:
            raise ValueError(f"unsupported thinking_control: {thinking_control}")
    return payload


def extract_choice_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("chat response has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("chat response first choice has no message")
    return message


class OpenAICompatibleChatClient:
    def __init__(self, *, base_url: str, api_key: str, timeout: int = 120):
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key or "local-token")
        self.timeout = int(timeout)
        if is_private_base_url(self.base_url):
            self._opener = request.build_opener(request.ProxyHandler({}))
        else:
            self._opener = request.build_opener()

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.chat_completions_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with self._opener.open(req, timeout=self.timeout) as response:
            response_body = response.read().decode("utf-8")
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise OpenAIResponseDecodeError(response_body, exc) from exc
