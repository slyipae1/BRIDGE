from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib import request

from auto_construct_graph.llm_client import is_private_base_url


DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_EMBEDDING_API_KEY = "EMPTY"
DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    api_key: str
    model: str


def resolve_embedding_config(env: Mapping[str, str] | None = None) -> EmbeddingConfig:
    values = os.environ if env is None else env
    return EmbeddingConfig(
        base_url=str(values.get("EMBEDDING_BASE_URL") or values.get("BASE_URL") or DEFAULT_EMBEDDING_BASE_URL).rstrip("/"),
        api_key=str(values.get("EMBEDDING_API_KEY") or values.get("API_KEY") or DEFAULT_EMBEDDING_API_KEY),
        model=str(values.get("EMBEDDING_MODEL_NAME") or DEFAULT_EMBEDDING_MODEL),
    )


def build_embeddings_payload(*, model: str, inputs: list[str]) -> dict[str, Any]:
    return {"model": model, "input": list(inputs)}


def extract_embeddings_from_response(response: dict[str, Any]) -> list[list[float]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise ValueError("embedding response has no data list")
    ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
    embeddings: list[list[float]] = []
    for item in ordered:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"embedding response item has no embedding vector: {item!r}")
        embeddings.append([float(value) for value in embedding])
    return embeddings


class OpenAICompatibleEmbeddingClient:
    def __init__(self, *, base_url: str, api_key: str, timeout: int = 120):
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.timeout = int(timeout)
        if is_private_base_url(self.base_url):
            self._opener = request.build_opener(request.ProxyHandler({}))
        else:
            self._opener = request.build_opener()

    @property
    def embeddings_url(self) -> str:
        return f"{self.base_url}/embeddings"

    def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.embeddings_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with self._opener.open(req, timeout=self.timeout) as response:
            response_body = response.read().decode("utf-8")
        return json.loads(response_body)

    def embed_texts(self, *, model: str, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), int(batch_size)):
            batch = texts[start:start + int(batch_size)]
            response = self.embeddings(build_embeddings_payload(model=model, inputs=batch))
            batch_embeddings = extract_embeddings_from_response(response)
            if len(batch_embeddings) != len(batch):
                raise ValueError(f"embedding response length mismatch: {len(batch_embeddings)} != {len(batch)}")
            embeddings.extend(batch_embeddings)
        return embeddings
