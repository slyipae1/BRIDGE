from __future__ import annotations


RETRIEVAL_ABLATION_DISABLE_CHANNELS = ("none", "column")
DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL = "none"


def validate_retrieval_ablation_disable_channel(channel: str | None) -> str:
    normalized = str(channel or DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL).strip().lower()
    if normalized not in RETRIEVAL_ABLATION_DISABLE_CHANNELS:
        raise ValueError(
            "retrieval ablation disable channel must be one of "
            f"{RETRIEVAL_ABLATION_DISABLE_CHANNELS}, got {channel!r}"
        )
    return normalized


def retrieval_channel_enabled(*, disable_channel: str | None, channel: str) -> bool:
    normalized_disable_channel = validate_retrieval_ablation_disable_channel(disable_channel)
    normalized_channel = str(channel).strip().lower()
    if normalized_channel not in {"column", "value"}:
        raise ValueError(f"Unsupported retrieval channel: {channel!r}")
    return normalized_disable_channel != normalized_channel
