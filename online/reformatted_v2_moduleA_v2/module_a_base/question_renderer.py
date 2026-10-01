from __future__ import annotations

from typing import Any, Dict, List


def render_module_a_items(
    entries: List[Dict[str, Any]],
    *,
    feedback_render_mode: str,
) -> List[Dict[str, Any]]:
    """Pass the public Module A ambiguity entries directly to feedback generation."""
    if feedback_render_mode != "AmbiModel_direct":
        raise ValueError("The public runtime supports only AmbiModel_direct feedback rendering.")
    return list(entries)
