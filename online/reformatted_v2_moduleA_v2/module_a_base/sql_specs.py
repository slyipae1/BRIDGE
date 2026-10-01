from __future__ import annotations

from typing import Any


def build_module_a_specs(
    *,
    entries: list[dict[str, Any]],
    rendered_items: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build question_text / user_text pairs from entries and LLM decisions.

    The public ``AmbiModel_direct`` feedback route uses the following resolution
    logic:

    - ``question_text`` = ``{ambiguity_context} ({ambiguity_type})``
    - ``user_text`` = ``{nl_description} (`{sql_snippet}`)``
    """
    entry_by_id = {entry.get("entry_id", ""): entry for entry in entries}
    specs: list[dict[str, str]] = []

    for decision in decisions:
        entry_id = str(decision.get("entry_id", "") or "")
        entry = entry_by_id.get(entry_id, {})
        if not entry:
            continue

        question_text = _resolve_question_text(entry)
        user_text = _resolve_user_text(entry, decision)
        if not question_text or not user_text:
            continue

        specs.append(
            {
                "entry_id": entry_id,
                "question_text": question_text,
                "user_text": user_text,
            }
        )

    return specs


def _resolve_question_text(entry: dict[str, Any]) -> str:
    ambiguity_context = str(entry.get("ambiguity_context", "") or "").strip()
    ambiguity_type = str(entry.get("ambiguity_type", "") or "").strip()
    if ambiguity_context and ambiguity_type:
        return f"{ambiguity_context} ({ambiguity_type})"
    return ""


def _resolve_user_text(
    entry: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    if "selected_interpretation_indices" in decision:
        selected_text = _resolve_multi_selected_user_text(
            entry,
            decision.get("selected_interpretation_indices"),
        )
        if selected_text:
            return selected_text
        correct_choice = decision.get("correct_choice")
        if isinstance(correct_choice, dict):
            return _format_choice_text(correct_choice)
        return ""

    selected_index = _resolve_selected_index(entry, decision.get("selected_interpretation_index"))
    if selected_index >= 0:
        return _resolve_selected_user_text(entry, selected_index)
    correct_choice = decision.get("correct_choice")
    if not isinstance(correct_choice, dict):
        return ""
    return _format_choice_text(correct_choice)


def _resolve_selected_index(entry: dict[str, Any], raw_index: Any) -> int:
    """Return a usable choice index, or the protocol's ``-1`` fallback.

    Feedback responses occasionally contain ``null`` or an index outside the
    supplied choice list. Both mean that no supplied option can safely be
    selected, so they follow the prompt's ``-1`` / ``correct_choice`` branch
    rather than aborting feedback generation for every question in the batch.
    """
    try:
        selected_index = int(raw_index)
    except (TypeError, ValueError):
        return -1
    if selected_index < 0:
        return -1
    if _resolve_selected_user_text(entry, selected_index):
        return selected_index
    return -1


def _resolve_multi_selected_user_text(
    entry: dict[str, Any],
    selected_indices: Any,
) -> str:
    if not isinstance(selected_indices, list):
        return ""
    texts: list[str] = []
    for raw_index in selected_indices:
        try:
            selected_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if selected_index < 0:
            continue
        selected_text = _resolve_selected_user_text(entry, selected_index)
        if selected_text:
            texts.append(selected_text)
    return "; ".join(texts)


def _resolve_selected_user_text(
    entry: dict[str, Any],
    selected_index: int,
) -> str:
    for choice in entry.get("choices", []) or []:
        if int(choice.get("choice_index", -1)) == selected_index:
            return _format_choice_text(choice)
    return ""


def _format_choice_text(choice: dict[str, Any]) -> str:
    nl_description = str(choice.get("nl_description", "") or "").strip()
    sql_snippet = str(choice.get("sql_snippet", "") or "").strip()
    if not nl_description:
        return ""
    if not sql_snippet:
        return nl_description
    return f"{nl_description} (`{sql_snippet}`)"
