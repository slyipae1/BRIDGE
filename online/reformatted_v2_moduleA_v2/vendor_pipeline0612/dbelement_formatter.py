"""Prompt rendering for the public BRIDGE Module A contract."""

from __future__ import annotations

from typing import Any


COLUMN_GROUP_PROMPT_FORMATS = ("manual_target_reason",)
MANUAL_TARGET_REASON_PROMPT_FORMATS = COLUMN_GROUP_PROMPT_FORMATS


def resolve_column_group_prompt_format(
    column_group_prompt_format: str,
    column_group_version: str | None = None,
) -> str:
    del column_group_version
    if column_group_prompt_format != "manual_target_reason":
        raise ValueError("The public runtime supports only manual_target_reason rendering.")
    return column_group_prompt_format


def _anchor_for_entry(entry: dict[str, Any]) -> str:
    anchor = entry.get("anchor_column")
    if anchor:
        return str(anchor)
    for option in entry.get("options", []) or []:
        if option.get("anchor_column"):
            return str(option["anchor_column"])
    return str(entry.get("entity", "") or "column_group")


def _format_value_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": entry.get("entity", ""),
        "type": entry.get("type", ""),
        "options": [
            option.get("element", "")
            for option in entry.get("options", []) or []
            if option.get("element", "")
        ],
    }


def format_dbelement_options(
    dbelement_options: list[dict[str, Any]],
    *,
    include_column_relation_metadata: bool = True,
    column_group_prompt_format: str = "manual_target_reason",
    column_group_version: str | None = None,
    column_descriptive_names: dict[str, str] | None = None,
    column_descrip_mode: str = "empty",
) -> list[dict[str, Any]]:
    """Render graph candidates as target/options/reason entries for Module A."""
    del include_column_relation_metadata, column_descriptive_names
    if column_descrip_mode != "empty":
        raise ValueError("The public runtime renders columns as table.column only.")
    resolve_column_group_prompt_format(column_group_prompt_format, column_group_version)

    rendered: list[dict[str, Any]] = []
    for entry in dbelement_options or []:
        if entry.get("type", "") != "COLUMN":
            rendered.append(_format_value_entry(entry))
            continue

        anchor = _anchor_for_entry(entry)
        anchor_key = anchor.casefold()
        fallback_reason = str(
            entry.get("ambiguity_reason_concise_label") or entry.get("entity") or ""
        ).strip()
        options: list[dict[str, str]] = []
        seen: set[str] = set()
        for option in entry.get("options", []) or []:
            column = str(option.get("element", "")).strip()
            key = column.casefold()
            if not column or key == anchor_key or key in seen:
                continue
            seen.add(key)
            options.append(
                {
                    "column": column,
                    "ambiguity_reason": str(option.get("ambiguity_reason") or fallback_reason),
                }
            )
        rendered.append({"target": anchor, "type": "COLUMN", "options": options})
    return rendered
