from __future__ import annotations

import json
from typing import Any


DEFAULT_MODULE_A_DETECTION_PROMPT_MODE = "retrieval_elements"
MODULE_A_DETECTION_PROMPT_MODES = ("retrieval_elements",)

def build_module_a_detection_prompt(
    *,
    question: str,
    evidence: str,
    current_sql: str,
    schema_ddl: str,
    formatted_dbelement_list: str | list[dict[str, Any]],
    column_group_version: str | None = None,
    schema_section_title: str = "Database Schema (enriched)",
    target_column: str | None = None,
    require_retrieved_coverage: bool = False,
) -> str:
    """Build the prompt for the Module A ambiguity-detection LLM call.

    The LLM receives the user question, enriched DDL, current SQL (marked as
    rejected), and the retrieved DB element list, and returns a JSON list of
    detected ambiguities. ``require_retrieved_coverage`` is the prompt-only
    no-prune ablation: it changes the coverage instruction but not the prompt
    shape or output schema.
    """
    del column_group_version
    relation_guidance = ""
    target_focus_section = _target_focus_section(target_column)
    coverage_instruction = _detection_coverage_instruction(require_retrieved_coverage)
    return """You are analyzing why a previously generated SQL query was incorrect.

Based on the user question, database schema, and retrieved candidate elements, identify any ambiguities that may have led to the wrong SQL.

Focus on:
- AmbiColumn: Which column(s) should be used for a given condition or output?
- AmbiValue: Which specific value(s) should be used in a filter condition or others?

For each ambiguity, provide the options that the SQL author could have chosen.
{coverage_instruction}

Return only a JSON list in this format:
[
  {{
    "ambiguity_type": "AmbiColumn or AmbiValue",
    "ambiguity_context": "brief description of what is ambiguous",
    "choices": [
      {{
        "nl_description": "human-readable description of this choice",
        "sql_snippet": "the corresponding SQL fragment"
      }}
    ]
  }}
]

## User Question
{question}

## Evidence
{evidence}

## Current SQL (rejected by user)
```sql
{current_sql}
```

{target_focus_section}
## {schema_section_title}
{schema_ddl}

## Retrieved DB Element List
{relation_guidance}
{formatted_dbelement_list}
""".format(
        question=question,
        evidence=evidence,
        current_sql=current_sql,
        target_focus_section=target_focus_section,
        schema_section_title=schema_section_title,
        schema_ddl=schema_ddl,
        relation_guidance=relation_guidance,
        formatted_dbelement_list=_json_blob(formatted_dbelement_list),
        coverage_instruction=coverage_instruction,
    )


def _detection_coverage_instruction(require_retrieved_coverage: bool) -> str:
    if require_retrieved_coverage:
        return (
            "You are expected to cover all retrieved potentially ambiguous elements in the list of ambiguity entries."
        )
    return (
        "You do NOT need to convert every retrieved element into an ambiguity — only the ones you judge as genuinely ambiguous given the question context."
    )


def _target_focus_section(target_column: str | None) -> str:
    target_column = str(target_column or "").strip()
    if not target_column:
        return ""
    return f"""## Current Targeted Column
{target_column}

This detection call is focused on the targeted column above, which was extracted from the current SQL prediction. Other columns in the predicted SQL will be analyzed in separate detection calls. Only report ambiguities that involve this targeted column. You may include other columns only when the ambiguity directly involves the targeted column together with those columns.
"""


def build_module_a_target_column_detection_prompt(
    *,
    question: str,
    evidence: str,
    current_sql: str,
    schema_ddl: str,
    target_column: str,
    require_retrieved_coverage: bool = False,
) -> str:
    """Build the no-retrieval COLUMN-slice detection prompt."""
    return """You are analyzing why a previously generated SQL query was incorrect.

Based on the user question, database schema, and target column,
identify any ambiguities that may have led to the wrong SQL.

Focus on:
- AmbiColumn: Which column should be used for a given condition or output?
- AmbiValue: Which specific value should be used in a filter condition?

For each ambiguity, provide the options that the SQL author could have chosen.
{coverage_instruction}
This detection call is focused on the targeted column, which was extracted from the current SQL prediction. Other columns in the predicted SQL will be analyzed in separate detection calls. Only report ambiguities that involve this targeted column. You may include other columns only when the ambiguity directly involves the targeted column together with those columns

Return only a JSON list in this format:
[
  {{
    "ambiguity_type": "AmbiColumn or AmbiValue",
    "ambiguity_context": "brief description of what is ambiguous",
    "choices": [
      {{
        "nl_description": "human-readable description of this choice",
        "sql_snippet": "the corresponding minimal SQL fragment"
      }}
    ]
  }}
]

## User Question
{question}

## Evidence
{evidence}

## Current SQL (rejected by user)
```sql
{current_sql}
```

## Database Schema
{schema_ddl}

## Target Column
{target_column}
""".format(
        question=question,
        evidence=evidence,
        current_sql=current_sql,
        schema_ddl=schema_ddl,
        target_column=target_column,
        coverage_instruction=(
            _detection_coverage_instruction(True)
            if require_retrieved_coverage
            else (
                "You do NOT need to convert every possible schema element into an ambiguity - only\n"
                "the ones you judge as genuinely ambiguous given the question context."
            )
        ),
    )


def build_module_a_feedback_prompt(
    *,
    question: str,
    evidence: str,
    gold_sql: str,
    schema_ddl: str,
    rendered_items: list[dict[str, Any]],
    allow_multi_select: bool = False,
) -> str:
    """Build the prompt for the Module A feedback LLM call.

    The LLM simulates a user who knows the ground-truth SQL and selects the
    correct interpretation for each ambiguity entry.

    The ``reason`` field appears BEFORE ``selected_interpretation_index`` so
    that the LLM thinks (generates text) before committing to a choice.
    """
    if allow_multi_select:
        return """You are simulating a user who knows the correct SQL answer.

For each ambiguity item below, choose all options that exactly match the gold SQL.
The options are intentionally raw database elements; do not add schema facts.
Return only one JSON list. Keep the same order as input.

Return format:
[
  {{
    "entry_id": "string",
    "reason": "short explanation, think before you decide",
    "selected_interpretation_indices": [0],
    "correct_choice": null
  }}
]

If none of the provided options exactly match the gold SQL, set
`selected_interpretation_indices` to [] and provide:
{{
  "nl_description": "...",
  "db_element_list": [["table", "column", "value-or-null"]],
  "sql_snippet": "..."
}}
in `correct_choice`. If the correct answer is that none of the options should
be selected and no replacement is needed, use [] and set `correct_choice` to null.

## User Question
{question}

## Evidence
{evidence}

## Ground Truth SQL
```sql
{gold_sql}
```

## Module A Items
{rendered_items_json}
""".format(
            question=question,
            evidence=evidence,
            gold_sql=gold_sql,
            rendered_items_json=_json_blob(rendered_items),
        )

    return """You are simulating a user who knows the correct SQL answer.

For each ambiguity item below, choose the option that exactly matches the gold SQL.
Return only one JSON list. Keep the same order as input.

Return format:
[
  {{
    "entry_id": "string",
    "reason": "short explanation, think before you decide",
    "selected_interpretation_index": 0,
    "correct_choice": null
  }}
]

If none of the provided options exactly match the gold SQL, set
`selected_interpretation_index` to -1 and provide:
{{
  "nl_description": "...",
  "db_element_list": [["table", "column", "value-or-null"]],
  "sql_snippet": "..."
}}
in `correct_choice`.

## User Question
{question}

## Evidence
{evidence}

## Ground Truth SQL
```sql
{gold_sql}
```

## DDL
{schema_ddl}

## Module A Items
{rendered_items_json}
""".format(
        question=question,
        evidence=evidence,
        gold_sql=gold_sql,
        schema_ddl=schema_ddl,
        rendered_items_json=_json_blob(rendered_items),
    )


def parse_json_list_response(payload: str) -> list[dict[str, Any]]:
    cleaned = payload.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fence(cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # MOD: vLLM occasionally returns raw control characters inside JSON strings.
        # Python's strict=False preserves the response text while accepting those
        # characters; structurally broken JSON still raises and is handled by the
        # caller as an unrecoverable malformed response.
        parsed = json.loads(cleaned, strict=False)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list response")
    return parsed


def _json_blob(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _strip_code_fence(value: str) -> str:
    lines = value.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
