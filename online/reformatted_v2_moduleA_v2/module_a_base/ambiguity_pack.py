from __future__ import annotations

from typing import Any, Dict, List


def auto_fill_entries(
    *,
    llm_output: List[Dict[str, Any]],
    question_id: int,
    db_id: str,
) -> List[Dict[str, Any]]:
    """Take raw LLM detection output and normalize it into full Module A entries.

    The LLM output (from the detection prompt) contains ambiguity_type,
    ambiguity_context, and choices without entry_id or choice_index.
    This function auto-fills those fields.

    LLM output structure (input):
        [
          {
            "ambiguity_type": "AmbiValue",
            "ambiguity_context": "locally funded",
            "choices": [
              {"nl_description": "Use ...", "sql_snippet": "..."},
              ...
            ]
          }
        ]

    Normalized structure (output):
        [
          {
            "entry_id": "moda_0028_00",
            "question_id": 28,
            "db_id": "california_schools",
            "ambiguity_type": "AmbiValue",
            "ambiguity_context": "locally funded",
            "source": "db_retrieval_sql_full_llm",
            "choices": [
              {"choice_index": 0, "nl_description": "Use ...", "sql_snippet": "..."},
              ...
            ]
          }
        ]
    """
    entries: List[Dict[str, Any]] = []

    for entry_index, raw_entry in enumerate(llm_output or []):
        ambiguity_type = str(raw_entry.get("ambiguity_type", "AmbiColumn") or "").strip()
        ambiguity_context = str(raw_entry.get("ambiguity_context", "") or "").strip()
        raw_choices = raw_entry.get("choices", []) or []

        if not ambiguity_context or not raw_choices:
            continue

        packed_choices = []
        for choice_index, choice in enumerate(raw_choices):
            if not isinstance(choice, dict):
                continue
            nl_description = str(choice.get("nl_description", "") or "").strip()
            sql_snippet = str(choice.get("sql_snippet", "") or "").strip()
            if not nl_description and not sql_snippet:
                continue

            packed: Dict[str, Any] = {
                "choice_index": choice_index,
                "nl_description": nl_description,
                "sql_snippet": sql_snippet,
            }
            if "db_element_list" in choice:
                packed["db_element_list"] = choice["db_element_list"]
            packed_choices.append(packed)

        if not packed_choices:
            continue

        entries.append(
            {
                "entry_id": f"moda_{int(question_id):04d}_{entry_index:02d}",
                "question_id": int(question_id),
                "db_id": db_id,
                "ambiguity_type": ambiguity_type,
                "ambiguity_context": ambiguity_context,
                "source": str(raw_entry.get("source") or "db_retrieval_sql_full_llm"),
                "choices": packed_choices,
            }
        )

    return entries
