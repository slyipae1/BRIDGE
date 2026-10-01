from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ModuleARetrievalStagePayload:
    """Read and normalize a live module_a_retrieval stage payload."""

    def __init__(self, payload: Dict[str, Any], source_path: Path) -> None:
        self.payload = payload
        self.source_path = Path(source_path)
        self._questions_by_id = self._build_question_index(payload.get("questions", {}))

    @classmethod
    def from_file(cls, path: str | Path) -> "ModuleARetrievalStagePayload":
        source_path = Path(path)
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload=payload, source_path=source_path)

    def get_question_record(self, question_id: int) -> Optional[Dict[str, Any]]:
        return self._questions_by_id.get(int(question_id))

    def get_retrieval_payload(self, question_id: int) -> Optional[Dict[str, Any]]:
        record = self.get_question_record(question_id)
        if not record:
            return None
        return record.get("db_retrieval_sql_full")

    @staticmethod
    def _build_question_index(raw_questions: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        indexed: Dict[int, Dict[str, Any]] = {}
        for key, raw_record in (raw_questions or {}).items():
            if not isinstance(raw_record, dict):
                continue
            qid_value = raw_record.get("question_id", key)
            try:
                qid = int(qid_value)
            except (TypeError, ValueError):
                continue
            normalized = dict(raw_record)
            normalized["question_id"] = qid
            normalized.setdefault("db_id", "")
            indexed[qid] = normalized
        return indexed
