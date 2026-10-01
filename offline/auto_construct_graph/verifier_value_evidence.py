"""Bounded, reproducible literal-value evidence for active all-pairs verification."""

from __future__ import annotations

import hashlib
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .channels.value_collision_channel import is_missing_normalized_value, normalize_value_for_collision
from .schema_profile import quote_identifier
from .types import make_pair_key, split_column_ref


MODEL_VALUE_SAMPLE_LIMIT = 5
MODEL_VALUE_MAX_CHARS = 25


def truncate_verifier_value(value: object, *, max_chars: int = MODEL_VALUE_MAX_CHARS) -> str:
    """Render one literal value without letting a long value dominate a prompt."""
    text = str(value or "").strip()
    if len(text) <= int(max_chars):
        return text
    return f"{text[:int(max_chars)]}...({len(text) - int(max_chars)} more chars)"


def _seeded_cycle_samples(
    values: tuple[str, ...],
    *,
    excluded_domain: dict[str, str],
    seed: str,
    limit: int = MODEL_VALUE_SAMPLE_LIMIT,
) -> list[str]:
    """Sample without constructing ``values - overlap`` for high-cardinality IDs."""
    if not values:
        return []
    rng = random.Random(int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16))
    length = len(values)
    start = rng.randrange(length)
    step = rng.randrange(1, length) if length > 1 else 1
    while math.gcd(step, length) != 1:
        step = (step + 1) % length or 1
    selected: list[str] = []
    rendered: set[str] = set()
    for offset in range(length):
        normalized = values[(start + offset * step) % length]
        if normalized in excluded_domain:
            continue
        display = truncate_verifier_value(normalized)
        if display in rendered:
            continue
        rendered.add(display)
        selected.append(normalized)
        if len(selected) >= int(limit):
            break
    return selected


@dataclass(frozen=True)
class VerifierValueEvidenceIndex:
    representatives_by_column: dict[str, dict[str, str]]
    normalized_values_by_column: dict[str, tuple[str, ...]]


def build_verifier_value_domains(sqlite_path: Path, schema_profile: dict[str, Any]) -> VerifierValueEvidenceIndex:
    """Return normalized value -> stable original-text representative per column.

    The domain stays in memory. A lexicographically minimal original spelling is
    selected for each normalized value so SQLite row order cannot affect a
    later prompt or a resumed run.
    """
    domains: dict[str, dict[str, str]] = {}
    with sqlite3.connect(Path(sqlite_path)) as conn:
        for column_ref in sorted(schema_profile.get("columns") or {}):
            table_name, column_name = split_column_ref(column_ref)
            table_sql = quote_identifier(table_name)
            column_sql = quote_identifier(column_name)
            representatives: dict[str, str] = {}
            for row in conn.execute(f"SELECT {column_sql} FROM {table_sql} WHERE {column_sql} IS NOT NULL"):
                display_value = str(row[0]).strip()
                normalized = normalize_value_for_collision(display_value)
                if not display_value or is_missing_normalized_value(normalized):
                    continue
                previous = representatives.get(normalized)
                if previous is None or display_value < previous:
                    representatives[normalized] = display_value
            domains[column_ref] = representatives
    return VerifierValueEvidenceIndex(
        representatives_by_column=domains,
        normalized_values_by_column={column_ref: tuple(sorted(values)) for column_ref, values in domains.items()},
    )


def _overlap_count_and_samples(
    *,
    left_domain: dict[str, str],
    right_domain: dict[str, str],
    seed: str,
) -> tuple[int, list[str]]:
    small, large = (left_domain, right_domain) if len(left_domain) <= len(right_domain) else (right_domain, left_domain)
    count = 0
    retained: list[tuple[int, str]] = []
    for normalized in small:
        if normalized not in large:
            continue
        count += 1
        score = int(hashlib.sha256(f"{seed}|{normalized}".encode("utf-8")).hexdigest(), 16)
        if len(retained) < MODEL_VALUE_SAMPLE_LIMIT:
            retained.append((score, normalized))
            continue
        largest_index = max(range(len(retained)), key=lambda index: retained[index][0])
        if score < retained[largest_index][0]:
            retained[largest_index] = (score, normalized)
    return count, [normalized for _, normalized in sorted(retained)]


def build_pair_verifier_value_evidence(
    *,
    db_id: str,
    left_ref: str,
    right_ref: str,
    value_domains: VerifierValueEvidenceIndex,
) -> dict[str, Any]:
    """Create the sole model-visible value evidence for one unordered pair."""
    left_ref, right_ref = make_pair_key(left_ref, right_ref)
    left_domain = value_domains.representatives_by_column.get(left_ref) or {}
    right_domain = value_domains.representatives_by_column.get(right_ref) or {}
    left_values = value_domains.normalized_values_by_column.get(left_ref) or ()
    right_values = value_domains.normalized_values_by_column.get(right_ref) or ()
    pair_seed = f"{db_id}|{left_ref}|{right_ref}"
    overlap_count, overlap_values = _overlap_count_and_samples(
        left_domain=left_domain, right_domain=right_domain, seed=f"{pair_seed}|overlap"
    )
    left_sample_values = _seeded_cycle_samples(
        left_values, excluded_domain=right_domain, seed=f"{pair_seed}|{left_ref}|endpoint"
    )
    right_sample_values = _seeded_cycle_samples(
        right_values, excluded_domain=left_domain, seed=f"{pair_seed}|{right_ref}|endpoint"
    )
    return {
        "column_sample_values": {
            left_ref: [truncate_verifier_value(left_domain[value]) for value in left_sample_values],
            right_ref: [truncate_verifier_value(right_domain[value]) for value in right_sample_values],
        },
        "value_overlapping_stat": {
            "unique_overlap_count": overlap_count,
            "samples": [truncate_verifier_value(min(left_domain[value], right_domain[value])) for value in overlap_values],
        },
    }
