#!/usr/bin/env python3
"""Evaluate text-to-SQL predictions with BRIDGE's robust EX_col metric.

The evaluator accepts either ``{question_id: sql_or_sql_list}`` or a list of
``{"id": question_id, "pred_sqls": [sql, ...]}`` records. It evaluates the
first SQL per question, matching the greedy evaluation used by BRIDGE.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sqlite3
from pathlib import Path
from typing import Any

import sqlglot
from func_timeout import FunctionTimedOut, func_timeout


EVAL_NAME = "robustEX_col"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Prediction JSON.")
    parser.add_argument("--gold", type=Path, required=True, help="BIRD gold JSON.")
    parser.add_argument("--db-path", type=Path, required=True, help="Directory containing {db_id}/{db_id}.sqlite.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the prediction file directory.")
    parser.add_argument("--num-cpus", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600, help="Per-question SQL evaluation timeout in seconds.")
    return parser.parse_args()


def _prediction_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = []
        for question_id, sqls in payload.items():
            rows.append({"id": int(question_id), "pred_sqls": sqls if isinstance(sqls, list) else [sqls]})
        return rows
    if not isinstance(payload, list):
        raise ValueError("--pred must be a question_id mapping or a list of prediction records")
    rows = []
    for row in payload:
        if not isinstance(row, dict) or "id" not in row or not isinstance(row.get("pred_sqls"), list):
            raise ValueError("prediction records must contain integer id and list pred_sqls")
        rows.append({"id": int(row["id"]), "pred_sqls": row["pred_sqls"]})
    return rows


def _compare_sql(db_file: str, ground_truth: str | list[str], pred_sql: str) -> tuple[int, str, str]:
    """Apply the original robust EX_col greedy comparison to one prediction."""
    try:
        normalized_prediction = sqlglot.parse_one(pred_sql, read="sqlite").sql(dialect="sqlite")
    except Exception as exc:
        return 0, str(pred_sql), f"prediction parse error: {exc}"

    connection = sqlite3.connect(db_file)
    try:
        cursor = connection.cursor()
        connection.execute("BEGIN TRANSACTION;")
        cursor.execute(normalized_prediction)
        predicted_rows = cursor.fetchall()
        gold_sqls = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        for gold_sql in gold_sqls:
            try:
                cursor.execute(gold_sql)
                gold_rows = cursor.fetchall()
            except Exception:
                continue
            if set(predicted_rows) == set(gold_rows):
                return 1, normalized_prediction, ""
            if not predicted_rows or not gold_rows:
                continue
            if len(predicted_rows[0]) != len(gold_rows[0]):
                continue
            for permutation in itertools.permutations(range(len(predicted_rows[0]))):
                reordered_prediction = [tuple(row[index] for index in permutation) for row in predicted_rows]
                if set(reordered_prediction) == set(gold_rows):
                    return 1, normalized_prediction, ""
        return 0, normalized_prediction, ""
    except Exception as exc:
        return 0, normalized_prediction, f"prediction execution error: {exc}"
    finally:
        try:
            connection.rollback()
        except Exception:
            pass
        connection.close()


def _evaluate_one(task: tuple[int, str, str | list[str], str, int]) -> dict[str, Any]:
    question_id, db_file, ground_truth, pred_sql, timeout = task
    try:
        correctness, normalized_prediction, remark = func_timeout(
            timeout, _compare_sql, args=(db_file, ground_truth, pred_sql),
        )
    except FunctionTimedOut:
        correctness, normalized_prediction, remark = 0, pred_sql, "Time Out"
    except Exception as exc:
        correctness, normalized_prediction, remark = 0, pred_sql, str(exc)
    return {
        "question_id": question_id,
        "pred_sql": normalized_prediction,
        "correctness": correctness,
        "remark": remark,
    }


def main() -> None:
    args = parse_args()
    if args.num_cpus < 1 or args.timeout < 1:
        raise ValueError("--num-cpus and --timeout must be positive")
    gold_rows = json.loads(args.gold.read_text(encoding="utf-8"))
    if not isinstance(gold_rows, list):
        raise ValueError("--gold must contain a BIRD JSON list")
    predictions = _prediction_rows(json.loads(args.pred.read_text(encoding="utf-8")))
    prediction_by_id = {row["id"]: row for row in predictions}
    gold_by_id = {int(row["question_id"]): row for row in gold_rows}
    common_ids = sorted(set(prediction_by_id) & set(gold_by_id))
    print(f"Common question_ids: {len(common_ids)}")
    if not common_ids:
        raise ValueError("no common question_ids between --pred and --gold")

    tasks = []
    for question_id in common_ids:
        prediction_sqls = prediction_by_id[question_id]["pred_sqls"]
        if not prediction_sqls:
            prediction_sqls = ["Error SQL"]
        gold = gold_by_id[question_id]
        db_file = args.db_path / str(gold["db_id"]) / f"{gold['db_id']}.sqlite"
        if not db_file.is_file():
            raise FileNotFoundError(f"database not found: {db_file}")
        tasks.append((question_id, str(db_file), gold["SQL"], str(prediction_sqls[0]), args.timeout))

    with mp.Pool(processes=min(args.num_cpus, len(tasks))) as pool:
        results = list(pool.imap(_evaluate_one, tasks))
    results.sort(key=lambda row: row["question_id"])
    correct = sum(int(row["correctness"]) for row in results)
    print(f"robust EX_col accuracy: {correct}/{len(results)} = {correct / len(results):.6f}")

    output_dir = args.output_dir or args.pred.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.pred.stem}_{args.gold.stem}_evalRes_gd_{EVAL_NAME}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
