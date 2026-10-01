"""
Utility functions: SQL cleaning, DB schema, evaluation, prompt building.
"""
import os, sqlite3, traceback, json, sys
from multiprocessing import Process, Queue

try:
    from .vendor_sphinteract import fewshot_utils
    from .vendor_sphinteract import query_module
    from .vendor_sphinteract.prompts import (
        fewshot_prefix, feedback_prefix_v1, cq_prefix_v1,
        selfdebug_few_shot, sql_generation_selfdebug,
        sql_generation_v2, feedback_v2, SRA, SRA_ES, fix_invalid_v1,
    )
except ImportError:
    from vendor_sphinteract import fewshot_utils
    from vendor_sphinteract import query_module
    from vendor_sphinteract.prompts import (
        fewshot_prefix, feedback_prefix_v1, cq_prefix_v1,
        selfdebug_few_shot, sql_generation_selfdebug,
        sql_generation_v2, feedback_v2, SRA, SRA_ES, fix_invalid_v1,
    )


def clean_query(sql_query: str) -> str:
    """Remove markdown fences and strip."""
    sql_query = sql_query.replace("```sql", '').replace("```", '')
    return sql_query.strip()


def first_gold_sql(value):
    """If gold SQL is stored as a list, take the first element."""
    if isinstance(value, list):
        return value[0] if value else ''
    return value


def generate_db_schema(dbname: str, source: str = 'bird', db_file: str = '') -> str:
    return fewshot_utils.generate_db_schema(dbname, source=source, db_file=db_file)


def resolve_db_path(database: str, source: str = 'bird', db_file: str = '') -> str:
    return fewshot_utils.resolve_db_path(database, source=source, db_file=db_file)


def evalfunc(sql_source, sql_target, database, source='bird', db_file=None):
    """Evaluate SQL against gold. Returns (execution_bool, exception_list)."""
    return fewshot_utils.evalfunc(sql_source, sql_target, database, source=source, db_file=db_file)


def build_cqas_text(cqs_and_answers: list, evidence: str, with_metadata: bool) -> str:
    """Format CQ/answer pairs for prompt context. Same as reformatted/_build_cqas_text."""
    cqas = ""
    if with_metadata and evidence:
        cqas = 'user: ' + evidence + '\n'
    for idx, value in enumerate(cqs_and_answers):
        if idx % 2 == 0:
            cqas += "multiple choice clarification question: " + value + '\n'
        else:
            cqas += "user: " + value + '\n'
    if cqas == '':
        cqas = 'no previous clarification question.\n'
    return cqas


def build_seed_prompt(dbschema: str, nlq: str, k_shot: int, evidence: str) -> str:
    return (fewshot_prefix + selfdebug_few_shot[k_shot - 1] +
            sql_generation_selfdebug.format(schema=dbschema, question=nlq, sqls='', metadata=evidence))


def build_selfdebug_prompt(dbschema: str, nlq: str, query_set: set, evidence: str, k_shot: int) -> str:
    return (fewshot_prefix + selfdebug_few_shot[k_shot - 1] +
            sql_generation_selfdebug.format(schema=dbschema, question=nlq,
                                            sqls=";\n".join(query_set), metadata=evidence))


def build_cq_prompt(dbschema: str, nlq: str, query_set: set, cqs_and_answers: list,
                    evidence: str, with_metadata: bool, break_on_no_amb: bool) -> str:
    cqas = build_cqas_text(cqs_and_answers, evidence, with_metadata)
    template = SRA_ES if break_on_no_amb else SRA
    prompt = template.format(schema=dbschema, question=nlq,
                             sqls=";\n".join(query_set), cqs=cqas)
    return cq_prefix_v1 + prompt


def build_feedback_prompt(gold_sql: str, cq: str, nlq: str) -> str:
    original_prompt = feedback_prefix_v1 + feedback_v2.format(query=gold_sql, question=cq, nlq=nlq)
    return (
        "You are simulating a user who knows the correct SQL answer.\n\n"
        "Answer the clarification question by following the original task below, "
        "but return only one JSON list in this format:\n"
        "[\n"
        '  {"module": "sphinteract", "question_text": "string", "answer_text": "string"}\n'
        "]\n\n"
        "## Original Sphinteract Feedback Task\n"
        f"{original_prompt}"
    )


def build_fix_invalid_prompt(dbschema: str, invalid_sql: str, exception_msg: str) -> str:
    return fix_invalid_v1.format(schema=dbschema, invalidSQL=invalid_sql, ex=exception_msg)


def load_pred_cache(path: str) -> dict[int, list[str]]:
    """Load prediction cache file. Returns dict mapping question_id -> list of SQL strings."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    if isinstance(data, dict):
        for qid_str, sqls in data.items():
            try:
                qid = int(qid_str)
            except (ValueError, TypeError):
                continue
            if isinstance(sqls, list):
                result[qid] = sqls
            elif isinstance(sqls, str):
                result[qid] = [sqls]
    elif isinstance(data, list):
        for item in data:
            qid = int(item.get("question_id", item.get("id", 0)))
            pred = item.get("pred", item.get("SQL", item.get("pred_sql", "")))
            if pred:
                result.setdefault(qid, []).append(pred)
    return result
