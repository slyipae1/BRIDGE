import json
import re
import uuid
from pathlib import Path

from .batch_client import BatchRequest
from .stage_artifacts import STAGE_ORDER, is_stage_complete, load_latest_stage, load_response_json, stage_dir
from .utils import build_fix_invalid_prompt, clean_query, evalfunc


_SEED_RESPONSE_RE = re.compile(
    r"=+\sLLM Response: seed_response\s=+\n(.*?)(?=\n={6,}|\Z)",
    re.DOTALL,
)

_ROUND_STAGES = (
    "cq_generation",
    "feedback_generation",
    "sql_regeneration",
    "sql_evaluation",
)


def collect_resume_question_ids(run_dir):
    run_dir = Path(run_dir)

    question_files = sorted((run_dir / "questions").glob("*.json"))
    if question_files:
        return sorted(int(path.stem) for path in question_files if path.stem.isdigit())

    log_files = sorted((run_dir / "logs").glob("*.log"))
    if log_files:
        return sorted(int(path.stem) for path in log_files if path.stem.isdigit())

    qids = set()
    for response_path in sorted(run_dir.glob("stage_artifacts/**/*response.json")):
        payload = load_response_json_from_path(response_path)
        for item in payload.get("requests", []):
            if item.get("question_id") is not None:
                qids.add(int(item["question_id"]))
        for item in payload.get("responses", []):
            if item.get("question_id") is not None:
                qids.add(int(item["question_id"]))
    return sorted(qids)


def load_response_json_from_path(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _discover_latest_completed_stage(run_dir):
    latest = None
    for stage_path in Path(run_dir).glob("stage_artifacts/**/*stage.json"):
        payload = json.loads(stage_path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            continue
        stage = payload.get("stage")
        round_idx = int(payload.get("round", 0) or 0)
        if stage == "pipeline_complete":
            stage_idx = len(STAGE_ORDER)
        elif stage in STAGE_ORDER:
            stage_idx = STAGE_ORDER.index(stage)
        else:
            continue
        key = (round_idx, stage_idx)
        if latest is None or key > latest[0]:
            latest = (key, {"stage": stage, "round": round_idx, "status": payload.get("status", "ok")})
    return latest[1] if latest else {}


def extract_seed_sql_from_log(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8")
    match = _SEED_RESPONSE_RE.search(text)
    if not match:
        return None
    return clean_query(match.group(1))


def _seed_event(qs, sql, source):
    return {
        "event_type": "seed_initial_generation",
        "prompt": "",
        "response": sql,
        "perplexity": 0.0,
        "model": qs.get("model", ""),
        "question_id": qs.get("question_id"),
        "internal_index": qs.get("internal_index"),
        "source": source,
    }


def _reconstruct_seed_generation(run_dir, questions, pred_cache):
    response_data = load_response_json(run_dir, "seed_generation", 0)
    if response_data and response_data.get("responses"):
        for response in response_data.get("responses", []):
            qid = response.get("question_id")
            if qid not in questions:
                continue
            qs = questions[qid]
            if not response.get("ok"):
                qs["status"] = "failed"
                qs["error_stage"] = "seed_generation"
                qs["error_message"] = response.get("error", "Unknown resume error")
                continue
            sql = clean_query(response.get("content", ""))
            qs["internal_index"] = response.get("internal_index", qs.get("internal_index"))
            qs["sql_log"] = [[0, "resume_artifact", sql, response.get("perplexity", 0.0)]]
            qs["events"] = [_seed_event(qs, sql, "resume_artifact")]
        return

    missing = []
    for qid, qs in questions.items():
        sql = None
        source = None
        cached_sqls = pred_cache.get(qid) if pred_cache else None
        if cached_sqls:
            sql = clean_query(cached_sqls[0])
            source = "pred_cache"
        if not sql:
            sql = extract_seed_sql_from_log(Path(run_dir) / "logs" / f"{int(qid):04d}.log")
            source = "resume_log"
        if not sql:
            missing.append(qid)
            continue
        qs["sql_log"] = [[0, source, sql, 0.0]]
        qs["events"] = [_seed_event(qs, sql, source)]

    if missing:
        raise ValueError(
            "Could not reconstruct seed SQL for question_ids "
            f"{missing[:10]} from stage artifacts, pred_cache, or logs."
        )


def _replay_seed_evaluation(questions, data_source, dry, batch_client_sql=None):
    for qid, qs in questions.items():
        if qs.get("status") != "active":
            continue
        if not qs.get("sql_log"):
            qs["status"] = "failed"
            qs["error_stage"] = "seed_evaluation"
            qs["error_message"] = "No seed SQL found"
            continue

        order, prompt, sql, pscore = qs["sql_log"][0]
        sql = clean_query(sql)
        qs.setdefault("events", []).append(
            {
                "event_type": "initial_sql_candidate",
                "order": 0,
                "prompt": prompt,
                "response": sql,
                "perplexity": pscore,
            }
        )

        execution = True
        exception = None
        if not dry:
            kwargs = {"source": data_source}
            if qs.get("db_file"):
                kwargs["db_file"] = qs["db_file"]
            execution, exception = evalfunc(sql, qs["gold_sql"], qs["db_id"], **kwargs)
            if exception:
                query_set = qs.get("query_set", set())
                if sql in query_set:
                    query_set.discard(sql)
                fixed_sql = None
                if batch_client_sql is not None:
                    invalid_prompt = build_fix_invalid_prompt(qs["dbschema"], sql, str(exception[0]))
                    fix_req = BatchRequest(
                        request_id=uuid.uuid4().hex,
                        question_id=qid,
                        internal_index=qs.get("internal_index", 0),
                        prompt=invalid_prompt,
                        stage="sql_fix_seed",
                        turn=0,
                        stop_thinking=True,
                        temperature=0.0,
                        max_tokens=16384,
                    )
                    fix_resp = batch_client_sql.batch_generate([fix_req])[0]
                    if fix_resp.ok:
                        fixed_sql = clean_query(fix_resp.content)
                        qs.setdefault("sql_log", []).append((0, invalid_prompt, fixed_sql, fix_resp.perplexity))
                        qs.setdefault("events", []).append(
                            {
                                "event_type": "sql_fix_seed",
                                "order": 0,
                                "turn": 0,
                                "prompt": invalid_prompt,
                                "response": fixed_sql,
                                "perplexity": fix_resp.perplexity,
                                "reasoning": fix_resp.reasoning,
                                "usage": fix_resp.usage,
                                "llm_request_logs": [fix_resp.request_log_path] if fix_resp.request_log_path else [],
                            }
                        )
                if fixed_sql:
                    query_set.add(fixed_sql)
                    kwargs = {"source": data_source}
                    if qs.get("db_file"):
                        kwargs["db_file"] = qs["db_file"]
                    execution, _ = evalfunc(fixed_sql, qs["gold_sql"], qs["db_id"], **kwargs)
                qs["query_set"] = set(query_set)

        qs.setdefault("events", []).append(
            {
                "event_type": "initial_sql_evaluation",
                "order": 0,
                "sql": sql,
                "execution": execution,
            }
        )

        if execution:
            qs["status"] = "completed"
            qs["completed_reason"] = "seed_correct"
            qs["final_sql"] = qs.get("sql_log", [[None, None, "", None]])[-1][2]
            qs["num_cq_asked"] = 0
        else:
            qs["query_set"] = set(qs.get("query_set", set()) or {sql})
            qs["query_set"].add(sql)
            qs.setdefault("cqs_and_answers", [])
            qs["order"] = 1


def _reconstruct_module_a_retrieval(run_dir, questions, turn=0):
    payload_path = stage_dir(str(run_dir), "module_a_retrieval", turn) / "module_a_retrieval_payload.json"
    if not payload_path.exists():
        return
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    for qid_text, record in payload.get("questions", {}).items():
        try:
            qid = int(qid_text)
        except (TypeError, ValueError):
            continue
        target_key = qid if qid in questions else str(qid) if str(qid) in questions else None
        if target_key is not None:
            questions[target_key]["module_a_retrieval"] = record


def _resume_prompt(stage_name, request_id):
    return f"resume_artifact:{stage_name}:{request_id}"


def _replay_cq_generation(response_data, questions, turn):
    request_map = {item.get("request_id"): item for item in response_data.get("requests", [])}
    for response in response_data.get("responses", []):
        qid = response.get("question_id")
        qs = questions.get(qid)
        if qs is None or qs.get("status") != "active":
            continue
        order = qs.get("order", 0)
        if not response.get("ok"):
            qs["status"] = "failed"
            qs["error_stage"] = f"cq_generation_turn_{turn}"
            qs["error_message"] = response.get("error", "Unknown resume error")
            continue

        cq = response.get("content", "")
        req = request_map.get(response.get("request_id"), {})
        qs["internal_index"] = response.get("internal_index", qs.get("internal_index"))
        qs.setdefault("cq_log", []).append((order, _resume_prompt("cq_generation", response.get("request_id")), cq, response.get("perplexity", 0.0)))
        qs.setdefault("events", []).append(
            {
                "event_type": "cq_generation",
                "order": order,
                "turn": turn,
                "prompt": req.get("prompt", ""),
                "response": cq,
                "perplexity": response.get("perplexity", 0.0),
                "reasoning": response.get("reasoning", ""),
                "usage": response.get("usage"),
                "llm_request_logs": [response.get("request_log_path")] if response.get("request_log_path") else [],
            }
        )
        order += 1

        if "NO AMBIGUITY" in cq:
            qs.setdefault("events", []).append(
                {
                    "event_type": "cq_no_ambiguity",
                    "order": order - 1,
                    "turn": turn,
                    "response": cq,
                }
            )
            qs["status"] = "completed"
            qs["completed_reason"] = "no_ambiguity"
            qs["num_cq_asked"] = turn + 1
            qs["final_sql"] = qs.get("sql_log", [[None, None, "", None]])[-1][2]
            continue

        if "mul_choice_cq = " in cq:
            cq = cq.split("mul_choice_cq = ")[-1]
        qs["last_cq"] = cq
        qs["order"] = order


def _replay_feedback_generation(response_data, questions, turn):
    for response in response_data.get("responses", []):
        qid = response.get("question_id")
        qs = questions.get(qid)
        if qs is None or qs.get("status") != "active":
            continue
        order = qs.get("order", 0)
        if not response.get("ok"):
            qs["status"] = "failed"
            qs["error_stage"] = f"feedback_generation_turn_{turn}"
            qs["error_message"] = response.get("error", "Unknown resume error")
            continue

        feedback = response.get("content", "")
        qs.setdefault("feedback_log", []).append(
            (order, _resume_prompt("feedback_generation", response.get("request_id")), feedback, response.get("perplexity", 0.0))
        )
        qs.setdefault("events", []).append(
            {
                "event_type": "feedback_generation",
                "order": order,
                "turn": turn,
                "prompt": "",
                "response": feedback,
                "perplexity": response.get("perplexity", 0.0),
                "reasoning": response.get("reasoning", ""),
                "usage": response.get("usage"),
                "llm_request_logs": [response.get("request_log_path")] if response.get("request_log_path") else [],
            }
        )
        order += 1

        if "answer_to_cq =" in feedback:
            feedback = feedback.split("answer_to_cq =")[-1].strip()

        cqs_and_answers = qs.get("cqs_and_answers", [])
        cqs_and_answers.append(qs.get("last_cq", ""))
        cqs_and_answers.append(feedback)
        qs["cqs_and_answers"] = cqs_and_answers
        qs["order"] = order


def _replay_sql_regeneration(response_data, questions, turn):
    for response in response_data.get("responses", []):
        qid = response.get("question_id")
        qs = questions.get(qid)
        if qs is None or qs.get("status") != "active":
            continue
        order = qs.get("order", 0)
        if not response.get("ok"):
            qs["status"] = "failed"
            qs["error_stage"] = f"sql_regeneration_turn_{turn}"
            qs["error_message"] = response.get("error", "Unknown resume error")
            continue

        sql = clean_query(response.get("content", ""))
        qs.setdefault("sql_log", []).append(
            (order, _resume_prompt("sql_regeneration", response.get("request_id")), sql, response.get("perplexity", 0.0))
        )
        qs.setdefault("events", []).append(
            {
                "event_type": "sql_generation",
                "order": order,
                "turn": turn,
                "prompt": "",
                "response": sql,
                "perplexity": response.get("perplexity", 0.0),
                "reasoning": response.get("reasoning", ""),
                "usage": response.get("usage"),
                "llm_request_logs": [response.get("request_log_path")] if response.get("request_log_path") else [],
            }
        )
        order += 1

        query_set = qs.get("query_set", set())
        query_set.add(sql)
        qs["query_set"] = query_set
        qs["order"] = order
        qs["_last_sql"] = sql


def _replay_sql_evaluation(questions, data_source, turn, batch_client_sql, dry):
    for qid, qs in list(questions.items()):
        if qs.get("status") != "active":
            continue
        sql = qs.get("_last_sql", "")
        if not sql and qs.get("sql_log"):
            sql = qs["sql_log"][-1][2]
        if not sql:
            qs["status"] = "failed"
            qs["error_stage"] = f"sql_evaluation_turn_{turn}"
            qs["error_message"] = "No SQL to evaluate"
            continue

        if dry:
            execution, exception = True, None
        else:
            try:
                kwargs = {"source": data_source}
                if qs.get("db_file"):
                    kwargs["db_file"] = qs["db_file"]
                execution, exception = evalfunc(sql, qs["gold_sql"], qs["db_id"], **kwargs)
            except Exception as exc:
                qs["status"] = "failed"
                qs["error_stage"] = f"sql_evaluation_turn_{turn}"
                qs["error_message"] = str(exc)
                continue

        order = qs.get("order", 0)
        qs.setdefault("events", []).append(
            {
                "event_type": "sql_evaluation",
                "order": order - 1,
                "turn": turn,
                "sql": sql,
                "execution": execution,
                "exception": [str(ex) for ex in exception] if exception else [],
            }
        )

        if exception and batch_client_sql is not None:
            most_recent = clean_query(qs.get("sql_log", [[None, None, "", None]])[-1][2])
            query_set = qs.get("query_set", set())
            if most_recent in query_set:
                query_set.discard(most_recent)
            fix_prompt = build_fix_invalid_prompt(qs["dbschema"], most_recent, str(exception[0]))
            fix_req = BatchRequest(
                request_id=uuid.uuid4().hex,
                question_id=qid,
                internal_index=qs.get("internal_index", 0),
                prompt=fix_prompt,
                stage="sql_fix",
                turn=turn,
                stop_thinking=True,
                temperature=0.0,
                max_tokens=16384,
            )
            fix_resp = batch_client_sql.batch_generate([fix_req])[0]
            if fix_resp.ok:
                fixed_sql = clean_query(fix_resp.content)
                query_set.add(fixed_sql)
                qs["query_set"] = query_set
                qs.setdefault("sql_log", []).append((order, fix_prompt, fixed_sql, fix_resp.perplexity))
                qs.setdefault("events", []).append(
                    {
                        "event_type": "sql_fix",
                        "order": order,
                        "turn": turn,
                        "prompt": fix_prompt,
                        "response": fixed_sql,
                        "perplexity": fix_resp.perplexity,
                        "reasoning": fix_resp.reasoning,
                        "usage": fix_resp.usage,
                        "llm_request_logs": [fix_resp.request_log_path] if fix_resp.request_log_path else [],
                    }
                )
                order += 1
                kwargs = {"source": data_source}
                if qs.get("db_file"):
                    kwargs["db_file"] = qs["db_file"]
                try:
                    execution, _ = evalfunc(fixed_sql, qs["gold_sql"], qs["db_id"], **kwargs)
                except Exception:
                    execution = False

        if execution:
            qs["status"] = "completed"
            qs["completed_reason"] = "regen_correct"
            qs["num_cq_asked"] = turn + 1
            qs["final_sql"] = qs.get("sql_log", [[None, None, "", None]])[-1][2]
        else:
            qs["order"] = order


def _clarification_stage_reached(stage_name, turn, latest_stage, latest_round):
    if latest_stage in (None, "seed_generation", "seed_evaluation"):
        return False
    latest_idx = STAGE_ORDER.index(latest_stage) if latest_stage in STAGE_ORDER else len(STAGE_ORDER)
    stage_idx = STAGE_ORDER.index(stage_name)
    if turn < latest_round:
        return True
    if turn > latest_round:
        return False
    return stage_idx <= latest_idx


def reconstruct_run_state(run_dir, questions, data_source, dry, pred_cache=None, batch_client_sql=None, debug=False):
    del debug
    run_dir = str(run_dir)
    pred_cache = pred_cache or {}

    if is_stage_complete(run_dir, "seed_generation", 0):
        _reconstruct_seed_generation(run_dir, questions, pred_cache)
    if is_stage_complete(run_dir, "seed_evaluation", 0):
        _replay_seed_evaluation(questions, data_source, dry, batch_client_sql=batch_client_sql)

    latest_stage = load_latest_stage(run_dir) or _discover_latest_completed_stage(run_dir)
    latest_name = latest_stage.get("stage")
    latest_round = int(latest_stage.get("round", 0) or 0)

    for turn in range(latest_round + 1):
        if _clarification_stage_reached("module_a_retrieval", turn, latest_name, latest_round) and is_stage_complete(run_dir, "module_a_retrieval", turn):
            _reconstruct_module_a_retrieval(run_dir, questions, turn)
        if _clarification_stage_reached("cq_generation", turn, latest_name, latest_round) and is_stage_complete(run_dir, "cq_generation", turn):
            response_data = load_response_json(run_dir, "cq_generation", turn)
            if response_data:
                _replay_cq_generation(response_data, questions, turn)
        if _clarification_stage_reached("feedback_generation", turn, latest_name, latest_round) and is_stage_complete(run_dir, "feedback_generation", turn):
            response_data = load_response_json(run_dir, "feedback_generation", turn)
            if response_data:
                _replay_feedback_generation(response_data, questions, turn)
        if _clarification_stage_reached("sql_regeneration", turn, latest_name, latest_round) and is_stage_complete(run_dir, "sql_regeneration", turn):
            response_data = load_response_json(run_dir, "sql_regeneration", turn)
            if response_data:
                _replay_sql_regeneration(response_data, questions, turn)
        if _clarification_stage_reached("sql_evaluation", turn, latest_name, latest_round) and is_stage_complete(run_dir, "sql_evaluation", turn):
            _replay_sql_evaluation(questions, data_source, turn, batch_client_sql, dry)
