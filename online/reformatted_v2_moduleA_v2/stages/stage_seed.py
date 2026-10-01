"""
Stage: seed_generation [LLM batch] + seed_evaluation [pipeline].
"""
import uuid, sys
from ..batch_client import BatchRequest
from ..utils import clean_query, build_seed_prompt, evalfunc
from ..stage_artifacts import is_stage_complete, write_stage_json, write_response_json
from ..output_formatter import append_log_pair
from ..telemetry import Timer, append_cpu_step, stage_timing, summarize_usage


def _append_sql_eval_cpu_step(run_dir, *, qid, qs, status, timing, candidate, execution, exception):
    append_cpu_step(run_dir, {
        "step": "sql_evaluation",
        "stage": "seed_evaluation",
        "turn": 0,
        "question_id": int(qid),
        "db_id": qs.get("db_id", ""),
        "status": status,
        "elapsed_s": float(timing.get("elapsed_s") or 0.0),
        "metadata": {
            "candidate": candidate,
            "execution": bool(execution),
            "exception_count": len(exception or []),
            "timeout_s": None,
        },
    })


def run_seed_generation(run_dir, questions, batch_client, k_shot, with_metadata,
                        data_source, pred_cache, dry, debug):
    """Stage 0: LLM batch seed generation. Uses pred_cache if available."""
    if is_stage_complete(run_dir, "seed_generation"):
        print("[SEED] seed_generation already complete, skipping.")
        return

    timer = Timer("seed_generation")
    pending = []
    for qid, qs in questions.items():
        if qs['status'] != 'active':
            continue
        # Check pred_cache
        if qid in pred_cache:
            sql = clean_query(pred_cache[qid][0])
            qs['sql_log'] = [[0, 'pred_cache', sql, 0.0]]
            qs['events'].append({
                'event_type': 'seed_initial_generation',
                'response': sql, 'perplexity': 0.0, 'model': 'pred_cache',
                'source': 'pred_cache', 'question_id': qid,
                'internal_index': qs.get('internal_index'),
            })
            append_log_pair(qid, f"{run_dir}/logs", 'seed_prompt (pred_cache)',
                            '', 'seed_response', sql, debug_print=debug)
            if debug:
                print(f"[SEED] Cached question {qid}: {sql[:100]!r}")
            continue

        prompt = build_seed_prompt(qs['dbschema'], qs['question'], k_shot, qs.get('evidence', ''))
        pending.append(BatchRequest(
            request_id=uuid.uuid4().hex, question_id=qid,
            internal_index=qs.get('internal_index', 0),
            prompt=prompt, stage='seed', turn=0,
            stop_thinking=True, temperature=0.0, max_tokens=16384,
        ))

    if not pending:
        print(f"[SEED] All questions cached or complete ({len(questions)}).")
        write_stage_json(run_dir, "seed_generation", 0, {
            "status": "ok", "total": len(questions), "active": 0, "request_count": 0,
            "usage_summary": summarize_usage([]),
            "timing": stage_timing(timer),
        })
        return

    print(f"[SEED] Generating seeds for {len(pending)} questions via LLM ...")
    responses = batch_client.batch_generate(pending)
    ok_count = 0
    for req, resp in zip(pending, responses):
        qid = req.question_id
        qs = questions.get(qid)
        if qs is None:
            continue
        if not resp.ok:
            qs['status'] = 'failed'
            qs['error_stage'] = 'seed_generation'
            qs['error_message'] = resp.error
            continue
        sql = clean_query(resp.content)
        qs['sql_log'] = [[0, 'generated', sql, resp.perplexity]]
        qs['events'].append({
            'event_type': 'seed_initial_generation',
            'prompt': req.prompt, 'response': sql,
            'perplexity': resp.perplexity, 'model': batch_client._model_name,
            'question_id': qid, 'internal_index': qs.get('internal_index'),
            'reasoning': resp.reasoning, 'usage': resp.usage,
            'llm_request_logs': [resp.request_log_path] if resp.request_log_path else [],
        })
        append_log_pair(qid, f"{run_dir}/logs", 'seed_prompt',
                        req.prompt, 'seed_response', sql, debug_print=debug)
        ok_count += 1
        if debug:
            print(f"[SEED] Generated qid={qid}: {sql[:100]!r}")

    write_response_json(run_dir, "seed_generation", 0, pending, responses)
    write_stage_json(run_dir, "seed_generation", 0, {
        "status": "ok", "total": len(questions),
        "active": len(pending), "request_count": len(pending),
        "ok_count": ok_count, "failed_count": len(pending) - ok_count,
        "usage_summary": summarize_usage(responses),
        "timing": stage_timing(timer),
    })
    print(f"[SEED] Done: {ok_count}/{len(pending)} OK")


def run_seed_evaluation(run_dir, questions, data_source, dry, debug, batch_client_sql=None):
    """Stage 1: Evaluate seed SQL against gold. Mark correct ones as completed."""
    if is_stage_complete(run_dir, "seed_evaluation"):
        print("[EVAL] seed_evaluation already complete, skipping.")
        return

    timer = Timer("seed_evaluation")
    completed = 0
    failed_eval = 0
    for qid, qs in questions.items():
        if qs['status'] != 'active':
            continue
        if not qs.get('sql_log'):
            qs['status'] = 'failed'
            qs['error_stage'] = 'seed_evaluation'
            qs['error_message'] = 'No seed SQL found'
            continue

        order, prompt, sql, pscore = qs['sql_log'][0]
        sql = clean_query(sql)
        qs['events'].append({
            'event_type': 'initial_sql_candidate', 'order': 0,
            'prompt': prompt, 'response': sql, 'perplexity': pscore,
        })

        eval_timer = Timer("sql_evaluation")
        exception = None
        if dry:
            execution = True
            eval_timing = stage_timing(eval_timer, {"status": "dry_run"})
            eval_timing["elapsed_s"] = 0.0
        else:
            kwargs = {'source': data_source}
            if qs.get('db_file'):
                kwargs['db_file'] = qs['db_file']
            execution, exception = evalfunc(sql, qs['gold_sql'], qs['db_id'], **kwargs)
            eval_timing = stage_timing(eval_timer, {"status": "ok"})
            _append_sql_eval_cpu_step(
                run_dir,
                qid=qid,
                qs=qs,
                status="ok",
                timing=eval_timing,
                candidate="seed",
                execution=execution,
                exception=exception,
            )
            # fix_invalid path (matches notebook: if exception, try to fix via LLM)
            if exception:
                from ..utils import build_fix_invalid_prompt
                from ..batch_client import BatchRequest as _BR
                most_recent_sql = clean_query(qs.get('sql_log', [[None, None, '', None]])[-1][2])
                query_set = qs.get('query_set', set())
                if most_recent_sql in query_set:
                    query_set.discard(most_recent_sql)
                invalid_prompt = build_fix_invalid_prompt(qs['dbschema'], most_recent_sql, str(exception[0]))
                # Inline non-batched LLM call for seed fix
                if batch_client_sql:
                    fix_req = _BR(
                        request_id=uuid.uuid4().hex, question_id=qid,
                        internal_index=qs.get('internal_index', 0),
                        prompt=invalid_prompt, stage='sql_fix_seed', turn=0,
                        stop_thinking=True, temperature=0.0, max_tokens=16384,
                    )
                    fix_resp = batch_client_sql.batch_generate([fix_req])[0]
                    if fix_resp.ok:
                        fixed_sql = clean_query(fix_resp.content)
                        qs.setdefault('sql_log', []).append((0, invalid_prompt, fixed_sql, fix_resp.perplexity))
                        qs.setdefault('events', []).append({
                            'event_type': 'sql_fix_seed', 'order': 0, 'turn': 0,
                            'prompt': invalid_prompt, 'response': fixed_sql, 'perplexity': fix_resp.perplexity,
                            'reasoning': fix_resp.reasoning, 'usage': fix_resp.usage,
                            'timing': fix_resp.timing,
                            'llm_request_logs': [fix_resp.request_log_path] if fix_resp.request_log_path else [],
                        })
                        query_set.add(fixed_sql)
                        qs["query_set"] = set(query_set)
                    query_set.add(fixed_sql)
                qs['query_set'] = set(query_set) if isinstance(query_set, set) else query_set
        if dry:
            _append_sql_eval_cpu_step(
                run_dir,
                qid=qid,
                qs=qs,
                status="dry_run",
                timing=eval_timing,
                candidate="seed",
                execution=execution,
                exception=exception,
            )

        qs['events'].append({
            'event_type': 'initial_sql_evaluation', 'order': 0,
            'sql': sql, 'execution': execution,
            'exception': [str(ex) for ex in exception] if exception else [],
            'timing': eval_timing,
        })

        if execution:
            qs['status'] = 'completed'
            qs['completed_reason'] = 'seed_correct'
            qs['final_sql'] = sql
            qs['num_cq_asked'] = 0
            completed += 1
        else:
            # Set up for clarification rounds
            if 'query_set' not in qs:
                qs["query_set"] = {sql}
            if 'cqs_and_answers' not in qs:
                qs['cqs_and_answers'] = []
            qs['order'] = 1
            if debug:
                print(f"[EVAL] qid={qid} seed wrong, entering clarification.")

    write_stage_json(run_dir, "seed_evaluation", 0, {
        "status": "ok", "total": len(questions),
        "completed": completed, "failed_eval": failed_eval,
        "timing": stage_timing(timer),
    })
    print(f"[EVAL] Done: {completed} correct at seed, {failed_eval} eval errors")
