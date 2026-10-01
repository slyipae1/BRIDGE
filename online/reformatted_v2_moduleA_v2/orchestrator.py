"""
Orchestrator: stage-based pipeline loop with resume support.
"""
import os, time
from pathlib import Path

from .stage_artifacts import (
    is_stage_complete, load_stage_json, load_latest_stage,
    write_stage_json,
)
from .stages.stage_seed import run_seed_generation, run_seed_evaluation
from .stages.stage_clarification import (
    run_module_a_feedback_generation,
    run_cq_generation, run_feedback_generation,
    run_sql_regeneration, run_sql_evaluation,
)
from .stages.stage_module_a_retrieval import run_module_a_retrieval
from .module_a_base.config import (
    DEFAULT_DDL_MODE,
    DEFAULT_FEEDBACK_RENDER_MODE,
    DEFAULT_INTEGRATION_MODE,
    DEFAULT_MODULE_A_COLUMN_DESCRIP,
    DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
    DEFAULT_MODULE_A_CONTEXTUALIZATION_MODE,
    DEFAULT_MODULE_A_DB_ROOT,
    should_run_module_a_turn,
    should_run_sphinteract_turn,
    should_stop_after_turn,
)
from .module_a_base.feedback_prompts import DEFAULT_MODULE_A_DETECTION_PROMPT_MODE
from .module_a_base.schema_filter import SCHEMA_FILTER_NONE
from .module_a_base.retrieval_ablation import DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL
from .output_formatter import (
    finalize_run_summary,
    flush_all_question_jsons,
    make_summary_entry,
    write_question_json,
)
from .telemetry import (
    Timer,
    append_round_timing,
    build_cost_summary,
    elapsed_s,
    iso_now,
    monotonic_s,
    stage_timing,
)


def init_question_state(question_id, internal_index, row, data_source,
                         dbschema, model_name, db_file, with_metadata):
    """Initialize a per-question state dict."""
    from .utils import first_gold_sql
    gold = first_gold_sql(row.get('SQL', row.get('gold_query', '')))
    nlq = row.get('question', row.get('nl', ''))
    db_id = row.get('db_id', row.get('target_db', ''))
    evidence = row.get('evidence', '') if with_metadata else ''

    return {
        'status': 'active',
        'internal_index': internal_index,
        'question_id': question_id,
        'question': nlq,
        'db_id': db_id,
        'gold_sql': gold,
        'data_source': data_source,
        'dbschema': dbschema,
        'evidence': evidence,
        'module_a_column_retrieval_source': DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
        'db_file': db_file or '',
        'model': model_name,
        'sql_log': [],
        'cq_log': [],
        'feedback_log': [],
        'events': [],
        'num_cq_asked': 0,
        'final_sql': '',
        'completed_reason': None,
        'error_stage': None,
        'error_type': None,
        'error_message': None,
        'order': 0,
        'query_set': set(),
        'cqs_and_answers': [],
        'last_cq': '',
        '_last_sql': '',
        'module_a_entries': [],
        'module_a_rendered_items': [],
        'module_a_feedback_log': [],
        'module_a_specs': [],
        'module_a_decisions': [],
        'module_a_status': 'not_run',
        'module_a_mode': '',
        'module_a_integration_mode': '',
        'module_a_ddl_mode': '',
        'module_a_subset_only': False,
        'module_a_column_descrip': DEFAULT_MODULE_A_COLUMN_DESCRIP,
        'module_a_contextualization_mode': DEFAULT_MODULE_A_CONTEXTUALIZATION_MODE,
        'module_a_detection_prompt_mode': DEFAULT_MODULE_A_DETECTION_PROMPT_MODE,
        'module_a_schema_filter_mode': SCHEMA_FILTER_NONE,
    }


def _reconstruct_from_stage(questions, stage_data, turn):
    """If resuming after a completed LLM stage, restore per-question state
    from the response data so that subsequent stages have the correct state."""
    if not stage_data:
        return
    responses = stage_data.get("responses", [])
    for r in responses:
        qid = r.get("question_id")
        qs = questions.get(qid)
        if qs is None or qs.get('status') != 'active':
            continue
        if r.get("ok"):
            qs['_last_sql'] = r.get("content", "")
            if r.get("stage") == "cq_gen":
                qs['last_cq'] = r.get("content", "")
                qs['order'] = qs.get('order', 0) + 1
            elif r.get("stage") == "sql_gen":
                qs['order'] = qs.get('order', 0) + 1


def run_pipeline(run_dir, all_questions, batch_client_sql, batch_client_cq,
                 rounds, data_source, k_shot, with_metadata,
                 break_on_no_amb, sql_gen_few_shot, dry, debug, pred_cache=None,
                 integration_mode=DEFAULT_INTEGRATION_MODE,
                 module_a_db_root=DEFAULT_MODULE_A_DB_ROOT,
                 module_a_db_mode="dev",
                 module_a_lsh_top_n=20,
                 module_a_column_retrieval_source=DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
                 module_a_enable_llm_fallback=True,
                 module_a_column_group_version="manual",
                 module_a_column_group_artifact_root=None,
                 retrieval_ablation_disable_channel=DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
                 module_a_include_column_relation_metadata=True,
                 module_a_column_descrip=DEFAULT_MODULE_A_COLUMN_DESCRIP,
                 module_a_column_group_prompt_format="plain_list",
                 module_a_contextualization_mode=DEFAULT_MODULE_A_CONTEXTUALIZATION_MODE,
                 module_a_detection_mode="by_all",
                 module_a_detection_prompt_mode=DEFAULT_MODULE_A_DETECTION_PROMPT_MODE,
                 module_a_schema_filter_mode=SCHEMA_FILTER_NONE,
                 module_a_col_lit_aggregation=None,
                 feedback_render_mode=DEFAULT_FEEDBACK_RENDER_MODE,
                 ddl_mode=DEFAULT_DDL_MODE,
                 subset_only=False,
                 batch_client_feedback=None):
    """Main pipeline loop with resume support."""
    pipeline_timer = Timer("pipeline")
    # ADD: User-simulator feedback may use a fixed-capability endpoint while
    # detection, CQ generation, and all SQL stages retain the system client.
    if batch_client_feedback is None:
        batch_client_feedback = batch_client_cq

    for qs in all_questions.values():
        qs['module_a_integration_mode'] = integration_mode
        qs['module_a_ddl_mode'] = ddl_mode
        qs['module_a_subset_only'] = bool(subset_only)
        qs['module_a_contextualization_mode'] = module_a_contextualization_mode
        qs['module_a_detection_mode'] = module_a_detection_mode
        qs['module_a_detection_prompt_mode'] = module_a_detection_prompt_mode
        qs['module_a_schema_filter_mode'] = module_a_schema_filter_mode
        qs['module_a_column_descrip'] = module_a_column_descrip
        qs['module_a_column_retrieval_source'] = module_a_column_retrieval_source
        if module_a_col_lit_aggregation is not None:
            qs['module_a_col_lit_aggregation'] = module_a_col_lit_aggregation

    # ---- Stage 0: Seed Generation ----
    if pred_cache is None:
        pred_cache = {}

    run_seed_generation(run_dir, all_questions, batch_client_sql,
                        k_shot, with_metadata, data_source, pred_cache, dry, debug)
    seed_generation_completed_m = monotonic_s()
    flush_all_question_jsons(run_dir, all_questions)

    # ---- Stage 1: Seed Evaluation ----
    run_seed_evaluation(run_dir, all_questions, data_source, dry, debug, batch_client_sql)
    flush_all_question_jsons(run_dir, all_questions)

    # ---- Rounds ----
    for turn in range(rounds):
        active = {qid: qs for qid, qs in all_questions.items()
                  if qs.get('status') == 'active'}
        if not active:
            print(f"[PIPE] All questions completed by round {turn}")
            break

        round_started_at = iso_now()
        round_started_m = monotonic_s()
        round_stages = []
        sql_regeneration_completed_at = None
        sql_regeneration_completed_m = None
        round_timing_written = False

        def _append_round_record():
            nonlocal round_timing_written
            if round_timing_written:
                return
            completed_m = monotonic_s()
            record = {
                "turn": turn,
                "started_at": round_started_at,
                "completed_at": iso_now(),
                "sql_regeneration_completed_at": sql_regeneration_completed_at,
                "round_full_elapsed_s": elapsed_s(round_started_m, completed_m),
                "stages": list(round_stages),
            }
            if sql_regeneration_completed_m is not None:
                record["round_to_sql_regeneration_elapsed_s"] = elapsed_s(
                    round_started_m,
                    sql_regeneration_completed_m,
                )
                record["seed_generation_end_to_sql_regeneration_elapsed_s"] = elapsed_s(
                    seed_generation_completed_m,
                    sql_regeneration_completed_m,
                )
            else:
                record["round_to_sql_regeneration_elapsed_s"] = None
                record["seed_generation_end_to_sql_regeneration_elapsed_s"] = None
            append_round_timing(run_dir, record)
            round_timing_written = True

        if should_run_module_a_turn(turn, integration_mode=integration_mode):
            run_module_a_retrieval(
                run_dir,
                all_questions,
                turn,
                db_root_path=module_a_db_root,
                db_mode=module_a_db_mode,
                lsh_top_n=module_a_lsh_top_n,
                column_retrieval_source=module_a_column_retrieval_source,
                batch_client=batch_client_cq,
                enable_llm_fallback=module_a_enable_llm_fallback,
                column_group_version=module_a_column_group_version,
                column_group_artifact_root=module_a_column_group_artifact_root,
                retrieval_ablation_disable_channel=retrieval_ablation_disable_channel,
                debug=debug,
            )
            round_stages.append("module_a_retrieval")
            flush_all_question_jsons(run_dir, all_questions)

            run_module_a_feedback_generation(
                run_dir, all_questions, batch_client_cq, turn,
                feedback_batch_client=batch_client_feedback,
                feedback_render_mode=feedback_render_mode,
                ddl_mode=ddl_mode,
                subset_only=subset_only,
                include_column_relation_metadata=module_a_include_column_relation_metadata,
                column_descrip_mode=module_a_column_descrip,
                column_group_prompt_format=module_a_column_group_prompt_format,
                column_group_version=module_a_column_group_version,
                contextualization_mode=module_a_contextualization_mode,
                detection_mode=module_a_detection_mode,
                detection_prompt_mode=module_a_detection_prompt_mode,
                schema_filter_mode=module_a_schema_filter_mode,
                col_lit_aggregation_mode=module_a_col_lit_aggregation,
                debug=debug,
            )
            round_stages.append("module_a_feedback_generation")
            flush_all_question_jsons(run_dir, all_questions)

            active = {qid: qs for qid, qs in all_questions.items()
                      if qs.get('status') == 'active'}
            if not active:
                _append_round_record()
                continue

        if should_run_sphinteract_turn(turn, integration_mode=integration_mode):
            run_cq_generation(run_dir, all_questions, batch_client_cq,
                              turn, with_metadata, break_on_no_amb, debug)
            round_stages.append("cq_generation")
            flush_all_question_jsons(run_dir, all_questions)

            active = {qid: qs for qid, qs in all_questions.items()
                      if qs.get('status') == 'active'}
            if not active:
                _append_round_record()
                continue

            run_feedback_generation(run_dir, all_questions, batch_client_feedback,
                                    turn, debug)
            round_stages.append("feedback_generation")
            flush_all_question_jsons(run_dir, all_questions)

            active = {qid: qs for qid, qs in all_questions.items()
                      if qs.get('status') == 'active'}
            if not active:
                _append_round_record()
                continue

        # Stage: SQL Regeneration
        run_sql_regeneration(run_dir, all_questions, batch_client_sql,
                             turn, sql_gen_few_shot, debug)
        round_stages.append("sql_regeneration")
        sql_regeneration_completed_m = monotonic_s()
        sql_regeneration_completed_at = iso_now()
        flush_all_question_jsons(run_dir, all_questions)

        active = {qid: qs for qid, qs in all_questions.items()
                  if qs.get('status') == 'active'}
        if not active:
            _append_round_record()
            continue

        # Stage: SQL Evaluation + fix_invalid
        run_sql_evaluation(run_dir, all_questions, data_source,
                           turn, batch_client_sql, dry, debug)
        round_stages.append("sql_evaluation")
        flush_all_question_jsons(run_dir, all_questions)
        _append_round_record()

        if should_stop_after_turn(turn, integration_mode=integration_mode):
            print(f"[PIPE] Stopping after Module A single turn at round {turn}")
            break

    # ---- Finalize incomplete questions ----
    for qid, qs in all_questions.items():
        if qs.get('status') == 'active':
            qs['status'] = 'failed'
            qs['completed_reason'] = 'exhausted'
            qs['num_cq_asked'] = "Failed"
            qs['final_sql'] = qs.get('sql_log', [[None, None, '', None]])[-1][2]

    # ---- Write per-question artifacts ----
    questions_dir = os.path.join(run_dir, 'questions')
    logs_dir = os.path.join(run_dir, 'logs')
    for qid, qs in all_questions.items():
        write_question_json(questions_dir, qid, qs)

    # ---- Mark pipeline complete ----
    write_stage_json(run_dir, "pipeline_complete", 0, {
        "status": "ok",
        "total": len(all_questions),
        "completed": sum(1 for q in all_questions.values() if q.get('status') == 'completed'),
        "failed": sum(1 for q in all_questions.values() if q.get('status') == 'failed'),
        "timing": stage_timing(pipeline_timer),
    })

    # ---- Write cost summary and run_summary ----
    cost_summary = build_cost_summary(run_dir)
    summaries = [make_summary_entry(qid, qs) for qid, qs in all_questions.items()]
    finalize_run_summary(run_dir, summaries, metrics_summary={
        "cost_summary_path": cost_summary.get("cost_summary_path"),
        "llm_total_tokens": sum(
            bucket.get("total_tokens", 0)
            for bucket in cost_summary.get("llm_usage_by_model", {}).values()
        ),
        "total_stage_elapsed_s": cost_summary.get("timing_summary", {}).get("total_stage_elapsed_s", 0.0),
    })
    print(f"\n[PIPE] Pipeline complete. "
          f"{sum(1 for q in all_questions.values() if q.get('status') == 'completed')} completed, "
          f"{sum(1 for q in all_questions.values() if q.get('status') == 'failed')} failed")
