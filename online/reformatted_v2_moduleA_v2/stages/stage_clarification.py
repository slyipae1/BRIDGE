"""
Stage: clarification rounds (CQ generation, feedback, SQL regeneration, evaluation).
"""
import json
import uuid
from pathlib import Path
from ..batch_client import BatchRequest
from ..module_a_base.ambiguity_pack import auto_fill_entries
from ..module_a_base.config import (
    COL_LIT_AGGREGATION_MODES,
    DEFAULT_MODULE_A_COLUMN_DESCRIP,
    MODULE_A_COLUMN_DESCRIP_CHOICES,
    DEFAULT_MODULE_A_CONTEXTUALIZATION_MODE,
    MODULE_A_CONTEXTUALIZATION_MODE_CHOICES,
)
from ..module_a_base.col_lit_aggregation import (
    build_col_lit_aggregation_plan,
    validate_col_lit_aggregation_mode,
)
from ..module_a_base.ddl_enricher import enrich_schema_ddl
from ..module_a_base.feedback_prompts import (
    DEFAULT_MODULE_A_DETECTION_PROMPT_MODE,
    MODULE_A_DETECTION_PROMPT_MODES,
    build_module_a_detection_prompt,
    build_module_a_feedback_prompt,
    parse_json_list_response,
)
from ..module_a_base.detection_slices import (
    MODULE_A_DETECTION_MODES,
    build_detection_slices,
    dedupe_raw_detection_entries,
)
from ..module_a_base.schema_filter import (
    MODULE_A_SCHEMA_FILTER_MODES,
    SCHEMA_FILTER_NONE,
    filter_schema_for_detection,
)
from ..module_a_base.question_renderer import render_module_a_items
from ..module_a_base.sql_specs import build_module_a_specs
from ..utils import (
    clean_query, build_cq_prompt, build_feedback_prompt,
    build_fix_invalid_prompt, build_cqas_text, evalfunc,
)
from ..vendor_pipeline0612.dbelement_formatter import (
    MANUAL_TARGET_REASON_PROMPT_FORMATS,
    format_dbelement_options,
    resolve_column_group_prompt_format,
)
from ..vendor_pipeline0612.schema_description_loader import build_additional_info_by_column
from ..stage_artifacts import (
    is_stage_complete,
    stage_dir,
    write_stage_json,
    write_response_json,
)
from ..output_formatter import append_log_pair
from ..telemetry import Timer, append_cpu_step, elapsed_s, monotonic_s, stage_timing, summarize_usage


def _active_questions(questions):
    return {qid: qs for qid, qs in questions.items() if qs.get('status') == 'active'}


def _append_sql_eval_cpu_step(
    run_dir,
    *,
    turn,
    qid,
    qs,
    status,
    timing,
    candidate,
    execution,
    exception,
):
    append_cpu_step(run_dir, {
        "step": "sql_evaluation",
        "stage": "sql_evaluation",
        "turn": turn,
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


def _current_sql_for_question(qs):
    if qs.get('_last_sql'):
        return qs.get('_last_sql', '')
    sql_log = qs.get('sql_log', [])
    if sql_log:
        return sql_log[-1][2]
    query_set = qs.get('query_set', set())
    return next(iter(query_set), '')


def _resolve_db_dir(qs):
    db_file = str(qs.get('db_file', '') or '').strip()
    if not db_file:
        return None
    path = Path(db_file).expanduser()
    return path.parent if path.parent.exists() else None


def _description_info_for_question(qs):
    db_dir = _resolve_db_dir(qs)
    if db_dir is None:
        return {}
    return build_additional_info_by_column(db_dir, qs.get('db_id', ''))


def _additional_info_from_description_info(info, ddl_mode):
    del info
    if ddl_mode != "Sphinteract_plain":
        raise ValueError("The public runtime supports only Sphinteract_plain DDL.")
    return {}


def _column_descriptive_names_from_description_info(info):
    return {
        key: str(payload.get("descriptive_name") or "").strip()
        for key, payload in info.items()
        if str(payload.get("descriptive_name") or "").strip()
    }


def _additional_info_for_question(qs, ddl_mode):
    return _additional_info_from_description_info(
        _description_info_for_question(qs),
        ddl_mode,
    )


def _append_module_a_specs(qs, specs):
    cqs_and_answers = qs.get('cqs_and_answers', [])
    for spec in specs:
        cqs_and_answers.append(spec.get('question_text', ''))
        cqs_and_answers.append(spec.get('user_text', ''))
    qs['cqs_and_answers'] = cqs_and_answers


def _extract_sphinteract_feedback_answer(feedback_text):
    try:
        parsed = parse_json_list_response(feedback_text)
        if parsed and isinstance(parsed[0], dict):
            answer_text = str(parsed[0].get("answer_text", "") or "").strip()
            if answer_text:
                return answer_text
    except Exception:
        pass

    if "answer_to_cq =" in feedback_text:
        return feedback_text.split("answer_to_cq =")[-1].strip()
    return feedback_text


def _summarize_direct_render_stats(stats_by_qid):
    summary = {
        "question_count": len(stats_by_qid or {}),
        "anchor_slice_count": 0,
        "entry_count": 0,
        "choice_count": 0,
        "duplicate_choice_count": 0,
        "skipped_single_choice_count": 0,
        "elapsed_s": 0.0,
    }
    for stats in (stats_by_qid or {}).values():
        summary["anchor_slice_count"] += int(stats.get("anchor_slice_count") or 0)
        summary["entry_count"] += int(stats.get("entry_count") or 0)
        summary["choice_count"] += int(stats.get("choice_count") or 0)
        summary["duplicate_choice_count"] += int(stats.get("duplicate_choice_count") or 0)
        summary["skipped_single_choice_count"] += int(stats.get("skipped_single_choice_count") or 0)
        summary["elapsed_s"] += float(stats.get("elapsed_s") or 0.0)
    summary["elapsed_s"] = round(summary["elapsed_s"], 6)
    return summary


def _summarize_col_lit_aggregation(audits_by_qid):
    summary = {
        "question_count": len(audits_by_qid),
        "component_count": 0,
        "prune_eligible_component_count": 0,
        "pruning_component_count": 0,
        "no_action_component_count": 0,
        "singleton_skip_component_count": 0,
        "detection_slice_count": 0,
    }
    for audit in audits_by_qid.values():
        summary["component_count"] += int(audit.get("component_count") or 0)
        summary["prune_eligible_component_count"] += int(
            audit.get("prune_eligible_component_count") or 0
        )
        summary["pruning_component_count"] += int(audit.get("pruning_component_count") or 0)
        summary["no_action_component_count"] += int(audit.get("no_action_component_count") or 0)
        summary["singleton_skip_component_count"] += len(
            audit.get("singleton_skip_component_ids") or []
        )
        summary["detection_slice_count"] += int(audit.get("detection_slice_count") or 0)
    return summary


def _write_col_lit_aggregation_artifact(run_dir, stage_name, turn, *, mode, audits_by_qid):
    artifact_dir = stage_dir(run_dir, stage_name, turn)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "col_lit_aggregation.json"
    payload = {
        "mode": mode,
        "summary": _summarize_col_lit_aggregation(audits_by_qid),
        "questions": {str(qid): audit for qid, audit in audits_by_qid.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def run_module_a_feedback_generation(
    run_dir,
    questions,
    batch_client,
    turn,
    *,
    feedback_batch_client=None,
    feedback_render_mode,
    ddl_mode,
    subset_only,
    debug,
    include_column_relation_metadata=True,
    column_descrip_mode=DEFAULT_MODULE_A_COLUMN_DESCRIP,
    column_group_prompt_format="plain_list",
    column_group_version=None,
    contextualization_mode=DEFAULT_MODULE_A_CONTEXTUALIZATION_MODE,
    detection_mode="by_all",
    detection_prompt_mode=DEFAULT_MODULE_A_DETECTION_PROMPT_MODE,
    schema_filter_mode=SCHEMA_FILTER_NONE,
    col_lit_aggregation_mode=None,
):
    del debug
    detection_batch_client = batch_client
    if feedback_batch_client is None:
        feedback_batch_client = detection_batch_client
    if detection_mode not in MODULE_A_DETECTION_MODES:
        raise ValueError(f"Unsupported Module A detection mode: {detection_mode}")
    if contextualization_mode not in MODULE_A_CONTEXTUALIZATION_MODE_CHOICES:
        raise ValueError(f"Unsupported Module A contextualization mode: {contextualization_mode}")
    if detection_prompt_mode not in MODULE_A_DETECTION_PROMPT_MODES:
        raise ValueError(f"Unsupported Module A detection prompt mode: {detection_prompt_mode}")
    if schema_filter_mode not in MODULE_A_SCHEMA_FILTER_MODES:
        raise ValueError(f"Unsupported Module A schema filter mode: {schema_filter_mode}")
    if column_descrip_mode not in MODULE_A_COLUMN_DESCRIP_CHOICES:
        raise ValueError(f"Unsupported Module A column description mode: {column_descrip_mode}")
    if ddl_mode != "Sphinteract_plain":
        raise ValueError("The public runtime supports only Sphinteract_plain DDL.")
    if column_group_prompt_format not in MANUAL_TARGET_REASON_PROMPT_FORMATS:
        raise ValueError("The public runtime supports only manual_target_reason rendering.")
    col_lit_aggregation_mode = validate_col_lit_aggregation_mode(col_lit_aggregation_mode)
    if col_lit_aggregation_mode is not None:
        if col_lit_aggregation_mode not in COL_LIT_AGGREGATION_MODES:
            raise ValueError(f"Unsupported column-literal aggregation mode: {col_lit_aggregation_mode}")
        if detection_prompt_mode != "retrieval_elements":
            raise ValueError(
                "column-literal aggregation requires "
                "module_a_detection_prompt_mode=retrieval_elements"
            )

    stage_name = "module_a_feedback_generation"
    if is_stage_complete(run_dir, stage_name, turn):
        print(f"[MODA] round_{turn}/{stage_name} already complete, skipping.")
        return _active_questions(questions)

    timer = Timer(stage_name)
    active = _active_questions(questions)
    detection_requests = []
    feedback_requests = []
    detection_context_by_request_id = {}
    question_context = {}
    question_detection_order = {}
    schema_filter_action_counts = {}
    col_lit_aggregation_audits = {}
    # Direct-retrieval rendering is intentionally absent from the public route;
    # retain an empty audit map so the shared stage summary stays schema-stable.
    direct_render_stats_by_qid = {}
    effective_column_group_prompt_format = resolve_column_group_prompt_format(
        column_group_prompt_format,
        column_group_version,
    )

    # ---- Step 2a: Load live retrieval output, enrich DDL, LLM detection ----
    for qid, qs in active.items():
        order = qs.get('order', 0)
        retrieval_record = qs.get("module_a_retrieval") or {}
        retrieval_payload = retrieval_record.get("db_retrieval_sql_full")
        qs['module_a_mode'] = feedback_render_mode
        qs['module_a_ddl_mode'] = ddl_mode
        qs['module_a_subset_only'] = bool(subset_only)
        qs['module_a_column_descrip'] = column_descrip_mode
        qs['module_a_column_group_prompt_format'] = effective_column_group_prompt_format
        qs['module_a_contextualization_mode'] = contextualization_mode
        qs['module_a_detection_mode'] = detection_mode
        qs['module_a_detection_prompt_mode'] = detection_prompt_mode
        qs['module_a_schema_filter_mode'] = schema_filter_mode
        if col_lit_aggregation_mode is not None:
            qs['module_a_col_lit_aggregation'] = col_lit_aggregation_mode
        qs.setdefault('events', []).append(
            {
                'event_type': 'module_a_retrieval_load',
                'order': order,
                'turn': turn,
                'artifact_source': 'stage_artifacts/module_a_retrieval',
                'has_payload': retrieval_payload is not None,
            }
        )

        if not retrieval_payload:
            qs['module_a_status'] = 'skipped_no_payload'
            qs.setdefault('events', []).append(
                {
                    'event_type': 'module_a_skipped',
                    'order': order,
                    'turn': turn,
                    'reason': 'no_retrieval_payload',
                }
            )
            continue

        # ---- Enrich DDL (happens BEFORE detection) ----
        dbelement_options = retrieval_payload.get("dbelement_options", []) or []
        sql_parse_meta = retrieval_payload.get("sql_parse_meta") or {}
        parsed_sql_columns_by_table = (
            sql_parse_meta.get("columns_dict")
            if isinstance(sql_parse_meta, dict)
            else None
        )
        current_sql = _current_sql_for_question(qs)
        description_info_by_column = _description_info_for_question(qs)
        additional_info_by_column = _additional_info_from_description_info(
            description_info_by_column,
            ddl_mode,
        )
        column_descriptive_names = _column_descriptive_names_from_description_info(
            description_info_by_column
        )
        schema_ddl = enrich_schema_ddl(
            qs.get('dbschema', ''),
            additional_info_by_column=additional_info_by_column,
            ddl_mode=ddl_mode,
            subset_only=subset_only,
            current_sql=current_sql,
            dbelement_options=dbelement_options,
        )
        question_context[qid] = {
            "schema_ddl": schema_ddl,
            "current_sql": current_sql,
            "dbelement_options": dbelement_options,
        }
        if col_lit_aggregation_mode is None:
            # MOD: The absent flag preserves the legacy slice builder and its
            # request metadata without invoking the aggregation planner.
            detection_slices = build_detection_slices(
                dbelement_options,
                detection_mode=detection_mode,
            )
        else:
            aggregation_plan = build_col_lit_aggregation_plan(
                dbelement_options=dbelement_options,
                current_sql=current_sql,
                db_path=qs.get("db_file"),
                detection_mode=detection_mode,
                mode=col_lit_aggregation_mode,
            )
            detection_slices = aggregation_plan.detection_slices
            col_lit_aggregation_audits[qid] = aggregation_plan.audit
            qs.setdefault('events', []).append(
                {
                    'event_type': 'module_a_col_lit_aggregation',
                    'order': order,
                    'turn': turn,
                    'mode': col_lit_aggregation_mode,
                    'component_count': aggregation_plan.audit['component_count'],
                    'prune_eligible_component_count': aggregation_plan.audit[
                        'prune_eligible_component_count'
                    ],
                    'pruning_component_count': aggregation_plan.audit['pruning_component_count'],
                    'singleton_skip_component_ids': aggregation_plan.audit[
                        'singleton_skip_component_ids'
                    ],
                    'detection_slice_count': aggregation_plan.audit['detection_slice_count'],
                }
            )
        question_detection_order[qid] = [slice_info.anchor_key for slice_info in detection_slices]

        if not detection_slices:
            qs['module_a_status'] = 'skipped_no_detection_slices'
            qs.setdefault('events', []).append(
                {
                    'event_type': 'module_a_skipped',
                    'order': order,
                    'turn': turn,
                    'reason': 'no_detection_slices',
                    'detection_mode': detection_mode,
                    'detection_prompt_mode': detection_prompt_mode,
                }
            )
            continue

        # ---- Step 2c: LLM-based ambiguity detection ----
        for slice_info in detection_slices:
            if schema_filter_mode == SCHEMA_FILTER_NONE:
                detection_schema_ddl = schema_ddl
                _, schema_filter_meta = filter_schema_for_detection(
                    qs.get('dbschema', ''),
                    db_path=qs.get('db_file', ''),
                    current_sql=current_sql,
                    prompt_dbelement_options=dbelement_options,
                    mode=schema_filter_mode,
                    sql_columns_by_table=parsed_sql_columns_by_table,
                )
            else:
                filtered_raw_schema_ddl, schema_filter_meta = filter_schema_for_detection(
                    qs.get('dbschema', ''),
                    db_path=qs.get('db_file', ''),
                    current_sql=current_sql,
                    prompt_dbelement_options=slice_info.dbelement_options,
                    mode=schema_filter_mode,
                    sql_columns_by_table=parsed_sql_columns_by_table,
                )
                detection_schema_ddl = enrich_schema_ddl(
                    filtered_raw_schema_ddl,
                    additional_info_by_column=additional_info_by_column,
                    ddl_mode=ddl_mode,
                    subset_only=subset_only,
                    current_sql=current_sql,
                    dbelement_options=slice_info.dbelement_options,
                )
            schema_filter_action = schema_filter_meta.get("filter_action", "unknown")
            schema_filter_action_counts[schema_filter_action] = (
                schema_filter_action_counts.get(schema_filter_action, 0) + 1
            )

            formatted_dbelement_list = format_dbelement_options(
                slice_info.dbelement_options,
                include_column_relation_metadata=include_column_relation_metadata,
                column_group_prompt_format=column_group_prompt_format,
                column_group_version=column_group_version,
                column_descriptive_names=column_descriptive_names,
                column_descrip_mode=column_descrip_mode,
            )
            detection_prompt = build_module_a_detection_prompt(
                question=qs.get('question', ''),
                evidence=qs.get('evidence', ''),
                current_sql=current_sql,
                schema_ddl=detection_schema_ddl,
                formatted_dbelement_list=formatted_dbelement_list,
                column_group_version=column_group_version,
                schema_section_title="Database Schema",
                target_column=None,
                require_retrieved_coverage=False,
            )
            request = BatchRequest(
                request_id=uuid.uuid4().hex,
                question_id=qid,
                internal_index=qs.get('internal_index', 0),
                prompt=detection_prompt,
                stage='module_a_detection',
                turn=turn,
                stop_thinking=False,
                temperature=0.0,
                max_tokens=16384,
                metadata={
                    "module_a_contextualization_mode": contextualization_mode,
                    "module_a_detection_mode": detection_mode,
                    "module_a_detection_prompt_mode": detection_prompt_mode,
                    "module_a_schema_filter_mode": schema_filter_mode,
                    "module_a_column_descrip": column_descrip_mode,
                    "schema_filter_meta": schema_filter_meta,
                    **slice_info.metadata(),
                },
            )
            detection_requests.append(request)
            detection_context_by_request_id[request.request_id] = {
                "qid": qid,
                "order": order,
                "slice": slice_info,
                "schema_filter_meta": schema_filter_meta,
            }

    # ---- Execute LLM detection call ----
    detection_responses = (
        detection_batch_client.batch_generate(detection_requests) if detection_requests else []
    )

    detection_results_by_qid = {
        qid: []
        for qid, anchor_keys in question_detection_order.items()
        if anchor_keys
    }
    detection_failed_qids = set()
    detection_parse_failed_qids = set()

    for req, resp in zip(detection_requests, detection_responses):
        qid = req.question_id
        qs = questions.get(qid)
        if qs is None or qs.get('status') != 'active':
            continue
        context = detection_context_by_request_id.get(req.request_id, {})
        slice_info = context.get("slice")
        schema_filter_meta = context.get("schema_filter_meta", req.metadata.get("schema_filter_meta", {}))
        order = qs.get('order', 0)

        if not resp.ok:
            detection_failed_qids.add(qid)
            qs.setdefault('events', []).append(
                {
                    'event_type': 'module_a_detection_failed',
                    'order': order,
                    'turn': turn,
                    'detection_mode': detection_mode,
                    'detection_prompt_mode': detection_prompt_mode,
                    'module_a_column_descrip': column_descrip_mode,
                    'anchor_key': getattr(slice_info, "anchor_key", None),
                    'schema_filter_meta': schema_filter_meta,
                    'error': resp.error,
                }
            )
            continue

        # ---- Parse detection output ----
        try:
            llm_detection = parse_json_list_response(resp.content)
        except Exception as exc:
            detection_parse_failed_qids.add(qid)
            qs.setdefault('events', []).append(
                {
                    'event_type': 'module_a_detection_parse_failed',
                    'order': order,
                    'turn': turn,
                    'detection_mode': detection_mode,
                    'detection_prompt_mode': detection_prompt_mode,
                    'module_a_column_descrip': column_descrip_mode,
                    'anchor_key': getattr(slice_info, "anchor_key", None),
                    'schema_filter_meta': schema_filter_meta,
                    'error_type': exc.__class__.__name__,
                    'error_message': str(exc),
                    'detection_response': resp.content,
                }
            )
            continue

        qs.setdefault('events', []).append(
            {
                'event_type': 'module_a_detection',
                'order': order,
                'turn': turn,
                'detection_mode': detection_mode,
                'detection_prompt_mode': detection_prompt_mode,
                'module_a_column_descrip': column_descrip_mode,
                'anchor_type': getattr(slice_info, "anchor_type", None),
                'anchor_key': getattr(slice_info, "anchor_key", None),
                'anchor_label': getattr(slice_info, "anchor_label", None),
                'schema_filter_meta': schema_filter_meta,
                'detection_prompt': req.prompt,
                'detection_response': resp.content,
            }
        )

        detection_results_by_qid.setdefault(qid, []).append(
            {
                "request": req,
                "response": resp,
                "slice": slice_info,
                "llm_detection": llm_detection,
            }
        )

    for qid, qs in active.items():
        if qid not in detection_results_by_qid:
            continue
        if qs.get('status') != 'active':
            continue
        order = qs.get('order', 0)

        if qid in detection_failed_qids:
            qs['status'] = 'failed'
            qs['error_stage'] = f'module_a_detection_turn_{turn}'
            qs['error_message'] = 'one or more Module A detection anchor calls failed'
            continue

        if qid in detection_parse_failed_qids:
            qs['status'] = 'failed'
            qs['error_stage'] = f'module_a_detection_parse_turn_{turn}'
            qs['error_message'] = 'one or more Module A detection anchor responses failed to parse'
            continue

        question_results = detection_results_by_qid.get(qid, [])
        merged_raw_detection = []
        for result in question_results:
            merged_raw_detection.extend(result["llm_detection"])
        merged_raw_detection = dedupe_raw_detection_entries(merged_raw_detection)

        # ---- Auto-fill entries ----
        entries = auto_fill_entries(
            llm_output=merged_raw_detection,
            question_id=qid,
            db_id=qs.get('db_id', ''),
        )
        qs['module_a_entries'] = entries
        qs.setdefault('events', []).append(
            {
                'event_type': 'module_a_entry_build',
                'order': order,
                'turn': turn,
                'entry_count': len(entries),
            }
        )
        qs.setdefault('events', []).append(
            {
                'event_type': 'module_a_detection_merge',
                'order': order,
                'turn': turn,
                'contextualization_mode': contextualization_mode,
                'detection_mode': detection_mode,
                'detection_prompt_mode': detection_prompt_mode,
                'anchor_slice_count': len(question_detection_order.get(qid, [])),
                'merged_raw_entry_count': len(merged_raw_detection),
                'entry_count': len(entries),
            }
        )

        if not entries:
            qs['module_a_status'] = 'skipped_no_entries'
            qs.setdefault('events', []).append(
                {
                    'event_type': 'module_a_skipped',
                    'order': order,
                    'turn': turn,
                    'reason': 'no_entries_after_detection',
                }
            )
            continue

        # ---- Step 2d: Render items (deterministic, no LLM call) ----
        rendered_items = render_module_a_items(
            entries, feedback_render_mode=feedback_render_mode,
        )
        qs['module_a_rendered_items'] = rendered_items
        qs.setdefault('events', []).append(
            {
                'event_type': 'module_a_question_render',
                'order': order,
                'turn': turn,
                'render_mode': feedback_render_mode,
                'rendered_count': len(rendered_items),
            }
        )

        # ---- Step 2e: Build feedback prompt ----
        schema_ddl = question_context[qid]["schema_ddl"]
        feedback_prompt = build_module_a_feedback_prompt(
            question=qs.get('question', ''),
            evidence=qs.get('evidence', ''),
            gold_sql=qs.get('gold_sql', ''),
            schema_ddl=schema_ddl,
            rendered_items=rendered_items,
            allow_multi_select=False,
        )
        feedback_requests.append(
            BatchRequest(
                request_id=uuid.uuid4().hex,
                question_id=qid,
                internal_index=qs.get('internal_index', 0),
                prompt=feedback_prompt,
                stage=stage_name,
                turn=turn,
                stop_thinking=False,
                temperature=0.0,
                max_tokens=16384,
                metadata={
                    'schema_ddl': schema_ddl,
                    'module_a_contextualization_mode': contextualization_mode,
                    'direct_render_stats': direct_render_stats_by_qid.get(qid),
                },
            )
        )

    # ---- Step 2f: LLM feedback call ----
    feedback_responses = (
        feedback_batch_client.batch_generate(feedback_requests) if feedback_requests else []
    )
    ok_count = 0
    parse_failed_count = 0

    for req, resp in zip(feedback_requests, feedback_responses):
        qid = req.question_id
        qs = questions.get(qid)
        if qs is None or qs.get('status') != 'active':
            continue
        order = qs.get('order', 0)

        if not resp.ok:
            qs['status'] = 'failed'
            qs['error_stage'] = f'{stage_name}_turn_{turn}'
            qs['error_message'] = resp.error
            continue

        try:
            decisions = parse_json_list_response(resp.content)
        except Exception as exc:
            parse_failed_count += 1
            qs['module_a_status'] = 'feedback_parse_failed'
            qs['module_a_decisions'] = []
            qs['module_a_specs'] = []
            qs.setdefault('module_a_feedback_log', []).append((order, req.prompt, resp.content, resp.perplexity))
            qs.setdefault('events', []).append(
                {
                    'event_type': 'module_a_feedback_parse_failed',
                    'order': order,
                    'turn': turn,
                    'prompt': req.prompt,
                    'response': resp.content,
                    'perplexity': resp.perplexity,
                    'reasoning': resp.reasoning,
                    'usage': resp.usage,
                    'error_type': exc.__class__.__name__,
                    'error_message': str(exc),
                    'llm_request_logs': [resp.request_log_path] if resp.request_log_path else [],
                }
            )
            continue

        qs['module_a_decisions'] = decisions
        qs.setdefault('module_a_feedback_log', []).append((order, req.prompt, resp.content, resp.perplexity))
        qs.setdefault('events', []).append(
            {
                'event_type': 'module_a_feedback_generation',
                'order': order,
                'turn': turn,
                'prompt': req.prompt,
                'response': resp.content,
                'perplexity': resp.perplexity,
                'reasoning': resp.reasoning,
                'usage': resp.usage,
                'llm_request_logs': [resp.request_log_path] if resp.request_log_path else [],
            }
        )

        # ---- Step 2g: Build specs ----
        specs = build_module_a_specs(
            entries=qs.get('module_a_entries', []),
            rendered_items=qs.get('module_a_rendered_items', []),
            decisions=decisions,
        )
        qs['module_a_specs'] = specs
        _append_module_a_specs(qs, specs)
        qs.setdefault('events', []).append(
            {
                'event_type': 'module_a_spec_injection',
                'order': order,
                'turn': turn,
                'spec_count': len(specs),
                'specs': specs,
            }
        )
        qs['module_a_status'] = 'completed'
        qs['order'] = order + 1
        ok_count += 1

    write_response_json(run_dir, stage_name, turn, feedback_requests, feedback_responses)
    write_response_json(
        run_dir,
        'module_a_detection',
        turn,
        detection_requests,
        detection_responses,
    )
    detection_usage_summary = summarize_usage(detection_responses)
    feedback_usage_summary = summarize_usage(feedback_responses)
    direct_render_summary = _summarize_direct_render_stats(direct_render_stats_by_qid)
    stage_payload = {
        "status": "ok",
        "total": len(active),
        "request_count": len(feedback_requests),
        "ok_count": ok_count,
        "parse_failed_count": parse_failed_count,
        "contextualization_mode": contextualization_mode,
        "column_descrip_mode": column_descrip_mode,
        "detection_mode": detection_mode,
        "detection_prompt_mode": detection_prompt_mode,
        "schema_filter_mode": schema_filter_mode,
        "schema_filter_action_counts": schema_filter_action_counts,
        "detection_request_count": len(detection_requests),
        "direct_placeholder_request_count": 0,
        "direct_render_summary": direct_render_summary,
        "detection_question_count": len(question_context),
        "detection_anchor_slice_count": sum(len(keys) for keys in question_detection_order.values()),
        "detection_failed_question_count": len(detection_failed_qids),
        "detection_parse_failed_question_count": len(detection_parse_failed_qids),
        "usage_summary": summarize_usage([*detection_responses, *feedback_responses]),
        "substage_usage_summary": {
            "module_a_detection": detection_usage_summary,
            "module_a_feedback_generation": feedback_usage_summary,
        },
        "timing": stage_timing(timer),
    }
    if col_lit_aggregation_mode is not None:
        col_lit_aggregation_summary = _summarize_col_lit_aggregation(
            col_lit_aggregation_audits
        )
        col_lit_aggregation_artifact = _write_col_lit_aggregation_artifact(
            run_dir,
            stage_name,
            turn,
            mode=col_lit_aggregation_mode,
            audits_by_qid=col_lit_aggregation_audits,
        )
        stage_payload["col_lit_aggregation"] = {
            "mode": col_lit_aggregation_mode,
            "summary": col_lit_aggregation_summary,
            "artifact": col_lit_aggregation_artifact,
        }
    write_stage_json(
        run_dir,
        stage_name,
        turn,
        stage_payload,
    )
    print(f"[MODA] Done turn {turn}: {ok_count} Module A feedback batches, {parse_failed_count} parse failures skipped")
    return _active_questions(questions)


def run_cq_generation(run_dir, questions, batch_client, turn,
                      with_metadata, break_on_no_amb, debug):
    stage_name = "cq_generation"
    if is_stage_complete(run_dir, stage_name, turn):
        print(f"[CQ] round_{turn}/{stage_name} already complete, skipping.")
        return _active_questions(questions)

    timer = Timer(stage_name)
    active = _active_questions(questions)
    requests = []
    for qid, qs in active.items():
        prompt = build_cq_prompt(
            qs['dbschema'], qs['question'], qs.get('query_set', set()),
            qs.get('cqs_and_answers', []), qs.get('evidence', ''),
            with_metadata, break_on_no_amb,
        )
        requests.append(BatchRequest(
            request_id=uuid.uuid4().hex, question_id=qid,
            internal_index=qs.get('internal_index', 0),
            prompt=prompt, stage='cq_gen', turn=turn,
            stop_thinking=False, temperature=0.0, max_tokens=16384,
        ))

    print(f"[CQ] Generating CQ for {len(requests)} questions (turn {turn}) ...")
    responses = batch_client.batch_generate(requests) if requests else []
    ok_count = 0
    no_amb_count = 0

    for req, resp in zip(requests, responses):
        qid = req.question_id
        qs = questions.get(qid)
        if qs is None or qs.get('status') != 'active':
            continue
        order = qs.get('order', 0)

        if not resp.ok:
            qs['status'] = 'failed'
            qs['error_stage'] = f'cq_generation_turn_{turn}'
            qs['error_message'] = resp.error
            continue

        cq = resp.content
        qs.setdefault('cq_log', []).append((order, req.prompt, cq, resp.perplexity))
        qs.setdefault('events', []).append({
            'event_type': 'cq_generation', 'order': order, 'turn': turn,
            'prompt': req.prompt, 'response': cq, 'perplexity': resp.perplexity,
            'reasoning': resp.reasoning, 'usage': resp.usage,
            'llm_request_logs': [resp.request_log_path] if resp.request_log_path else [],
        })
        order += 1

        if "NO AMBIGUITY" in cq:
            qs.setdefault('events', []).append({
                'event_type': 'cq_no_ambiguity', 'order': order - 1, 'turn': turn, 'response': cq,
            })
            qs['status'] = 'completed'
            qs['completed_reason'] = 'no_ambiguity'
            qs['num_cq_asked'] = turn + 1
            qs['final_sql'] = qs.get('sql_log', [[None, None, '', None]])[-1][2]
            no_amb_count += 1
            if debug:
                print(f"[CQ] qid={qid} NO AMBIGUITY")
            continue

        if "mul_choice_cq = " in cq:
            cq = cq.split("mul_choice_cq = ")[-1]

        qs['last_cq'] = cq
        qs['order'] = order
        ok_count += 1

    write_response_json(run_dir, stage_name, turn, requests, responses)
    write_stage_json(run_dir, stage_name, turn, {
        "status": "ok", "total": len(active), "request_count": len(requests),
        "ok_count": ok_count, "no_ambiguity": no_amb_count,
        "usage_summary": summarize_usage(responses),
        "timing": stage_timing(timer),
    })
    print(f"[CQ] Done turn {turn}: {ok_count} CQs, {no_amb_count} NO AMBIGUITY")
    return _active_questions(questions)


def run_feedback_generation(run_dir, questions, batch_client, turn, debug):
    stage_name = "feedback_generation"
    if is_stage_complete(run_dir, stage_name, turn):
        print(f"[FB] round_{turn}/{stage_name} already complete, skipping.")
        return _active_questions(questions)

    timer = Timer(stage_name)
    active = _active_questions(questions)
    requests = []
    for qid, qs in active.items():
        cq = qs.get('last_cq', '')
        if not cq:
            qs['status'] = 'failed'
            qs['error_stage'] = f'feedback_generation_turn_{turn}'
            qs['error_message'] = 'No CQ available'
            continue
        prompt = build_feedback_prompt(qs['gold_sql'], cq, qs['question'])
        requests.append(BatchRequest(
            request_id=uuid.uuid4().hex, question_id=qid,
            internal_index=qs.get('internal_index', 0),
            prompt=prompt, stage='feedback', turn=turn,
            stop_thinking=False, temperature=0.0, max_tokens=16384,
        ))

    print(f"[FB] Generating feedback for {len(requests)} questions (turn {turn}) ...")
    responses = batch_client.batch_generate(requests) if requests else []
    ok_count = 0

    for req, resp in zip(requests, responses):
        qid = req.question_id
        qs = questions.get(qid)
        if qs is None or qs.get('status') != 'active':
            continue
        order = qs.get('order', 0)

        if not resp.ok:
            qs['status'] = 'failed'
            qs['error_stage'] = f'feedback_generation_turn_{turn}'
            qs['error_message'] = resp.error
            continue

        feedback = resp.content
        qs.setdefault('feedback_log', []).append((order, req.prompt, feedback, resp.perplexity))
        qs.setdefault('events', []).append({
            'event_type': 'feedback_generation', 'order': order, 'turn': turn,
            'prompt': req.prompt, 'response': feedback, 'perplexity': resp.perplexity,
            'reasoning': resp.reasoning, 'usage': resp.usage,
            'llm_request_logs': [resp.request_log_path] if resp.request_log_path else [],
        })
        order += 1

        feedback_answer = _extract_sphinteract_feedback_answer(feedback)

        cqs_and_answers = qs.get('cqs_and_answers', [])
        cqs_and_answers.append(qs.get('last_cq', ''))
        cqs_and_answers.append(feedback_answer)
        qs['cqs_and_answers'] = cqs_and_answers
        qs['order'] = order
        ok_count += 1

    write_response_json(run_dir, stage_name, turn, requests, responses)
    write_stage_json(run_dir, stage_name, turn, {
        "status": "ok", "total": len(active), "request_count": len(requests),
        "ok_count": ok_count,
        "usage_summary": summarize_usage(responses),
        "timing": stage_timing(timer),
    })
    print(f"[FB] Done turn {turn}: {ok_count} feedbacks")
    return _active_questions(questions)


def run_sql_regeneration(run_dir, questions, batch_client, turn,
                          sql_gen_few_shot, debug):
    stage_name = "sql_regeneration"
    if is_stage_complete(run_dir, stage_name, turn):
        print(f"[SQL] round_{turn}/{stage_name} already complete, skipping.")
        return _active_questions(questions)

    timer = Timer(stage_name)
    active = _active_questions(questions)
    requests = []
    for qid, qs in active.items():
        if qs.get('module_a_specs'):
            qs.setdefault('events', []).append({
                'event_type': 'module_a_sql_regen_context',
                'order': qs.get('order', 0),
                'turn': turn,
                'spec_count': len(qs.get('module_a_specs', [])),
            })
        cqas = build_cqas_text(qs.get('cqs_and_answers', []), qs.get('evidence', ''), True)
        query_set = qs.get('query_set', set())
        sql_prompt = sql_gen_few_shot.format(
            schema=qs['dbschema'], question=qs['question'],
            sqls=";\n".join(query_set),
            cqas=cqas, metadata=qs.get('evidence', ''),
        )
        sql_prompt = "/* some examples are provided */\n" + sql_prompt
        requests.append(BatchRequest(
            request_id=uuid.uuid4().hex, question_id=qid,
            internal_index=qs.get('internal_index', 0),
            prompt=sql_prompt, stage='sql_gen', turn=turn,
            stop_thinking=True, temperature=0.0, max_tokens=16384,
        ))

    print(f"[SQL] Regenerating SQL for {len(requests)} questions (turn {turn}) ...")
    responses = batch_client.batch_generate(requests) if requests else []
    ok_count = 0

    for req, resp in zip(requests, responses):
        qid = req.question_id
        qs = questions.get(qid)
        if qs is None or qs.get('status') != 'active':
            continue
        order = qs.get('order', 0)

        if not resp.ok:
            qs['status'] = 'failed'
            qs['error_stage'] = f'sql_regeneration_turn_{turn}'
            qs['error_message'] = resp.error
            continue

        sql = clean_query(resp.content)
        qs.setdefault('sql_log', []).append((order, req.prompt, sql, resp.perplexity))
        qs.setdefault('events', []).append({
            'event_type': 'sql_generation', 'order': order, 'turn': turn,
            'prompt': req.prompt, 'response': sql, 'perplexity': resp.perplexity,
            'reasoning': resp.reasoning, 'usage': resp.usage,
            'llm_request_logs': [resp.request_log_path] if resp.request_log_path else [],
        })
        order += 1

        query_set = qs.get('query_set', set())
        query_set.add(sql)
        qs['query_set'] = query_set
        qs['order'] = order
        qs['_last_sql'] = sql
        ok_count += 1

    write_response_json(run_dir, stage_name, turn, requests, responses)
    write_stage_json(run_dir, stage_name, turn, {
        "status": "ok", "total": len(active), "request_count": len(requests),
        "ok_count": ok_count,
        "usage_summary": summarize_usage(responses),
        "timing": stage_timing(timer),
    })
    print(f"[SQL] Done turn {turn}: {ok_count} SQLs")
    return _active_questions(questions)


def run_sql_evaluation(run_dir, questions, data_source, turn, batch_client, dry, debug):
    stage_name = "sql_evaluation"
    if is_stage_complete(run_dir, stage_name, turn):
        print(f"[EVAL] round_{turn}/{stage_name} already complete, skipping.")
        return _active_questions(questions)

    timer = Timer(stage_name)
    active = _active_questions(questions)
    completed = 0
    failed = 0

    for qid, qs in active.items():
        sql = qs.get('_last_sql', '')
        if not sql:
            sl = qs.get('sql_log', [])
            if sl:
                sql = sl[-1][2]
            else:
                qs['status'] = 'failed'
                qs['error_stage'] = f'sql_evaluation_turn_{turn}'
                qs['error_message'] = 'No SQL to evaluate'
                failed += 1
                continue

        eval_timer = Timer("sql_evaluation")
        exception = None
        if dry:
            execution, exception = True, None
            eval_timing = stage_timing(eval_timer, {"status": "dry_run"})
            eval_timing["elapsed_s"] = 0.0
            _append_sql_eval_cpu_step(
                run_dir,
                turn=turn,
                qid=qid,
                qs=qs,
                status="dry_run",
                timing=eval_timing,
                candidate="regen",
                execution=execution,
                exception=exception,
            )
        else:
            try:
                kwargs = {'source': data_source}
                if qs.get('db_file'):
                    kwargs['db_file'] = qs['db_file']
                execution, exception = evalfunc(sql, qs['gold_sql'], qs['db_id'], **kwargs)
                eval_timing = stage_timing(eval_timer, {"status": "ok"})
                _append_sql_eval_cpu_step(
                    run_dir,
                    turn=turn,
                    qid=qid,
                    qs=qs,
                    status="ok",
                    timing=eval_timing,
                    candidate="regen",
                    execution=execution,
                    exception=exception,
                )
            except Exception as exc:
                eval_timing = stage_timing(
                    eval_timer,
                    {"status": "failed", "error": f"{exc.__class__.__name__}: {exc}"},
                )
                _append_sql_eval_cpu_step(
                    run_dir,
                    turn=turn,
                    qid=qid,
                    qs=qs,
                    status="failed",
                    timing=eval_timing,
                    candidate="regen",
                    execution=False,
                    exception=[exc],
                )
                qs['status'] = 'failed'
                qs['error_stage'] = f'sql_evaluation_turn_{turn}'
                qs['error_message'] = str(exc)
                failed += 1
                continue

        order = qs.get('order', 0)
        qs.setdefault('events', []).append({
            'event_type': 'sql_evaluation', 'order': order - 1, 'turn': turn,
            'sql': sql, 'execution': execution,
            'exception': [str(ex) for ex in exception] if exception else [],
            'timing': eval_timing,
        })

        if exception:
            most_recent = clean_query(qs.get('sql_log', [[None, None, '', None]])[-1][2])
            query_set = qs.get('query_set', set())
            if most_recent in query_set:
                query_set.discard(most_recent)
            fix_prompt = build_fix_invalid_prompt(qs['dbschema'], most_recent, str(exception[0]))
            fix_req = BatchRequest(
                request_id=uuid.uuid4().hex, question_id=qid,
                internal_index=qs.get('internal_index', 0),
                prompt=fix_prompt, stage='sql_fix', turn=turn,
                stop_thinking=True, temperature=0.0, max_tokens=16384,
            )
            fix_resp = batch_client.batch_generate([fix_req])[0]
            if fix_resp.ok:
                fixed_sql = clean_query(fix_resp.content)
                query_set.add(fixed_sql)
                qs['query_set'] = query_set
                qs.setdefault('sql_log', []).append((order, fix_prompt, fixed_sql, fix_resp.perplexity))
                qs.setdefault('events', []).append({
                    'event_type': 'sql_fix', 'order': order, 'turn': turn,
                    'prompt': fix_prompt, 'response': fixed_sql,
                    'perplexity': fix_resp.perplexity,
                    'reasoning': fix_resp.reasoning, 'usage': fix_resp.usage,
                    'timing': fix_resp.timing,
                    'llm_request_logs': [fix_resp.request_log_path] if fix_resp.request_log_path else [],
                })
                order += 1
                fix_eval_timer = Timer("sql_evaluation")
                try:
                    kwargs = {'source': data_source}
                    if qs.get('db_file'):
                        kwargs['db_file'] = qs['db_file']
                    execution, fix_exception = evalfunc(fixed_sql, qs['gold_sql'], qs['db_id'], **kwargs)
                    fix_eval_timing = stage_timing(fix_eval_timer, {"status": "ok"})
                    _append_sql_eval_cpu_step(
                        run_dir,
                        turn=turn,
                        qid=qid,
                        qs=qs,
                        status="ok",
                        timing=fix_eval_timing,
                        candidate="fix",
                        execution=execution,
                        exception=fix_exception,
                    )
                except Exception as exc:
                    fix_eval_timing = stage_timing(
                        fix_eval_timer,
                        {"status": "failed", "error": f"{exc.__class__.__name__}: {exc}"},
                    )
                    _append_sql_eval_cpu_step(
                        run_dir,
                        turn=turn,
                        qid=qid,
                        qs=qs,
                        status="failed",
                        timing=fix_eval_timing,
                        candidate="fix",
                        execution=False,
                        exception=[exc],
                    )
                    execution = False

        if execution:
            qs['status'] = 'completed'
            qs['completed_reason'] = 'regen_correct'
            qs['num_cq_asked'] = turn + 1
            qs['final_sql'] = qs.get('sql_log', [[None, None, '', None]])[-1][2]
            completed += 1
            if debug:
                print(f"[EVAL] qid={qid} solved at turn {turn}")
        else:
            qs['order'] = order

    write_stage_json(run_dir, stage_name, turn, {
        "status": "ok", "total": len(active),
        "completed": completed, "failed": failed,
        "timing": stage_timing(timer),
    })
    print(f"[EVAL] Done turn {turn}: {completed} solved, {failed} failed, "
          f"{len(active) - completed - failed} continuing")
    return _active_questions(questions)
