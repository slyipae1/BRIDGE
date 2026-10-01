"""
Output formatter — writes per-question JSON, logs, and run_summary.
Format matches old Sphinteract baseline exactly.
"""
import json, os
from pathlib import Path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=str)]
    import numpy as np
    if isinstance(value, (np.generic, np.ndarray)):
        return value.item() if isinstance(value, np.generic) else value.tolist()
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def write_question_json(questions_dir, question_id, qstate):
    """Write questions/{qid}.json in the same format as old Sphinteract."""
    os.makedirs(questions_dir, exist_ok=True)
    path = os.path.join(questions_dir, f'{int(question_id):04d}.json')

    module_a_state = {
        'entries': qstate.get('module_a_entries', []),
        'rendered_items': qstate.get('module_a_rendered_items', []),
        'feedback_log': qstate.get('module_a_feedback_log', []),
        'specs': qstate.get('module_a_specs', []),
        'decisions': qstate.get('module_a_decisions', []),
        'retrieval': qstate.get('module_a_retrieval'),
        'status': qstate.get('module_a_status', 'not_run'),
        'mode': qstate.get('module_a_mode', ''),
        'integration_mode': qstate.get('module_a_integration_mode', ''),
        'ddl_mode': qstate.get('module_a_ddl_mode', ''),
        'subset_only': bool(qstate.get('module_a_subset_only', False)),
        'column_descrip': qstate.get('module_a_column_descrip', ''),
        'contextualization_mode': qstate.get('module_a_contextualization_mode', ''),
        'detection_mode': qstate.get('module_a_detection_mode', ''),
        'detection_prompt_mode': qstate.get('module_a_detection_prompt_mode', ''),
        'schema_filter_mode': qstate.get('module_a_schema_filter_mode', 'none'),
    }
    # ADD: Preserve aggregation provenance only for an opt-in run; ordinary
    # question artifact shapes stay byte-for-byte compatible at this level.
    if 'module_a_col_lit_aggregation' in qstate:
        module_a_state['col_lit_aggregation'] = qstate['module_a_col_lit_aggregation']

    record = {
        'question_id': question_id,
        'internal_index': qstate.get('internal_index'),
        'question': qstate.get('question', ''),
        'db_id': qstate.get('db_id', ''),
        'gold_sql': qstate.get('gold_sql', ''),
        'data_source': qstate.get('data_source', ''),
        'model': qstate.get('model', ''),
        'events': qstate.get('events', []),
        'sql_log': qstate.get('sql_log', []),
        'cq_log': qstate.get('cq_log', []),
        'feedback_log': qstate.get('feedback_log', []),
        'num_cq_asked': qstate.get('num_cq_asked', 'Failed'),
        'final_sql': qstate.get('final_sql', ''),
        'run_status': qstate.get('status', 'incomplete'),
        'dbschema': qstate.get('dbschema', ''),
        'evidence': qstate.get('evidence', ''),
        'db_file': qstate.get('db_file', ''),
        'resume_state': {
            'order': qstate.get('order', 0),
            'query_set': list(sorted(qstate.get('query_set', set()), key=str)),
            'cqs_and_answers': qstate.get('cqs_and_answers', []),
            'last_cq': qstate.get('last_cq', ''),
            'last_sql': qstate.get('_last_sql', ''),
            'completed_reason': qstate.get('completed_reason'),
        },
        'module_a_state': module_a_state,
    }
    # Include error info if present
    for k in ('error_stage', 'error_type', 'error_message', 'error_traceback'):
        if qstate.get(k):
            record[k] = qstate[k]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(record), f, ensure_ascii=False, indent=2)
        f.write('\n')


def load_question_json(path):
    """Load questions/{qid}.json and restore resume-critical fields."""
    with open(path, 'r', encoding='utf-8') as f:
        record = json.load(f)

    resume_state = record.get('resume_state', {})
    module_a_state = record.get('module_a_state', {})
    state = {
        'status': record.get('run_status', 'incomplete'),
        'internal_index': record.get('internal_index'),
        'question_id': record.get('question_id'),
        'question': record.get('question', ''),
        'db_id': record.get('db_id', ''),
        'gold_sql': record.get('gold_sql', ''),
        'data_source': record.get('data_source', ''),
        'dbschema': record.get('dbschema', ''),
        'evidence': record.get('evidence', ''),
        'db_file': record.get('db_file', ''),
        'model': record.get('model', ''),
        'events': record.get('events', []),
        'sql_log': record.get('sql_log', []),
        'cq_log': record.get('cq_log', []),
        'feedback_log': record.get('feedback_log', []),
        'num_cq_asked': record.get('num_cq_asked', 'Failed'),
        'final_sql': record.get('final_sql', ''),
        'completed_reason': resume_state.get('completed_reason'),
        'error_stage': record.get('error_stage'),
        'error_type': record.get('error_type'),
        'error_message': record.get('error_message'),
        'order': int(resume_state.get('order', 0) or 0),
        'query_set': set(resume_state.get('query_set', [])),
        'cqs_and_answers': list(resume_state.get('cqs_and_answers', [])),
        'last_cq': resume_state.get('last_cq', ''),
        '_last_sql': resume_state.get('last_sql', ''),
        'module_a_entries': list(module_a_state.get('entries', [])),
        'module_a_rendered_items': list(module_a_state.get('rendered_items', [])),
        'module_a_feedback_log': list(module_a_state.get('feedback_log', [])),
        'module_a_specs': list(module_a_state.get('specs', [])),
        'module_a_decisions': list(module_a_state.get('decisions', [])),
        'module_a_retrieval': module_a_state.get('retrieval'),
        'module_a_status': module_a_state.get('status', 'not_run'),
        'module_a_mode': module_a_state.get('mode', ''),
        'module_a_integration_mode': module_a_state.get('integration_mode', ''),
        'module_a_ddl_mode': module_a_state.get('ddl_mode', ''),
        'module_a_subset_only': bool(module_a_state.get('subset_only', False)),
        'module_a_column_descrip': module_a_state.get('column_descrip', ''),
        'module_a_contextualization_mode': module_a_state.get('contextualization_mode', ''),
        'module_a_detection_mode': module_a_state.get('detection_mode', ''),
        'module_a_detection_prompt_mode': module_a_state.get('detection_prompt_mode', ''),
        'module_a_schema_filter_mode': module_a_state.get('schema_filter_mode', 'none'),
    }
    if 'col_lit_aggregation' in module_a_state:
        state['module_a_col_lit_aggregation'] = module_a_state['col_lit_aggregation']
    return state


def flush_question_json(run_dir, question_id, qstate):
    write_question_json(Path(run_dir) / 'questions', question_id, qstate)


def flush_all_question_jsons(run_dir, questions):
    questions_dir = Path(run_dir) / 'questions'
    for qid, qstate in questions.items():
        write_question_json(questions_dir, qid, qstate)


def append_log_pair(question_id, logs_dir, prompt_label, prompt_text,
                    response_label, response_text, debug_print=False):
    """Append to logs/{qid}.log — matches old format."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f'{int(question_id):04d}.log'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"============== Prompt: {prompt_label} ===============\n")
        f.write(str(prompt_text).rstrip() + "\n")
        f.write(f"============== LLM Response: {response_label} ===============\n")
        f.write(str(response_text).rstrip() + "\n\n\n")
    if debug_print:
        print(f"[DEBUG] Wrote log pair for question {question_id} -> {path}")


def append_log_section(question_id, logs_dir, section_label, section_text, debug_print=False):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f'{int(question_id):04d}.log'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"============== {section_label} ===============\n")
        f.write(str(section_text).rstrip() + "\n\n\n")
    if debug_print:
        print(f"[DEBUG] Wrote log section for question {question_id} -> {path}")


def finalize_run_summary(run_dir, question_summaries, metrics_summary=None):
    """Write run_summary.json sorted by question_id."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    question_summaries.sort(key=lambda item: int(item['question_id']))
    summary = {
        'run_dir': str(run_dir),
        'num_questions': len(question_summaries),
        'questions': question_summaries,
    }
    if metrics_summary is not None:
        summary['metrics_summary'] = metrics_summary
    path = run_dir / 'run_summary.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
        f.write('\n')
    return str(path)


def make_summary_entry(question_id, qstate):
    """Create a run_summary entry dict from a question state."""
    nq = qstate.get('num_cq_asked', 'Failed')
    status = qstate.get('status', 'incomplete')
    return {
        'question_id': int(question_id),
        'internal_index': qstate.get('internal_index', 0),
        'question_json': '',
        'question_log': '',
        'num_cq_asked': nq,
        'final_sql': qstate.get('final_sql', ''),
        'status': 'complete' if status == 'completed' else 'error' if status == 'failed' else status,
        'error_stage': qstate.get('error_stage'),
        'error_type': qstate.get('error_type'),
    }
