"""
Main entry point for stage-based Sphinteract batch pipeline.
"""
import os, re, sys, json
from datetime import datetime
from pathlib import Path

from .vendor_sphinteract.prompts import sql_generation_v2

from .batch_client import OnlineBatchClient
from .module_a_base.config import (
    COL_LIT_AGGREGATION_MODES,
    DEFAULT_DDL_MODE,
    DEFAULT_FEEDBACK_RENDER_MODE,
    DEFAULT_INTEGRATION_MODE,
    DEFAULT_MODULE_A_COLUMN_DESCRIP,
    DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
    DEFAULT_MODULE_A_CONTEXTUALIZATION_MODE,
    DEFAULT_MODULE_A_DB_ROOT,
    DDL_MODE_CHOICES,
    FEEDBACK_RENDER_MODE_CHOICES,
    INTEGRATION_MODE_CHOICES,
    MODULE_A_CONTEXTUALIZATION_MODE_CHOICES,
    MODULE_A_COLUMN_DESCRIP_CHOICES,
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES,
    MODULE_A_COLUMN_RETRIEVAL_SOURCE_REALTIME_REASONING,
    default_results_root,
)
from .module_a_base.detection_slices import MODULE_A_DETECTION_MODES
from .module_a_base.feedback_prompts import (
    DEFAULT_MODULE_A_DETECTION_PROMPT_MODE,
    MODULE_A_DETECTION_PROMPT_MODES,
)
from .module_a_base.schema_filter import MODULE_A_SCHEMA_FILTER_MODES, SCHEMA_FILTER_NONE
from .module_a_base.retrieval_ablation import (
    DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
    RETRIEVAL_ABLATION_DISABLE_CHANNELS,
    validate_retrieval_ablation_disable_channel,
)
from .module_a_base.sql_regen_template import load_sql_generation_template
from .vendor_pipeline0612.live_retrieval.dbelement_options import (
    COLUMN_GROUP_VERSIONS,
)
from .vendor_pipeline0612.dbelement_formatter import COLUMN_GROUP_PROMPT_FORMATS
from .utils import generate_db_schema, load_pred_cache, first_gold_sql
from .orchestrator import init_question_state, run_pipeline
from .output_formatter import load_question_json
from .resume_support import collect_resume_question_ids, reconstruct_run_state


def _artifact_question_id(data_frame, index):
    try:
        if 'question_id' in data_frame.columns:
            return int(data_frame.iloc[index]['question_id'])
    except Exception:
        pass
    return int(index)


def _first_gold_sql(value):
    if isinstance(value, list):
        return value[0] if value else ''
    return value


def _infer_resume_layout(resume_run_dir):
    if not resume_run_dir:
        return None, None
    run_dir = Path(resume_run_dir).expanduser().resolve()
    mode = run_dir.parent.name if run_dir.parent else None
    dataset_folder = run_dir.parent.parent.name if run_dir.parent and run_dir.parent.parent else None
    dataset = None
    if dataset_folder:
        lowered = dataset_folder.lower()
        if lowered == 'bird':
            dataset = 'bird'
        elif lowered in {'kaggle', 'clambsql'}:
            dataset = lowered
    if mode not in {'baseline', 'askClarificationQuestions', 'askCQsBreakNoAmb'}:
        mode = None
    return mode, dataset


def _rows_by_question_id(df):
    rows = {}
    for idx in range(len(df)):
        qid = _artifact_question_id(df, idx)
        rows[qid] = df.iloc[idx]
    return rows


def _nonempty_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _resolve_db_file_for_question(
    row,
    *,
    dataset: str,
    module_a_db_root: str,
    module_a_db_mode: str,
) -> str:
    explicit_db_file = _nonempty_text(row.get('db_file', ''))
    if explicit_db_file:
        return explicit_db_file

    dbname = _nonempty_text(row.get('db_id', row.get('target_db', '')))
    if (dataset or "").lower() == "bird" and dbname:
        candidate = (
            Path(module_a_db_root).expanduser()
            / f"{module_a_db_mode}_databases"
            / dbname
            / f"{dbname}.sqlite"
        )
        if candidate.is_file():
            return str(candidate)
    return ""


def _ensure_run_dirs(run_dir):
    questions_dir = run_dir / 'questions'
    logs_dir = run_dir / 'logs'
    llm_req_dir = run_dir / 'llm_requests'
    questions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    llm_req_dir.mkdir(parents=True, exist_ok=True)
    return questions_dir, logs_dir, llm_req_dir


def _is_27b_model_name(model_name):
    """Recognize the explicit 27B model names used by guarded sensitivity runs."""
    return bool(re.search(r"(?<!\d)27\s*b(?![a-z0-9])", str(model_name or "").casefold()))


def _optional_positive_int(value, *, name):
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def resolve_user_feedback_client_config(args, *, chat_model, chat_base_url, chat_api_key):
    """Resolve the optional simulator endpoint without exposing API keys in artifacts."""
    model = str(args.user_feedback_model or os.getenv("USER_FEEDBACK_MODEL_NAME") or "").strip()
    base_url = str(args.user_feedback_base_url or os.getenv("USER_FEEDBACK_BASE_URL") or "").strip()
    api_key = str(args.user_feedback_api_key or os.getenv("USER_FEEDBACK_API_KEY") or "").strip()
    raw_concurrency = (
        args.user_feedback_batch_concurrency
        if args.user_feedback_batch_concurrency is not None
        else os.getenv("USER_FEEDBACK_BATCH_CONCURRENCY")
    )
    concurrency = _optional_positive_int(
        raw_concurrency,
        name="--user-feedback-batch-concurrency / USER_FEEDBACK_BATCH_CONCURRENCY",
    )

    has_model_or_url = bool(model or base_url)
    if has_model_or_url and not (model and base_url):
        raise ValueError(
            "a dedicated user-feedback client requires both "
            "--user-feedback-model (or USER_FEEDBACK_MODEL_NAME) and "
            "--user-feedback-base-url (or USER_FEEDBACK_BASE_URL)"
        )
    if api_key and not has_model_or_url:
        raise ValueError(
            "--user-feedback-api-key / USER_FEEDBACK_API_KEY requires a dedicated "
            "user-feedback model and base URL"
        )
    if concurrency is not None and not has_model_or_url:
        raise ValueError(
            "--user-feedback-batch-concurrency / USER_FEEDBACK_BATCH_CONCURRENCY "
            "requires a dedicated user-feedback model and base URL"
        )

    dedicated = bool(model and base_url)
    effective_model = model if dedicated else chat_model
    effective_base_url = base_url if dedicated else chat_base_url
    effective_api_key = api_key or chat_api_key
    effective_concurrency = concurrency if concurrency is not None else args.batch_concurrency
    provider_profile = str(os.getenv("USER_FEEDBACK_PROVIDER_PROFILE", "generic")).strip() or "generic"
    api_timeout_s = str(os.getenv("USER_FEEDBACK_API_TIMEOUT_SECONDS", "600")).strip() or "600"

    if args.require_user_feedback_27b and not _is_27b_model_name(effective_model):
        if dedicated:
            raise ValueError(
                "--require-user-feedback-27b requires the configured user-feedback "
                f"model to be 27B; got {effective_model!r}"
            )
        raise ValueError(
            "--require-user-feedback-27b requires a dedicated 27B user-feedback "
            "model when the global model is not 27B"
        )

    return {
        "dedicated": dedicated,
        "model": effective_model,
        "base_url": effective_base_url,
        "api_key": effective_api_key,
        "concurrency": effective_concurrency,
        "provider_profile": provider_profile,
        "api_timeout_s": api_timeout_s,
    }


def _write_runtime_llm_client_config(run_dir, *, chat_model, chat_base_url, chat_concurrency, feedback):
    """Persist effective routes for later audit without serializing API keys."""
    payload = {
        "system_client": {
            "model": chat_model,
            "base_url": chat_base_url,
            "concurrency": chat_concurrency,
            "provider_profile": str(os.getenv("SYSTEM_LLM_PROVIDER_PROFILE", "generic")),
            "api_timeout_s": str(os.getenv("SYSTEM_LLM_API_TIMEOUT_SECONDS", "600")),
        },
        "user_feedback_client": {
            "dedicated": bool(feedback["dedicated"]),
            "model": feedback["model"],
            "base_url": feedback["base_url"],
            "concurrency": feedback["concurrency"],
            "provider_profile": feedback["provider_profile"],
            "api_timeout_s": feedback["api_timeout_s"],
            "stages": ["module_a_feedback_generation", "feedback"],
        },
    }
    path = Path(run_dir) / "runtime_llm_client_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _load_resumed_questions(run_dir, dataset_rows, args):
    questions_dir = Path(run_dir) / 'questions'
    question_files = sorted(questions_dir.glob('*.json'))
    if not question_files:
        return None

    all_questions = {}
    for index, question_file in enumerate(question_files):
        qstate = load_question_json(question_file)
        qid = int(qstate['question_id'])
        row = dataset_rows.get(qid)
        if row is not None:
            db_file = _resolve_db_file_for_question(
                row,
                dataset=args.dataset,
                module_a_db_root=args.module_a_db_root,
                module_a_db_mode=args.module_a_db_mode,
            )
            qstate['question'] = qstate.get('question') or row.get('question', row.get('nl', ''))
            qstate['db_id'] = qstate.get('db_id') or row.get('db_id', row.get('target_db', ''))
            qstate['gold_sql'] = qstate.get('gold_sql') or _first_gold_sql(row.get('SQL', row.get('gold_query', '')))
            qstate['db_file'] = qstate.get('db_file') or db_file
            qstate['evidence'] = qstate.get('evidence') or row.get('evidence', '')
            if qstate.get('internal_index') is None:
                qstate['internal_index'] = index
        if not qstate.get('dbschema') and qstate.get('db_id') and not args.dry:
            qstate['dbschema'] = generate_db_schema(
                qstate['db_id'],
                source=qstate.get('data_source') or 'bird',
                db_file=qstate.get('db_file') or '',
            )
        all_questions[qid] = qstate
    return all_questions


def _initialize_questions_for_qids(qids, dataset_rows, args):
    all_questions = {}
    for position, qid in enumerate(qids):
        row = dataset_rows.get(qid)
        if row is None:
            raise ValueError(f'question_id {qid} was not found in the {args.dataset} dataset')
        dbname = row.get('db_id', row.get('target_db', ''))
        db_file = _resolve_db_file_for_question(
            row,
            dataset=args.dataset,
            module_a_db_root=args.module_a_db_root,
            module_a_db_mode=args.module_a_db_mode,
        )
        dbschema = generate_db_schema(dbname, source=args.dataset, db_file=db_file) if not args.dry else ''
        all_questions[qid] = init_question_state(
            question_id=qid,
            internal_index=position,
            row=row,
            data_source=args.dataset,
            dbschema=dbschema,
            model_name=args.model,
            db_file=db_file,
            with_metadata=args.with_metadata,
        )
    return all_questions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['askClarificationQuestions'], default='askClarificationQuestions')
    parser.add_argument('--model', default=os.getenv('CHAT_MODEL_NAME', 'Qwen3.5-27B'))
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--k_shot', type=int, default=3)
    parser.add_argument('--with_metadata', action='store_true')
    parser.add_argument('--dataset', choices=['bird', 'kaggle', 'clambsql'], default='bird')
    parser.add_argument(
        '--dataset-file',
        default=None,
        help='Explicit BIRD dataset JSON.',
    )
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--subset-file', help='JSON subset file with question_id values')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--out', default=str(default_results_root()))
    parser.add_argument('--resume-run-dir')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pred-cache', help='JSON file mapping question_id to pre-computed SQL')
    parser.add_argument('--batch-concurrency', type=int, default=5)
    parser.add_argument('--user-feedback-model', default=None)
    parser.add_argument('--user-feedback-base-url', default=None)
    parser.add_argument('--user-feedback-api-key', default=None)
    parser.add_argument('--user-feedback-batch-concurrency', type=int, default=None)
    parser.add_argument(
        '--require-user-feedback-27b',
        action='store_true',
        help=(
            'Fail before API calls unless the effective user-feedback simulator '
            'uses a model name containing 27B. Use for 4B/9B sensitivity runs.'
        ),
    )
    parser.add_argument('--integration-mode', choices=INTEGRATION_MODE_CHOICES, default=DEFAULT_INTEGRATION_MODE)
    parser.add_argument('--feedback-render-mode', choices=FEEDBACK_RENDER_MODE_CHOICES, default=DEFAULT_FEEDBACK_RENDER_MODE)
    parser.add_argument('--ddl-mode', choices=DDL_MODE_CHOICES, default=DEFAULT_DDL_MODE)
    parser.add_argument('--subset-only', action='store_true')
    parser.add_argument(
        '--module-a-db-root',
        default=DEFAULT_MODULE_A_DB_ROOT,
        help='Root containing dev_databases/ for Module A live retrieval',
    )
    parser.add_argument('--module-a-db-mode', default='dev')
    parser.add_argument('--module-a-lsh-top-n', type=int, default=20)
    parser.add_argument(
        '--module-a-column-retrieval-source',
        choices=MODULE_A_COLUMN_RETRIEVAL_SOURCE_CHOICES,
        default=DEFAULT_MODULE_A_COLUMN_RETRIEVAL_SOURCE,
        help=(
            'Source of Module A COLUMN candidates. column_group preserves the '
            'offline graph lookup; realtime_reasoning replaces only that lookup '
            'with one full-schema LLM request per parsed SQL COLUMN anchor.'
        ),
    )
    parser.add_argument('--column-group-version', choices=COLUMN_GROUP_VERSIONS, default='manual')
    parser.add_argument(
        '--column-group-artifact-root',
        default=None,
        help='External root containing the public manual graph archive.',
    )
    parser.add_argument(
        '--module-a-column-group-prompt-format',
        choices=COLUMN_GROUP_PROMPT_FORMATS,
        default='manual_target_reason',
        help='Fixed public rendering for graph candidates.',
    )
    parser.add_argument(
        '--module-a-contextualization-mode',
        choices=MODULE_A_CONTEXTUALIZATION_MODE_CHOICES,
        default=DEFAULT_MODULE_A_CONTEXTUALIZATION_MODE,
        help='Use the public LLM contextualization route.',
    )
    parser.add_argument(
        '--module-a-detection-mode',
        choices=('by_all',),
        default='by_all',
        help='Use one Module A detection request per question.',
    )
    parser.add_argument(
        '--apply-col-lit-aggregation',
        '--apply-ColLit-aggregation',
        dest='module_a_col_lit_aggregation',
        choices=COL_LIT_AGGREGATION_MODES,
        default=None,
        help='Apply the public COLUMN-literal integration policy.',
    )
    parser.add_argument(
        '--module-a-detection-prompt-mode',
        choices=MODULE_A_DETECTION_PROMPT_MODES,
        default=DEFAULT_MODULE_A_DETECTION_PROMPT_MODE,
        help='Use retrieved DB elements as Module A detection context.',
    )
    parser.add_argument(
        '--module-a-schema-filter-mode',
        choices=MODULE_A_SCHEMA_FILTER_MODES,
        default=SCHEMA_FILTER_NONE,
        help=(
            'Optional schema pruning for Module A detection prompts. none keeps '
            'the full schema; prompt_columns keeps current-SQL columns, global '
            'PK/FK/UNIQUE columns, and columns mentioned by DB elements in the '
            'current detection prompt slice.'
        ),
    )
    parser.add_argument('--module-a-disable-llm-fallback', action='store_true')
    parser.add_argument(
        '--module-a-column-descrip',
        choices=MODULE_A_COLUMN_DESCRIP_CHOICES,
        default=DEFAULT_MODULE_A_COLUMN_DESCRIP,
        help='Use table.column presentation in Module A prompts.',
    )
    parser.add_argument(
        '--retrieval-ablation-disable-channel',
        choices=RETRIEVAL_ABLATION_DISABLE_CHANNELS,
        default=DEFAULT_RETRIEVAL_ABLATION_DISABLE_CHANNEL,
        help=(
            'Disable one retrieval candidate channel for an ablation: column skips '
            'COLUMN-group candidate generation. SQL anchors remain available in '
            'additive retrieval audit metadata.'
        ),
    )
    parser.add_argument('--allow-empty-few-shot-fallback', action='store_true')
    args = parser.parse_args()
    try:
        args.retrieval_ablation_disable_channel = validate_retrieval_ablation_disable_channel(
            args.retrieval_ablation_disable_channel
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.module_a_column_retrieval_source == MODULE_A_COLUMN_RETRIEVAL_SOURCE_REALTIME_REASONING:
        if args.retrieval_ablation_disable_channel == 'column':
            parser.error(
                'realtime COLUMN retrieval is incompatible with '
                '--retrieval-ablation-disable-channel column'
            )

    resume_mode, resume_dataset = _infer_resume_layout(args.resume_run_dir)
    if resume_mode:
        args.mode = resume_mode
    if resume_dataset:
        args.dataset = resume_dataset

    debug_enabled = args.debug or os.getenv('RUNNER_DEBUG', '').lower() in ('1', 'true', 'yes')

    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir).expanduser().resolve()
        if not run_dir.exists():
            parser.error(f'--resume-run-dir does not exist: {run_dir}')
    else:
        dataset_folder = args.dataset.upper() if args.dataset.lower() == 'bird' else args.dataset
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = Path(args.out).expanduser().resolve() / dataset_folder / args.mode / timestamp

    questions_dir, logs_dir, llm_req_dir = _ensure_run_dirs(run_dir)
    os.environ['LLM_REQUEST_LOG_DIR'] = str(llm_req_dir)
    if debug_enabled:
        os.environ['RUNNER_DEBUG'] = '1'

    # ---- Load dataset ----
    if args.dataset == 'bird':
        df_paths = ([args.dataset_file] if args.dataset_file else []) + [
            './dataset/BIRD/dev.json',
            os.path.join(os.path.dirname(__file__), '..', 'dataset', 'BIRD', 'dev.json'),
        ]
        df = None
        for p in df_paths:
            if os.path.exists(p):
                import pandas as pd
                df = pd.read_json(p)
                df = df.copy()
                if 'SQL' in df.columns:
                    df['SQL'] = df['SQL'].apply(_first_gold_sql)
                df.reset_index(level=0, inplace=True)
                break
        if df is None:
            print("ERROR: Cannot find BIRD dev.json")
            sys.exit(1)
    elif args.dataset == 'clambsql':
        import pandas as pd
        df = pd.read_json('./dataset/CLAMBSQL/clambsql.json')
        if 'question_id' not in df.columns:
            if 'index' not in df.columns:
                raise ValueError('CLAMBSQL requires question_id or index column')
            df['question_id'] = df['index'].astype(int)
        else:
            df['question_id'] = df['question_id'].astype(int)
        df = df.sort_values('question_id').reset_index(drop=True)
    else:
        import pandas as pd
        if os.path.exists('kaggle_dataset.csv'):
            df = pd.read_csv('kaggle_dataset.csv')
        else:
            print("ERROR: Kaggle dataset not found")
            sys.exit(1)

    dataset_rows = _rows_by_question_id(df)

    # ---- Apply subset ----
    if not args.resume_run_dir:
        if args.subset_file:
            with open(args.subset_file) as f:
                subset_rows = json.load(f)
            subset_ids = []
            for item in subset_rows:
                if isinstance(item, dict) and 'question_id' in item:
                    subset_ids.append(int(item['question_id']))
                else:
                    subset_ids.append(int(item))
            subset_order = {qid: pos for pos, qid in enumerate(subset_ids)}
            if 'question_id' not in df.columns:
                raise ValueError('--subset-file requires a question_id column')
            df = df[df['question_id'].astype(int).isin(subset_order)].copy()
            df['_subset_order'] = df['question_id'].astype(int).map(subset_order)
            df = df.sort_values('_subset_order').drop(columns=['_subset_order']).reset_index(drop=True)

        if args.start:
            df = df.iloc[max(args.start, 0):].reset_index(drop=True)

        sample_limit = min(max(args.limit, 0), len(df))
        df = df.iloc[:sample_limit].reset_index(drop=True)

    # ---- Load pred_cache ----
    pred_cache = {}
    if args.pred_cache:
        pred_cache = load_pred_cache(args.pred_cache)
        print(f"[MAIN] Loaded {len(pred_cache)} cached predictions from {args.pred_cache}")

    print(f"[MAIN] Module A live retrieval DB root: {args.module_a_db_root}")

    # ---- Init batch clients ----
    chat_api_key = os.getenv("CHAT_API_KEY") or "local-token"
    chat_base_url = os.getenv("CHAT_BASE_URL") or "http://127.0.0.1:3005/v1"
    chat_model = os.getenv("CHAT_MODEL_NAME") or args.model
    try:
        user_feedback_config = resolve_user_feedback_client_config(
            args,
            chat_model=chat_model,
            chat_base_url=chat_base_url,
            chat_api_key=chat_api_key,
        )
    except ValueError as exc:
        parser.error(str(exc))
    _write_runtime_llm_client_config(
        run_dir,
        chat_model=chat_model,
        chat_base_url=chat_base_url,
        chat_concurrency=args.batch_concurrency,
        feedback=user_feedback_config,
    )

    if not args.dry:

        batch_client_sql = OnlineBatchClient(
            model_name=chat_model,
            base_url=chat_base_url,
            api_key=chat_api_key,
            temperature=0.0, max_tokens=16384,
            concurrency=args.batch_concurrency,
            provider_profile=os.getenv("SYSTEM_LLM_PROVIDER_PROFILE", "generic"),
            api_timeout_s=os.getenv("SYSTEM_LLM_API_TIMEOUT_SECONDS", "600"),
        )
        batch_client_cq = batch_client_sql
        if user_feedback_config["dedicated"]:
            batch_client_feedback = OnlineBatchClient(
                model_name=user_feedback_config["model"],
                base_url=user_feedback_config["base_url"],
                api_key=user_feedback_config["api_key"],
                temperature=0.0,
                max_tokens=16384,
                concurrency=user_feedback_config["concurrency"],
                provider_profile=user_feedback_config["provider_profile"],
                api_timeout_s=user_feedback_config["api_timeout_s"],
            )
        else:
            batch_client_feedback = batch_client_cq
        print(f"[MAIN] System OnlineBatchClient: {chat_model} @ {chat_base_url}")
        print(
            "[MAIN] User-feedback OnlineBatchClient: "
            f"{user_feedback_config['model']} @ {user_feedback_config['base_url']} "
            f"(dedicated={user_feedback_config['dedicated']})"
        )
    else:
        batch_client_sql = None
        batch_client_cq = None
        batch_client_feedback = None

    # ---- Initialize question states ----
    resumed_from_question_files = False
    if args.resume_run_dir:
        all_questions = _load_resumed_questions(run_dir, dataset_rows, args)
        if all_questions:
            resumed_from_question_files = True
        else:
            qids = collect_resume_question_ids(run_dir)
            if not qids:
                parser.error(f'Could not infer question_ids from resume directory: {run_dir}')
            all_questions = _initialize_questions_for_qids(qids, dataset_rows, args)
            reconstruct_run_state(
                run_dir=run_dir,
                questions=all_questions,
                data_source=args.dataset,
                dry=args.dry,
                pred_cache=pred_cache,
                batch_client_sql=batch_client_sql,
                debug=debug_enabled,
            )
    else:
        all_questions = {}
        for idx in range(len(df)):
            row = df.iloc[idx]
            qid = _artifact_question_id(df, idx)
            dbname = row.get('db_id', row.get('target_db', ''))
            db_file = _resolve_db_file_for_question(
                row,
                dataset=args.dataset,
                module_a_db_root=args.module_a_db_root,
                module_a_db_mode=args.module_a_db_mode,
            )
            dbschema = generate_db_schema(dbname, source=args.dataset, db_file=db_file) if not args.dry else ''

            qs = init_question_state(
                question_id=qid,
                internal_index=idx,
                row=row,
                data_source=args.dataset,
                dbschema=dbschema,
                model_name=args.model,
                db_file=db_file,
                with_metadata=args.with_metadata,
            )

            all_questions[qid] = qs

    print(f"[MAIN] Initialized {len(all_questions)} questions, output -> {run_dir}")
    if args.resume_run_dir:
        source = 'question checkpoints' if resumed_from_question_files else 'stage artifacts'
        print(f"[MAIN] Resumed state from {source}: {run_dir}")

    # ---- Build feedback few-shot prompt (for SQL regeneration stages) ----
    sql_gen_few_shot = None
    if args.mode in ('askClarificationQuestions', 'askCQsBreakNoAmb'):
        try:
            embedding_api_key = os.getenv("EMBEDDING_API_KEY") or "EMPTY"
            embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or "http://127.0.0.1:8001/v1"
            embedding_model = os.getenv("EMBEDDING_MODEL_NAME") or "Qwen3-Embedding-0.6B"
            persist_dir = os.getenv("USERSTUDY_CHROMA_DIR") or "./userstudy_chroma_qwen0.6"
            sql_gen_few_shot = load_sql_generation_template(
                k_shot=args.k_shot,
                embedding_api_key=embedding_api_key,
                embedding_base_url=embedding_base_url,
                embedding_model=embedding_model,
                persist_dir=persist_dir,
                allow_empty_few_shot_fallback=args.allow_empty_few_shot_fallback,
                force_zero_shot=args.allow_empty_few_shot_fallback,
            )
            if args.allow_empty_few_shot_fallback:
                print(f"[MAIN] SQL regeneration template prepared (vectorstore or zero-shot fallback): {persist_dir}")
            else:
                print(f"[MAIN] Vectorstore loaded: {persist_dir}")
        except Exception as exc:
            print(f"[MAIN] WARNING: Could not load vectorstore: {exc}")
            if args.mode != 'baseline':
                print("[MAIN] Clarification modes require langchain + Chroma + embeddings")
                sys.exit(1)

    # ---- Run pipeline ----
    break_on_no_amb = (args.mode == 'askCQsBreakNoAmb')
    run_pipeline(
        run_dir=str(run_dir),
        all_questions=all_questions,
        batch_client_sql=batch_client_sql,
        batch_client_cq=batch_client_cq,
        batch_client_feedback=batch_client_feedback,
        rounds=args.rounds,
        data_source=args.dataset,
        k_shot=args.k_shot,
        with_metadata=args.with_metadata,
        break_on_no_amb=break_on_no_amb,
        sql_gen_few_shot=sql_gen_few_shot,
        dry=args.dry,
        debug=debug_enabled,
        pred_cache=pred_cache,
        integration_mode=args.integration_mode,
        module_a_db_root=args.module_a_db_root,
        module_a_db_mode=args.module_a_db_mode,
        module_a_lsh_top_n=args.module_a_lsh_top_n,
        module_a_column_retrieval_source=args.module_a_column_retrieval_source,
        module_a_enable_llm_fallback=not args.module_a_disable_llm_fallback,
        module_a_column_group_version=args.column_group_version,
        module_a_column_group_artifact_root=args.column_group_artifact_root,
        module_a_column_group_prompt_format=args.module_a_column_group_prompt_format,
        module_a_contextualization_mode=args.module_a_contextualization_mode,
        module_a_detection_mode=args.module_a_detection_mode,
        module_a_detection_prompt_mode=args.module_a_detection_prompt_mode,
        module_a_schema_filter_mode=args.module_a_schema_filter_mode,
        module_a_col_lit_aggregation=args.module_a_col_lit_aggregation,
        retrieval_ablation_disable_channel=args.retrieval_ablation_disable_channel,
        module_a_include_column_relation_metadata=False,
        module_a_column_descrip=args.module_a_column_descrip,
        feedback_render_mode=args.feedback_render_mode,
        ddl_mode=args.ddl_mode,
        subset_only=args.subset_only,
    )

    print(f"[MAIN] Results: {run_dir}/run_summary.json")


if __name__ == '__main__':
    main()
