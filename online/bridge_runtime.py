"""Shared runtime contract for the public BRIDGE online entrypoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from online.reformatted_v2_moduleA_v2 import main as internal_main  # noqa: E402


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add portable data, model, and simulated-user arguments."""
    parser.add_argument("--dataset-file", type=Path, required=True, help="BIRD dev JSON file.")
    parser.add_argument("--db-root", type=Path, required=True, help="Directory containing dev_databases/.")
    parser.add_argument("--column-group-root", type=Path, required=True, help="Offline graph archive root containing one DB directory per database.")
    parser.add_argument("--subset-file", type=Path, required=True)
    parser.add_argument("--pred-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--few-shot-store", type=Path, default=None, help="External Chroma store for SQL regeneration examples.")
    parser.add_argument("--model", default=None, help="System model, or MODEL_NAME / CHAT_MODEL_NAME.")
    parser.add_argument("--base-url", default=None, help="System endpoint, or BASE_URL / CHAT_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="System API key, or API_KEY / CHAT_API_KEY.")
    parser.add_argument("--batch-concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="0 runs every question in --subset-file.")
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--user-feedback-model", default=None)
    parser.add_argument("--user-feedback-base-url", default=None)
    parser.add_argument("--user-feedback-api-key", default=None)
    parser.add_argument("--user-feedback-batch-concurrency", type=int, default=None)
    parser.add_argument("--require-user-feedback-27b", action="store_true")
    parser.add_argument("--allow-empty-few-shot-fallback", action="store_true")


def _env_or_argument(value: str | None, *names: str) -> str:
    resolved = value or next((os.getenv(name) for name in names if os.getenv(name)), None)
    if not resolved:
        raise ValueError(f"provide the corresponding CLI argument or set {' or '.join(names)}")
    return resolved


def validate_paths(args: argparse.Namespace) -> None:
    for name in ("dataset_file", "subset_file", "pred_cache"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")
    if not args.db_root.is_dir() or not (args.db_root / "dev_databases").is_dir():
        raise FileNotFoundError("--db-root must exist and contain dev_databases/")
    if not args.column_group_root.is_dir():
        raise FileNotFoundError(f"--column-group-root does not exist: {args.column_group_root}")
    if args.few_shot_store is not None and not args.few_shot_store.is_dir():
        raise FileNotFoundError(f"--few-shot-store does not exist: {args.few_shot_store}")
    if args.batch_concurrency < 1 or args.k_shot < 0 or args.limit < 0:
        raise ValueError("batch concurrency must be positive; k-shot and limit must be non-negative")


def _configure_environment(args: argparse.Namespace) -> str:
    model = _env_or_argument(args.model, "MODEL_NAME", "CHAT_MODEL_NAME")
    base_url = _env_or_argument(args.base_url, "BASE_URL", "CHAT_BASE_URL")
    api_key = args.api_key or os.getenv("API_KEY") or os.getenv("CHAT_API_KEY") or "local-token"
    if args.few_shot_store is None and not args.allow_empty_few_shot_fallback:
        raise ValueError("--few-shot-store is required unless --allow-empty-few-shot-fallback is explicitly selected")
    os.environ.update({
        "MODEL_NAME": model,
        "CHAT_MODEL_NAME": model,
        "CHAT_BASE_URL": base_url,
        "CHAT_API_KEY": api_key,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": api_key,
        "FORCE_DISABLE_THINKING": "1",
    })
    if args.few_shot_store is not None:
        os.environ["USERSTUDY_CHROMA_DIR"] = str(args.few_shot_store)
    return model


def _append_user_feedback_options(forwarded: list[str], args: argparse.Namespace) -> None:
    if args.user_feedback_model or args.user_feedback_base_url:
        if not (args.user_feedback_model and args.user_feedback_base_url):
            raise ValueError("dedicated user feedback requires both model and base URL")
        forwarded.extend(["--user-feedback-model", args.user_feedback_model])
        forwarded.extend(["--user-feedback-base-url", args.user_feedback_base_url])
        if args.user_feedback_api_key:
            forwarded.extend(["--user-feedback-api-key", args.user_feedback_api_key])
        if args.user_feedback_batch_concurrency:
            forwarded.extend(["--user-feedback-batch-concurrency", str(args.user_feedback_batch_concurrency)])
    if args.require_user_feedback_27b:
        forwarded.append("--require-user-feedback-27b")


def build_forwarded_args(
    args: argparse.Namespace,
    *,
    variant: str = "main",
    include_bird_evidence: bool = False,
) -> list[str]:
    """Build the exact internal command for one public, documented variant."""
    model = _configure_environment(args)
    forwarded = [
        "bridge-internal", "--dataset", "bird", "--dataset-file", str(args.dataset_file),
        "--mode", "askClarificationQuestions", "--model", model, "--rounds", "4",
        "--k_shot", str(args.k_shot), "--limit", str(args.limit or 10**9),
        "--subset-file", str(args.subset_file), "--pred-cache", str(args.pred_cache),
        "--out", str(args.out), "--batch-concurrency", str(args.batch_concurrency),
        "--integration-mode", "frontloaded_handoff", "--feedback-render-mode", "AmbiModel_direct",
        "--ddl-mode", "Sphinteract_plain", "--module-a-db-root", str(args.db_root),
        "--module-a-db-mode", "dev", "--module-a-lsh-top-n", "20",
        "--column-group-version", "manual",
        "--column-group-artifact-root", str(args.column_group_root),
        "--module-a-column-group-prompt-format", "manual_target_reason",
        "--module-a-contextualization-mode", "llm_detection", "--module-a-detection-mode", "by_all",
        "--module-a-detection-prompt-mode", "retrieval_elements", "--module-a-schema-filter-mode", "prompt_columns",
        "--module-a-column-descrip", "empty", "--apply-col-lit-aggregation", "prune_combine",
    ]
    if variant == "no_integration":
        start = forwarded.index("--apply-col-lit-aggregation")
        del forwarded[start : start + 2]
    elif variant == "no_graph":
        start = forwarded.index("--module-a-schema-filter-mode")
        forwarded[start + 1] = "none"
        forwarded.extend(["--retrieval-ablation-disable-channel", "column"])
    elif variant == "realtime_retrieval":
        forwarded.extend(["--module-a-column-retrieval-source", "realtime_reasoning"])
    elif variant != "main":
        raise ValueError(f"unsupported public online variant: {variant}")
    if include_bird_evidence:
        forwarded.append("--with_metadata")
    if args.resume_run_dir:
        forwarded.extend(["--resume-run-dir", str(args.resume_run_dir)])
    _append_user_feedback_options(forwarded, args)
    if args.allow_empty_few_shot_fallback:
        forwarded.append("--allow-empty-few-shot-fallback")
    return forwarded


def execute(args: argparse.Namespace, *, variant: str = "main", include_bird_evidence: bool = False) -> None:
    validate_paths(args)
    forwarded = build_forwarded_args(args, variant=variant, include_bird_evidence=include_bird_evidence)
    previous = sys.argv
    try:
        sys.argv = forwarded
        internal_main.main()
    finally:
        sys.argv = previous
