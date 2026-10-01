#!/usr/bin/env bash
# Construct BRIDGE graph and value-retrieval artifacts for one or more BIRD DBs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_offline.sh --db-root PATH --output-root PATH --db-id ID [--db-id ID ...] [options]

Options:
  --verification-universe proposal_union|all_pairs   Default: proposal_union
  --disable-proposal-view VIEW                       schema_semantic_similarity,
                                                     surface_name_similarity, or value_domain_collision
  --verifier-concurrency N                           Default: 1
  --embedding-batch-size N                           Default: 32
  --allow-paid-api                                   Permit an explicitly configured paid verifier endpoint
EOF
}

DB_ROOT=""
OUTPUT_ROOT=""
DB_IDS=()
VERIFICATION_UNIVERSE="proposal_union"
DISABLE_PROPOSAL_VIEW=""
VERIFIER_CONCURRENCY=1
EMBEDDING_BATCH_SIZE=32
ALLOW_PAID_API=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db-root) DB_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --db-id) DB_IDS+=("$2"); shift 2 ;;
    --verification-universe) VERIFICATION_UNIVERSE="$2"; shift 2 ;;
    --disable-proposal-view) DISABLE_PROPOSAL_VIEW="$2"; shift 2 ;;
    --verifier-concurrency) VERIFIER_CONCURRENCY="$2"; shift 2 ;;
    --embedding-batch-size) EMBEDDING_BATCH_SIZE="$2"; shift 2 ;;
    --allow-paid-api) ALLOW_PAID_API=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${DB_ROOT}" && -n "${OUTPUT_ROOT}" && ${#DB_IDS[@]} -gt 0 ]] || { usage >&2; exit 2; }
[[ -n "${MODEL_NAME:-}" && -n "${BASE_URL:-}" ]] || { echo "MODEL_NAME and BASE_URL must be set in .env or the environment." >&2; exit 2; }
[[ -n "${EMBEDDING_MODEL_NAME:-}" && -n "${EMBEDDING_BASE_URL:-}" ]] || { echo "EMBEDDING_MODEL_NAME and EMBEDDING_BASE_URL must be set in .env or the environment." >&2; exit 2; }

VERIFIER_PAID_ARGS=()
if [[ "${ALLOW_PAID_API}" == "1" ]]; then
  VERIFIER_PAID_ARGS+=(--allow-paid-api)
fi
QUEUE_VARIANT_ARGS=(--verification-universe "${VERIFICATION_UNIVERSE}")
if [[ -n "${DISABLE_PROPOSAL_VIEW}" ]]; then
  QUEUE_VARIANT_ARGS+=(--disable-proposal-view "${DISABLE_PROPOSAL_VIEW}")
fi

for DB_ID in "${DB_IDS[@]}"; do
  DB_DIR="${DB_ROOT}/${DB_ID}"
  SQLITE_PATH="${DB_DIR}/${DB_ID}.sqlite"
  WORK_DIR="${OUTPUT_ROOT}/work/${DB_ID}"
  PROFILE_DIR="${WORK_DIR}/profile"
  SCHEMA_CACHE_DIR="${WORK_DIR}/embeddings/schema_semantic"
  SURFACE_CACHE_DIR="${WORK_DIR}/embeddings/surface_name"
  CANDIDATE_DIR="${WORK_DIR}/candidate_views"
  QUEUE_DIR="${WORK_DIR}/verifier_queue"
  VERIFIER_DIR="${WORK_DIR}/verifier"

  [[ -f "${SQLITE_PATH}" ]] || { echo "Missing SQLite database: ${SQLITE_PATH}" >&2; exit 1; }
  echo "[BRIDGE offline] ${DB_ID}"

  "${PYTHON_BIN}" "${ROOT}/offline/scripts/generate_descriptive_names.py" \
    --db-dir "${DB_DIR}" --resume --model "${MODEL_NAME}" --base-url "${BASE_URL}" --api-key "${API_KEY:-local-token}"
  "${PYTHON_BIN}" "${ROOT}/offline/scripts/build_db_schema_profile.py" \
    --db-id "${DB_ID}" --db-dir "${DB_DIR}" --sqlite "${SQLITE_PATH}" --output-dir "${PROFILE_DIR}"
  "${PYTHON_BIN}" "${ROOT}/offline/scripts/build_embedding_caches.py" \
    --profile "${PROFILE_DIR}/schema_profile.json" --proposal-view schema_semantic --output-dir "${SCHEMA_CACHE_DIR}" \
    --embedding-model "${EMBEDDING_MODEL_NAME}" --embedding-base-url "${EMBEDDING_BASE_URL}" --embedding-api-key "${EMBEDDING_API_KEY:-local-token}" --batch-size "${EMBEDDING_BATCH_SIZE}"
  "${PYTHON_BIN}" "${ROOT}/offline/scripts/build_embedding_caches.py" \
    --profile "${PROFILE_DIR}/schema_profile.json" --proposal-view surface_name --output-dir "${SURFACE_CACHE_DIR}" \
    --embedding-model "${EMBEDDING_MODEL_NAME}" --embedding-base-url "${EMBEDDING_BASE_URL}" --embedding-api-key "${EMBEDDING_API_KEY:-local-token}" --batch-size "${EMBEDDING_BATCH_SIZE}"
  "${PYTHON_BIN}" "${ROOT}/offline/scripts/materialize_candidate_views.py" \
    --db-id "${DB_ID}" --schema-profile "${PROFILE_DIR}/schema_profile.json" --sqlite-path "${SQLITE_PATH}" \
    --schema-cache-dir "${SCHEMA_CACHE_DIR}" --surface-cache-dir "${SURFACE_CACHE_DIR}" \
    --channel-config "${ROOT}/offline/configs/candidate_channels_mainline_v1.json" --output-dir "${CANDIDATE_DIR}"
  "${PYTHON_BIN}" "${ROOT}/offline/scripts/build_candidate_verifier_queue.py" \
    --db-id "${DB_ID}" --schema-profile "${PROFILE_DIR}/schema_profile.json" --sqlite-path "${SQLITE_PATH}" \
    --candidate-artifact "schema_embedding=${CANDIDATE_DIR}/schema_embedding/candidate_pairs.jsonl" \
    --candidate-artifact "lexical=${CANDIDATE_DIR}/lexical/candidate_pairs.jsonl" \
    --candidate-artifact "surface_name_embedding=${CANDIDATE_DIR}/surface_name_embedding/candidate_pairs.jsonl" \
    --candidate-artifact "value_collision=${CANDIDATE_DIR}/value_collision/candidate_pairs.jsonl" \
    --output-dir "${QUEUE_DIR}" "${QUEUE_VARIANT_ARGS[@]}"
  "${PYTHON_BIN}" "${ROOT}/offline/scripts/run_pair_verifier.py" \
    --queue-path "${QUEUE_DIR}/queue.jsonl" --bundles-path "${QUEUE_DIR}/candidate_bundles.jsonl" --output-dir "${VERIFIER_DIR}" \
    --model "${MODEL_NAME}" --base-url "${BASE_URL}" --api-key "${API_KEY:-local-token}" --max-concurrency "${VERIFIER_CONCURRENCY}" "${VERIFIER_PAID_ARGS[@]}"
  "${PYTHON_BIN}" "${ROOT}/offline/scripts/compile_verified_graph.py" \
    --db-id "${DB_ID}" --sqlite-path "${SQLITE_PATH}" \
    --descriptive-name-dir "${DB_DIR}/preprocessed/ColGrp_artifacts/descriptive_names" \
    --queue-path "${QUEUE_DIR}/queue.jsonl" --verifier-ledger "${VERIFIER_DIR}/ledger/verifier_decisions.jsonl" --output-root "${OUTPUT_ROOT}"
  if [[ ! -f "${DB_DIR}/preprocessed/${DB_ID}_lsh_manifest.json" ]]; then
    "${PYTHON_BIN}" "${ROOT}/offline/scripts/build_value_lsh_index.py" --db-dir "${DB_DIR}"
  else
    echo "[BRIDGE offline] Reusing existing value-LSH artifacts for ${DB_ID}."
  fi
done
