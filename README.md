# BRIDGE

BRIDGE is a text-to-SQL repair pipeline that constructs a reusable database-specific ambiguity graph offline and uses it during a four-turn interactive repair process online.

## Release Status

This repository is being prepared as the public, slimmed BRIDGE release. The portable offline construction path is self-contained; historical results, private data, API keys, and machine-specific launch wrappers are intentionally excluded.

The frozen primary method contract is defined in [`offline/configs/bridge_offline_mainline_v1.json`](offline/configs/bridge_offline_mainline_v1.json). The release boundary, retained ablations, and excluded development branches are documented in [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md).

## Main Method

Offline construction creates an ambiguity graph for each database:

1. Generate descriptive names for all tables and columns.
2. Build column profiles from schema, available documentation, samples, and lightweight value statistics.
3. Propose pairs through Schema-Semantic Similarity, Surface-Name Similarity, and Value-Domain Collision.
4. Verify only the union of proposed pairs with a taxonomy-guided LLM and retain `ADD_EDGE` decisions with their ambiguity trigger contexts.
5. Build a MinHash-LSH index over eligible text values for online literal retrieval.

The online pipeline retrieves graph alternatives and literal alternatives, asks clarification questions, incorporates feedback, regenerates SQL, and evaluates the resulting query over four turns.

The frozen public online entrypoint is documented in [`online/README.md`](online/README.md).
The same document lists the small set of retained, paper-aligned online
ablations; historical development switches are not exposed.

Deterministic FK/ER edge injection is not part of the released main method. An FK-related pair may still be retained when it is proposed by the three general views and independently verified by the LLM.

## Model Configuration

All LLM and embedding clients use portable OpenAI-compatible configuration. Copy `.env.example` to a private `.env` file or export the variables directly; never commit API keys.

```bash
export MODEL_NAME="your-chat-model"
export BASE_URL="https://your-provider/v1"
export API_KEY="your-private-key"
```

The eventual online runner also supports a separate user-feedback client. This lets a smaller system model be evaluated while keeping the simulated user on a fixed stronger model.

Install the sole non-standard dependency required by the public offline path:

```bash
pip install -r requirements-offline.txt
```

## Descriptive-Name Preparation

The current component writes exactly the cache format consumed by the offline profile loader and online fallback loader:

```text
{db_dir}/preprocessed/ColGrp_artifacts/descriptive_names/{table}.json
```

Validate an existing cache without model calls:

```bash
python offline/scripts/generate_descriptive_names.py \
  --db-dir /path/to/dev_databases/example_db \
  --check
```

Generate or resume names for missing tables:

```bash
python offline/scripts/generate_descriptive_names.py \
  --db-dir /path/to/dev_databases/example_db \
  --resume \
  --model "$MODEL_NAME" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY"
```

The script calls one table at a time, validates complete schema coverage, and leaves completed table artifacts intact if a later table fails.

## Offline Construction

The public offline path is proposal-first. It deliberately has no deterministic FK or ER injection stage.

```bash
# 1. Build a descriptive-name-backed profile.
python offline/scripts/build_db_schema_profile.py \
  --db-id example_db --db-dir /path/to/dev_databases/example_db \
  --sqlite /path/to/dev_databases/example_db/example_db.sqlite \
  --output-dir work/profile/example_db

# 2. Build cached embeddings for the two embedding proposal views.
python offline/scripts/build_embedding_caches.py --db-id example_db \
  --profile work/profile/example_db/schema_profile.json \
  --proposal-view schema_semantic --output-dir work/embeddings/schema_semantic/example_db
python offline/scripts/build_embedding_caches.py --db-id example_db \
  --profile work/profile/example_db/schema_profile.json \
  --proposal-view surface_name --output-dir work/embeddings/surface_name/example_db

# 3. Materialize the four internal proposal channels, then union them by pair key.
python offline/scripts/materialize_candidate_views.py --db-id example_db \
  --schema-profile work/profile/example_db/schema_profile.json \
  --sqlite-path /path/to/dev_databases/example_db/example_db.sqlite \
  --schema-cache-dir work/embeddings/schema_semantic/example_db \
  --surface-cache-dir work/embeddings/surface_name/example_db \
  --channel-config offline/configs/candidate_channels_mainline_v1.json \
  --output-dir work/candidate_views/example_db
python offline/scripts/build_candidate_verifier_queue.py --db-id example_db \
  --schema-profile work/profile/example_db/schema_profile.json \
  --sqlite-path /path/to/dev_databases/example_db/example_db.sqlite \
  --candidate-artifact schema_embedding=work/candidate_views/example_db/schema_embedding \
  --candidate-artifact lexical=work/candidate_views/example_db/lexical \
  --candidate-artifact surface_name_embedding=work/candidate_views/example_db/surface_name_embedding \
  --candidate-artifact value_collision=work/candidate_views/example_db/value_collision \
  --output-dir work/verifier_queue/example_db

# 4. Verify the proposal union and compile ADD_EDGE decisions into compatible groups.
python offline/scripts/run_pair_verifier.py --queue-path work/verifier_queue/example_db/queue.jsonl \
  --bundles-path work/verifier_queue/example_db/candidate_bundles.jsonl \
  --output-dir work/verifier/example_db \
  --model "$MODEL_NAME" --base-url "$BASE_URL" --api-key "$API_KEY"
python offline/scripts/compile_verified_graph.py --db-id example_db \
  --sqlite-path /path/to/dev_databases/example_db/example_db.sqlite \
  --descriptive-name-dir /path/to/dev_databases/example_db/preprocessed/ColGrp_artifacts/descriptive_names \
  --queue-path work/verifier_queue/example_db/queue.jsonl \
  --verifier-ledger work/verifier/example_db/ledger/verifier_decisions.jsonl \
  --output-root work/column_groups

# 5. Build reusable literal-retrieval artifacts.
python offline/scripts/build_value_lsh_index.py --db-dir /path/to/dev_databases/example_db
```

`schema_embedding` and `surface_name_embedding` map to the paper's Schema-Semantic Similarity and Surface-Name Similarity views. `lexical` is the deterministic component of Surface-Name Similarity; `value_collision` is Value-Domain Collision. The default schema selector is `rank <= 10` and `score >= 0.55`.

For the retained ablations, pass `--verification-universe all_pairs`, or disable exactly one paper view with `--disable-proposal-view schema_semantic_similarity`, `surface_name_similarity`, or `value_domain_collision`.

## Data and Reproducibility

BRIDGE expects users to provide their own licensed BIRD checkout, including the SQLite database and optional `database_description/*.csv` files for each database. Derived BIRD artifacts, seed SQL files, result folders, embeddings, and Chroma stores are not redistributed here.

See [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md) before using the research ablations or reproducing paper-specific settings.
