# BRIDGE

BRIDGE constructs a database-specific ambiguity graph offline, then uses it to
ask clarification questions and repair text-to-SQL predictions online.

## Setup

Install the offline dependency and configure OpenAI-compatible endpoints in a
private `.env` file:

```bash
pip install -r requirements-offline.txt
cp .env.example .env
```

Set the system-chat variables (`MODEL_NAME`, `BASE_URL`, `API_KEY`) and the
embedding variables (`EMBEDDING_MODEL_NAME`, `EMBEDDING_BASE_URL`,
`EMBEDDING_API_KEY`). The online runner can optionally use a separate fixed
user-feedback model through the `USER_FEEDBACK_*` variables or CLI options.

BRIDGE expects a licensed BIRD checkout. Its database root must contain
`dev_databases/{db_id}/{db_id}.sqlite` and the accompanying BIRD dev JSON.

## 1. Construct the Graph Offline

Run one command for one or more databases:

```bash
bash scripts/run_offline.sh \
  --db-root /path/to/BIRD_dev/dev_databases \
  --output-root /path/to/bridge_graph \
  --db-id example_db
```

The script performs these steps in order:

1. Generates loader-compatible descriptive-name caches in each database.
2. Builds schema profiles and cached embeddings.
3. Proposes pairs with schema-semantic similarity, surface-name similarity,
   and value-domain collision.
4. Unions candidate pairs, verifies them with the taxonomy-guided LLM, and
   writes accepted pairs as column groups.
5. Builds the text-value MinHash-LSH files used by online literal retrieval.

The graph consumed online is:

```text
/path/to/bridge_graph/{db_id}/preprocessed/{db_id}_column_groups_manual.json
```

The script defaults to the main proposal-union setting. For the retained
offline ablations, pass `--verification-universe all_pairs` or
`--disable-proposal-view schema_semantic_similarity`,
`surface_name_similarity`, or `value_domain_collision`.

## 2. Run Online Repair

Use the graph directory from step 1 and a seed-failure subset plus matching
seed-SQL cache:

```bash
bash scripts/run_online.sh \
  --dataset-file /path/to/BIRD_dev/dev.json \
  --db-root /path/to/BIRD_dev \
  --column-group-root /path/to/bridge_graph \
  --subset-file /path/to/seed_failures.json \
  --pred-cache /path/to/seed_predictions.json \
  --few-shot-store /path/to/few_shot_chroma \
  --out /path/to/bridge_results
```

The online script runs four turns: graph/value retrieval and front-loaded
clarification, followed by three clarification-and-SQL-repair turns. Results
are written under the supplied `--out` directory, with one timestamped run
directory containing `questions/` artifacts and `run_summary.json`.

To evaluate a smaller system model while keeping the simulated user fixed,
append:

```bash
  --user-feedback-model "your-27B-user-model" \
  --user-feedback-base-url "https://your-user-endpoint/v1" \
  --require-user-feedback-27b
```

## Evaluation

`data/gold/dev_ourfix.json` is the gold JSON used by BRIDGE. The portable copy
of the official robust `EX_col` evaluator is in `eval/`. Install its
dependencies and pass the final prediction JSON in `question_id -> [SQL]`
format:

```bash
pip install -r eval/requirements.txt
python eval/evaluate_EX_col.py \
  --pred /path/to/final_predictions.json \
  --gold data/gold/dev_ourfix.json \
  --db_path /path/to/BIRD_dev/dev_databases \
  --mode greedy_search \
  --output_dir /path/to/evaluation
```
