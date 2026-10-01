# Provenance For Vendored `pipeline_0612` Helpers

This folder contains minimal local copies of upstream helpers that the Stage 4 baseline needs at runtime.

Copied sources:

- `dbelement_formatter.py`
  - source: `src/pipeline_0612/core/utils/DBeleOptions.py`
  - reason: preserve the exact `format_dbelement_options` behavior locally

- `schema_description_loader.py`
  - source logic distilled from the Stage 1 / current-method DB-grounding preprocessing utilities
  - reason: load descriptive names plus official column and value descriptions without importing external project code at runtime

These files are intentionally vendored so the copied baseline runner remains self-contained.

Live retrieval additions:

- `live_retrieval/dbelement_options.py`
  - source: `src/pipeline_0612/core/utils/DBeleOptions.py`
  - reason: preserve Version 0 full-SQL DB-element retrieval locally

- `live_retrieval/database_manager.py`
  - source: `src/pipeline_0612/runner/database_manager.py`
  - reason: load DB paths, preprocessed LSH/index artifacts, schema strings, and SQLite helpers without importing `src/pipeline_0612`

- `live_retrieval/sql_parser.py`, `fixed_parse_one.py`, `execution.py`, `db_info.py`, `database_profiler.py`, `schema.py`, `schema_generator.py`, `schema_constraints.py`
  - source: matching files under `src/pipeline_0612/database_utils/`
  - reason: keep the parser/schema/profiler dependency chain self-contained for live retrieval

- `live_retrieval/db_values/search.py`
  - source: `src/pipeline_0612/database_utils/db_values/search.py`
  - reason: load preprocessed MinHash LSH files and query similar DB values

- `live_retrieval/db_values/minhash.py`
  - source logic: `_create_minhash(...)` from `src/pipeline_0612/database_utils/db_values/preprocess.py`
  - reason: reuse the runtime query signature without copying offline preprocessing

- `live_retrieval/llm_fallback.py`
  - source logic: local replacement for the old `llm.model_factory` fallback in `DBeleOptions.py`
  - reason: keep parser fallback OpenAI-compatible while avoiding a dependency on the original pipeline source tree

- `live_retrieval/runner.py`
  - source logic: new wrapper around the vendored `get_DBeleOptions(...)`
  - reason: normalize DB env vars and emit cache-compatible `db_retrieval_sql_full` records for `stage_artifacts/module_a_retrieval/`

Offline LSH build additions:

- `live_retrieval/db_values/preprocess.py`
  - source: `related_resources/DeltaRefinement_260416/src/database_utils/db_values/preprocess.py`
  - v2 modification: build-side MinHash now reuses local `_create_minhash(...)`, including whole-string updates for values shorter than `n_gram`
  - reason: keep LSH build/query parameter behavior self-contained in this v2 workspace

- `source_code/offline/build_lsh_index.py`
  - source logic: separated LSH-only portion of `related_resources/DeltaRefinement_260416/src/preprocess_index.py`
  - reason: rebuild value LSH artifacts without also building DeltaRefinement context-vector DB artifacts
