# BRIDGE Online Repair

Run the primary public pipeline through `run_bridge.py`. It freezes the
paper's primary online algorithm while keeping data locations and
OpenAI-compatible model endpoints configurable. A separate,
paper-aligned `run_bridge_ablation.py` exposes only the retained experiment
variants documented below.

## Frozen Algorithm

The public route uses four turns: one front-loaded Module A turn followed by
three clarification/repair turns. Module A uses the constructed column graph,
MinHash-LSH value retrieval (top 20), query-aware schema filtering, taxonomy
relation guidance, `by_all` LLM contextualization, empty column presentation,
and COLUMN/literal `prune_combine` aggregation. BIRD evidence is intentionally
not supplied to prompts.

## Required Inputs

The caller supplies licensed BIRD files and derived artifacts:

- BIRD dev JSON and its root containing `dev_databases/`;
- a JSON subset of question IDs and a compatible seed-SQL prediction cache;
- the graph archive created by `offline/scripts/compile_verified_graph.py`;
- LSH files under each database's `preprocessed/` directory, created by
  `offline/scripts/build_value_lsh_index.py`; and
- a compatible few-shot Chroma store for `k_shot > 0`, or an explicit
  `--allow-empty-few-shot-fallback` for a zero-shot smoke test.

The few-shot store is a derived artifact and is not bundled with this source
release. Its builder/documentation is the remaining online-release task.

## Run

Configure the system client without putting secrets in commands:

```bash
export MODEL_NAME="your-system-model"
export BASE_URL="https://your-provider/v1"
export API_KEY="your-private-key"
```

Then run:

```bash
python online/run_bridge.py \
  --dataset-file /path/to/bird/dev.json \
  --db-root /path/to/bird \
  --column-group-root /path/to/bridge_column_groups \
  --subset-file /path/to/subset.json \
  --pred-cache /path/to/seed_predictions.json \
  --few-shot-store /path/to/few_shot_chroma \
  --out /path/to/results
```

For model-sensitivity experiments, route simulated-user feedback to a fixed
endpoint while the system model changes:

```bash
python online/run_bridge.py ... \
  --user-feedback-model "your-27B-user-model" \
  --user-feedback-base-url "https://your-user-provider/v1" \
  --require-user-feedback-27b
```

`run_bridge.py` always disables reasoning requests. It does not accept
historical oracle, direct-contextualization, NoPrune, channel-drop, or
alternate graph-format switches.

## Retained Online Ablations

Use the separate entrypoint so the primary command remains a frozen contract:

```bash
python online/run_bridge_ablation.py ... \
  --variant no_integration
```

The only retained variants are:

| Variant | Exact deviation from the primary route |
| --- | --- |
| `no_integration` | Omits only COLUMN/literal `prune_combine`; graph and value retrieval remain unchanged. |
| `no_graph` | Disables only graph-backed COLUMN retrieval, retains live VALUE-LSH retrieval, and supplies the full plain schema to the detector. |
| `realtime_retrieval` | Replaces graph-backed COLUMN lookup with one batched, full-schema LLM request per parsed SQL COLUMN anchor; the remaining pipeline is unchanged. |

Add `--variant main --bird-evidence` for the with-BIRD-evidence comparison,
or add `--bird-evidence` to another retained variant. It supplies the dataset
evidence wherever the online pipeline already consumes evidence; it is not the
default method.

The trigger-context ablation is intentionally not exposed yet: the current
historical code path does not implement a narrow, audited removal of only the
per-option trigger context.
