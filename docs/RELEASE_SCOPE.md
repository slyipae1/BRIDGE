# BRIDGE Release Scope

## Default Algorithm

The public default is the paper's three-view, proposal-first construction route:

```text
descriptive names
  -> column profiles
  -> {schema-semantic, surface-name, value-domain} proposals
  -> pair-keyed union
  -> taxonomy-guided LLM verification
  -> verified ambiguity graph
```

The default verifies only the proposal union. It does not construct an all-pairs universe, inject deterministic FK/ER groups, or perform two-hop graph expansion.

## Retained Paper Ablations

The following are retained as optional, explicitly named experiment modes because they correspond to the paper's offline-construction experiments:

| Experiment | Allowed deviation from default |
| --- | --- |
| All-Pairs Verification | verify every unordered column pair instead of only the proposal union |
| w/o Schema-Semantic Similarity | disable only this proposal view |
| w/o Surface-Name Similarity | disable the complete public surface-name view, including lexical and raw-name embedding submethods |
| w/o Value-Domain Collision | disable only this proposal view |

The all-pairs setting and a disabled view are never the implicit default. Each run must record its effective frozen config, model endpoint, and input DBs.

The following online ablations are retained through
`online/run_bridge_ablation.py` because their existing implementations match
the paper's controlled conditions:

| Experiment | Allowed deviation from the primary online route |
| --- | --- |
| w/o Integration | omit only COLUMN/literal `prune_combine` |
| w/o Graph | disable graph-backed COLUMN retrieval, preserve VALUE-LSH retrieval, and use the full plain schema for detection |
| rp/w Real-Time | replace graph-backed COLUMN lookup with batched, per-parsed-column full-schema LLM retrieval |
| with BIRD evidence | opt in to existing evidence fields throughout online prompt contexts; the primary route remains without evidence |

Online model-size sensitivity remains operational rather than a separate
algorithm branch: choose the system model normally and pass the dedicated
user-feedback endpoint plus `--require-user-feedback-27b` to keep the user
simulator fixed.

## Operational Flexibility

Users may choose a compatible chat model, verifier model, embedding model, endpoint, API key, batch concurrency, and thinking-control implementation. Those are operational choices, not separate method definitions. The online runner may also use a separate user-feedback model; this is required for fair online model-size sensitivity experiments where the user simulator stays fixed.

## Excluded Development Lines

The first public release excludes deterministic FK/ER injection, ER mining, ER safe projection, two-hop expansion, oracle filters, candidate-channel prompt evidence, Chroma value embedding, DB-specific lexical normalization, and private result/queue artifacts. The retained real-time replacement is restricted to the documented online ablation entrypoint.

The trigger-context ablation is excluded for now. The available historical
relation-metadata switch is broader than the paper condition and does not
provide an audited removal of only the option-level trigger context.

An LLM-verified edge can describe an FK relationship. Such an edge remains valid only because it was proposed by the general views and accepted by the pair verifier; it is not injected by a deterministic FK rule.

## Data and Artifacts

The repository does not redistribute BIRD data, BIRD database symlinks, private seed SQL predictions, graph result archives, model weights, API keys, or generated Chroma/SQLite stores. Users must provide their own lawful data checkout and build derived artifacts locally.
