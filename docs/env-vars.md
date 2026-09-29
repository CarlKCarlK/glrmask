# GLRMASK Environment Variables

This document lists all `GLRMASK_*` environment variables used in this crate, grouped by functional area.

## Value Parsers

- **Strict `"1"` bool**: only value `1` enables; anything else disables.
- **Truthy bool**: enabled unless value is empty or one of `0`, `false`, `no`, `off` (case-insensitive).
- **Presence toggle**: enabled when variable is present, regardless of value.
- **Compact mode**: `none|0|off|skip`, `fast`, `full|1|on`.
- **Minimize strategy**: `full`, `fast`, or `threshold:<n>`.

## Compiler Pipeline

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_COMPILE_THREADS` | positive integer (`usize > 0`) | auto (available parallelism capped at 6 on Windows, 10 elsewhere) |
| `GLRMASK_DISABLE_MACRO_PARALLELISM` | truthy bool | off; serializes coarse compiler/composition fan-out (compiler DAG lanes, terminal partitions and L1/L2P siblings, template/characterization items, parser-family/bundle siblings, composition table/boundary/parser lanes, and per-component/template/shard work) while preserving parallelism inside each individual operation. With compile/composition profiling enabled, would-be parallel sibling sets emit `[glrmask/profile][macro_fanout]` summaries with count, total work, maximum item time (the ideal abundant-core critical path for that fan-out), mean, p50, p90, and `ideal_parallelism=total/max`. |
| `GLRMASK_PROFILE_COMPILE` | truthy bool | off |
| `GLRMASK_PROFILE_COMPILE_SUMMARY` | truthy bool | off |
| `GLRMASK_DISABLE_TERMINAL_COLORING` | truthy bool | off |
| `GLRMASK_COMPACT_POSSIBLE_MATCHES_BEFORE_RECONCILE` | truthy bool | on |

## Terminal DWA / ID Map (L1/L2P/Merge)

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_FORCE_ALL_L2P` | strict `1` bool | off |
| `GLRMASK_PROFILE_DETERMINIZE` | strict `1` bool | off |
| `GLRMASK_DISABLE_DIVERSITY_STATE_ORDER` | truthy bool | off |
| `GLRMASK_DISABLE_TRIE_WALK` | truthy bool | off |
| `GLRMASK_VOCAB_UNGROUPED_BATCH` | truthy bool | off |
| `GLRMASK_VOCAB_EQUIV_BATCH_SIZE` | positive integer (`usize > 0`) | auto |

### Adaptive exact compile optimizations

These paths are exact language-preserving optimizations. The positive-name
variables below remain force-on/debug controls; production normally relies on
the automatic gates. `DISABLE_...` variables are presence toggles and are
intended as diagnostic kill switches.

| Variable | Valid values | Default behavior |
|---|---|---|
| `GLRMASK_DETERMINIZE_PARALLEL_DIRECT_WAVES` | presence toggle | auto: on with >1 Rayon worker for NWAs with at least 128 states |
| `GLRMASK_DISABLE_DETERMINIZE_PARALLEL_DIRECT_WAVES` | presence toggle | off |
| `GLRMASK_DETERMINIZE_WAVE_MAX` | positive integer (`usize > 0`) | `1000` |
| `GLRMASK_L2P_PARALLEL_ROOT_TRIE` | presence toggle | auto: on with >1 Rayon worker for L2P tries with at least 512 reachable tokens and at least 2 root children |
| `GLRMASK_DISABLE_L2P_PARALLEL_ROOT_TRIE` | presence toggle | off |
| `GLRMASK_L2P_ROOT_TRIE_TASKS` | positive integer (`usize > 0`) | current Rayon worker count, capped by root-child count |
| `GLRMASK_DISABLE_L2P_DEFER_UNCOLORED_FUTURE_LEAVES` | presence toggle | off; exact uncolored future-leaf token contributions are batched per NWA source/scanner state and expanded once during flush |
| `GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR` | truthy bool | on; exact first-transition vocabulary prepartitioning uses a 5% preliminary-work limit normally, or 10% for 1024–8192-token vocabularies whose ordinary authority fits in one state batch |
| `GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_MAX_WORK_RATIO` | positive finite float | adaptive as above; explicit value overrides the automatic 5%/10% limit |
| `GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_MIN_BUCKET_TOKENS` | integer (`usize >= 2`) | `2` |
| `GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_PARALLEL_BUCKETS` | truthy bool | auto; factors admitted only by the moderate-vocabulary 10% extension use sequential buckets to avoid nested Rayon contention |
| `GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_STRICT_REFERENCE` | truthy bool | off; recompute the ordinary exact partition and require equality |
| `GLRMASK_ENABLE_L2P_COMMON_ATOM_PRECLASS` | truthy-ish bool (`0`/`false` disable) | on |
| `GLRMASK_L2P_COMMON_ATOM_MAX_TOKENS` | integer (`usize >= 4096`) | `100000` |
| `GLRMASK_PREPARE_COMMON_ATOM_SUFFIX_INDEX` | presence toggle | auto: prepare vocabularies with 16384–100000 entries |
| `GLRMASK_DISABLE_PREPARE_COMMON_ATOM_SUFFIX_INDEX` | presence toggle | off |
| `GLRMASK_COMMON_ATOM_PARALLEL_PHASED` | presence toggle | auto: phased path with >1 worker for at least 32768 eligible tokens |
| `GLRMASK_DISABLE_COMMON_ATOM_PARALLEL_PHASED` | presence toggle | off |
| `GLRMASK_DISABLE_COMMON_ATOM_CUT_FILTERED_ROOT` | presence toggle | off; cut-filtered root semantics are on by default |
| `GLRMASK_DISABLE_COMMON_ATOM_PARALLEL_GROUP` | presence toggle | off; exact grouping is parallel for sufficiently large groups with >1 worker |
| `GLRMASK_PARSER_SUPPORT_DIRECT_UNION` | presence toggle | auto: direct multiway union for at least 5 meaningful operands |
| `GLRMASK_DISABLE_PARSER_SUPPORT_DIRECT_UNION` | presence toggle | off |
| `GLRMASK_PARSER_SUPPORT_DEFER_EDGE_UNIONS` | presence toggle | auto: defer independent edge unions with >1 worker for parser NWAs with at least 512 states |
| `GLRMASK_DISABLE_PARSER_SUPPORT_DEFER_EDGE_UNIONS` | presence toggle | off |
| `GLRMASK_PARSER_FINAL_DIRECT_UNION` | presence toggle | auto: direct multiway final union for at least 5 components |
| `GLRMASK_DISABLE_PARSER_FINAL_DIRECT_UNION` | presence toggle | off |
| `GLRMASK_SPECULATIVE_P2_L2P` | presence toggle | off; experimental/explicit only |
| `GLRMASK_SPECULATIVE_P2_POOL_THREADS` | positive integer (`usize > 0`) | unset; no dedicated speculative pool unless configured |

### Minimize strategy vars

| Variable | Valid values | Default behavior |
|---|---|---|
| `GLRMASK_MINIMIZE_BUNDLE` | minimize strategy | callsite default (`minimize_fast` for multi-group bundles) |
| `GLRMASK_MINIMIZE_L2P` | minimize strategy | callsite default (`minimize_with_threshold(..., 50)`) |
| `GLRMASK_MINIMIZE_MERGE` | minimize strategy | callsite default (`minimize`) |
| `GLRMASK_MINIMIZE_MERGE_GLOBAL` | minimize strategy | callsite default (`minimize`) |
| `GLRMASK_MINIMIZE_PARSER_DWA` | minimize strategy | callsite default (`minimize_fast`) |

### Compact mode vars

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_COMPACT_L1` | compact mode | `fast` |
| `GLRMASK_COMPACT_MERGE` | compact mode | `fast` |
| `GLRMASK_COMPACT_MERGE_GLOBAL` | compact mode | `fast` |

## Segmented Static Linker

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_DISABLE_STATIC_BOUNDARY_SHARDS` | presence toggle | off; the static boundary route attempts the signed walk link and declines loudly where it cannot serve (nullable nested links, already-composed parents, virtual-residual components). When set, static requests quietly install exact dynamic shards instead — the only quiet all-dynamic route on the static entry point. Diagnostic fallback; production callers needing dynamic behavior should use the explicit Dynamic backend. |
| `GLRMASK_STRICT_STATIC_TRAP_DYNAMIC` | presence toggle | off; test-only proof flag. When set, evaluating any `DynamicDirect` boundary shard (or dynamic mask fallback) on a strict-static path panics loudly instead of silently succeeding, so differentials cannot pass vacuously through a hidden dynamic fallback. |
| `GLRMASK_SIGNED_SHARD_NO_CONTROLS` | presence toggle | off; diagnostic size oracle only. When set, control (Entry/Finish) fragments are omitted and only ordinary transfers compose, isolating the control-closure size contribution. The result is not a valid shard (controls are required for exactness). |
| `GLRMASK_SIGNED_SHARD_MINIMIZE_ORDER` | `descending` (case-insensitive) or unset | unset (`stable` order); diagnostic representation control. `descending` places denser partial behavior functions first for greedy absorption during signed-shard minimization. Affects only representation choices among already compatible classes, never the accepted weighted language; the DynamicDirect differential remains the arbiter. |

## JSON Schema Import

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_ADDPROP_NO_EXCLUSIONS` | truthy bool | off |
| `GLRMASK_AP_KEY_ANY_STRING` | truthy bool | off |
| `GLRMASK_SHARED_STRING_VALUE_EXCLUSIONS` | truthy bool | on |
| `GLRMASK_GLOBAL_SHARED_STRING_VALUE_EXCLUSIONS` | truthy bool | off |
| `GLRMASK_SHARED_STRING_VALUE_EXCLUSION_LIMIT` | integer (`usize`) | unset |
| `GLRMASK_STRING_VALUE_EXCLUSIONS_COMPAT` | `abdcffb6b` | unset |
| `GLRMASK_MERGE_ANYOF` | strict `1` bool | off |
| `GLRMASK_STRING_REPEAT_CHUNK` | integer (`usize`) | `256` |
| `GLRMASK_JSON_SCHEMA_PRESERVE_PATTERN_MAX_LENGTH` | truthy bool | `on` |
| `GLRMASK_JSON_SCHEMA_PATTERN_MAX_LENGTH_COMPLEXITY_LIMIT` | integer (`usize`) | `8000` |
| `GLRMASK_JSON_SCHEMA_SPLIT_COMPLEX_PATTERNS` | truthy bool | `off` |

`GLRMASK_JSON_SCHEMA_PATTERN_MAX_LENGTH_COMPLEXITY_LIMIT` is a static regex-HIR budget used only when preserving `maxLength` on patterned strings. If the score is above the budget, the importer drops the upper length envelope before terminal DFA construction; cheap `minLength` lower bounds are still kept. There is no separate hard `maxLength` cap: a sufficiently simple large bound can remain exact.

`GLRMASK_JSON_SCHEMA_SPLIT_COMPLEX_PATTERNS` controls the importer-level exact decomposition of sufficiently complex, fully anchored string `pattern` expressions containing one large bounded repetition. It is disabled by default. Set it to `1`, `true`, `yes`, or `on` to enable the optimization; leaving it off retains the original monolithic pattern terminal.

`GLRMASK_STRING_VALUE_EXCLUSIONS_COMPAT=abdcffb6b` restores the old capped
schema-wide string-value exclusion profile from `abdcffb6b`: global shared
string-value exclusions are enabled, the default cap is 32 excluded
literals/patterns, and the newer local anyOf string-value exclusions are
disabled. The same pieces can be configured manually with
`GLRMASK_GLOBAL_SHARED_STRING_VALUE_EXCLUSIONS=1`,
`GLRMASK_SHARED_STRING_VALUE_EXCLUSION_LIMIT=32`, and
`GLRMASK_SHARED_STRING_VALUE_EXCLUSIONS=0`.

## Grammar AST Lowering

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_REPEAT_TREE_SHAPE` | `left`, `balanced`, `leftbalanced`, `left_balanced` (other set value falls to right) | `balanced` when unset |
| `GLRMASK_MAX_RUNTIME_REDUCTION_LEN` | positive integer (`usize > 0`) | `5` |

## Notes

- For minimize strategy vars, invalid set values panic with a validation error.
- For compact mode vars, unknown set values silently fall back to the per-callsite default.
- `GLRMASK_AP_KEY_ANY_STRING` is effectively enabled if either itself or `GLRMASK_ADDPROP_NO_EXCLUSIONS` is enabled.

## Bounded dynamic-boundary runtime paths

These runtime paths use the existing shared token-trie walker and preserve its
parser, lexer, and token-mask semantics. Their defaults were promoted together;
setting an individual flag to `0` retains its comparison/fallback implementation.

| Variable | Type | Default and behavior |
|---|---|---|
| `GLRMASK_EPSILON_ONLY_RESETS` | truthy bool | on; at most four bounded proofs per mask may represent a completed leaf as an epsilon-only fresh reset. Token-end acceptance is retained. Any uncertainty keeps ordinary reset processing. |
| `GLRMASK_BOUNDARY_PACKED_VOCAB` | truthy bool | on; prepares boundary vocabularies using borrowed token bytes and flat alias storage. Preparation remains charged to the first use; no hidden prewarming. |
| `GLRMASK_BOUNDED_HIGH_RAW_MEMO` | truthy bool | on; raw state IDs of at least 1,024 use query-local full-key tagged caches instead of directories sized by the highest raw ID. Small IDs retain dense lookups. Cache collisions recompute exactly. |

The shared walker also initializes owned traversal storage only at depths it
actually saves, starting with eight inline slots and growing within the trie's
certified depth. This does not cache parser results between mask calls.

These changes reduce measured boundary costs; they do not guarantee that every
boundary invocation is as cheap as its ordinary non-boundary contribution.
