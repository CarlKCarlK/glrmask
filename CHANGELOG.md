# Changelog

## Unreleased

### Improved

- Static masks bound speculative concrete-stack expansion per parser graph,
  retaining shared-graph evaluation for wide ambiguities. Deterministic
  reductions reuse common output prefixes, single-top commits avoid a duplicate
  fast-path attempt, and fully accepted stack paths stop without redundant work.
- Static compilation reuses support keys, final-weight signatures, and token
  remaps; large two-parser unions use the exact direct construction. Token-set
  intersections use a memoized interval sweep while preserving the previous
  operand representative, including on reversed cache hits.
- Dynamic compilation overlaps lexer factoring with parser-table construction.
  Shared regex discovery traverses unique DAG nodes, and choice rewrites retain
  already-factored children instead of recursively factoring them again.
- Dynamic master-trie admission compares its cost with the original vocabulary
  trie before replacing the traversal, avoiding an expanded-trie comparison that
  could select a slower path.

- Static constraint serialization now primes canonical artifact bytes during public compilation, making the first subsequent `save()` a bulk copy rather than a full re-encode. Current artifacts also persist the packed-DWA dense-mask cache needed by the runtime, substantially reducing load-time cache reconstruction. Rust now exposes one `Constraint::load(...)` entry point that automatically takes the zero-copy backing path for owned `Vec<u8>` input while continuing to accept borrowed bytes.

- Direct dynamic masking now walks the vocabulary trie once with interned
  correlated lexer/parser recognizer states instead of repeating the trie walk
  for each live branch. Runtime vocabulary data uses a flat preorder walk,
  flattened token aliases, and compact per-node mask operations; reusable trie
  construction moves into vocabulary preparation. Small lexer NFAs may be
  determinized for faster masks, while large source automata skip that optional
  product construction before it becomes a build-time cliff.
- Dynamic tokenizer construction now keeps very large exact binary
  intersections in compressed runtime form, uses a compact fail-closed engine
  for long zero-minimum bounded string residuals, and isolates nullable roots
  before assembling the global lexer union. One-word automaton bitsets stay
  inline and large single-group metadata is built in parallel. Representative
  former 0.5–1.0 second dynamic builds now complete below 200 ms without
  weakening lexer semantics or precomputing vocabulary-specific token programs.
- Static constraint saving now serializes directly into the compressed
  artifact and borrows tokenizer DFA storage instead of materializing a full
  raw bincode payload and cloned wire automaton. Large compile-once artifacts
  therefore avoid the previous hundreds-of-megabytes serialization spike;
  loading also bounds decompression by the declared raw length before parsing.
- Runtime mask and commit paths now reuse bounded parser, tokenizer, accumulator, and bitmap storage for common deterministic and small-frontier states, avoiding allocator activity during ordinary decoding. Tokenizer epsilon-closure data is finalized during compile/load rather than on the first commit. The Python extension keeps delayed automatic mimalloc purging enabled but defaults purges to reset (`MADV_FREE`/`MEM_RESET`) rather than synchronous decommit. Pages remain OS-reclaimable without requiring caller-managed trimming; `MIMALLOC_PURGE_DECOMMITS=1` restores immediate RSS-oriented decommit behaviour.
- Constraints with at most 16 initially admissible tokens now exercise those initial commit transitions during runtime-cache finalization. This moves a bounded amount of cold parser/tokenizer execution into compile or load, reducing first-token TBM without adding work to `Constraint::start()` or displacing it to the second token; larger initial masks skip the step entirely.
- Large bounded JSON Schema string patterns now retain exact `maxLength`
  semantics by compiling terminal/parser automata against a certified smaller
  residual representative while keeping the full exact lexer for runtime
  state. Pathological bounded-repeat intersections no longer force the former
  multi-second terminal-DWA construction path.
- Exact runtime tokenizer finalization now overlaps terminal/parser automaton
  construction. Large protected residual products publish their synthesized
  compile tokenizer before full future analysis and byte-transition
  materialization, and runtime component assembly moves transition storage
  instead of copying it a second time.
- Bounded-terminal synthesis candidate discovery now runs before terminal
  materialization, uses cached vocabulary length/alphabet statistics, reuses
  language-canonical repeat-horizon proofs, and parallelizes only the small set
  of qualified candidates. Grammars with no viable candidate avoid synthesis
  allocations entirely, and the diagnostic opt-out shares the same eligibility
  scan so no-candidate ON/OFF builds follow the same planning path.
- Expensive constrained array items are no longer duplicated into separate
  first/next contextual terminals when one item already exceeds the importer
  product budget. The item is compiled once and its count remains enforced at
  grammar level.

### Changed

- `DynamicConstraint` once again performs vocabulary/lexer/parser analysis in
  the decoding path instead of compiling whole-vocabulary token programs or
  continuation partitions. Dynamic artifact format version 9 no longer stores
  those precomputed structures; version-7 and version-8 dynamic artifacts must
  be rebuilt rather than silently restoring the removed backend.
- Static `Constraint` artifacts now use compressed format version 8. Current
  code continues to load uncompressed version-7 artifacts; newly saved
  artifacts require a version-8-capable loader.
- Finite `maxLength` constraints on patterned JSON Schema strings are preserved
  by default even when their estimated pattern/length product is large. The
  complexity budget now selects lowering strategy only; it no longer permits a
  semantically weaker grammar.
- Exact bounded-terminal synthesis is enabled by default. Runtime keeps the
  full exact tokenizer while terminal/parser DWA construction may use a
  certified smaller representative. Set
  `GLRMASK_SYNTHETIC_BOUNDED_TERMINALS=0` only for diagnostics.

### Fixed

- Fallback tokenizer serialization preserves packed compressed segments when
  they cannot use the contiguous-suffix encoding. Previously affected large
  mixed tokenizers could lose transitions after saving and loading; affected
  artifacts written by older code must be rebuilt from their grammar.
- Direct slice and byte-family proofs on compact loaded bounded-repeat
  tokenizers use the retained liveness metadata rather than indexing construction
  tables omitted from the artifact. Missing proof data declines the shortcut
  instead of inventing liveness or panicking.

## 0.1.1 — 2026-07-19 — runtime, integration, and tail-latency update

### Added

- Grammar-level end-token IDs for JSON Schema, EBNF, Lark, and GLRM constructors. End tokens are exact parser terminals rather than byte spellings or metadata stored on `Vocab`.
- Bounded token-level rollback for speculative decoding, with zero retained history by default.
- Non-mutating proposal validation that returns the longest admissible token prefix.
- Explicit failed-state inspection for recovery after an invalid commit.
- A llama.cpp-oriented vocabulary construction path and expanded integration examples.

### Improved

- Dynamic masking now precompiles and caches exact residual token programs, selects overlays by structural family, and avoids redundant parser simulation and continuation-partition construction.
- Dynamic mask and artifact paths received additional indexing, cache, serialization, and tail-latency work.
- README performance figures, dark-mode assets, runtime-mode documentation, and full-corpus benchmark links were revised.

### Changed

- `Vocab` no longer owns a distinguished EOS field. Consumers pass one or more `end_token_ids` when compiling a constraint; those tokens may also retain ordinary byte semantics if present in the byte vocabulary.
- Dynamic constraint artifacts use a new format version. Older artifacts without Vocab-level EOS metadata are migrated; artifacts that depended on the removed EOS metadata fail explicitly and must be rebuilt.
- Importer-level complex anchored-pattern splitting is available through `GLRMASK_JSON_SCHEMA_SPLIT_COMPLEX_PATTERNS=1` but is disabled by default.

### Integration compatibility

- The frozen vLLM backend requires `glrmask >= 0.1.1` for bounded rollback, non-mutating validation, failed-state inspection, and grammar-level end-token support.
- Public `glrmask 0.1.0` remains installable but is not compatible with that backend.

## 0.1.0 — 2026-07-15 — Shingleback initial release

### Highlights

- Public project brand: Shingleback; the Rust crate, PyPI distribution, and Python import remain `glrmask`.
- Vocabulary-specific grammar-constrained decoding for EBNF, Lark, and a documented pragmatic subset of JSON Schema.
- Reusable compiled `Constraint` objects with incremental mask, commit, completion, and forced-prefix operations.
- GLR-based parsing for ambiguous and genuinely context-free grammars, including tokenizations that cross grammar-terminal boundaries.
- Rust and Python APIs for incremental mask, commit, completion, and forced-prefix operations.
- Constraint serialization for compile-once, load-and-run deployments, plus a smaller execution-only runtime crate for serving artifacts.
- A build-only Python wheel workflow covering Python 3.9–3.13 across manylinux x86_64/aarch64, macOS x86_64/arm64, and Windows x86_64.

### Release evidence and caveats

- The bounded v0.1 `make example-slow-all` comparison is documented in [`docs/benchmark-0.1.md`](docs/benchmark-0.1.md), including exact scope, environment, backend versions, methodology, and caveats.
- JSON Schema support is not full specification conformance; see [`docs/json-schema-semantic-deviations.md`](docs/json-schema-semantic-deviations.md).
