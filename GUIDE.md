GLRMask is a grammar-constrained generation library for high-throughput LLM decoding. It is optimized for fast next-token mask computation with extremely low tail latency, even for complex grammars.

> **Interim benchmark:** the figures below are the corrected official-9,558-schema view of the 20 August 2026 engineering run. They supersede the July figures, but are not the final native publication benchmark.

<p align="center">
  <img src="https://raw.githubusercontent.com/IsaacBreen/glrmask/2f8b1505d0cba2467a458eb8b45c4879710468dd/docs/assets/benchmark-tbm-tail-2026-08-21.webp" alt="TBM latency tail curves for GLRMask and LLGuidance in the corrected 20 August 2026 engineering run" width="100%">
</p>

## Installation

### Python

```bash
python -m pip install glrmask
```

Published wheels include the native extension. Building from source requires a Rust toolchain and the platform's native build tools.

### Rust

```bash
cargo add glrmask
```

**Documentation:** [Python](https://github.com/IsaacBreen/glrmask/blob/main/python/README.md) · [Rust](https://docs.rs/glrmask)

## Usage

GLRMask has three ordinary public layers:

- `Grammar` is a source or mixed description. It can contain source children, compiled children, and vocabulary-qualified exact-token bindings.
- `UnlinkedConstraint` is reusable compiled machinery for one exact vocabulary. It may deliberately remain open and is not runnable.
- `Constraint` is closed, rooted, and immediately runnable. `ConstraintState` is the mutable per-sequence state.

Bindings are immutable. Calling `bind(...)` returns a new `Grammar` or `UnlinkedConstraint`; the original remains reusable.

At runtime, call `constraint.start()` once per generated sequence. Compute the next-token mask, sample an allowed model token, then commit that token. If the constraint was built with end tokens, those IDs become maskable only when the grammar body is accepting; committing one marks the state terminated.

```text
state = constraint.start()

while generating:
    in parallel:
        logits = llm.forward(...)
        mask = state.mask()

    logits = apply_mask(logits, mask)
    token_id = sample(logits)
    state.commit_token(token_id)
    if state.is_terminated():
        break
```

### Python quickstart

```bash
python -m pip install glrmask llama-cpp-python torch
```

```python
import numpy as np
from llama_cpp import Llama
from torch import from_numpy
from torch.distributions import Categorical

import glrmask

llm = Llama(model_path="model.gguf", logits_all=True)
vocab = glrmask.Vocab.from_llama_cpp(llm)
end_token_ids = vocab.llama_cpp_end_token_ids

schema = {"type": "string", "enum": ["positive", "negative", "neutral"]}
constraint = glrmask.Grammar.from_json_schema(schema).compile(
    vocab,
    end_tokens=end_token_ids,
    optimization=glrmask.Optimization.AUTO,
)

prompt = "Classify this review: The story dragged badly. Sentiment: "
input_tokens = llm.tokenize(prompt.encode())
llm.reset()
llm.eval(input_tokens)

state = constraint.start()
generated = []

for _ in range(64):
    logits = llm.scores[llm.n_tokens - 1]
    mask = state.mask(llm.n_vocab())
    logits[~mask] = -np.inf
    token_id = Categorical(logits=from_numpy(logits)).sample().item()

    llm.eval([token_id])
    generated.append(token_id)
    state.commit_token(token_id)
    if state.is_terminated():
        break

print(llm.detokenize(generated).decode())
```

`state.mask()` returns a NumPy Boolean array indexed by model token ID. Pass a size when the model's logits vector is wider than the constraint's natural token coordinate.

### Rust quickstart

```rust
use glrmask::{BuildOptions, Grammar, Optimization, Vocab};

let vocab = Vocab::new(vec![
    (0, b"\"yes\"".to_vec()),
    (1, b"\"no\"".to_vec()),
    (2, b"<eos>".to_vec()),
]);
let schema = r#"{"type":"string","enum":["yes","no"]}"#;
let constraint = Grammar::from_json_schema(schema).compile_with(
    &vocab,
    BuildOptions::default()
        .end_tokens([2])
        .optimization(Optimization::Auto),
)?;

let mut state = constraint.start();
let mask = state.mask();
state.commit_token(0)?;
assert!(state.is_accepting());
# Ok::<(), glrmask::Error>(())
```

Rust masks are packed `u32` bitsets. Bit `token_id % 32` of word `token_id / 32` indicates whether that token is allowed.

### Source, compiled, and exact-token bindings

GLRM declares child slots with `extern grammar NAME;` and exact-token slots with `extern token NAME;`. `Grammar::bind` is the one operation for both source and compiled attachments:

```rust
use glrmask::{Grammar, Result, Vocab};

# fn demo(vocab: &Vocab) -> Result<()> {
let parent = Grammar::from_glrm(
    "glrm 1; start document; extern grammar payload; nt document = payload;",
);
let source_child = Grammar::from_json_schema(r#"{"type":"null"}"#);
let source_composition = parent.bind("payload", &source_child)?;

let compiled_child = source_child.compile(vocab)?;
let mixed_composition = parent.bind("payload", &compiled_child)?;

let _a = source_composition.compile(vocab)?;
let _b = mixed_composition.compile(vocab)?;
# Ok(())
# }
```

Exact token bindings come from the vocabulary rather than from naked IDs:

```rust
# use glrmask::{Grammar, Result, Vocab};
# fn demo(vocab: &Vocab, tool_call_id: u32) -> Result<()> {
let grammar = Grammar::from_glrm(
    "glrm 1; start message; extern token TOOL_CALL; nt message = TOOL_CALL;",
);
let grammar = grammar.bind("TOOL_CALL", vocab.token(tool_call_id)?)?;
let _constraint = grammar.compile(vocab)?;
# Ok(())
# }
```

`Vocab::token(...)` and `Vocab::tokens(...)` retain the complete vocabulary identity. A value from an incompatible vocabulary is rejected even if the bound numeric ID happens to exist in both vocabularies.

Model control/tool tokens do not need fake byte spellings. When constructing a
Rust vocabulary manually, use `Vocab::new_with_exact_token_ids(byte_entries,
exact_only_ids)` for IDs that are valid exact-token bindings but must never
enter the byte language. Python's `Vocab.from_llama_cpp()` records omitted
control/EOG/empty-piece IDs this way automatically.

### Cached parents with `UnlinkedConstraint`

Use `compile_unlinked` when a compiled parent will be reused with request-specific children. A `UnlinkedConstraint` may remain open, can be saved and loaded, and is deliberately not runnable.

```rust
# use glrmask::{BuildOptions, Grammar, Optimization, Result, Vocab};
# fn demo(vocab: &Vocab) -> Result<()> {
let parent = Grammar::from_glrm(
    "glrm 1; start document; extern grammar payload; nt document = payload;",
);
let host = parent.compile_unlinked(vocab)?;

let child_a = Grammar::from_ebnf(r#"start ::= "a""#).compile(vocab)?;
let child_b = Grammar::from_ebnf(r#"start ::= "b""#).compile(vocab)?;

let a = host.bind("payload", &child_a)?;
let b = host.bind("payload", &child_b)?;

let _constraint_a = a.link_with(
    BuildOptions::default().optimization(Optimization::FastRuntime),
)?;
let _constraint_b = b.link()?;
# Ok(())
# }
```

`UnlinkedConstraint::bind` is compiled-only: it accepts a compiled `Constraint` or a vocabulary-qualified exact-token value. It does not accept an unresolved `UnlinkedConstraint`, and it does not parse or compile source children. Link a child first before binding it to the parent. Composition stays deferred until `link`/`link_with`, so the final optimization preference can choose the boundary construction strategy.

Python uses the same lifecycle:

```python
parent = glrmask.Grammar.from_glrm(
    'glrm 1; start document; extern grammar payload; nt document = payload;'
)
host = parent.compile_unlinked(vocab)
child = glrmask.Grammar.from_json_schema(payload_schema).compile(vocab)
constraint = host.bind("payload", child).link(
    optimization=glrmask.Optimization.FAST_RUNTIME,
)
```

### Build/runtime trade-off

`Optimization` expresses intent rather than exposing internal engine names:

- `AUTO` / `Auto`: let GLRMask choose.
- `FAST_BUILD` / `FastBuild`: minimize compile/link work, leaving more work for runtime where useful.
- `FAST_RUNTIME` / `FastRuntime`: spend more build work to favor lower mask latency where supported.

All three modes preserve accepted-language semantics and produce the same public `Constraint` type.

### Optional table-free parser backend

The parser backend is a separate choice from the build/runtime preference. The
default remains `LR_TABLE` / `LrTable`. Select the acyclic template backend
explicitly for a standalone constraint:

```python
constraint = grammar.compile(
    vocab,
    optimization=glrmask.Optimization.FAST_RUNTIME,
    parser_backend=glrmask.ParserBackend.TEMPLATE_DFA,
)
assert constraint.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
```

```rust
# use glrmask::{BuildOptions, Grammar, Optimization, ParserBackend, Result, Vocab};
# fn demo(grammar: &Grammar, vocab: &Vocab) -> Result<()> {
let constraint = grammar.compile_with(
    vocab,
    BuildOptions::default()
        .optimization(Optimization::FastRuntime)
        .parser_backend(ParserBackend::TemplateDfa),
)?;
assert_eq!(constraint.parser_backend(), ParserBackend::TemplateDfa);
# Ok(())
# }
```

The selected runtime and its artifact contain the template relations, not an
LR table. Mask generation and token commitment use the existing shared engines;
parser advancement and admissibility use those relations. Built-in grammar
compilation can still use LR machinery to derive the program. Data-only
`ParserProgram` providers bypass that frontend and support both static and
dynamic mask compilation.

Compiled-component composition is not supported with this backend. An
unsupported request returns an error; it does not retain a hidden table or
silently switch backends. Performance and load-time tradeoffs depend on the
grammar and mode, so template selection is not an automatic speed guarantee.
See [the template parser contract](docs/template-parser.md) for provider
examples, exact POP/READ/PUSH semantics, validation, and persistence.
The [validation report](docs/template-parser-validation-2026-09-30.md) records
measured runtime and storage results, build/load costs, and the reasons the LR
backend remains the default.

### End tokens are final-root policy

End-token policy belongs to the final `compile`/`compile_with` or `link`/`link_with` operation. It is not inherited when a completed `Constraint` is embedded as a child.

```python
constraint = grammar.compile(vocab, end_tokens=[eos_id])
state = constraint.start()
# eos_id is allowed only when the grammar body is accepting.
```

A state reports `is_accepting()` for grammar-body acceptance, `is_rejected()` for an irrecoverable invalid prefix, and `is_terminated()` after an allowed final end token has been committed.

### Persistence

Both compiled object types are serializable:

```python
module_bytes = host.save()
host = glrmask.UnlinkedConstraint.load(module_bytes)

constraint_bytes = constraint.save()
constraint = glrmask.Constraint.load(constraint_bytes)
```

A loaded LR-backed `Constraint` remains composable as a child. Its standalone end-token policy is stripped when embedded; its compiled grammar body is retained. Template-backed constraints retain their backend through loading, but compiled-component composition is not yet supported for them.

## Grammar formats

GLRMask accepts JSON Schema, GLRM, Lark, and EBNF. GLRM is the native composition format. A grammar begins with `glrm 1;` and a `start` declaration:

```glrm
glrm 1;
start value;

t NUMBER = /-?(0|[1-9][0-9]*)/;
nt value = NUMBER | "null";
```

Declarations use `=`, epsilon is written as `eps`, and regexes use full-match semantics. Inline `g name = { ... };` grammars and externally bound `extern grammar name;` slots have the same language semantics, including scope-local ignores.

Special model-token IDs are declared with `extern token NAME;` and bound with vocabulary-qualified values:

```python
grammar = glrmask.Grammar.from_glrm('''
glrm 1;
start message;
extern token TOOL_CALL;
nt message = TOOL_CALL "lookup()";
''')
constraint = grammar.bind("TOOL_CALL", vocab.token(tool_call_token_id)).compile(vocab)
```

Lark and EBNF also support explicit `@token(<id>)` atoms when the numeric ID is deliberately part of the grammar source.


## How it works

GLRMask maintains a GLR parser state for the generated prefix, updating it as tokens are committed. To compute the next-token mask, a precomputed deterministic weighted automaton reads each parser stack one symbol at a time.

Each transition carries a Boolean mask over the model vocabulary. These masks are intersected along each stack traversal and unioned across alternative paths.

## Performance

Latest corrected engineering result: the **9,558 official JSONSchemaBench schemas**, using their corresponding MaskBench replay payloads. The historical run originally contained 705 additional MaskBench-only cases; those are excluded from every number and graph shown here. The original full sweep used AWS M8azn, and the corrected GLRMask runtime tail was refreshed on the same CPU family after fixing a deterministic post-deserialization first-commit bug. This is intentionally not presented as the final native publication run.

| TBM | GLRMask | LLGuidance |
|---|---:|---:|
| p50 | **3 µs** | 10 µs |
| p99 | **10 µs** | 223 µs |
| p99.9 | **18 µs** | 788 µs |
| p99.99 | **24 µs** | 2,290 µs |
| maximum | **70 µs** | 14,426 µs |

| TTFM | GLRMask | LLGuidance |
|---|---:|---:|
| p50 | 10,421 µs | **1,109 µs** |
| p90 | 59,350 µs | **2,738 µs** |
| p99 | 229,809 µs | **12,023 µs** |
| maximum | 779,588 µs | **81,104 µs** |

<p align="center">
  <img src="https://raw.githubusercontent.com/IsaacBreen/glrmask/fbc60288c7d86701a03a3100fa7acfa3dc8fc5fd/docs/assets/benchmark-tbm-2026-08-21-v3.webp" alt="TBM latency comparison for GLRMask and LLGuidance in the corrected 20 August 2026 engineering run" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/IsaacBreen/glrmask/fbc60288c7d86701a03a3100fa7acfa3dc8fc5fd/docs/assets/benchmark-ttfm-2026-08-21-v3.webp" alt="TTFM comparison for GLRMask and LLGuidance in the corrected 20 August 2026 engineering run" width="100%">
</p>

The old CFA runner used llguidance 1.6.1 and Linux thread-CPU timing; the final native runner uses a different, stricter methodology. See the [20 August engineering benchmark report](https://github.com/IsaacBreen/glrmask/blob/main/docs/benchmark-cfa-full-2026-08-20.md) for the exact corpus, hardware, fix/rerun provenance, build numbers, and interpretation limits.
