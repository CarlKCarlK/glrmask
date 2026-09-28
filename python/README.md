# GLRMask for Python

Extremely fast grammar-constrained decoding for LLMs.

The `glrmask` package compiles a grammar together with a model vocabulary and exposes an incremental next-token mask for use inside a decoding loop.

## Allocator policy

The Python extension uses mimalloc with delayed automatic purging enabled. It
does not override `MIMALLOC_PURGE_DELAY`, whose mimalloc v3 default is 1000 ms.

GLRMask defaults each purge to a memory **reset** (`MADV_FREE` on supported Unix
systems and `MEM_RESET` on Windows) rather than a synchronous decommit. Reset
pages remain reclaimable by the operating system and reusable by mimalloc, but
process RSS may not decrease immediately. This avoids charging immediate page
decommit work to an otherwise unrelated runtime allocation.

Most runtime work also uses bounded preallocated parser, tokenizer, accumulator,
and mask storage. Ordinary applications therefore need no allocator lifecycle
calls or manual trimming. Set `MIMALLOC_PURGE_DECOMMITS=1` before importing
GLRMask when immediate RSS reduction is more important than allocator tail
latency.

The unstable `glrmask._internal.mimalloc_purge_delay()`,
`glrmask._internal.mimalloc_purge_decommits()`, and
`glrmask._internal.collect_allocator(force=True)` helpers remain available for
diagnostics and controlled experiments.

## Installation

```bash
python -m pip install glrmask
```

Published wheels include the native extension and support Python 3.9 through 3.13.

## Quickstart

```python
import glrmask

vocab = glrmask.Vocab.from_dict({
    b"hello": 0,
    b" ": 1,
    b"world": 2,
    b"<eos>": 3,
})
constraint = glrmask.Grammar.from_ebnf(
    'start ::= "hello" " " "world"'
).compile(vocab, end_tokens=[3])
state = constraint.start()

assert state.mask().tolist() == [True, False, False, False]
state.commit_token(0)
assert state.mask().tolist() == [False, True, False, False]
state.commit_token(1)
state.commit_token(2)
assert state.is_accepting()
assert state.mask()[3]
state.commit_token(3)
assert state.is_terminated()
```

`state.mask()` returns a NumPy Boolean array indexed by model token ID. Pass `state.mask(size)` when the model's logits vector is wider than the constraint's natural token coordinate.

## Public model

The ordinary API has four layers:

- `Grammar`: immutable source description with source-grammar/exact-token bindings.
- `UnlinkedConstraint`: reusable compiled machinery for one exact vocabulary in pre-link form; it may intentionally remain open.
- `Constraint`: closed, rooted, immediately runnable compiled constraint.
- `ConstraintState`: mutable state for one generated sequence.

`Grammar.bind(...)` and `UnlinkedConstraint.bind(...)` return new values and leave the receiver reusable.

### Vocabulary and exact tokens

Create a vocabulary from token bytes to IDs or IDs to bytes:

```python
vocab = glrmask.Vocab.from_dict({b"yes": 0, b"no": 1})
vocab = glrmask.Vocab.from_id_to_bytes({0: b"yes", 1: b"no"})
```

For `llama-cpp-python`:

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf", logits_all=True)
vocab = glrmask.Vocab.from_llama_cpp(llm)
end_token_ids = vocab.llama_cpp_end_token_ids
```

`from_llama_cpp()` keeps EOG, control, unused, and empty-piece IDs as
**exact-only** model tokens even though they are omitted from the byte
vocabulary. They can therefore be used with `vocab.token(id)` /
`vocab.tokens(ids)` without inventing fake bytes, while EOG IDs can be supplied
directly as `end_tokens`.

Use `vocab.token(id)` or `vocab.tokens(ids)` for `extern token` bindings. These values retain the complete vocabulary identity; a binding from an incompatible vocabulary is rejected even if its numeric ID happens to match.

```python
grammar = glrmask.Grammar.from_glrm('''
glrm 1;
start message;
extern token TOOL_CALL;
nt message = TOOL_CALL "lookup()";
''')
grammar = grammar.bind("TOOL_CALL", vocab.token(tool_call_token_id))
constraint = grammar.compile(vocab)
```

### Compile a grammar

Construct descriptions with:

```python
glrmask.Grammar.from_json_schema(schema)
glrmask.Grammar.from_glrm(grammar)
glrmask.Grammar.from_lark(grammar)
glrmask.Grammar.from_ebnf(grammar)
```

Then compile a complete description:

```python
constraint = grammar.compile(
    vocab,
    end_tokens=end_token_ids,
    optimization=glrmask.Optimization.AUTO,
)
```

The intent-level optimization choices are:

- `Optimization.AUTO`
- `Optimization.FAST_BUILD`
- `Optimization.FAST_RUNTIME`

They preserve language semantics and all return the same `Constraint` type. They do not expose GLRMask's internal static/dynamic/O1/O2/O3 engines.

### Bind source children

Source composition stays in the source world:

```python
parent = glrmask.Grammar.from_glrm('''
glrm 1;
start document;
extern grammar payload;
nt document = "{" payload "}";
''')

source_child = glrmask.Grammar.from_json_schema(payload_schema)
a = parent.bind("payload", source_child).compile(vocab)
```

`Grammar.bind` does not accept a compiled `Constraint`. To reuse a compiled parent
with request-specific compiled children, use `compile_unlinked` instead.

### Cache an unlinked parent

Use `compile_unlinked` when a parent is reused across requests:

```python
host = parent.compile_unlinked(vocab)
child = glrmask.Grammar.from_json_schema(payload_schema).compile(vocab)

bound = host.bind("payload", child)
constraint = bound.link(optimization=glrmask.Optimization.FAST_RUNTIME)
```

`UnlinkedConstraint.bind` is compiled-only: accepted grammar children are runnable
`Constraint` values (plus `ExactToken`/`ExactTokens` for token slots). It does not
accept a source `Grammar` or another `UnlinkedConstraint`. Composition remains
deferred until `link`, so the terminal optimization preference can choose the link strategy.

Unlinked constraints are serializable:

```python
artifact = host.save()
host = glrmask.UnlinkedConstraint.load(artifact)
```

### Decode

Create one state per generated sequence:

```python
state = constraint.start()

while generating:
    mask = state.mask(model_vocab_size)
    token_id = sample_with_mask(logits, mask)
    state.commit_token(token_id)
    if state.is_terminated():
        break
```

The main state operations are:

- `mask(size=None)`: return the allowed-token mask.
- `fill_mask(words)`: fill a writable, aligned, contiguous NumPy `int32` buffer of packed mask words. An existing `uint32` array can be passed as `words.view(np.int32)` without copying.
- `commit_token(token_id)`: advance by one model token.
- `commit_bytes(data)`: advance by raw bytes.
- `forced()`: return a forced token sequence when one can be determined.
- `is_accepting()`: grammar-body acceptance at the current prefix.
- `is_rejected()`: irrecoverably invalid prefix.
- `is_terminated()`: an allowed final end token has been committed.

### End tokens

End tokens are final-root policy, not grammar-child semantics:

```python
constraint = grammar.compile(vocab, end_tokens=[eos_id])
```

An end token is allowed only when the grammar body is accepting. A child's previous end-token policy is not inherited when that compiled constraint is linked into an unlinked parent.

### Constraint persistence

`Constraint` objects are immutable/shareable and remain composable after loading:

```python
artifact = constraint.save()
constraint = glrmask.Constraint.load(artifact)
```

Passing `vocab=` to `Constraint.load` or `UnlinkedConstraint.load` is optional and validates/shares an already-existing exact vocabulary object.

Compiled artifacts are caches, not a stable cross-version interchange format.
Keep the original grammar and exact vocabulary so incompatible caches can be
rebuilt. Loading an older compiled artifact does not re-run the compiler or
automatically incorporate later compiler correctness fixes.

## Grammar formats

GLRMask accepts JSON Schema, GLRM, Lark, and EBNF. GLRM is the native composition format:

```glrm
glrm 1;
start value;
t NUMBER = /-?(0|[1-9][0-9]*)/;
nt value = NUMBER | "null";
```

External compiled children use `extern grammar NAME;`; exact model-token slots use `extern token NAME;`. Inline `g name = { ... };` grammars and externally bound child grammars have the same language semantics, including scope-local ignores.

Lark and EBNF also support explicit `@token(<id>)` atoms when a numeric model token ID is deliberately part of the grammar source.

The top-level package intentionally exposes intent-level `Optimization`, not the historical engine-specific dynamic/vocabulary-partition types. Repository experiments remain available through explicit internal/submodule imports and carry no public compatibility guarantee.


## Source builds

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install ./python
```

Building from source requires a Rust toolchain and the platform's native linker and build tools. On Windows, activate the environment with `.venv\Scripts\activate`.
