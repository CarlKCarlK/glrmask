# Table-free parsers from Python

The Python API uses the same Rust compiler, artifact loader and runtime as the
Rust API. Backend selection is per constraint. It does not change process-wide
environment flags or install Python callbacks in token generation.

## Select a backend for an ordinary grammar

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"ab", 2: b"b"})
grammar = glrmask.Grammar.from_ebnf('start ::= "a" "b"?')
constraint = grammar.compile(
    vocab,
    optimization=glrmask.Optimization.FAST_RUNTIME,
    parser_backend=glrmask.ParserBackend.TEMPLATE_DFA,
)
assert constraint.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA

state = constraint.start()
state.commit_token(1)
assert state.is_accepting()
```

`FAST_RUNTIME` selects static masking. `FAST_BUILD` selects the existing
vocabulary-partitioned dynamic masking path for built-in template grammars.
Neither choice retains an LR table in the resulting template constraint.
Omitting `parser_backend` retains the existing `LR_TABLE` default. Pass the enum,
not a string, integer or Boolean flag.

`constraint.parser_backend` is read-only and remains valid after loading.
The built-in compiler may use LR machinery while deriving the templates; this
differs from the data-only provider below, which never needs an LR grammar.

## Compile a parser definition directly

`glrmask.ParserProgram(definition)` accepts a JSON string or a JSON-compatible
mapping with the fields of Rust's public `ParserDefinition`. All input becomes
owned, immutable Rust data during construction. Mutating or deleting the
original mapping after construction cannot change a compiled parser.

This minimal example accepts any number of `a` terminals. Its action relation
is identity, and completion is allowed at every terminal boundary:

```python
import glrmask

def graph(accepting):
    return {"start": 0, "states": [
        {"accepting": accepting, "transitions": []}
    ]}

identity = {
    "pop": graph(True),
    "read": graph(False),
    "push": graph(False),
    "pop_to_read": [],
    "pop_to_push": [],
    "read_to_push": [],
}
program = glrmask.ParserProgram({
    "stack_symbol_count": 1,
    "terminals": [identity],
    "completion": identity,
})
vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"aa", 2: b"b"})
constraint = program.compile(
    vocab,
    [b"a"],
    optimization=glrmask.Optimization.FAST_RUNTIME,
    end_tokens=[63],
)
state = constraint.start()
assert state.mask()[0] and state.mask()[1] and not state.mask()[2]
state.commit_token(1)
assert state.is_accepting() and state.mask()[63]
```

Each transition has `{"label": {"Symbol": 0}, "target": 1}` shape. A POP default
label is the string `"Default"`. Every state supplies `accepting` and
`transitions`; each graph supplies `start` and `states`. Optional phase links
are arrays of target IDs or `None`; missing trailing entries mean no link.

There is one terminal relation per lexer pattern, in the same order.
`terminal_patterns` uses `bytes` for exact literal bytes and `str` for a UTF-8
regular expression. For a Unicode literal, pass its UTF-8 bytes explicitly.
Other element types are rejected; an accidental string is not silently
interpreted as a literal. Empty or nullable terminals are invalid.

The initial concrete stack is `[0]`. POP consumes a top symbol, READ observes
the current top without consuming it, and PUSH appends a symbol. Explicit POP
edges shadow DEFAULT even when they lead to rejection. Completion queries the
domain of its relation without executing output actions. Acyclic graphs may
still represent exponentially many output stacks; they remain shared graphs.
See `template-parser.md` for the full relation contract and static normalization
requirements, including valid empty concrete stacks.

`ignore_terminal=N` is permitted only when terminal N has the canonical
identity relation shown above. This prevents a lexer shortcut from discarding
a meaningful parser action.

For data-only programs, `FAST_RUNTIME` uses the shared static compiler;
`FAST_BUILD`, `AUTO` and omission of `optimization` use the shared dynamic
mask engine. All choices remain table-free. Excessive static expansion raises
`ValueError` rather than silently falling back to dynamic masking or LR.

Definition text is checked against a 64 MiB limit before the owned Rust copy
and JSON decoder. Python's serialization of a mapping may allocate before this
check. The Rust graph validator and compiler apply their own resource bounds;
this is not a whole-process memory guarantee.

## Save, load and vocabulary identity

```python
raw = constraint.save()
loaded = glrmask.Constraint.load(raw)
assert loaded.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
assert loaded.save() == raw

external = constraint.save_with_external_vocab()
loaded = glrmask.Constraint.load(external, vocab=vocab)
assert loaded.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
```

An external-vocabulary artifact requires the exact original vocabulary
mapping. The same number of tokens with different IDs or byte strings is not
sufficient. Self-contained and external forms retain the end-token policy.

## Composition and compatibility

Compiled table-free children and compiled table-free linkage are not yet
supported. Binding records an immutable attachment; final linking raises
`ValueError` for unsupported composition rather than quietly attaching a hidden
LR table. `UnlinkedConstraint.link` accepts the typed `parser_backend` keyword so
the request can be checked explicitly. Ordinary LR composition is
unchanged. This compatibility gap is a reason not to switch the global default
based only on faster single-component averages.
