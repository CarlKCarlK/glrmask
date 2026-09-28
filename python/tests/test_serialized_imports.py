"""Source-to-artifact adapters preserve language, aliases and explicit end tokens."""
import numpy as np
import pytest
import glrmask
from glrmask import _glrmask as native

SOURCES = {
    "ebnf": 'start ::= "1" | "2"',
    "lark": 'start: "1" | "2"',
    "json_schema": '{"enum": [1, 2]}',
    "glrm": 'glrm 1; start root; nt root = "1" | "2";',
}
ROUTES = ("plain", "profiled", "partition-profiled")
TOKENS = {i: bytes([i]) for i in range(256)} | {
    256: b"1", 257: b"2", 258: b"12", 259: b" 1", 319: b"<end>", 511: b"1",
}


def compile_artifact(kind, route, source, vocab, end_ids):
    name = f"compile_{kind}_serialized" + ("" if route == "plain" else "_profiled")
    compile_source = getattr(native._internal, name)
    if route == "plain":
        data = compile_source(source, vocab, end_ids)
    else:
        data, compile_ns, save_ns = compile_source(
            source, vocab, end_ids, route == "partition-profiled"
        )
        assert isinstance(compile_ns, int) and compile_ns >= 0
        assert isinstance(save_ns, int) and save_ns >= 0
    return bytes(data)


def mask(state, constraint):
    # Public boolean masks and caller-owned packed masks must use model IDs.
    boolean = np.asarray(state.mask((constraint.mask_len() + 2) * 32))
    storage = np.full(constraint.mask_len() + 6, -314159, dtype=np.int32)
    packed = storage[2:-2]
    state.fill_mask(packed)
    bits = ((packed.view(np.uint32)[:, None] >> np.arange(32, dtype=np.uint32)) & 1)
    np.testing.assert_array_equal(bits.astype(bool).ravel(), boolean)
    np.testing.assert_array_equal(storage[[0, 1, -2, -1]], [-314159] * 4)
    assert not packed[-2:].any()
    return boolean


@pytest.mark.parametrize("kind", SOURCES)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("with_end", (False, True))
@pytest.mark.parametrize("resaved", (False, True))
def test_serialized_source_adapter_roundtrip(kind, route, with_end, resaved):
    vocab = glrmask.Vocab.from_id_to_bytes(TOKENS)
    ends = [319] if with_end else []
    artifact = compile_artifact(kind, route, SOURCES[kind], vocab, ends)
    constraint = native.DynamicConstraint.load(artifact, vocab)
    if resaved:
        constraint = native.DynamicConstraint.load(bytes(constraint.save()), vocab)
    state = constraint.start()
    initial = mask(state, constraint)
    assert all(initial[i] for i in (49, 50, 256, 257, 511))
    assert not initial[258] and not initial[319]
    # These adapters lower the root value directly, without surrounding whitespace.
    assert not initial[259]
    state.commit_token(511)
    assert not state.is_rejected()
    assert bool(state.is_accepting()) == (not with_end)
    current = mask(state, constraint)
    if with_end:
        assert current[319]
        state.commit_token(319)
        assert state.is_accepting()
        current = mask(state, constraint)
    assert not current[ord("3")]


@pytest.mark.parametrize("kind", SOURCES)
@pytest.mark.parametrize("route", ROUTES)
def test_serialized_source_adapter_rejects_malformed_grammar(kind, route):
    vocab = glrmask.Vocab.from_id_to_bytes(TOKENS)
    source = "{not json}" if kind == "json_schema" else "@@@ definitely not a grammar"
    with pytest.raises(ValueError):
        compile_artifact(kind, route, source, vocab, [])
