"""The empty literal is a grammar epsilon, not a zero-byte model token."""
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native

SOURCES={
    "ebnf": 'start ::= "" | "a"',
    "lark": 'start: "" | "a"',
    "glrm": 'glrm 1; start root; nt root = "" | "a";',
}
TOKENS={0:b"", 1:b"a", 2:b"b", 31:b"", 63:b"a"}

@pytest.mark.parametrize("kind", SOURCES)
@pytest.mark.parametrize("mode", ("FAST_BUILD", "FAST_RUNTIME", "AUTO", "O2"))
@pytest.mark.parametrize("loaded", (False,True))
def test_empty_literal_accepts_epsilon_without_admitting_empty_tokens(kind,mode,loaded):
    vocab=g.Vocab.from_id_to_bytes(TOKENS)
    if mode == "O2":
        ctor=getattr(native.DynamicConstraint, "from_glrm_grammar" if kind == "glrm" else "from_" + kind)
        value=ctor(SOURCES[kind],vocab,vocab_partition=True)
        if loaded:value=native.DynamicConstraint.load(value.save(),vocab)
    else:
        value=getattr(g.Grammar,"from_" + kind)(SOURCES[kind]).compile(
            vocab,optimization=getattr(g.Optimization,mode))
        if loaded:value=g.Constraint.load(value.save())
    state=value.start()
    assert state.is_accepting()
    assert not state.is_rejected()
    for _ in range(2):
        actual=np.full(value.mask_len()+2,-1,dtype=np.int32)
        state.fill_mask(actual)
        expected=np.zeros_like(actual,dtype=np.uint32)
        for token in (1,63):expected[token//32] |= np.uint32(1 << (token%32))
        np.testing.assert_array_equal(actual.view(np.uint32),expected)
    state.commit_bytes(b"")
    assert state.is_accepting()
    state.commit_token(1)
    assert state.is_accepting()
    actual=np.full(value.mask_len()+2,-1,dtype=np.int32)
    state.fill_mask(actual)
    assert not actual.any()
