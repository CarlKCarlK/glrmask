"""Exercise wide output coordinates together with compiled-child metadata.

The expected masks come from an independently enumerated finite language,
not from another GLRMask engine or a previously captured output.
"""
from __future__ import annotations

import glrmask as g
import numpy as np
import pytest


MODES = ("FAST_BUILD", "FAST_RUNTIME", "AUTO")
HIGH = 1 << 21
CHILD_END = 4094
PARENT_END = HIGH + 95
WORDS = (b"<>", b"<ab>", b"<ac>")
ENTRIES = {i: bytes([i]) for i in range(256)} | {
    256: b"<>", 257: b"<ab>", 258: b"<ac>",
    HIGH + 31: b"a", HIGH + 32: b"ab", HIGH + 33: b"ac>",
}


def grammar(body: str):
    return g.Grammar.from_glrm("glrm 1; start root; " + body)


def restored(value):
    # No external vocabulary: the artifact must carry its own coordinates.
    return type(value).load(value.save())


def assert_mask(state, constraint, prefix: bytes) -> None:
    allowed = {token for token, data in ENTRIES.items()
               if any(word.startswith(prefix + data) for word in WORDS)}
    if prefix in WORDS:
        allowed.add(PARENT_END)
    expected = np.zeros(constraint.mask_len() + 3, dtype=np.uint32)
    for token in allowed:
        expected[token // 32] |= np.uint32(1 << (token % 32))
    buffer = np.empty(len(expected), dtype=np.int32)
    for poison in (-1, 0x55555555):
        buffer.fill(poison)
        state.fill_mask(buffer)
        mismatch = np.flatnonzero(buffer.view(np.uint32) != expected)
        assert not len(mismatch), (prefix, mismatch[:8].tolist())
    assert state.is_accepting() == (prefix in WORDS)
    assert not state.is_rejected()


def assert_language(constraint) -> None:
    prefixes = sorted({word[:i] for word in WORDS for i in range(len(word) + 1)})
    for prefix in prefixes:
        state = constraint.start()
        if prefix:
            state.commit_bytes(prefix)
        assert_mask(state, constraint, prefix)
        if prefix in WORDS:
            state.commit_token(PARENT_END)
            assert state.is_terminated()
            buffer = np.full(constraint.mask_len() + 3, -1, dtype=np.int32)
            state.fill_mask(buffer)
            assert not np.any(buffer)
    # Distinct segmentations cross the child boundary or use a high-ID alias.
    for history in ((256,), (257,), (258,), (60, 62),
                    (60, HIGH + 31, 98, 62), (60, HIGH + 32, 62),
                    (60, HIGH + 33)):
        state, prefix = constraint.start(), b""
        for token in history:
            assert_mask(state, constraint, prefix)
            state.commit_token(token)
            prefix += ENTRIES[token]
        assert prefix in WORDS
        assert_mask(state, constraint, prefix)


@pytest.mark.parametrize("child_mode", MODES)
@pytest.mark.parametrize("parent_mode", MODES)
def test_nullable_child_roundtrips_preserve_wide_masks(child_mode, parent_mode):
    vocab = g.Vocab.from_id_to_bytes(ENTRIES)
    child_option = getattr(g.Optimization, child_mode)
    parent_option = getattr(g.Optimization, parent_mode)
    child = grammar('nt root = ("a" ("b" | "c"))?;').compile(
        vocab, optimization=child_option, end_tokens=[CHILD_END])
    child = restored(child)
    assert not child.start().is_accepting()  # Standalone nonempty-generation policy.

    # Passing through a compiled, saved, loaded wrapper must preserve nullability.
    wrapper = restored(grammar('extern grammar leaf; nt root = leaf;')
                       .compile_unlinked(vocab))
    wrapped = restored(wrapper.bind("leaf", child).link(
        optimization=child_option, end_tokens=[CHILD_END]))
    parent = restored(grammar('extern grammar child; nt root = "<" child ">";')
                      .compile_unlinked(vocab))
    for payload in (child, wrapped):
        value = parent.bind("child", payload).link(
            optimization=parent_option, end_tokens=[PARENT_END])
        cold = restored(value)  # Save before the first mask query.
        assert_language(value)
        assert_language(cold)
        assert_language(restored(value))  # Save again after masks populated caches.
