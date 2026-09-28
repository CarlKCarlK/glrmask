"""Exact-token and byte-path union at commitment boundaries.

The oracle operates on finite sequences of bytes and a distinguished exact-token
symbol. It does not call a second grammar engine or assume that diagnostic reset
policies are interchangeable. Each wrapper must preserve this language.
"""
from itertools import product
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native

MARK = 256  # Oracle-only symbol, not a model token or byte.
END = 131071
UNKNOWN = 65534
SPECIAL_IDS = frozenset((300, 301, 511))
TOKENS = {i: bytes([i]) for i in (97, 98, 99, 112, 113, 114, 120, 122)}
TOKENS.update({257: b"ax", 258: b"xb", 259: b"axb", 260: b"axc", 261: b"cp",
               262: b"px", 263: b"pq", 264: b"pxr", 265: b"xq", 266: b"ap",
               300: b"x", 301: b"zz", 511: b""})
SOURCE = '''glrm 1; start root; extern token MARK;
nt root = ("a" MARK "c" | "axb") ("p" MARK "q" | "pxr");'''
LANGUAGE = frozenset(left + right for left, right in product(
    ((97, MARK, 99), tuple(b"axb")), ((112, MARK, 113), tuple(b"pxr"))))
HISTORIES = (
    (97, 300, 99, 112, 301, 113),    # Both byte and exact branches, then only exact.
    (97, 301, 261, 511, 113),        # Failed byte path rescued by exact; empty MARK.
    (97, 511, 261, 300, 113),        # Zero-byte exact marker advances grammar.
    (97, 300, 98, 264),             # The same x token also has ordinary-byte meaning.
    (259, 264),                    # Fully ordinary fused tokens, no exact symbol.
    (257, 98, 262, 114),            # Fused ax/px cannot substitute for an exact MARK.
    (b"a", 301, b"cp", 511, b"q"),
    (b"axb", b"p", 300, b"q"),
)


def advance_bytes(residuals, data):
    if not data:
        return residuals
    prefix = tuple(data)
    return frozenset(row[len(prefix):] for row in residuals if row[:len(prefix)] == prefix)


def advance_token(residuals, token):
    piece = TOKENS[token]
    result = set(advance_bytes(residuals, piece)) if piece else set()
    if token in SPECIAL_IDS:
        result.update(row[1:] for row in residuals if row and row[0] == MARK)
    return frozenset(result)


def expected_mask(residuals, width, terminated=False):
    result = np.zeros(width, dtype=np.uint32)
    if terminated:
        return result
    allowed = [token for token in TOKENS if advance_token(residuals, token)]
    if () in residuals:
        allowed.append(END)
    for token in allowed:
        result[token // 32] |= np.uint32(1 << (token % 32))
    return result


def check(state, value, residuals, *, terminated=False):
    want = expected_mask(residuals, value.mask_len() + 2, terminated)
    for poison in (-1, 0x55555555):
        output = np.full(len(want), poison, dtype=np.int32)
        state.fill_mask(output)
        np.testing.assert_array_equal(output.view(np.uint32), want)
    assert state.is_accepting() == (() in residuals)
    assert state.is_terminated() == terminated
    assert not state.is_rejected()


def commit(state, token, wrapper):
    if wrapper == "ordinary":
        state.commit_token(token)
    elif wrapper == "timed":
        assert native._internal.commit_token_timed_ns(state, token) >= 0
    elif wrapper == "profiled":
        profile = native._internal.commit_token_profiled(state, token)
        assert profile["total_ns"] >= profile["mask_cache_reuse_ns"]
    else:
        result = native._internal.commit_token_per_advance(state, token)
        profile = result["commit_profile"]
        assert profile["total_ns"] >= profile["mask_cache_reuse_ns"]


@pytest.mark.parametrize("mode", ("FAST_BUILD", "FAST_RUNTIME", "AUTO"))
@pytest.mark.parametrize("loaded", (False, True))
@pytest.mark.parametrize("wrapper", ("ordinary", "timed", "profiled", "per_advance"))
@pytest.mark.parametrize("cache_policy", ("eager", "terminal_only"))
def test_exact_control_and_byte_paths_remain_a_union(mode, loaded, wrapper, cache_policy):
    vocab = g.Vocab.from_id_to_bytes(TOKENS)
    value = (g.Grammar.from_glrm(SOURCE).bind("MARK", vocab.tokens(sorted(SPECIAL_IDS)))
             .compile(vocab, optimization=getattr(g.Optimization, mode), end_tokens=[END]))
    if loaded:
        value = g.Constraint.load(value.save())
    for history in HISTORIES:
        state = value.start()
        residuals = LANGUAGE
        for step, action in enumerate(history):
            if cache_policy == "eager":
                check(state, value, residuals)
            # A failed unknown-ID call must not alter any live branch, including
            # cached exact-token admission. The ID is neither byte-backed nor MARK.
            with pytest.raises(ValueError):
                commit(state, UNKNOWN, wrapper)
            if isinstance(action, bytes):
                residuals = advance_bytes(residuals, action)
                assert residuals, (mode, wrapper, history, step, action)
                state.commit_bytes(action)
            else:
                residuals = advance_token(residuals, action)
                assert residuals, (mode, wrapper, history, step, action)
                commit(state, action, wrapper)
        check(state, value, residuals)
        assert residuals == frozenset(((),))
        commit(state, END, wrapper)
        check(state, value, residuals, terminated=True)
