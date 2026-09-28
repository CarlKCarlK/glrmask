"""Independent finite-language checks for exact markers in nullable compiled children.

The oracle uses byte symbols and one exact-marker symbol. Empty token payloads
never add byte transitions. It deliberately does not require forced() to find
all forced tokens: any returned sequence must merely be sound and nonmutating.
"""
from itertools import product

import numpy as np
import pytest

MARK = 256
CHILD_END = 63
ROOT_END = 65535
UNKNOWN = 65534
EXACT_IDS = frozenset((31, 32))
TOKENS = {
    0: b"", 1: b"a", 2: b"b", 3: b"(", 4: b")", 5: b"[", 6: b"]",
    7: b"([", 8: b"])", 9: b"[a]", 10: b"([])", 11: b"([a])a",
    12: b"([b])a", 13: b"([a])", 14: b"a", 31: b"", 32: b"b",
    CHILD_END: b"", ROOT_END: b"",
}
LEAF_LANGUAGE = ((), (97,), (MARK,))
LANGUAGE = frozenset((40, 91) + left + (93, 41) + right
                     for left, right in product(LEAF_LANGUAGE, repeat=2))
HISTORIES = (
    (7, 31, 8, 31),
    (11,),
    (10,),
    (7, 1, 8, 14),
    (7, 32, 8, 31),
    (b"([", 31, b"])", b"a"),
)


def advance_bytes(residuals, piece):
    prefix = tuple(piece)
    return frozenset(row[len(prefix):] for row in residuals
                     if row[:len(prefix)] == prefix)


def advance_token(residuals, token):
    piece = TOKENS[token]
    result = set(advance_bytes(residuals, piece)) if piece else set()
    if token in EXACT_IDS:
        result.update(row[1:] for row in residuals if row and row[0] == MARK)
    return frozenset(result)


def expected(residuals, width, terminated=False):
    words = np.zeros(width, dtype=np.uint32)
    if not terminated:
        for token in TOKENS:
            allowed = (() in residuals) if token == ROOT_END else bool(advance_token(residuals, token))
            if allowed:
                words[token // 32] |= np.uint32(1 << (token % 32))
    return words


def build(mode, loaded):
    import glrmask as g
    vocab = g.Vocab.from_id_to_bytes(TOKENS)
    optimization = getattr(g.Optimization, mode)
    leaf = (g.Grammar.from_glrm(
        'glrm 1; start leaf; extern token MARK; nt leaf = "" | "a" | MARK;')
        .bind('MARK', vocab.tokens(sorted(EXACT_IDS)))
        .compile(vocab, optimization=optimization, end_tokens=[CHILD_END]))
    wrapper = (g.Grammar.from_glrm(
        'glrm 1; start wrap; extern grammar leaf; nt wrap = "[" leaf "]";')
        .compile_unlinked(vocab).bind('leaf', leaf)
        .link(optimization=optimization, end_tokens=[CHILD_END]))
    value = (g.Grammar.from_glrm(
        'glrm 1; start root; extern grammar wrap; extern grammar leaf;'
        ' nt root = "(" wrap ")" leaf;')
        .compile_unlinked(vocab).bind('wrap', wrapper).bind('leaf', leaf)
        .link(optimization=optimization, end_tokens=[ROOT_END]))
    return g.Constraint.load(value.save()) if loaded else value


def check(value, state, residuals, native, profiled_first, terminated=False):
    want = expected(residuals, value.mask_len() + 2, terminated)
    outputs = []
    for route in (('profiled', 'ordinary') if profiled_first else ('ordinary', 'profiled')):
        storage = np.full(len(want) + 4, -314159, dtype=np.int32)
        output = storage[2:-2]
        output[:] = -1
        if route == 'profiled':
            native._internal.fill_mask_profiled(state, output)
        else:
            state.fill_mask(output)
        np.testing.assert_array_equal(output.view(np.uint32), want,
                                      err_msg=f'{route=} {residuals=}')
        np.testing.assert_array_equal(storage[[0, 1, -2, -1]], [-314159] * 4)
        outputs.append(output.copy())
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert state.is_accepting() == (() in residuals)
    assert state.is_terminated() == terminated
    assert not state.is_rejected()


def commit(state, token, native, profiled):
    if not profiled:
        state.commit_token(token)
    else:
        result = native._internal.commit_token_profiled(state, token)
        assert result['total_ns'] >= result['mask_cache_reuse_ns']


@pytest.mark.parametrize('mode', ('FAST_BUILD', 'FAST_RUNTIME', 'AUTO'))
@pytest.mark.parametrize('loaded', (False, True))
@pytest.mark.parametrize('profiled_first', (False, True))
def test_nested_nullable_children_preserve_exact_token_domain(mode, loaded, profiled_first, monkeypatch):
    from glrmask import _glrmask as native
    monkeypatch.setenv('GLRMASK_PROFILE_ALLOW_FAST_PATHS', '1')
    value = build(mode, loaded)
    for profiled in (False, True):
        for history in HISTORIES:
            state = value.start()
            residuals = LANGUAGE
            for action in history:
                check(value, state, residuals, native, profiled_first)
                before = residuals
                with pytest.raises(ValueError):
                    commit(state, UNKNOWN, native, profiled)
                check(value, state, before, native, profiled_first)
                if isinstance(action, bytes):
                    residuals = advance_bytes(residuals, action)
                    state.commit_bytes(action)
                else:
                    residuals = advance_token(residuals, action)
                    commit(state, action, native, profiled)
                assert residuals, (mode, history, action)
            check(value, state, residuals, native, profiled_first)
            assert () in residuals
            # Accepting states may also have viable continuations. Their forced
            # sequence may be empty but must never contain an unbound empty ID.
            forced = state.forced()
            remaining = residuals
            for token in forced:
                assert token != ROOT_END
                remaining = advance_token(remaining, token)
                assert remaining, (mode, history, forced)
            check(value, state, residuals, native, profiled_first)
            commit(state, ROOT_END, native, profiled)
            check(value, state, residuals, native, profiled_first, terminated=True)
