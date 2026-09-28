"""Diagnostic commitment must preserve the public compiled-child language."""
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native

TOKENS = {0: b'a', 1: b'b', 2: b'c', 3: b'<', 4: b'>',
          7: b'ab', 8: b'ac', 10: b'<ab>', 11: b'<ac>', 65535: b'ab'}
ROOT_END = 131071
CHILD_END = 511
LANGUAGE = (b'<ab>', b'<ac>')
HISTORIES = ((3, 7, 4), (10,), (b'<', 0, 2, 4), (3, 65535, b'>'))


def mask_oracle(value, state, prefix, terminated=False):
    expected = np.zeros(value.mask_len() + 2, dtype=np.uint32)
    if not terminated:
        for token, piece in TOKENS.items():
            if any(word.startswith(prefix + piece) for word in LANGUAGE):
                expected[token // 32] |= np.uint32(1 << (token % 32))
        if prefix in LANGUAGE:
            expected[ROOT_END // 32] |= np.uint32(1 << (ROOT_END % 32))
    actual = np.full(len(expected), -1, dtype=np.int32)
    state.fill_mask(actual)
    np.testing.assert_array_equal(actual.view(np.uint32), expected)
    assert state.is_accepting() == (prefix in LANGUAGE)
    assert state.is_terminated() == terminated
    assert not state.is_rejected()


@pytest.mark.parametrize('mode', ('FAST_BUILD', 'FAST_RUNTIME', 'AUTO'))
@pytest.mark.parametrize('loaded', (False, True))
@pytest.mark.parametrize('observe_each_step', (False, True))
@pytest.mark.parametrize('wrapper', ('profiled', 'per_advance', 'per_advance_fast'))
def test_profiled_commit_keeps_compiled_child_control_closure(
    mode, loaded, observe_each_step, wrapper, monkeypatch
):
    monkeypatch.setenv('GLRMASK_PROFILE_ALLOW_FAST_PATHS',
                       '1' if wrapper == 'per_advance_fast' else '0')
    vocab = g.Vocab.from_id_to_bytes(TOKENS)
    optimization = getattr(g.Optimization, mode)
    child = g.Grammar.from_ebnf('start ::= "ab" | "ac"').compile(
        vocab, optimization=optimization, end_tokens=[CHILD_END])
    parent = g.Grammar.from_glrm(
        'glrm 1; start root; extern grammar child; nt root = "<" child ">";'
    ).compile_unlinked(vocab)
    value = parent.bind('child', child).link(
        optimization=optimization, end_tokens=[ROOT_END])
    if loaded:
        value = g.Constraint.load(value.save())

    def commit(state, token):
        if wrapper == 'profiled':
            profile = native._internal.commit_token_profiled(state, token)
        else:
            result = native._internal.commit_token_per_advance(state, token)
            profile = result['commit_profile']
            assert isinstance(result['advances'], list)
        assert profile['total_ns'] >= profile['mask_cache_reuse_ns']
        assert all(isinstance(v, int) and v >= 0 for v in profile.values())

    for history in HISTORIES:
        ordinary, diagnostic = value.start(), value.start()
        prefix = b''
        for action in history:
            if observe_each_step:
                for state in (ordinary, diagnostic):
                    mask_oracle(value, state, prefix)
            if isinstance(action, bytes):
                ordinary.commit_bytes(action)
                diagnostic.commit_bytes(action)
                prefix += action
            else:
                ordinary.commit_token(action)
                commit(diagnostic, action)
                prefix += TOKENS[action]
        for state in (ordinary, diagnostic):
            mask_oracle(value, state, prefix)
        ordinary.commit_token(ROOT_END)
        commit(diagnostic, ROOT_END)
        for state in (ordinary, diagnostic):
            mask_oracle(value, state, prefix, terminated=True)
