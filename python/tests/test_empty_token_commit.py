"""Empty byte payloads cannot consume a grammar transition by doing nothing."""
import numpy as np
import pytest
import glrmask as g
import test_control_byte_forks as t

MODES = ('FAST_BUILD', 'FAST_RUNTIME', 'AUTO')
WRAPPERS = ('ordinary', 'timed', 'profiled', 'per_advance')


def build(mode, loaded):
    entries = dict(t.TOKENS)
    entries[512] = b''  # Known to the vocabulary, not bound to an exact terminal.
    vocab = g.Vocab.from_id_to_bytes(entries)
    value = (g.Grammar.from_glrm(t.SOURCE).bind('MARK', vocab.tokens(sorted(t.SPECIAL_IDS)))
             .compile(vocab, optimization=getattr(g.Optimization, mode), end_tokens=[t.END]))
    return g.Constraint.load(value.save()) if loaded else value


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('loaded', (False, True))
@pytest.mark.parametrize('wrapper', WRAPPERS)
def test_empty_exact_marker_is_consumed_once(mode, loaded, wrapper):
    value = build(mode, loaded)
    state = value.start()
    t.commit(state, 97, wrapper)
    assert state.mask()[511]
    t.commit(state, 511, wrapper)
    mask = state.mask()
    assert np.flatnonzero(mask).tolist() == [99, 261]
    assert not mask[511], 'A second empty MARK is not a continuation of this grammar'
    # A forbidden second marker rejects, rather than leaving the first state live.
    try:
        t.commit(state, 511, wrapper)
    except ValueError:
        pass
    assert state.is_rejected()
    assert not state.is_accepting()
    assert not np.any(state.mask())


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('loaded', (False, True))
@pytest.mark.parametrize('wrapper', WRAPPERS)
def test_empty_byte_commit_is_noop_but_empty_model_token_is_not(mode, loaded, wrapper):
    value = build(mode, loaded)
    state = value.start()
    t.commit(state, 97, wrapper)
    before = state.mask().copy()
    state.commit_bytes(b'')
    np.testing.assert_array_equal(state.mask(), before)
    assert not before[512]
    try:
        t.commit(state, 512, wrapper)
    except ValueError:
        pass
    assert state.is_rejected(), 'Empty model-token bytes must not create an identity edge'
    assert not state.is_accepting()
    assert not np.any(state.mask())
