"""Empty byte payloads have no byte-language mask contribution.

They remain eligible as explicit exact-token terminals or generation-root ends.
Tests exercise first-use, per-state and shared caches without assuming polarity.
"""
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native
from test_finite_language_masks import accepted_prefix

TOKENS = {0:b'', 1:b'a', 2:b'b', 3:b'c', 4:b'x', 5:b'ab', 6:b'ac',
          31:b'', 63:b'', 300:b'a', 65535:b''}
EMPTY_IDS = (0,31,63,65535)
MODES=('FAST_BUILD','FAST_RUNTIME','AUTO','O2')


def mask(value,state,allowed):
    width=value.mask_len()+2
    expected=np.zeros(width,dtype=np.uint32)
    for token in allowed:expected[token//32] |= np.uint32(1 << (token%32))
    for poison in (-1,0x55555555):
        output=np.full(width,poison,dtype=np.int32)
        state.fill_mask(output)
        np.testing.assert_array_equal(output.view(np.uint32),expected)


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('loaded',(False,True))
@pytest.mark.parametrize('nullable',(False,True))
def test_empty_aliases_never_enter_byte_masks(mode,loaded,nullable):
    vocab=g.Vocab.from_id_to_bytes(TOKENS)
    words=(b'',b'ab',b'ac') if nullable else (b'ab',b'ac')
    source='start ::= '+ ' | '.join('"'+w.decode()+'"' for w in words)
    if mode=='O2':
        value=native.DynamicConstraint.from_ebnf(source,vocab,vocab_partition=True)
        if loaded:value=native.DynamicConstraint.load(value.save(),vocab)
    else:
        value=g.Grammar.from_ebnf(source).compile(vocab,optimization=getattr(g.Optimization,mode))
        if loaded:value=g.Constraint.load(value.save())
    for _ in range(3):
        state=value.start();prefix=b''
        for token in (1,2):
            allowed={i for i,b in TOKENS.items() if b and any(w.startswith(prefix+b) for w in words)}
            mask(value,state,allowed)
            assert state.is_accepting() == accepted_prefix(set(words), prefix)
            state.commit_token(token);prefix+=TOKENS[token]
        mask(value,state,set())
        assert state.is_accepting()


@pytest.mark.parametrize('mode',MODES[:-1])
@pytest.mark.parametrize('loaded',(False,True))
def test_empty_root_end_restores_only_selected_id(mode,loaded):
    vocab=g.Vocab.from_id_to_bytes(TOKENS)
    value=g.Grammar.from_ebnf('start ::= "ab"').compile(vocab,optimization=getattr(g.Optimization,mode),end_tokens=[31])
    if loaded:value=g.Constraint.load(value.save())
    for _ in range(3):
        state=value.start();mask(value,state,{1,5,300})
        state.commit_token(5);mask(value,state,{31})
        assert state.is_accepting()
        state.commit_token(31);mask(value,state,set())
        assert state.is_terminated()


@pytest.mark.parametrize('mode',MODES[:-1])
@pytest.mark.parametrize('loaded',(False,True))
def test_empty_exact_marker_preserves_identity_not_byte_aliases(mode,loaded):
    vocab=g.Vocab.from_id_to_bytes(TOKENS)
    source='glrm 1; start root; extern token MARK; nt root = MARK "ab";'
    value=g.Grammar.from_glrm(source).bind('MARK',vocab.token(63)).compile(vocab,optimization=getattr(g.Optimization,mode))
    if loaded:value=g.Constraint.load(value.save())
    for _ in range(3):
        state=value.start();mask(value,state,{63})
        state.commit_token(63);mask(value,state,{1,5,300})
        state.commit_token(5);mask(value,state,set())
        assert state.is_accepting()
