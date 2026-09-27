"""Python ownership and packed-buffer boundaries; finite-language expectations."""
import gc
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native

TOKENS={0:b'a',1:b'b',2:b'c',31:b'ab',63:b'ac',95:b'a',65535:b'ab'}
LANGUAGE=(b'ab',b'ac')
MODES=('FAST_BUILD','FAST_RUNTIME','AUTO')


def built(mode,loaded=False):
    vocab=g.Vocab.from_id_to_bytes(TOKENS)
    if mode=='O2':
        value=native.DynamicConstraint.from_ebnf('start ::= "ab" | "ac"',vocab,vocab_partition=True)
        loader=native.DynamicConstraint
    else:
        value=g.Grammar.from_ebnf('start ::= "ab" | "ac"').compile(vocab,optimization=getattr(g.Optimization,mode))
        loader=g.Constraint
    return loader.load(value.save(),vocab) if loaded else value


def expected(prefix,width):
    words=np.zeros(width,dtype=np.uint32)
    for token,data in TOKENS.items():
        if any(word.startswith(prefix+data) for word in LANGUAGE):
            words[token//32] |= np.uint32(1 << (token%32))
    return words


def fill(state,buffer,entry):
    if entry=='public':
        assert state.fill_mask(buffer) is None
    elif entry=='timed':
        ns=native._internal.fill_mask_timed_ns(state,buffer)
        assert isinstance(ns,int) and ns>=0
    else:
        profile=native._internal.fill_mask_profiled(state,buffer)
        assert isinstance(profile['total_ns'],int) and profile['total_ns']>=0


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('loaded',(False,True))
@pytest.mark.parametrize('entry',('public','timed','profiled'))
def test_checked_buffer_is_a_writable_view_not_a_copy(mode,loaded,entry):
    value=built(mode,loaded);state=value.start();width=value.mask_len()+2
    storage=np.full(width+4,-314159,dtype=np.int32)
    buffer=storage[2:-2] # Contiguous slice with independently guarded neighbors.
    prefix=b''
    for token in (0,1):
        buffer.fill(-1)
        fill(state,buffer,entry)
        np.testing.assert_array_equal(buffer.view(np.uint32),expected(prefix,width))
        np.testing.assert_array_equal(storage[[0,1,-2,-1]],[-314159]*4)
        assert bool(buffer[token//32] >> (token%32) & 1)
        state.commit_token(token);prefix+=TOKENS[token]
    buffer.fill(-1);fill(state,buffer,entry)
    np.testing.assert_array_equal(buffer.view(np.uint32),expected(prefix,width))
    assert state.is_accepting()


@pytest.mark.parametrize('entry',('public','timed','profiled'))
@pytest.mark.parametrize('kind',('strided','readonly','wrong_dtype','wrong_rank'))
def test_buffer_rejection_does_not_change_state_or_input(entry,kind):
    value=built('FAST_RUNTIME');state=value.start();width=value.mask_len()
    if kind=='strided':buffer=np.full(width*2,-1,dtype=np.int32)[::2]
    elif kind=='readonly':
        buffer=np.full(width,-1,dtype=np.int32);buffer.flags.writeable=False
    elif kind=='wrong_dtype':buffer=np.full(width,-1,dtype=np.int64)
    else:buffer=np.full((1,width),-1,dtype=np.int32)
    before=buffer.copy()
    with pytest.raises((ValueError,TypeError)) as caught:fill(state,buffer,entry)
    if kind=='strided':assert 'Array must be contiguous' in str(caught.value)
    np.testing.assert_array_equal(buffer,before)
    valid=np.full(width,-1,dtype=np.int32);fill(state,valid,entry)
    np.testing.assert_array_equal(valid.view(np.uint32),expected(b'',width))
    state.commit_token(65535)
    assert state.is_accepting()


@pytest.mark.parametrize('mode',MODES+('O2',))
@pytest.mark.parametrize('loaded',(False,True))
def test_multiple_states_outlive_constraint_and_do_not_share_frontiers(mode,loaded):
    value=built(mode,loaded);width=value.mask_len()+1
    first,second=value.start(),value.start()
    del value;gc.collect()
    def check(state,prefix):
        output=np.full(width,-1,dtype=np.int32);state.fill_mask(output)
        np.testing.assert_array_equal(output.view(np.uint32),expected(prefix,width))
    check(first,b'');check(second,b'')
    with pytest.raises(ValueError):first.commit_token(65534) # Unknown ID, not a rejection.
    check(first,b'')
    first.commit_token(95)
    check(first,b'a');check(second,b'')
    second.commit_token(65535);check(second,b'ab')
    check(first,b'a');first.commit_token(2);check(first,b'ac')
    assert first.is_accepting() and second.is_accepting()
