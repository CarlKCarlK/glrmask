"""Model-token length must not erase recursive parser stack obligations."""
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native

SOURCE = 'start root; nt root ::= "a" root "b" | "z";'


def language(prefix):
    """Exact prefix/acceptance predicates for the unbounded language a^n z b^n."""
    opening = len(prefix)-len(prefix.lstrip(b'a'))
    suffix = prefix[opening:]
    if not suffix:
        return True, False
    if not suffix.startswith(b'z'):
        return False, False
    closing = suffix[1:]
    viable = closing == b'b'*len(closing) and len(closing)<=opening
    return viable, viable and len(closing)==opening


def entries(kind):
    tokens = {0:b'a',1:b'b',2:b'z',3:b'x',31:b'azb',32:b'aazbb',63:b'zz'}
    if kind != 'short':
        tokens |= {7:b'aa',9:b'aaaa',11:b'a'*16,256:b'b'*16,1023:b'zbbbb'}
    if kind == 'nested':
        tokens |= {100+n:b'a'*n+b'z'+b'b'*n for n in range(1,9)}
        tokens |= {200+n:b'a'*n+b'z'+b'b'*(n+1) for n in range(1,9)}
    return tokens | {65535:b'a',65536:b'b',131071:b'aazbb'}


def compile_value(mode,tokens,loaded):
    vocab=g.Vocab.from_id_to_bytes(tokens)
    if mode=='O2':
        value=native.DynamicConstraint.from_glrm_grammar(SOURCE,vocab,vocab_partition=True)
        loader=native.DynamicConstraint
    else:
        value=g.Grammar.from_glrm(SOURCE).compile(vocab,optimization=getattr(g.Optimization,mode))
        loader=g.Constraint
    return loader.load(value.save(),vocab) if loaded else value


def check_mask(state,value,tokens,prefix):
    expected=np.zeros(value.mask_len()+3,dtype=np.uint32)
    for token,piece in tokens.items():
        if language(prefix+piece)[0]:
            expected[token//32] |= np.uint32(1 << (token%32))
    for poison in (-1,0x55555555):
        output=np.full(len(expected),poison,dtype=np.int32)
        state.fill_mask(output)
        assert np.array_equal(output.view(np.uint32),expected), (prefix, np.flatnonzero(output.view(np.uint32)!=expected).tolist())
    assert state.is_accepting()==language(prefix)[1]


@pytest.mark.parametrize('mode',['FAST_BUILD','O2','FAST_RUNTIME','AUTO'])
@pytest.mark.parametrize('loaded',[False,True])
@pytest.mark.parametrize('kind',['short','long','nested'])
def test_recursive_terminal_runs_preserve_full_token_masks(mode,loaded,kind):
    tokens=entries(kind)
    value=compile_value(mode,tokens,loaded)
    independent=value.start()
    check_mask(independent,value,tokens,b'')
    # Cross both inline-stack and direct-kernel depth boundaries with byte commits.
    for depth in (0,1,2,3,7,31,63,64,65,80):
        state=value.start()
        prefix=b''
        for token in [0]*depth+[2]+[1]*depth:
            check_mask(state,value,tokens,prefix)
            state.commit_token(token)
            prefix+=tokens[token]
        check_mask(state,value,tokens,prefix)
        assert state.is_accepting()
    # Fused tokens have the same complete meaning as their byte-wise histories.
    for token,piece in tokens.items():
        if not language(piece)[1]:
            continue
        state=value.start()
        check_mask(state,value,tokens,b'')
        state.commit_token(token)
        check_mask(state,value,tokens,piece)
        assert state.is_accepting()
    check_mask(independent,value,tokens,b'')


@pytest.mark.parametrize('mode',['FAST_BUILD','O2','FAST_RUNTIME','AUTO'])
def test_unrelated_long_token_cannot_change_existing_token_admission(mode):
    short=entries('short')
    extended=short | {12:b'a'*16}
    before=compile_value(mode,short,False)
    after=compile_value(mode,extended,False)
    for prefix in (b'',b'a',b'aa',b'az',b'aaz',b'aazb',b'aazbb'):
        for value,tokens in ((before,short),(after,extended)):
            state=value.start()
            if prefix:
                state.commit_bytes(prefix)
            check_mask(state,value,tokens,prefix)
