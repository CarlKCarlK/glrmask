"""Profiled masking has separate fast paths; it must obey the same token domain."""
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native
from test_empty_mask_domain import TOKENS

@pytest.mark.parametrize('mode',('FAST_BUILD','FAST_RUNTIME','AUTO'))
@pytest.mark.parametrize('loaded',(False,True))
@pytest.mark.parametrize('nullable',(False,True))
def test_all_mask_entrypoints_exclude_empty_byte_aliases(mode,loaded,nullable):
    vocab=g.Vocab.from_id_to_bytes(TOKENS)
    words=(b'',b'ab',b'ac') if nullable else (b'ab',b'ac')
    source='start ::= '+' | '.join('"'+w.decode()+'"' for w in words)
    value=g.Grammar.from_ebnf(source).compile(vocab,optimization=getattr(g.Optimization,mode))
    if loaded:value=g.Constraint.load(value.save())
    for entry in ('profiled','timed','ordinary'):
        state=value.start();prefix=b''
        for next_token in (1,2,None):
            expected=np.zeros(value.mask_len()+2,dtype=np.uint32)
            for token,data in TOKENS.items():
                if data and any(w.startswith(prefix+data) for w in words):
                    expected[token//32] |= np.uint32(1 << (token%32))
            for _ in range(2):
                output=np.full(len(expected),-1,dtype=np.int32)
                if entry=='profiled':native._internal.fill_mask_profiled(state,output)
                elif entry=='timed':native._internal.fill_mask_timed_ns(state,output)
                else:state.fill_mask(output)
                np.testing.assert_array_equal(output.view(np.uint32),expected,err_msg=f'{entry=} {prefix=}')
            assert state.is_accepting()==(prefix in words)
            if next_token is not None:state.commit_token(next_token);prefix+=TOKENS[next_token]
