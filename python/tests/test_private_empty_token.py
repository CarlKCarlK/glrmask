"""Supplemental private dynamic/O2 empty-token behavior; no timing claims."""
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native

@pytest.mark.parametrize('partition', (False, True))
@pytest.mark.parametrize('loaded', (False, True))
@pytest.mark.parametrize('wrapper', ('ordinary', 'profiled'))
@pytest.mark.parametrize('prefix', (b'', b'a'))
def test_private_dynamic_empty_token_cannot_be_identity(partition, loaded, wrapper, prefix):
    v=g.Vocab.from_id_to_bytes({0:b'a', 1:b'b', 2:b'', 300:b'a'})
    c=native.DynamicConstraint.from_ebnf('start ::= "a" "b"', v, vocab_partition=partition)
    if loaded:c=native.DynamicConstraint.load(c.save(), v)
    s=c.start()
    if prefix:s.commit_bytes(prefix)
    before=s.mask().copy()
    assert not before[2]
    s.commit_bytes(b'')
    np.testing.assert_array_equal(s.mask(),before)
    try:
        if wrapper=='ordinary':s.commit_token(2)
        else:s.commit_token_profiled(2)
    except ValueError:
        pass
    assert s.is_rejected()
    assert not np.any(s.mask())
    assert not s.is_accepting()
