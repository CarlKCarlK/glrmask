"""Checked NumPy layout contracts. Run only against the alignment-guarded build.

The unprotected predecessor must not be used as a runtime oracle here: forming
an aligned Rust slice from an unaligned array violates a precondition. The
predecessor finding is established by source inspection, not unsafe execution.
"""
import numpy as np
import pytest
from test_private_mask_boundaries import adapter

ENTRIES=('public','static_timed','static_profiled','dynamic','dynamic_timed','partition')


@pytest.mark.parametrize('entry,loaded',[(e,l) for e in ENTRIES for l in (False,True)
                                        if not (e=='partition' and l)])
@pytest.mark.parametrize('offset',(1,2,3))
def test_misaligned_mask_is_rejected_before_native_access(entry,loaded,offset):
    fill,width,expected,check=adapter(entry,loaded)
    storage=np.full((width+3)*4+8,0x5A,dtype=np.uint8)
    buffer=np.ndarray((width+3,),dtype=np.int32,buffer=storage,offset=offset)
    assert buffer.flags.c_contiguous and buffer.flags.writeable
    assert not buffer.flags.aligned and buffer.ctypes.data%4==offset
    before=storage.copy()
    with pytest.raises(ValueError,match='align'):
        fill(buffer)
    np.testing.assert_array_equal(storage,before)
    check()
    valid=np.full(width+3,-1,dtype=np.int32)
    fill(valid)
    np.testing.assert_array_equal(valid.view(np.uint32),expected)
