"""Independent Python input contracts for private masking and equivalence adapters.

No timing thresholds and no reference to native helper implementation. Public
errors must leave input storage and semantic state reusable. Oracle language:
{ab, ac}; partition expansion uses the explicitly returned disjoint classes.
"""
import gc
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native

TOKENS={0:b'a',1:b'b',2:b'c',31:b'ab',63:b'ac',95:b'a',65535:b'ab'}
LANGUAGE=(b'ab',b'ac')
ENTRIES=('static_timed','static_profiled','dynamic','dynamic_timed','partition')


def pack_ids(ids,width):
    out=np.zeros(width,dtype=np.uint32)
    for i in ids:out[i//32] |= np.uint32(1 << (i%32))
    return out


def adapter(entry,loaded):
    vocab=g.Vocab.from_id_to_bytes(TOKENS)
    source='start ::= "ab" | "ac"'
    if entry=='partition':
        partition=native._internal.VocabPartition.from_ebnf(source,vocab)
        classes=partition.classes()
        assert sorted(t for row in classes for t in row)==sorted(TOKENS)
        assert len(classes)==partition.num_classes
        selected=[i for i in range(len(classes)) if i%2==0]
        internal=[0]*((len(classes)+63)//64)
        for i in selected:internal[i//64] |= 1 << (i%64)
        expected=pack_ids([t for i in selected for t in classes[i]],partition.original_mask_len+3)
        def fill(out):return partition.fill_expanded_mask(internal,out)
        def check():
            assert partition.classes()==classes
            assert list(partition.expand_mask(internal))==expected[:partition.original_mask_len].tolist()
        return fill,partition.original_mask_len,expected,check
    if entry.startswith('dynamic'):
        value=native.DynamicConstraint.from_ebnf(source,vocab,vocab_partition=True)
        if loaded:value=native.DynamicConstraint.load(value.save(),vocab)
    else:
        value=g.Grammar.from_ebnf(source).compile(vocab,optimization=g.Optimization.FAST_RUNTIME)
        if loaded:value=g.Constraint.load(value.save())
    state=value.start();width=value.mask_len()
    allowed=[i for i,b in TOKENS.items() if any(word.startswith(b) for word in LANGUAGE)]
    expected=pack_ids(allowed,width+3)
    def fill(out):
        if entry=='static_timed':return native._internal.fill_mask_timed_ns(state,out)
        if entry=='static_profiled':return native._internal.fill_mask_profiled(state,out)
        if entry=='dynamic_timed':return state.fill_mask_timed_ns(out)
        return state.fill_mask(out)
    def check():
        actual=np.full(width+3,-1,dtype=np.int32);state.fill_mask(actual)
        np.testing.assert_array_equal(actual.view(np.uint32),expected)
        assert not state.is_rejected() and not state.is_accepting()
    del value,vocab
    gc.collect()
    return fill,width,expected,check


@pytest.mark.parametrize('entry,loaded', [(e,l) for e in ENTRIES for l in (False,True) if not (e=='partition' and l)])
@pytest.mark.parametrize('kind',('readonly','strided','wrong_dtype','wrong_rank','undersized'))
def test_private_buffer_failure_is_recoverable_without_mutation(entry,loaded,kind):
    fill,width,expected,check=adapter(entry,loaded)
    if kind=='readonly':
        buf=np.full(width+3,-1,dtype=np.int32);buf.flags.writeable=False
    elif kind=='strided':buf=np.full((width+3)*2,-1,dtype=np.int32)[::2]
    elif kind=='wrong_dtype':buf=np.full(width+3,(1<<32)-1,dtype=np.uint32)
    elif kind=='wrong_rank':buf=np.full((1,width+3),-1,dtype=np.int32)
    else:buf=np.full(width-1,-1,dtype=np.int32)
    old=buf.copy()
    with pytest.raises((TypeError,ValueError)):
        fill(buf)
    np.testing.assert_array_equal(buf,old)
    check()
    # Acquiring again after a failure must not encounter a retained borrow.
    storage=np.full(width+7,-314159,dtype=np.int32);view=storage[2:-2]
    fill(view)
    np.testing.assert_array_equal(view.view(np.uint32),expected)
    np.testing.assert_array_equal(storage[[0,1,-2,-1]],[-314159]*4)
    if kind=='readonly':
        buf.flags.writeable=True;fill(buf)
        np.testing.assert_array_equal(buf.view(np.uint32),expected)


@pytest.mark.parametrize('strategy',('automatic','compact','dedicated'))
def test_partition_overwrite_handles_every_class_without_retaining_previous_bits(strategy):
    vocab=g.Vocab.from_id_to_bytes(TOKENS)
    partition=native._internal.VocabPartition.from_ebnf('start ::= "ab" | "ac"',vocab,strategy=strategy)
    classes=partition.classes();width=partition.original_mask_len+3
    del vocab;gc.collect()
    out=np.full(width,-1,dtype=np.int32)
    selections=[list(range(len(classes))),[],*[[i] for i in range(len(classes))],[]]
    for selected in selections:
        words=[0]*((len(classes)+63)//64+1)
        for i in selected:words[i//64] |= 1 << (i%64)
        words[-1]=(1<<64)-1 # Extra class-space bits do not denote model tokens.
        partition.fill_expanded_mask(words,out)
        expected=pack_ids([t for i in selected for t in classes[i]],width)
        np.testing.assert_array_equal(out.view(np.uint32),expected)
        assert list(partition.expand_mask(words))==expected[:partition.original_mask_len].tolist()
