"""Cache ownership and replay contracts across cold, warm, and loaded constraints."""
import numpy as np
import pytest
import glrmask


MODES = [glrmask.Optimization.FAST_BUILD, glrmask.Optimization.AUTO,
         glrmask.Optimization.FAST_RUNTIME]
END = 511
MASK_SIZE = 640


def vocabulary(*, include_empty_end=True):
    entries = {i: bytes([i]) for i in range(256)}
    entries.update({256: b"ab", 257: b"ba", 300: b"a", END: b""})
    if not include_empty_end:
        del entries[END]
    return glrmask.Vocab.from_id_to_bytes(entries)


def checked_mask(state, constraint):
    expected = state.mask(MASK_SIZE)
    # Reusing a dirty caller-owned output must not retain bits from another state.
    packed = np.full(constraint.mask_len(), -1, dtype=np.int32)
    state.fill_mask(packed)
    unpacked = ((packed.view(np.uint32)[:, None]
                 >> np.arange(32, dtype=np.uint32)) & 1).astype(bool).ravel()
    assert np.array_equal(unpacked, expected[:len(unpacked)])
    assert not expected[len(unpacked):].any()
    assert bool(expected[ord("a")]) == bool(expected[300])
    return expected


def replay(constraint, steps, terminate=True):
    state = constraint.start()
    masks = []
    for step in steps:
        mask = checked_mask(state, constraint)
        masks.append(mask.tobytes())
        if isinstance(step, bytes):
            state.commit_bytes(step)
        else:
            assert mask[step], (step, steps)
            state.commit_token(step)
        assert not state.is_rejected()
    assert state.is_accepting()
    final_mask = checked_mask(state, constraint)
    masks.append(final_mask.tobytes())
    if terminate:
        assert final_mask[END]
        state.commit_token(END)
        assert state.is_terminated()
    return masks


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("steps", [[256], [ord("a"), ord("b")],
                                   [300, ord("b")], [b"a", ord("b")]])
def test_cold_warm_and_loaded_masks_match(mode, steps):
    vocab = vocabulary()
    constraint = glrmask.Grammar.from_ebnf('start ::= "ab" | "aab"').compile(
        vocab, optimization=mode, end_tokens=[END])
    before_first_mask = constraint.save()
    expected = replay(constraint, steps)
    assert replay(constraint, steps) == expected
    assert replay(glrmask.Constraint.load(before_first_mask), steps) == expected
    assert replay(glrmask.Constraint.load(constraint.save()), steps) == expected


@pytest.mark.parametrize("mode", MODES)
def test_constraints_sharing_a_vocab_do_not_share_mutable_frontiers(mode):
    vocab = vocabulary()
    left = glrmask.Grammar.from_ebnf('start ::= "ab"').compile(
        vocab, optimization=mode, end_tokens=[END])
    right = glrmask.Grammar.from_ebnf('start ::= "ba"').compile(
        vocab, optimization=mode, end_tokens=[END])
    a, b = left.start(), right.start()
    initial_right = checked_mask(b, right).copy()
    assert checked_mask(a, left)[ord("a")]
    assert not initial_right[ord("a")]
    a.commit_token(ord("a"))
    assert checked_mask(a, left)[ord("b")]
    assert np.array_equal(checked_mask(b, right), initial_right)
    b.commit_token(ord("b"))
    assert checked_mask(b, right)[ord("a")]
    assert not checked_mask(a, left)[ord("a")]
    a.commit_token(ord("b"))
    b.commit_token(ord("a"))
    assert a.is_accepting() and b.is_accepting()
    assert checked_mask(a, left)[END] and checked_mask(b, right)[END]


@pytest.mark.parametrize("mode", MODES)
def test_loaded_composition_keeps_child_termination_local(mode):
    # A generation control is not an ordinary zero-byte language token.
    # Root end IDs may extend the mask without entering the byte vocabulary.
    vocab = vocabulary(include_empty_end=False)
    child = glrmask.Grammar.from_ebnf('start ::= "ab"').compile(
        vocab, optimization=mode, end_tokens=[END])
    parent = glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern grammar child; nt root = "<" child ">";'
    ).compile_unlinked(vocab)
    constraint = parent.bind("child", child).link(optimization=mode)
    steps = [ord("<"), 256, ord(">")]
    expected = replay(constraint, steps, terminate=False)
    assert replay(glrmask.Constraint.load(constraint.save()), steps,
                  terminate=False) == expected
    state = constraint.start()
    state.commit_token(ord("<"))
    state.commit_token(256)
    assert not state.mask(MASK_SIZE)[END]
    assert state.mask(MASK_SIZE)[ord(">")]
