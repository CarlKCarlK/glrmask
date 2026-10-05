"""Mask/commit and exact-vocabulary contracts of the actual Python wrappers."""
import pytest
import glrmask
from glrmask._glrmask import DynamicConstraint

SOURCE = 'start ::= "(" start ")" start | ""'
TOKENS = {0: b"(", 2: b")", 7: b"()", 15: b"(", 20: b"x", 25: b"", 26: b""}


def version(raw):
    return int.from_bytes(bytes(raw)[8:10], "little")


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("partition", [False, True])
def test_dynamic_wrapper_empty_alias_masks_and_exact_external_vocab(monkeypatch, flag, partition):
    monkeypatch.setenv("GLRMASK_DYNAMIC_TEMPLATE_DFA", flag)
    monkeypatch.setenv("GLRMASK_ENABLE_TEMPLATE_DFA_ADVANCE", "1")
    vocab = glrmask.Vocab.from_id_to_bytes(TOKENS)
    compiled = DynamicConstraint.from_ebnf(SOURCE, vocab, vocab_partition=partition)
    raw = bytes(compiled.save())
    expected = 14 if partition or flag == "1" else 15
    assert version(raw) == expected
    # Changing the compile switch must not change existing or loaded objects.
    monkeypatch.setenv("GLRMASK_DYNAMIC_TEMPLATE_DFA", "1" if flag == "0" else "0")
    loaded = DynamicConstraint.load(raw, vocab)
    assert bytes(loaded.save()) == raw
    for constraint in [compiled, loaded]:
        for prefix, accepting in [(b"", True), (b"(", False), (b"()", True), (b"()(", False)]:
            state = constraint.start()
            state.commit_bytes(prefix)
            assert state.is_accepting() == accepting
            for _ in range(2):
                mask = state.mask()
                assert not mask[25] and not mask[26]
                assert mask[0] and mask[15]  # Sparse IDs with duplicate bytes.
            with pytest.raises(ValueError):
                state.commit_token(25)
            assert state.is_rejected()
    for changed in [{**TOKENS, 0: b"["}, {**TOKENS, 20: b"y"},
                    {key: value for key, value in TOKENS.items() if key != 15},
                    {**TOKENS, 15: b")"}]:
        with pytest.raises(ValueError, match="vocab"):
            DynamicConstraint.load(raw, glrmask.Vocab.from_id_to_bytes(changed))
    if expected == 15:
        corrupt = bytearray(raw)
        corrupt[26] ^= 1
        with pytest.raises(ValueError, match="vocab"):
            DynamicConstraint.load(bytes(corrupt), vocab)
        for old_version in range(1, 14):
            old = bytearray(raw)
            old[8:10] = old_version.to_bytes(2, "little")
            with pytest.raises(ValueError, match="lacks mandatory exact vocabulary identity"):
                DynamicConstraint.load(bytes(old), vocab)


@pytest.mark.parametrize("mode", [glrmask.Optimization.AUTO, glrmask.Optimization.FAST_BUILD,
                                  glrmask.Optimization.FAST_RUNTIME])
def test_public_empty_eos_and_empty_ordinary_alias_survive_reload(mode):
    vocab = glrmask.Vocab.from_id_to_bytes(TOKENS)
    compiled = glrmask.Grammar.from_ebnf(SOURCE).compile(vocab, optimization=mode, end_tokens=[26])
    for constraint in [compiled, glrmask.Constraint.load(compiled.save()),
                       glrmask.Constraint.load(compiled.save_with_external_vocab(), vocab)]:
        for prefix, accepting in [(b"", True), (b"(", False), (b"()", True)]:
            state = constraint.start()
            state.commit_bytes(prefix)
            for _ in range(2):
                assert not state.mask()[25]
                assert bool(state.mask()[26]) == accepting
            if accepting:
                state.commit_token(26)
                assert state.is_terminated()
                assert not state.mask().any()
            else:
                state.commit_token(26)
                assert state.is_rejected() and not state.is_terminated()


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("partition", [False, True])
def test_dynamic_profiled_wrapper_preserves_empty_eos(monkeypatch, flag, partition):
    monkeypatch.setenv("GLRMASK_DYNAMIC_TEMPLATE_DFA", flag)
    vocab = glrmask.Vocab.from_id_to_bytes(TOKENS)
    raw, _, _ = glrmask._internal.compile_ebnf_serialized_profiled(
        SOURCE, vocab, end_token_ids=[26], vocab_partition=partition)
    constraint = DynamicConstraint.load(bytes(raw), vocab)
    for prefix, accepting in [(b"", True), (b"(", False), (b"()", True)]:
        state = constraint.start()
        state.commit_bytes(prefix)
        assert not state.mask()[25]
        assert bool(state.mask()[26]) == accepting
        if accepting:
            state.commit_token(26)
            assert not state.mask().any()
        else:
            # The legacy Dynamic source-end-token wrapper rejects early EOS
            # with an error; the public root-end policy above returns Ok.
            with pytest.raises(ValueError):
                state.commit_token(26)
            assert state.is_rejected()


@pytest.mark.parametrize("flag", ["0", "1"])
def test_dynamic_empty_exact_special_id_remains_live(monkeypatch, flag):
    monkeypatch.setenv("GLRMASK_DYNAMIC_TEMPLATE_DFA", flag)
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 25: b"", 26: b""})
    compiled = DynamicConstraint.from_glrm_grammar(
        'glrm 1; start root; extern token MARK; nt root = "a" MARK;',
        vocab, bindings={"MARK": [25]})
    for constraint in [compiled, DynamicConstraint.load(bytes(compiled.save()), vocab)]:
        state = constraint.start()
        assert not state.mask()[25]
        state.commit_token(0)
        assert state.mask()[25] and not state.mask()[26]
        state.commit_token(25)
        assert state.is_accepting()
