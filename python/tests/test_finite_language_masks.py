"""Independent full-mask oracle for finite grammars; no reference mask engine.

Words are specified
independently of the grammar. Every reachable byte prefix is checked through
byte commits and several token segmentations, with sparse aliases and dirty
output buffers. This checks correctness, not performance or JSON conformance.
"""
from __future__ import annotations

import itertools

import glrmask as g
from glrmask import _glrmask as native
import numpy as np
import pytest


MODES = ["FAST_BUILD", "O2", "FAST_RUNTIME", "AUTO"]
END = 4095
CHILD_END = 4094
SPARSE_ALIAS = 131071


def glrm(body: str) -> str:
    return "glrm 1; start root; " + body


# These finite languages are the oracle, not outputs collected from GLRMask.
CASES = [
    ("prefix", glrm('nt root = "a" | "a" "b" | "a" "b" "c";'),
     ["a", "ab", "abc"]),
    ("ambiguous", glrm('nt root = left | right; nt left = "a" tail; '
                       'nt right = head "c"; nt tail = "b" | "c"; '
                       'nt head = "a" | "ab";'), ["ab", "ac", "abc"]),
    ("nullable", glrm('nt root = ("a" | "b")? ("a" | "c")? "z";'),
     [a + b + "z" for a in ["", "a", "b"] for b in ["", "a", "c"]]),
    ("empty", glrm('nt root = ("a" "b")?;'), ["", "ab"]),
    ("alternating", glrm('nt root = left "!" | right "?"; '
                         'nt left = "a" | "b" "a"; '
                         'nt right = "b" | "a" "b";'),
     ["a!", "ba!", "b?", "ab?"]),
    ("product", glrm('nt root = ("a" | "b") ("x" | "y") '
                     '("+" | "-") ("0" | "1");'),
     ["".join(parts) for parts in itertools.product("ab", "xy", "+-", "01")]),
    ("shared_suffix", glrm('nt root = "<" body ">"; '
                           'nt body = "a" "b" | "a" "c" | "b" "c";'),
     ["<ab>", "<ac>", "<bc>"]),
    ("utf8", glrm('nt root = ("é" | "한" | "🙂") ("!" | "?");'),
     [a + b for a in ["é", "한", "🙂"] for b in "!?"]),
    ("multi_word", glrm('nt root = "alpha" | "alpine" | "alphabet" | "beta";'),
     ["alpha", "alpine", "alphabet", "beta"]),
]


def token_bytes(words: list[str], *, wide=False) -> dict[int, bytes]:
    """All bytes, all word substrings, and duplicate bytes at sparse IDs."""
    entries = {i: bytes([i]) for i in range(256)}
    extras = {b"wrong", b"aa", b"ab", b"ac", b"abc", b"ba"}
    for word in words:
        data = word.encode()
        extras.update(data[a:b] for a in range(len(data))
                      for b in range(a + 1, len(data) + 1))
    entries.update({256 + i: data for i, data in enumerate(sorted(extras))})
    entries.update({1023: b"a", 2047: b"ab", SPARSE_ALIAS: b"a"})
    if wide:
        # Cross the u16 sparse-output-word coordinate without a huge vocabulary.
        entries[(1 << 21) + 31] = b"a"
    assert END not in entries and CHILD_END not in entries
    return entries


def constraint(source, entries, mode, *, end=False):
    vocab = g.Vocab.from_id_to_bytes(entries)
    if mode == "O2":
        assert not end
        return native.DynamicConstraint.from_glrm_grammar(
            source, vocab, vocab_partition=True), vocab
    return g.Grammar.from_glrm(source).compile(
        vocab, optimization=getattr(g.Optimization, mode),
        end_tokens=[END] if end else []), vocab


def restored(value, vocab, mode):
    loader = native.DynamicConstraint if mode == "O2" else g.Constraint
    return loader.load(value.save(), vocab)


def accepted_prefix(words: set[bytes], prefix: bytes) -> bool:
    # ParserTable::embedded_start_nullable documents the deliberate policy:
    # nullable child grammars work, but a standalone generation cannot finish
    # before producing any bytes. Do not change that API policy in a mask fix.
    return bool(prefix) and prefix in words


def expected_ids(words: set[bytes], prefix: bytes, entries, end=False):
    allowed = {i for i, data in entries.items()
               if data and any(word.startswith(prefix + data) for word in words)}
    if end and accepted_prefix(words, prefix):
        allowed.add(END)
    return allowed


def check_mask(state, value, allowed, *, label, verify_bool=True):
    # Check both minimum-size and oversized outputs. Alternate poison patterns
    # so mask-cache reuse cannot accidentally hide an incomplete buffer clear.
    for extra, poison in [(0, -1), (3, 0x55555555), (1, -1), (0, 0)]:
        buf = np.full(value.mask_len() + extra, poison, dtype=np.int32)
        expected = np.zeros(len(buf), dtype=np.uint32)
        for token in allowed:
            expected[token // 32] |= np.uint32(1 << (token % 32))
        state.fill_mask(buf)
        actual = buf.view(np.uint32)
        mismatch = np.flatnonzero(actual != expected)
        assert len(mismatch) == 0, (label, extra, mismatch[:8].tolist(),
            actual[mismatch[:8]].tolist(), expected[mismatch[:8]].tolist())
    if verify_bool:
        actual_ids = set(np.flatnonzero(state.mask(value.mask_len() * 32 + 7)).tolist())
        assert actual_ids == allowed, (label, "boolean", sorted(actual_ids ^ allowed)[:16])


def histories(prefix: bytes, entries):
    """Different tokenizations of the same bytes, including duplicate IDs."""
    paths = {tuple(prefix)}  # IDs 0..255 are individual bytes.
    for order in ["long_low", "long_high", "short_high"]:
        pending, path = prefix, []
        while pending:
            options = [(i, data) for i, data in entries.items()
                       if data and pending.startswith(data)]
            if order == "long_low":
                token, data = min(options, key=lambda p: (-len(p[1]), p[0]))
            elif order == "long_high":
                token, data = max(options, key=lambda p: (len(p[1]), p[0]))
            else:
                token, data = min(options, key=lambda p: (len(p[1]), -p[0]))
            path.append(token)
            pending = pending[len(data):]
        paths.add(tuple(path))
    return sorted(paths)


def check_language(value, words, entries, *, label, end=False, boolean=True):
    words = {word.encode() for word in words}
    prefixes = sorted({word[:i] for word in words for i in range(len(word) + 1)})
    checks = 0
    for prefix in prefixes:
        allowed = expected_ids(words, prefix, entries, end)
        byte_state = value.start()
        if prefix:
            byte_state.commit_bytes(prefix)
        assert byte_state.is_accepting() == accepted_prefix(words, prefix), (label, prefix, "accepting")
        check_mask(byte_state, value, allowed, label=(label, prefix, "bytes"), verify_bool=boolean)
        checks += 1
        for path in histories(prefix, entries):
            state = value.start()
            consumed = b""
            for token in path:
                check_mask(state, value, expected_ids(words, consumed, entries, end),
                           label=(label, prefix, path, consumed), verify_bool=False)
                state.commit_token(token)
                consumed += entries[token]
                assert not state.is_rejected(), (label, path, token)
                checks += 1
            assert consumed == prefix
            assert state.is_accepting() == accepted_prefix(words, prefix), (label, prefix, path)
            check_mask(state, value, allowed, label=(label, prefix, path), verify_bool=False)
            checks += 1
            # Advance a separate state to ensure mutable cache/frontier state is
            # not accidentally shared with the previously checked byte state.
            check_mask(byte_state, value, allowed, label=(label, prefix, "independent"),
                       verify_bool=False)
            checks += 1
            if end and accepted_prefix(words, prefix):
                state.commit_token(END)
                assert state.is_terminated()
                check_mask(state, value, set(), label=(label, prefix, "terminated"),
                           verify_bool=False)
                checks += 1
    return checks


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,source,words", CASES, ids=[c[0] for c in CASES])
def test_finite_language_masks(name, source, words, mode):
    entries = token_bytes(words)
    value, vocab = constraint(source, entries, mode, end=mode != "O2")
    cold = restored(value, vocab, mode)  # Save before the first mask query.
    for phase, item in [("fresh", value), ("loaded_cold", cold)]:
        check_language(item, words, entries, label=(name, mode, phase), end=mode != "O2")
    warm = restored(value, vocab, mode)
    check_language(warm, words, entries, label=(name, mode, "loaded_warm"), end=mode != "O2")


@pytest.mark.parametrize("mode", MODES)
def test_mask_words_beyond_sparse_u16_coordinate(mode):
    name, source, words = CASES[0]
    entries = token_bytes(words, wide=True)
    value, vocab = constraint(source, entries, mode, end=mode != "O2")
    cold = restored(value, vocab, mode)
    for phase, item in [("fresh", value), ("loaded", cold)]:
        check_language(item, words, entries, label=("wide", mode, phase),
                       end=mode != "O2", boolean=False)


@pytest.mark.parametrize("mode", ["FAST_BUILD", "FAST_RUNTIME", "AUTO"])
@pytest.mark.parametrize("nullable", [False, True])
def test_compiled_child_boundary_masks(mode, nullable):
    words = ["<ab>", "<ac>"] + (["<>"] if nullable else [])
    entries = token_bytes(words)
    vocab = g.Vocab.from_id_to_bytes(entries)
    option = getattr(g.Optimization, mode)
    child = g.Grammar.from_glrm(glrm('nt root = "a" ("b" | "c");')).compile(
        vocab, optimization=option, end_tokens=[CHILD_END])
    child = g.Constraint.load(child.save())
    body = 'nt root = "<" child' + ('?' if nullable else '') + ' ">";'
    host = g.Grammar.from_glrm(glrm('extern grammar child; ' + body)).compile_unlinked(vocab)
    host = g.UnlinkedConstraint.load(host.save())
    value = host.bind("child", child).link(optimization=option, end_tokens=[END])
    cold = g.Constraint.load(value.save())
    for phase, item in [("fresh", value), ("loaded", cold)]:
        check_language(item, words, entries, label=("composed", nullable, mode, phase), end=True)


def test_oracle_self_check():
    entries = {0: b"a", 1: b"a", 2: b"ab", 3: b"b", 4: b"wrong", 5: b""}
    assert expected_ids({b"a", b"ab"}, b"", entries) == {0, 1, 2}
    assert expected_ids({b"a", b"ab"}, b"a", entries, True) == {3, END}
    assert expected_ids({b"a", b"ab"}, b"ab", entries, True) == {END}


def test_nullable_root_follows_standalone_generation_policy():
    assert not accepted_prefix({b"", b"ab"}, b"")
    assert accepted_prefix({b"", b"ab"}, b"ab")
    assert END not in expected_ids({b"", b"ab"}, b"", {0: b"ab"}, end=True)
