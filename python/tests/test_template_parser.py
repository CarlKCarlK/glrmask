"""Public Python table-free API: masks are checked against literal languages."""
import copy
import gc
import json

import numpy as np
import pytest

import glrmask


def symbol(value):
    return {"Symbol": value}


def graph(accepting=False):
    return {"start": 0, "states": [{"accepting": accepting, "transitions": []}]}


def template():
    return {"pop": graph(), "read": graph(), "push": graph(),
            "pop_to_read": [], "pop_to_push": [], "read_to_push": []}


def identity():
    value = template()
    value["pop"]["states"][0]["accepting"] = True
    return value


def rewrite_top(mapping):
    """Literal relation: pop one top symbol and append its mapped output."""
    value = template()
    value["pop_to_push"] = [None]
    for label, pushes in mapping.items():
        pop_id = len(value["pop"]["states"])
        value["pop"]["states"][0]["transitions"].append(
            {"label": symbol(label), "target": pop_id})
        value["pop"]["states"].append({"accepting": not pushes, "transitions": []})
        if not pushes:
            value["pop_to_push"].append(None)
            continue
        start = len(value["push"]["states"])
        value["pop_to_push"].append(start)
        for offset, pushed in enumerate(pushes):
            value["push"]["states"].append({"accepting": False, "transitions": [
                {"label": symbol(pushed), "target": start + offset + 1}]})
        value["push"]["states"].append({"accepting": True, "transitions": []})
    return value


def balanced_parentheses():
    return {"stack_symbol_count": 2,
            "terminals": [rewrite_top({0: [0, 1], 1: [1, 1]}), rewrite_top({1: []}), identity()],
            "completion": rewrite_top({0: []})}


def words_and_vocab():
    words = [b"(", b")", b"()", b"((", b"))", b"(()", b"())", b"()()", b" ", b"( )", b"a", b"()"]
    return words, glrmask.Vocab.from_id_to_bytes(dict(enumerate(words)))


def depth_after(word, depth=0, ignore_whitespace=True):
    for byte in word:
        if byte == ord("("):
            depth += 1
        elif byte == ord(")") and depth:
            depth -= 1
        elif byte != ord(" ") or not ignore_whitespace:
            return None
    return depth


def representations(constraint, vocab):
    raw = constraint.save()
    loaded = glrmask.Constraint.load(raw)
    assert loaded.save() == raw
    external = constraint.save_with_external_vocab()
    with pytest.raises(ValueError):
        glrmask.Constraint.load(external)
    return [constraint, loaded, glrmask.Constraint.load(external, vocab=vocab)]


@pytest.mark.parametrize("mode", [glrmask.Optimization.FAST_RUNTIME, glrmask.Optimization.FAST_BUILD])
def test_python_builtin_template_backend_is_default_and_survives_reload(mode):
    words, vocab = words_and_vocab()
    grammar = glrmask.Grammar.from_ebnf('start ::= "(" start ")" start | ""')
    reference = grammar.compile(vocab, optimization=mode)
    candidate = grammar.compile(vocab, optimization=mode, parser_backend=glrmask.ParserBackend.TEMPLATE_DFA)
    assert reference.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
    reference_bytes = reference.save()
    for compiled in representations(reference, vocab) + representations(candidate, vocab):
        assert compiled.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
        for prefix in [b"", b"(", b"()", b"(()", b"((()))", b"()("]:
            left, right = reference.start(), compiled.start()
            left.commit_bytes(prefix)
            right.commit_bytes(prefix)
            assert np.array_equal(left.mask(), right.mask())
            assert left.is_accepting() == right.is_accepting()
            depth = depth_after(prefix, ignore_whitespace=False)
            # Built-in grammar normalization deliberately removes the root-only
            # empty generation path; embedded source nullability is retained.
            # Data-only ParserProgram completion below still accepts empty.
            assert right.is_accepting() == (bool(prefix) and depth == 0)
            for token_id, word in enumerate(words):
                expected = depth_after(word, depth, ignore_whitespace=False)
                assert bool(right.mask()[token_id]) == (expected is not None), (prefix, word)
                if expected is not None:
                    branch = compiled.start()
                    branch.commit_bytes(prefix)
                    branch.commit_token(token_id)
                    assert branch.is_accepting() == (expected == 0)
    assert reference.save() == reference_bytes


@pytest.mark.parametrize("mode", [glrmask.Optimization.FAST_RUNTIME, glrmask.Optimization.FAST_BUILD])
@pytest.mark.parametrize("as_json", [False, True])
def test_python_data_only_program_matches_literal_language(mode, as_json):
    definition = balanced_parentheses()
    program = glrmask.ParserProgram(json.dumps(definition) if as_json else definition)
    assert program.terminal_count == 3
    # The implementation owns the validated program, not the mutable mapping.
    definition["terminals"].clear()
    words, vocab = words_and_vocab()
    compiled = program.compile(vocab, [b"(", b")", "[ ]+"], ignore_terminal=2,
                               optimization=mode, end_tokens=[64])
    del program, definition
    gc.collect()
    for compiled in representations(compiled, vocab):
        assert compiled.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
        prefixes = [b""]
        while prefixes:
            prefix = prefixes.pop()
            depth = depth_after(prefix)
            state = compiled.start()
            state.commit_bytes(prefix)
            assert state.is_accepting() == (depth == 0)
            mask = state.mask()
            for token_id, word in enumerate(words):
                assert bool(mask[token_id]) == (depth_after(word, depth) is not None), (prefix, word)
            assert bool(mask[64]) == (depth == 0)
            if len(prefix) < 6:
                for byte in [b"(", b")"]:
                    if depth_after(byte, depth) is not None:
                        prefixes.append(prefix + byte)


def test_python_template_validation_is_early_and_does_not_retain_python_callbacks():
    definition = balanced_parentheses()
    cyclic = copy.deepcopy(definition)
    cyclic["terminals"][0]["pop"]["states"][0]["transitions"][0]["target"] = 0
    with pytest.raises(ValueError, match="(?i)cycl"):
        glrmask.ParserProgram(cyclic)
    with pytest.raises(ValueError):
        glrmask.ParserProgram("not valid JSON")
    with pytest.raises(TypeError):
        glrmask.ParserProgram(lambda: definition)
    program = glrmask.ParserProgram(definition)
    _, vocab = words_and_vocab()
    for bad in [[b"("], [b"(", b")", "[ ]+", b"extra"]]:
        with pytest.raises(ValueError):
            program.compile(vocab, bad)
    with pytest.raises(TypeError):
        program.compile(vocab, [object(), b")", "[ ]+"])
    with pytest.raises(ValueError):
        program.compile(vocab, [b"(", b")", "[ ]+"], ignore_terminal=0)
    with pytest.raises(ValueError):
        program.compile(vocab, [b"", b")", "[ ]+"])
    with pytest.raises(ValueError):
        program.compile(vocab, ["(", b")", "[ ]+"])


@pytest.mark.parametrize("mode", [glrmask.Optimization.AUTO, glrmask.Optimization.FAST_RUNTIME,
                                  glrmask.Optimization.FAST_BUILD])
def test_python_compiled_composition_uses_native_defaults(mode):
    tokens = {0: b"a", 1: b"aa", 2: b"b"}
    vocab = glrmask.Vocab.from_id_to_bytes(tokens)
    child = glrmask.Grammar.from_ebnf('start ::= "a"').compile(
        vocab, optimization=mode, parser_backend=glrmask.ParserBackend.TEMPLATE_DFA)
    parent = glrmask.Grammar.from_glrm('glrm 1; start root; extern grammar C; nt root = C;').compile_unlinked(vocab)
    assert child.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
    ordinary = glrmask.Grammar.from_ebnf('start ::= "a"').compile(vocab, optimization=mode)
    assert ordinary.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
    for child in [ordinary, child]:
        for source in representations(child, vocab):
            child_bytes = source.save()
            for selection in [{}, {"parser_backend": glrmask.ParserBackend.TEMPLATE_DFA}]:
                linked = parent.bind("C", source).link(optimization=mode, **selection)
                assert source.save() == child_bytes
                for compiled in representations(linked, vocab):
                    assert compiled.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
                    for prefix in [b"", b"a"]:
                        state = compiled.start()
                        state.commit_bytes(prefix)
                        assert state.is_accepting() == (prefix == b"a")
                        for token_id, word in tokens.items():
                            expected = b"a".startswith(prefix + word)
                            assert bool(state.mask()[token_id]) == expected, (prefix, word)
                            if expected:
                                branch = compiled.start()
                                branch.commit_bytes(prefix)
                                branch.commit_token(token_id)
                                assert branch.is_accepting() == (prefix + word == b"a")


@pytest.mark.parametrize("mode", [glrmask.Optimization.FAST_RUNTIME, glrmask.Optimization.FAST_BUILD])
def test_python_precompiled_nullable_composition_preserves_crossings_and_root_end(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"x", 1: b"a", 2: b"y", 3: b"xay",
                                          4: b"xy", 5: b"xx", 6: b""})
    child = glrmask.Grammar.from_ebnf('start ::= "a"?').compile(
        vocab, optimization=glrmask.Optimization.FAST_RUNTIME,
        parser_backend=glrmask.ParserBackend.TEMPLATE_DFA)
    parent = glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern grammar C; nt root = "x" C "y";').compile_unlinked(vocab)
    linked = parent.bind("C", child).link(optimization=mode,
        parser_backend=glrmask.ParserBackend.TEMPLATE_DFA, end_tokens=[6])
    for compiled in representations(linked, vocab):
        state = compiled.start()
        assert np.array_equal(state.mask(), np.array([True, False, False, True, True, False, False]))
        for token in [3, 4]:
            state = compiled.start()
            state.commit_token(token)
            assert state.is_accepting()
            assert state.mask()[6]
            state.commit_token(6)
            assert state.is_terminated()


def test_python_backend_selection_does_not_accept_untyped_flags():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a"})
    grammar = glrmask.Grammar.from_ebnf('start ::= "a"')
    for value in ["TEMPLATE_DFA", True, 1, object()]:
        with pytest.raises(TypeError):
            grammar.compile(vocab, parser_backend=value)


def test_python_explicit_lr_compile_and_link_requests_panic_loudly():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a"})
    grammar = glrmask.Grammar.from_ebnf('start ::= "a"')
    module = grammar.compile_unlinked(vocab)
    for operation in [lambda: grammar.compile(vocab, parser_backend=glrmask.ParserBackend.LR_TABLE),
                      lambda: module.link(parser_backend=glrmask.ParserBackend.LR_TABLE)]:
        with pytest.raises(BaseException, match="LR-BACKED CONSTRAINT REQUEST IS FORBIDDEN") as rejected:
            operation()
        # PyO3 exposes Rust panics as a BaseException, not an ordinary ValueError.
        assert type(rejected.value).__name__ == "PanicException"


@pytest.mark.parametrize("mode", [glrmask.Optimization.FAST_RUNTIME, glrmask.Optimization.FAST_BUILD])
def test_python_projected_virtual_child_preserves_exact_limits_and_reload(mode):
    tokens = {0: b"p", 1: b'"x:a"', 2: b"q", 3: b'p"x:a"q',
              4: b'a"q', 5: b"a", 6: b"aaa", 7: b'"q', 8: b""}
    vocab = glrmask.Vocab.from_id_to_bytes(tokens)
    child = glrmask.Grammar.from_json_schema(json.dumps({
        "type": "string", "format": "uri", "minLength": 1, "maxLength": 5000,
    })).compile(vocab, optimization=glrmask.Optimization.FAST_RUNTIME,
                parser_backend=glrmask.ParserBackend.TEMPLATE_DFA)
    child = glrmask.Constraint.load(child.save())
    parent = glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern grammar C; nt root = "p" C "q";').compile_unlinked(vocab)
    compiled = parent.bind("C", child).link(optimization=mode,
        parser_backend=glrmask.ParserBackend.TEMPLATE_DFA, end_tokens=[8])
    for compiled in representations(compiled, vocab):
        assert compiled.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
        state = compiled.start()
        assert state.mask()[3]
        state.commit_token(3)
        assert state.is_accepting() and state.mask()[8]
        state.commit_token(8)
        assert state.is_terminated()

        for count in [0, 1, 31, 4996, 4997, 4998]:
            state = compiled.start()
            state.commit_bytes(b'p"x:' + b"a" * count)
            mask = state.mask()
            for token, payload_size in [(5, 1), (6, 3)]:
                assert bool(mask[token]) == (2 + count + payload_size <= 5000), (count, token)
            assert mask[7]  # Completing the URI and returning to the parent.
            state.commit_token(7)
            assert state.is_accepting()
        state = compiled.start()
        with pytest.raises(ValueError):
            state.commit_bytes(b'p"x:' + b"a" * 4999)


@pytest.mark.parametrize("backend", [None, glrmask.ParserBackend.TEMPLATE_DFA])
@pytest.mark.parametrize("mode", [glrmask.Optimization.FAST_RUNTIME, glrmask.Optimization.FAST_BUILD])
def test_python_nullable_lexical_body_survives_compiled_child_binding(backend, mode):
    tokens = {0: b"x", 1: b"a", 2: b"y", 3: b"xay", 4: b"xy",
              5: b"ay", 6: b"aa", 7: b"yx", 8: b""}
    vocab = glrmask.Vocab.from_id_to_bytes(tokens)
    selection = {} if backend is None else {"parser_backend": backend}
    child = glrmask.Grammar.from_glrm(
        'start root; t A ::= /a?/; nt root ::= A;').compile(
            vocab, optimization=mode, **selection)
    parent = glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern grammar C; nt root = "x" C "y";').compile_unlinked(vocab)
    language = [b"xy", b"xay"]
    for child in representations(child, vocab):
        child_bytes = child.save()
        linked = parent.bind("C", child).link(
            optimization=mode, end_tokens=[8], **selection)
        assert child.save() == child_bytes
        for compiled in representations(linked, vocab):
            assert compiled.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
            for prefix in [b"", b"x", b"xa", b"xy", b"xay"]:
                state = compiled.start()
                state.commit_bytes(prefix)
                mask = state.mask()
                assert state.is_accepting() == (prefix in language)
                assert bool(mask[8]) == (prefix in language)
                for token_id, word in tokens.items():
                    if token_id == 8:
                        continue
                    expected = any(complete.startswith(prefix + word) for complete in language)
                    assert bool(mask[token_id]) == expected, (backend, mode, prefix, word)
                    if expected:
                        branch = compiled.start()
                        branch.commit_bytes(prefix)
                        branch.commit_token(token_id)
                        assert branch.is_accepting() == (prefix + word in language)
                if prefix in language:
                    state.commit_token(8)
                    assert state.is_terminated()
