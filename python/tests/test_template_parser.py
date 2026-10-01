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


def depth_after(word, depth=0):
    for byte in word:
        if byte == ord("("):
            depth += 1
        elif byte == ord(")") and depth:
            depth -= 1
        elif byte != ord(" "):
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
def test_python_builtin_template_backend_is_explicit_and_survives_reload(mode):
    words, vocab = words_and_vocab()
    grammar = glrmask.Grammar.from_ebnf('start ::= "(" start ")" start | ""')
    reference = grammar.compile(vocab, optimization=mode)
    candidate = grammar.compile(vocab, optimization=mode, parser_backend=glrmask.ParserBackend.TEMPLATE_DFA)
    assert reference.parser_backend == glrmask.ParserBackend.LR_TABLE
    for compiled in representations(candidate, vocab):
        assert compiled.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
        for prefix in [b"", b"(", b"()", b"(()", b"((()))", b"()("]:
            left, right = reference.start(), compiled.start()
            left.commit_bytes(prefix)
            right.commit_bytes(prefix)
            assert np.array_equal(left.mask(), right.mask())
            assert left.is_accepting() == right.is_accepting()


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


def test_python_compiled_composition_selects_templates_without_changing_the_default():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a"})
    child = glrmask.Grammar.from_ebnf('start ::= "a"').compile(
        vocab, parser_backend=glrmask.ParserBackend.TEMPLATE_DFA)
    parent = glrmask.Grammar.from_glrm('glrm 1; start root; extern grammar C; nt root = C;').compile_unlinked(vocab)
    # An unchanged LR default cannot reinterpret a table-free child as LR.
    bound = parent.bind("C", child)
    with pytest.raises(ValueError, match="template-parser|table-free"):
        bound.link()
    assert child.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
    ordinary = glrmask.Grammar.from_ebnf('start ::= "a"').compile(vocab)
    assert ordinary.parser_backend == glrmask.ParserBackend.LR_TABLE
    linked = parent.bind("C", ordinary).link()
    assert linked.parser_backend == glrmask.ParserBackend.LR_TABLE
    assert linked.start().mask()[0]
    for source in [ordinary, child]:
        linked = parent.bind("C", source).link(parser_backend=glrmask.ParserBackend.TEMPLATE_DFA)
        for compiled in representations(linked, vocab):
            assert compiled.parser_backend == glrmask.ParserBackend.TEMPLATE_DFA
            state = compiled.start()
            assert state.mask()[0]
            state.commit_token(0)
            assert state.is_accepting()


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
