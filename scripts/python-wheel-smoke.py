#!/usr/bin/env python3
"""Minimal public Python API smoke test for an installed glrmask wheel/sdist."""

import ctypes
import sys
import types
from pathlib import Path

import glrmask
from glrmask import _glrmask as native

# A fresh artifact install must not silently import a checkout or older wheel.
installed_root = Path(sys.prefix).resolve()
for module in (glrmask, native):
    assert Path(module.__file__).resolve().is_relative_to(installed_root), (
        f"expected installed package under {installed_root}, got {module.__file__}"
    )

vocab = glrmask.Vocab.from_dict(
    {
        b"hello": 0,
        b" ": 1,
        b"world": 2,
    }
)
constraint = glrmask.Grammar.from_ebnf(
    'start ::= "hello" " " "world"'
).compile(vocab)
state = constraint.start()

assert state.mask().tolist() == [True, False, False]
state.commit_token(0)
assert state.mask().tolist() == [False, True, False]
state.commit_token(1)
assert state.mask().tolist() == [False, False, True]
state.commit_token(2)
assert state.is_accepting()


# Exercise the optional llama-cpp-python adapter without installing or loading a
# real model. The fake module mirrors the small low-level API surface consumed by
# Vocab.from_llama_cpp.
pieces = {
    0: b"a",
    1: b"a",  # unused: must be omitted
    2: b"a",  # control: must be omitted
    3: b"a",  # EOG: exact-token-only, not a byte token
    4: b"d",
    5: b"a",  # a second EOG token
}
fake_llama_cpp = types.ModuleType("llama_cpp")
fake_llama_cpp.LLAMA_TOKEN_ATTR_UNUSED = 1 << 1
fake_llama_cpp.LLAMA_TOKEN_ATTR_CONTROL = 1 << 3
fake_vocab_handle = object()


def fake_model_get_vocab(model):
    assert model is fake_llm.model
    return fake_vocab_handle


def fake_is_eog(vocab_handle, token_id):
    assert vocab_handle is fake_vocab_handle
    return token_id in (3, 5)


def fake_get_attr(vocab_handle, token_id):
    assert vocab_handle is fake_vocab_handle
    if token_id == 1:
        return fake_llama_cpp.LLAMA_TOKEN_ATTR_UNUSED
    if token_id == 2:
        return fake_llama_cpp.LLAMA_TOKEN_ATTR_CONTROL
    return 0


def fake_token_to_piece(vocab_handle, token_id, buffer, capacity, _lstrip, _special):
    assert vocab_handle is fake_vocab_handle
    piece = pieces[token_id]
    if buffer is None or capacity < len(piece):
        return -len(piece)
    ctypes.memmove(buffer, piece, len(piece))
    return len(piece)


class FakeLlama:
    model = object()

    @staticmethod
    def n_vocab():
        return len(pieces)


fake_llm = FakeLlama()
fake_llama_cpp.llama_model_get_vocab = fake_model_get_vocab
fake_llama_cpp.llama_vocab_is_eog = fake_is_eog
fake_llama_cpp.llama_vocab_get_attr = fake_get_attr
fake_llama_cpp.llama_token_to_piece = fake_token_to_piece
sys.modules["llama_cpp"] = fake_llama_cpp

llama_vocab = glrmask.Vocab.from_llama_cpp(fake_llm)
assert llama_vocab.llama_cpp_end_token_ids == [3, 5]
assert llama_vocab.token(2).id == 2  # control token: exact-only, no byte route
control_constraint = (
    glrmask.Grammar.from_glrm(
        'glrm 1; start start; extern token CONTROL; nt start = CONTROL;'
    )
    .bind("CONTROL", llama_vocab.token(2))
    .compile(llama_vocab)
)
assert control_constraint.start().mask(6).tolist() == [False, False, True, False, False, False]

llama_constraint = glrmask.Grammar.from_ebnf(
    'start ::= "a"'
).compile(
    llama_vocab,
    end_tokens=llama_vocab.llama_cpp_end_token_ids,
)
llama_state = llama_constraint.start()
assert llama_state.mask(6).tolist() == [True, False, False, False, False, False]
llama_state.commit_token(0)
assert llama_state.mask(6).tolist() == [False, False, False, True, False, True]
llama_state.commit_token(3)
assert llama_state.is_terminated()

print("public Python API smoke test passed")
