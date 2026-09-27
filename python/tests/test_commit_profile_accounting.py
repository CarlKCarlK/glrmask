"""Exact finite-language and private profiling contract; no latency threshold."""
import numpy as np
import pytest
import glrmask as g
from glrmask import _glrmask as native

FIELDS = {'adv_n_nondet_waves', 'adv_det_exit_reason', 'adv_nondet_ns', 'fast_path_advance_ns', 'adv_n_nondet_branches', 'adv_n_nondet_merges', 'adv_gss_depth', 'n_advances', 'linear_fast_path_fuse_ns', 'adv_nondet_det_ns', 'adv_n_floor_crossings', 'adv_clone_ns', 'fast_path_total_ns', 'fast_path_state_update_ns', 'advance_core_ns', 'total_ns', 'adv_vstack_len', 'queue_enqueue_ns', 'adv_det_exit_state', 'fast_path_fuse_ns', 'linear_fast_path_action_lookup_ns', 'linear_fast_path_eligibility_ns', 'adv_fast_path_ns', 'linear_fast_path_carried_gate_ns', 'actionable_ns', 'adv_det_floor_cross_ns', 'linear_fast_path_materialize_ns', 'queue_exec_ns', 'linear_fast_path_profile_bookkeeping_ns', 'adv_nondet_det_floor_cross_ns', 'n_tokenizer_states', 'adv_n_nondet_isolates', 'adv_summary_ns', 'queue_match_ns', 'linear_fast_path_match_scan_ns', 'queue_bookkeeping_ns', 'adv_n_det_popn_ops', 'advance_may_check_ns', 'n_queue_entries', 'may_advance_ns', 'linear_fast_path_future_disallow_ns', 'linear_fast_path_state_update_ns', 'adv_n_det_action_lookups', 'fuse_ns', 'adv_n_det_goto_lookups', 'linear_fast_path_steps', 'adv_n_nondet_reduce_ops', 'prune_ns', 'linear_fast_path_exec_ns', 'linear_fast_path_total_ns', 'adv_n_reduces_above_floor', 'linear_fast_path_end_state_check_ns', 'fast_path_prune_ns', 'failed_fast_path_probe_ns', 'initial_exec_ns', 'exec_ns', 'fast_path_tokenizer_exec_ns', 'mask_cache_reuse_ns', 'queue_ns', 'linear_fast_path_advance_ns', 'advance_future_disallow_ns', 'fast_path_match_scan_ns', 'adv_det_ns', 'fast_path_end_state_check_ns', 'linear_fast_path_apply_action_wall_ns', 'advance_ns', 'adv_stack_shift_apply_ns', 'scan_ns', 'linear_fast_path_setup_ns', 'fast_path_future_disallow_ns'}
SOURCE = 'glrm 1; start root; t A = /a{1,80}/; nt root = A ("x" | "y") | "bbbbbbz";'
WORDS = {b"a" * n + suffix for n in range(1, 81) for suffix in (b"x", b"y")} | {b"bbbbbbz"}
TOKENS = {i: bytes([i]) for i in range(256)} | {256:b"aa", 257:b"aaaa", 258:b"ax", 259:b"ay", 131071:b"aa"}


def assert_mask(state, value, prefix):
    allowed = {i for i, data in TOKENS.items() if any(w.startswith(prefix + data) for w in WORDS)}
    expected = np.zeros(value.mask_len() + 3, dtype=np.uint32)
    for i in allowed:
        expected[i // 32] |= np.uint32(1 << (i % 32))
    for poison in [-1, 0x55555555]:
        buffer = np.full(len(expected), poison, dtype=np.int32)
        state.fill_mask(buffer)
        assert np.array_equal(buffer.view(np.uint32), expected), (prefix, sorted(allowed))
    assert state.is_accepting() == (prefix in WORDS)


@pytest.mark.parametrize("mode", ["FAST_BUILD", "O2", "FAST_RUNTIME", "AUTO"])
@pytest.mark.parametrize("loaded", [False, True])
def test_profile_fields_and_exact_masks_survive_commit_and_reload(mode, loaded):
    vocab = g.Vocab.from_id_to_bytes(TOKENS)
    if mode == "O2":
        value = native.DynamicConstraint.from_glrm_grammar(SOURCE, vocab, vocab_partition=True)
        loader = native.DynamicConstraint
    else:
        value = g.Grammar.from_glrm(SOURCE).compile(vocab, optimization=getattr(g.Optimization, mode))
        loader = g.Constraint
    if loaded:
        value = loader.load(value.save(), vocab)
    histories = [[97] * 80 + [120], [257] * 19 + [131071, 258], [98] * 6 + [122]]
    for tokens in histories:
        ordinary, profiled = value.start(), value.start()
        detailed = None if mode == "O2" else value.start()
        prefix = b""
        for token in tokens:
            for state in [ordinary, profiled] + ([] if detailed is None else [detailed]):
                assert_mask(state, value, prefix)
            ordinary.commit_token(token)
            profile = (profiled.commit_token_profiled(token) if mode == "O2" else
                       native._internal.commit_token_profiled(profiled, token))
            assert set(profile) == FIELDS
            assert all(isinstance(n, int) and n >= 0 for n in profile.values())
            assert profile["total_ns"] >= profile["mask_cache_reuse_ns"]
            if detailed is not None:
                result = native._internal.commit_token_per_advance(detailed, token)
                detail = result["commit_profile"]
                assert set(detail) == FIELDS
                assert all(isinstance(n, int) and n >= 0 for n in detail.values())
                assert detail["total_ns"] >= detail["mask_cache_reuse_ns"]
            prefix += TOKENS[token]
        for state in [ordinary, profiled] + ([] if detailed is None else [detailed]):
            assert_mask(state, value, prefix)
            assert state.is_accepting()
