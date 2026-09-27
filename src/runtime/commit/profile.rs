use crate::compiler::glr::parser::AdvanceProfile;
use crate::compiler::glr::table::Action;
use crate::ds::leveled_gss::LeveledGSSSummary;

use super::ParserGSS;

pub type GssProfileSummary = LeveledGSSSummary;

// Commit is a central runtime method and this profiling surface is used by
// CFA profile_step to choose optimization targets. Keep parent/child timing
// buckets on one wall-clock accounting tree; do not remove or repurpose
// fields without updating profile_step at the same time.
#[derive(Clone, Debug, Default)]
pub struct CommitProfile {
	pub total_ns: u64,
	/// Final generation update and optional mask-equality proof; included in total_ns.
	pub mask_cache_reuse_ns: u64,
	pub scan_ns: u64,
	pub prune_ns: u64,
	pub queue_ns: u64,
	pub fuse_ns: u64,
	pub initial_exec_ns: u64,
	pub exec_ns: u64,
	pub queue_exec_ns: u64,
	pub queue_match_ns: u64,
	pub queue_enqueue_ns: u64,
	pub queue_bookkeeping_ns: u64,
	pub advance_ns: u64,
	pub advance_may_check_ns: u64,
	pub advance_core_ns: u64,
	pub advance_future_disallow_ns: u64,
	pub actionable_ns: u64,
	pub may_advance_ns: u64,
	pub n_tokenizer_states: u64,
	pub n_queue_entries: u64,
	pub n_advances: u64,
	pub adv_n_reduces_above_floor: u64,
	pub adv_n_floor_crossings: u64,
	pub adv_n_nondet_waves: u64,
	pub adv_n_nondet_branches: u64,
	pub adv_clone_ns: u64,
	pub adv_summary_ns: u64,
	pub adv_fast_path_ns: u64,
        pub adv_stack_shift_apply_ns: u64,
        pub adv_det_ns: u64,
        pub adv_det_floor_cross_ns: u64,
        pub adv_nondet_ns: u64,
        pub adv_vstack_len: u64,
	pub adv_gss_depth: u64,
	pub adv_det_exit_reason: u64,
	pub adv_det_exit_state: u64,
	pub adv_n_det_action_lookups: u64,
	pub adv_n_det_goto_lookups: u64,
	pub adv_n_det_popn_ops: u64,
	pub adv_n_nondet_reduce_ops: u64,
        pub adv_n_nondet_merges: u64,
        pub adv_n_nondet_isolates: u64,
        pub adv_nondet_det_ns: u64,
        pub adv_nondet_det_floor_cross_ns: u64,
	pub fast_path_total_ns: u64,
	pub fast_path_tokenizer_exec_ns: u64,
	pub fast_path_match_scan_ns: u64,
	pub fast_path_end_state_check_ns: u64,
	pub fast_path_prune_ns: u64,
	pub fast_path_advance_ns: u64,
	pub fast_path_future_disallow_ns: u64,
	pub fast_path_fuse_ns: u64,
	pub fast_path_state_update_ns: u64,
	pub failed_fast_path_probe_ns: u64,
	pub linear_fast_path_total_ns: u64,
	pub linear_fast_path_exec_ns: u64,
	pub linear_fast_path_match_scan_ns: u64,
	pub linear_fast_path_end_state_check_ns: u64,
	pub linear_fast_path_advance_ns: u64,
	pub linear_fast_path_action_lookup_ns: u64,
	pub linear_fast_path_carried_gate_ns: u64,
	pub linear_fast_path_materialize_ns: u64,
	pub linear_fast_path_apply_action_wall_ns: u64,
	pub linear_fast_path_profile_bookkeeping_ns: u64,
	pub linear_fast_path_future_disallow_ns: u64,
	pub linear_fast_path_fuse_ns: u64,
	pub linear_fast_path_eligibility_ns: u64,
	pub linear_fast_path_setup_ns: u64,
	pub linear_fast_path_state_update_ns: u64,
	pub linear_fast_path_steps: u64,
}

#[derive(Clone, Debug)]
pub struct PerAdvanceEntry {
	pub terminal_id: u32,
	pub tokenizer_state: u32,
	pub gss_stacks_before: Vec<Vec<u32>>,
	pub gss_stacks_after: Vec<Vec<u32>>,
	pub gss_summary_before: GssProfileSummary,
	pub gss_summary_after: GssProfileSummary,
	pub match_start: usize,
	pub match_end: usize,
	pub token_bound: usize,
	pub match_bytes: Vec<u8>,
	pub profile: AdvanceProfile,
	pub summary_ns: u64,
}

pub(super) fn apply_advance_profile(commit_profile: &mut CommitProfile, profile: &AdvanceProfile) {
	commit_profile.adv_n_reduces_above_floor += profile.n_reduces_above_floor as u64;
	commit_profile.adv_n_floor_crossings += profile.n_floor_crossings as u64;
	commit_profile.adv_n_nondet_waves += profile.n_nondet_waves as u64;
	commit_profile.adv_n_nondet_branches += profile.n_nondet_branches as u64;
	commit_profile.adv_clone_ns += profile.clone_ns;
	commit_profile.adv_fast_path_ns += profile.fast_path_ns;
    commit_profile.adv_stack_shift_apply_ns += profile.stack_shift_apply_ns;
    commit_profile.adv_det_ns += profile.det_ns;
    commit_profile.adv_det_floor_cross_ns += profile.det_floor_cross_ns;
    commit_profile.adv_nondet_ns += profile.nondet_ns;
	commit_profile.adv_vstack_len = profile.vstack_len as u64;
	commit_profile.adv_gss_depth = profile.gss_depth as u64;
	commit_profile.adv_det_exit_reason = profile.det_exit_reason as u64;
	commit_profile.adv_det_exit_state = profile.det_exit_state as u64;
	commit_profile.adv_n_det_action_lookups += profile.n_det_action_lookups as u64;
	commit_profile.adv_n_det_goto_lookups += profile.n_det_goto_lookups as u64;
	commit_profile.adv_n_det_popn_ops += profile.n_det_popn_ops as u64;
	commit_profile.adv_n_nondet_reduce_ops += profile.n_nondet_reduce_ops as u64;
	commit_profile.adv_n_nondet_merges += profile.n_nondet_merges as u64;
    commit_profile.adv_n_nondet_isolates += profile.n_nondet_isolates as u64;
    commit_profile.adv_nondet_det_ns += profile.nondet_det_ns;
    commit_profile.adv_nondet_det_floor_cross_ns += profile.nondet_det_floor_cross_ns;
}

pub(super) fn fast_action_advance_profile(
	gss: &ParserGSS,
	action: &Action,
	elapsed_ns: u64,
) -> AdvanceProfile {
	AdvanceProfile {
		pure_shift: matches!(action, Action::Shift(..)),
		fast_path_ns: elapsed_ns,
		stack_shift_apply_ns: elapsed_ns,
		total_ns: elapsed_ns,
		top_states: gss.peek_values().len() as u32,
		gss_depth: gss.max_depth(),
		vstack_len: gss.try_virtual_stack().map_or(0, |vstack| vstack.len() as u32),
		..AdvanceProfile::default()
	}
}
