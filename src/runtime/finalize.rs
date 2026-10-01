use super::artifact::Constraint;

impl Constraint {
	pub(crate) fn rebuild_runtime_caches(&mut self) {
		self.rebuild_runtime_caches_impl(false, None);
	}

	pub(crate) fn rebuild_runtime_caches_preserving_packed_dwa_dense_masks(&mut self) {
		self.rebuild_runtime_caches_impl(true, None);
	}
}
