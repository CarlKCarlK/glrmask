//! Allocator policy and explicit diagnostic collection; no runtime-policy change.

#[cfg(feature = "allocation-tracking")]
use crate::allocation_tracking;
use pyo3::prelude::*;

#[cfg(feature = "allocation-tracking")]
#[global_allocator]
static GLOBAL: allocation_tracking::TrackingAllocator =
    allocation_tracking::TrackingAllocator(mimalloc::MiMalloc);

#[cfg(not(feature = "allocation-tracking"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// `libmimalloc-sys` intentionally does not expose constants for these advanced
// options. The values are pinned by the mimalloc v3 `mi_option_e` ordering.
const MIMALLOC_PURGE_DECOMMITS_OPTION: libmimalloc_sys::mi_option_t = 5;

const MIMALLOC_PURGE_DELAY_OPTION: libmimalloc_sys::mi_option_t = 15;

pub(super) fn configure_mimalloc_runtime_default() {
    // Keep delayed automatic purging, but reset unused pages with
    // MADV_FREE/MEM_RESET rather than synchronously decommitting them. Reset
    // pages remain reclaimable by the OS without charging decommit work to an
    // arbitrary runtime allocation. An explicit mimalloc environment setting
    // remains authoritative.
    if std::env::var_os("MIMALLOC_PURGE_DECOMMITS").is_none()
        && std::env::var_os("MIMALLOC_RESET_DECOMMITS").is_none()
    {
        unsafe {
            libmimalloc_sys::mi_option_set_enabled(MIMALLOC_PURGE_DECOMMITS_OPTION, false);
        }
    }
}

#[pyfunction]
pub(super) fn mimalloc_purge_delay() -> i64 {
    unsafe { libmimalloc_sys::mi_option_get(MIMALLOC_PURGE_DELAY_OPTION) as i64 }
}

#[pyfunction]
pub(super) fn mimalloc_purge_decommits() -> bool {
    unsafe { libmimalloc_sys::mi_option_is_enabled(MIMALLOC_PURGE_DECOMMITS_OPTION) }
}

#[pyfunction]
#[pyo3(signature = (force=true))]
pub(super) fn collect_allocator(force: bool) {
    unsafe {
        let purge_delay = libmimalloc_sys::mi_option_get(MIMALLOC_PURGE_DELAY_OPTION);
        if purge_delay < 0 {
            // An explicit no-purge policy should not make explicit collection
            // a no-op. Temporarily permit this caller-selected collection and
            // restore the configured policy before returning.
            libmimalloc_sys::mi_option_set(MIMALLOC_PURGE_DELAY_OPTION, 0);
        }
        libmimalloc_sys::mi_collect(force);
        if purge_delay < 0 {
            libmimalloc_sys::mi_option_set(MIMALLOC_PURGE_DELAY_OPTION, purge_delay);
        }
    }
}
