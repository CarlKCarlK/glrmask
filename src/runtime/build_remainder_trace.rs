//! Low-volume diagnostic timeline for build-remainder attribution.
//!
//! Enable before process startup:
//!   GLRMASK_TRACE_BUILD_REMAINDER=1
//!
//! Output is buffered until the last active Session finishes. Quiet production
//! compilation does not allocate events or read clocks at each disabled span.

use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

thread_local! {
    static CURRENT_SPAN: Cell<Option<u64>> = const { Cell::new(None) };
}

struct Event {
    id: u64,
    parent: Option<u64>,
    label: &'static str,
    begin_ns: u128,
    end_ns: u128,
    native_thread: Option<u64>,
    rust_thread: String,
}

struct State {
    epoch: Instant,
    next_id: AtomicU64,
    active_sessions: AtomicUsize,
    events: Mutex<Vec<Event>>,
}

static STATE: OnceLock<Option<State>> = OnceLock::new();

fn state() -> Option<&'static State> {
    STATE
        .get_or_init(|| {
            let enabled = std::env::var("GLRMASK_TRACE_BUILD_REMAINDER")
                .ok()
                .is_some_and(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                });
            enabled.then(|| State {
                epoch: Instant::now(),
                next_id: AtomicU64::new(1),
                active_sessions: AtomicUsize::new(0),
                events: Mutex::new(Vec::new()),
            })
        })
        .as_ref()
}

#[cfg(windows)]
fn native_thread_id() -> Option<u64> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThreadId() -> u32;
    }
    // SAFETY: this OS query has no arguments or caller-owned memory.
    Some(u64::from(unsafe { GetCurrentThreadId() }))
}

#[cfg(target_os = "linux")]
fn native_thread_id() -> Option<u64> {
    unsafe extern "C" {
        fn gettid() -> std::ffi::c_int;
    }
    // SAFETY: this OS query has no arguments or caller-owned memory.
    let id = unsafe { gettid() };
    (id >= 0).then_some(id as u64)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn native_thread_id() -> Option<u64> {
    None
}

pub(crate) struct Span {
    id: Option<u64>,
    parent: Option<u64>,
    previous: Option<u64>,
    label: &'static str,
    begin_ns: u128,
    // Spans must end on the thread where their TLS nesting was installed.
    not_send: PhantomData<Rc<()>>,
}

impl Span {
    pub(crate) fn new(label: &'static str, parent: Option<u64>) -> Self {
        let Some(state) = state() else {
            return Self {
                id: None,
                parent: None,
                previous: None,
                label,
                begin_ns: 0,
                not_send: PhantomData,
            };
        };

        let id = state.next_id.fetch_add(1, Ordering::Relaxed);
        let previous = CURRENT_SPAN.with(|current| current.replace(Some(id)));
        Self {
            id: Some(id),
            parent: parent.or(previous),
            previous,
            label,
            begin_ns: state.epoch.elapsed().as_nanos(),
            not_send: PhantomData,
        }
    }

    pub(crate) fn id(&self) -> Option<u64> {
        self.id
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let Some(id) = self.id else {
            return;
        };
        let state = state().expect("enabled span has trace state");
        let end_ns = state.epoch.elapsed().as_nanos();
        CURRENT_SPAN.with(|current| {
            debug_assert_eq!(current.get(), Some(id));
            current.set(self.previous);
        });
        let event = Event {
            id,
            parent: self.parent,
            label: self.label,
            begin_ns: self.begin_ns,
            end_ns,
            native_thread: native_thread_id(),
            rust_thread: format!("{:?}", std::thread::current().id()),
        };
        state
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event);
    }
}

pub(crate) struct Session {
    span: Option<Span>,
}

impl Session {
    pub(crate) fn new(label: &'static str) -> Self {
        if let Some(state) = state() {
            state.active_sessions.fetch_add(1, Ordering::AcqRel);
        }
        Self {
            span: Some(Span::new(label, None)),
        }
    }

    pub(crate) fn id(&self) -> Option<u64> {
        self.span.as_ref().and_then(Span::id)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        drop(self.span.take());

        let Some(state) = state() else {
            return;
        };
        if state.active_sessions.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }

        let mut events = {
            let mut buffered = state
                .events
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *buffered)
        };
        events.sort_unstable_by_key(|event| (event.begin_ns, event.id));

        for event in events {
            eprintln!(
                "[glrmask-build-event] pid={} id={} parent={} native_thread={} rust_thread={} begin_ns={} end_ns={} label={}",
                std::process::id(),
                event.id,
                event.parent.map_or_else(|| "-".to_owned(), |id| id.to_string()),
                event.native_thread.map_or_else(|| "-".to_owned(), |id| id.to_string()),
                event.rust_thread,
                event.begin_ns,
                event.end_ns,
                event.label,
            );
        }
    }
}

pub(crate) fn mark(label: &'static str, parent: Option<u64>) {
    drop(Span::new(label, parent));
}
