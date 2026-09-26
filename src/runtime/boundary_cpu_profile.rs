//! Internal diagnostic spans. Compiled out of normal dependency builds.
//! A caller supplies a thread-CPU clock; no logging, system calls, allocations,
//! parser work, or recursion are added by an inactive observer.
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryCpuReport {
    pub dispatch_ns: u64,
    pub dynamic_ns: u64,
    pub dispatch_calls: u64,
    pub dynamic_calls: u64,
    pub dynamic_max_call_ns: u64,
}
#[derive(Clone, Copy)]
pub(crate) enum Kind { Dispatch = 0, Dynamic = 1 }
struct Capture {
    clock: fn() -> u64,
    depth: [usize; 2],
    start: [u64; 2],
    report: BoundaryCpuReport,
}
thread_local! { static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) }; }
pub fn begin(clock: fn() -> u64) {
    CAPTURE.with(|cell| {
        let mut c = cell.borrow_mut();
        assert!(c.is_none(), "boundary CPU capture already active");
        *c = Some(Capture { clock, depth: [0; 2], start: [0; 2], report: BoundaryCpuReport::default() });
    });
}
pub fn take() -> BoundaryCpuReport {
    CAPTURE.with(|cell| {
        let mut c = cell.borrow_mut();
        if let Some(x) = c.as_ref() { assert_eq!(x.depth, [0, 0], "live boundary CPU span"); }
        c.take().map(|x| x.report).unwrap_or_default()
    })
}
pub(crate) struct Span { kind: Option<Kind>, _thread: PhantomData<Rc<()>> }
impl Span {
    #[inline]
    pub(crate) fn enter(kind: Kind) -> Self {
        let active = CAPTURE.with(|cell| {
            let mut c = cell.borrow_mut();
            let Some(c) = c.as_mut() else { return false; };
            let index = kind as usize;
            if c.depth[index] == 0 {
                c.start[index] = (c.clock)();
                match kind { Kind::Dispatch => c.report.dispatch_calls += 1, Kind::Dynamic => c.report.dynamic_calls += 1 }
            }
            c.depth[index] += 1;
            true
        });
        Self { kind: active.then_some(kind), _thread: PhantomData }
    }
}
impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        let Some(kind) = self.kind else { return; };
        CAPTURE.with(|cell| {
            let mut c = cell.borrow_mut();
            let Some(c) = c.as_mut() else { return; };
            let index = kind as usize;
            c.depth[index] -= 1;
            if c.depth[index] != 0 { return; }
            let elapsed = (c.clock)().saturating_sub(c.start[index]);
            match kind {
                Kind::Dispatch => c.report.dispatch_ns += elapsed,
                Kind::Dynamic => {
                    c.report.dynamic_ns += elapsed;
                    c.report.dynamic_max_call_ns = c.report.dynamic_max_call_ns.max(elapsed);
                }
            }
        });
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    thread_local! { static TICK: Cell<u64> = const { Cell::new(0) }; }
    fn now() -> u64 { TICK.with(|t| { let n = t.get(); t.set(n + 10); n }) }
    #[test]
    fn nested_spans_count_outer_interval_once() {
        begin(now);
        { let _a = Span::enter(Kind::Dispatch);
          { let _b = Span::enter(Kind::Dynamic); let _c = Span::enter(Kind::Dynamic); } }
        let r = take();
        assert_eq!((r.dispatch_ns, r.dynamic_ns), (30, 10));
        assert_eq!((r.dispatch_calls, r.dynamic_calls), (1, 1));
    }
    #[test]
    fn inactive_spans_do_not_count_and_threads_are_separate() {
        { let _a = Span::enter(Kind::Dynamic); }
        assert_eq!(take().dynamic_calls, 0);
        begin(now);
        let other = std::thread::spawn(|| { let _a = Span::enter(Kind::Dynamic); take() }).join().unwrap();
        assert_eq!(other.dynamic_calls, 0);
        { let _a = Span::enter(Kind::Dynamic); }
        assert_eq!(take().dynamic_calls, 1);
    }
    #[test]
    fn unwinding_closes_span_and_preserves_next_capture() {
        begin(now);
        let _ = std::panic::catch_unwind(|| { let _a = Span::enter(Kind::Dynamic); panic!("test unwind"); });
        assert_eq!(take().dynamic_ns, 10);
        begin(now); { let _a = Span::enter(Kind::Dynamic); }
        assert_eq!(take().dynamic_ns, 10);
    }
}
