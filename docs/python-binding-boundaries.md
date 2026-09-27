# Python binding boundaries

The extension keeps the public lifecycle in `api.rs`: a source `Grammar` becomes
an `UnlinkedConstraint` or a runnable `Constraint`. The registration entry point
does not own compilation or runtime algorithms.

## Responsibility map

| Module | Responsibility |
| --- | --- |
| `lib.rs` | Extension initialization and public class registration. |
| `api.rs` | Source grammars, immutable bindings, compile/link options. |
| `runtime.rs` | Compiled handles, owner-dependent lifetimes, sequence methods. |
| `vocab.rs` | Vocabulary construction and the llama.cpp adapter. |
| `conversion.rs` | Python errors, packed mask buffers, boolean mask expansion. |
| `profiling.rs` | Native diagnostic records converted to Python dictionaries. |
| `compiler.rs` | Private compilation adapters and compiler-cache diagnostics. |
| `internal.rs` | Registration and forwarding for unsupported internal tools. |
| `partition.rs` | Experimental vocabulary-equivalence analysis. |
| `allocator.rs` | Existing allocator defaults and explicit collection hooks. |

Implementation modules are private. Moving a Rust function does not make it a
supported Python API. Vocabulary partitioning remains under `_internal`.

## Sequence ownership

Each Python state contains a `self_cell` pairing the native state with its
`Arc<Constraint>` owner. The borrowed state cannot outlive its owner. A Python
caller may drop its original vocabulary and constraint references while keeping
the sequence alive. Different sequences never share their mutable parser
frontiers merely because they share a compiled constraint.

## Caller-owned mask buffers

Packed output uses a contiguous, one-dimensional NumPy `int32` array in model
token coordinates. The native operation sees its bits as `u32`; the sign bit is
an ordinary token bit. The output is a writable view, not a copied temporary.
An oversized output must have its unused tail cleared without touching storage
outside the supplied view.

`WritableMask` retains rust-numpy's exclusive borrow for the entire native call.
Its argument extractor uses `try_readwrite`, translating a failed borrow into a
Python `ValueError`. The ordinary rust-numpy extractor used previously called
the panicking `readwrite` convenience operation. A read-only input must not
escape as `PanicException` before the binding method begins.

Before any typed slice is formed, the argument wrapper also checks the raw
array pointer alignment. A contiguous NumPy view is not necessarily aligned.
The shared conversion then checks the required packed-word extent, including
private and O2 entry points; bad input returns a Python error without writing.

`bitmask_u32_view` is the single handwritten signed-to-unsigned mask view cast.
Its slice lifetime is tied to the held exclusive guard. No guard is dropped or
recreated around the native computation, and no extra mask copy is introduced.
Incorrect rank or dtype still fails during typed extraction; noncontiguous
arrays fail before native mask generation.

## Profiling is not the public contract

Private counters and profiled operations remain useful to repository tooling.
The shared profile conversion prevents ordinary, dynamic, and per-advance
diagnostics from silently diverging. Native timing boundaries are deliberately
unchanged by the module split.

Native timers do not include every part of Python argument extraction. A binding
performance review therefore needs both the existing native mask-plus-commit
measurements and an explicitly end-to-end Python-call check. A small cached-mask
microbenchmark measures wrapper overhead, not universal grammar performance or
the worst-case token latency.

## Review boundaries

Source inventory checks preserve method bodies, PyO3 attributes, field order,
and ownership macros except for the explicitly reviewed fallible argument guard
and shared conversion calls. They supplement, rather than replace, executable
tests. The complete Python suite, independent buffer/ownership regressions,
private partition checks, API reflection, and selected performance comparisons
are recorded with exact native hashes in the continuation evidence.

The binding split does not change the serialized artifact format or repair an
automaton that was compiled incorrectly by an older compiler. Such artifacts
still require recompilation. Passing the current-format bridge tests is not a
promise of long-term artifact compatibility.
