# Bounded deterministic provider reductions

The provider-driven GLR parser uses the existing `VirtualStack` to perform a
known, unambiguous sequence of reductions before materializing the resulting
persistent stack. The vocabulary walker, LR actions, and grammar semantics are
unchanged. This reduces intermediate stack allocation in dynamic boundary masks
and concrete commits. No vocabulary or grammar preprocessing is required.

## Exact fallback and bounds

The complete input frontier must be an Interface-backed graph with an empty
accumulator label. That is a constant-time structural certificate covering every
input stack, not merely one isolated branch. Mixed labels, nonempty labels, and
other representations retain the original general traversal. This matters
because accumulator union and reduction-wave grouping can otherwise interact;
changing the order is not justified just by a concrete-stack argument. The mask
walk keeps its lexer exclusions separately, so this restriction does not erase
or bypass exclusion semantics.

A reduction does not consume the lookahead. If a supported reduction sequence
reaches branching, guards, acceptance, extra stack effects, or an unknown lower
stack region, the completed intermediate stack re-enters the general **reduction
frontier**, not the consumed/shifted output. A prefix with no completed reduction
is not resumed. A failed mutating primitive discards the speculative cursor and
uses the original immutable input. Completion-only queries use the reference
traversal. Original accumulator labels and component-scoped goto mappings remain
attached throughout.

One 64-action allowance belongs to the entire enclosing provider advance, not
to each restarted branch. Exceeding it retains the exact general path; it does
not reject an otherwise valid parse or mask token.

## Defaults and diagnostic controls

Both bounded reduction fusion and retaining completed reduction prefixes are
on by default. `GLRMASK_PROVIDER_REDUCTION_PREFIX=0` selects the original general
provider traversal. `GLRMASK_PROVIDER_RESUME_REDUCTIONS=0`, with fusion enabled,
selects the transactional prefix policy that retries from the original input
when the entire prefix cannot finish. Overrides are read once per process.
Neither flag is required to obtain the default improvements.

Correctness tests compare complete stack languages and correlated accumulators,
including hidden floors, scoped actions, branch points, completion queries,
shared budget exhaustion, and exact continuation after suspension. Independent
finite-word and runtime replay checks supplement those mechanism tests.

Latency acceptance must use the complete dynamic boundary interval, including
initial preparation and cleanup, and compare it with non-boundary work from the
same mask invocation. Separate percentile maxima must not be subtracted. This
optimization does not itself establish that boundary cost is at parity with
ordinary static masking; the remaining gap must be measured.

## Predecessor isolation regression

The reduction-source shortcut now explicitly isolates its sole visible
predecessor. A graph with one visible top can also contain an epsilon stack;
that empty path has no predecessor and must not receive the visible state's
goto. Regression tests cover both append and replace gotos, uniform-empty and
mixed accumulator labels, and the reference, transactional, and resumable
execution policies. This fixes the common reduction-source operation rather
than depending on the optimization to hide the error.
