# Lexer compilation

The lexer crate retains one compilation boundary; this refactor does not split
hot algorithms into additional crates. The internal `lexer::compile` namespace
remains the entry point. The normal GLRMask public lifecycle is unchanged.

## Responsibilities

| Module under `crates/glrmask-lexer/src/automata/lexer/compile/` | Responsibility |
| --- | --- |
| `factor` | Language-preserving expression normalization and common factors. |
| `plan` | Shared subexpressions, nested language operations, and product plans. |
| `nfa`, `expression_graph` | Byte-NFA lowering and expression-labeled graph construction. |
| `component` | Compile one component, including finite-language shortcuts. |
| `bounded_repeat`, `repeat_suffix`, `virtual_repeat` | Exact bounded-repeat construction and eligible virtual representations. |
| `repeat_horizon` | Vocabulary-relative repeat horizons and cache ownership. |
| `product` | Product states, byte classes, construction, and coordinate traces. |
| `mapping` | Structural and vocabulary-relative state-map certificates. |
| `partition`, `partition_runtime`, `synthesis` | Component grouping, runtime publication, and smaller mask components. |
| `deferred` | Dense and virtual products that need not be materialized immediately. |
| `dfa_analysis`, `settings` | Shared automaton analyses and existing build policy. |

These are ownership boundaries, not a claim that compilation is a linear pass.
For example, lowering a nested intersection can recursively compile its operands.
Private imports name the collaborating owner explicitly. Shared data and methods
are visible only within the compilation subsystem unless an existing crate-level
or internal tooling contract requires wider access.

## Semantic boundaries

Expression factoring preserves the accepted byte language. A choice is a union,
a sequence concatenates languages, intersection retains both operands' accepted
strings, and exclusion subtracts the right operand. Rewriting a nullable repeat
requires particular care: eliminating an apparently redundant epsilon can change
where a terminal ends and which suffix can begin.

The exact commit automaton and a smaller mask-analysis automaton are different
objects. A state map between them must carry the structural or vocabulary-relative
proof required by its caller. Equal state counts, similar numbering, or identical
byte support are not such a proof. Preserve terminal finalizers and possible
futures as well as transitions.

Product traces retain the component coordinates needed to establish those maps.
A product optimization that merges coordinates must preserve this evidence or
explicitly decline the optimization. The absence of a fast-path certificate means
use the existing exact path, not assume the certificate holds.

A vocabulary-relative repeat horizon bounds how many body completions a model
token can cross. It does not replace the grammar's repetition bound. Keep horizon
caches associated with their owning vocabulary. Do not hold a cache lock while
spawning nested Rayon work; the existing publication policy deliberately permits
bounded duplicate cold computations to avoid worker starvation.

A deferred DFA is not an incomplete DFA: it is an exact representation whose
transition rows are materialized only by the operation that needs them. A cleanup
must not turn runtime publication into eager full construction, or discard the
compressed representation just to unify a return type.

## Partition policy

`PartitionOptions` names three independent choices: diagnostic labels, residual
isolation classes, and an optional adaptive-determinization override. `None` for
the override means use the existing environment/default policy, not `false`.
`build_regex_partitioned_with_options` forwards these values directly to the
partition compiler. The common default and explicit-adaptive convenience calls
remain for existing internal users; the six combinatorial overloads are gone.

`options_tests` checks all twelve combinations against the underlying compiler's
serialized DFA. Existing exhaustive language, shared-expression, finite-product,
repeat-bound, state-map and vocabulary-horizon regressions remain in `tests`.
The monolith's test implementation was moved, not discarded.

## Validation and further work

The initial extraction checksummed every original function body before the
explicit options simplification and reference-audited dead-wrapper removal.
Compiler checks and tests remain authoritative; the extraction audit alone does
not prove runtime performance. Use a separately preserved original extension and
the same vocabulary for paired build/mask/commit/serialization measurements.

Vocabulary equivalence remains an internal capability. No public API or saved
artifact format change is required by this lexer refactor. The terminal and
runtime kernels remain intact; do not interpret the file split as a new algorithm
or claim it improves latency without measurements.
