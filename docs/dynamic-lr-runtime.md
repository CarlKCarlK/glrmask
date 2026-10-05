# Ordinary Dynamic LR runtime

Ordinary Dynamic (O1) retains an executable LR table by default. O2/vocabulary
partition and public FastBuild explicitly use native templates; Static is
unchanged. The internal development switch `GLRMASK_DYNAMIC_TEMPLATE_DFA=1`
(or true/yes/on) selects ordinary Dynamic templates at compilation only. It is
resolved once for a complete source build before alternatives/worker pools.
No public API was added; artifact loading ignores this environment switch.

The shared compiler uses a narrow normalization choice and one runtime assembly
boundary. LR skips `PreparedTemplateParser::from_compiler_parts` entirely;
native derives immutable template programs and drops its temporary table before
Constraint materialization. Import/lowering, tokenizer/vocabulary work, masks
and commits remain shared. A direct-regular frontend retains its automaton for
exact LR table construction, then omits the runtime automaton bypass on LR.

Current nullable correctness is preserved using existing table source-nullability
metadata and an exact singleton initial-root completion guard. No EOF action or
pre-lowering admission row is rewritten; complex EOF actions retain their exact
semantics. Normal completion still follows the predecessor-feasible EOF closure.
An ordinary zero-byte model ID is not a byte transition; mask and commit reject
it, while a live exact special-token action remains valid. Known invalid tokens
enter the documented fail state; unknown IDs error without mutating it.

Self-contained retained-LR Constraints use v30; external-vocabulary LR uses v39
with a canonical exact-vocabulary digest. Native v37/v38 are unchanged. Dynamic
transfer artifacts use their existing self/external codecs. Load preserves the
built backend; mislabeled/mixed representations and mismatched vocabularies are
rejected. Native components/providers remain table-free, and native table access
cannot become a fallback.

Qualification: focused behavior/storage/persistence tests and native API checks
are required before the private commit. Production BUILD/TTFM/TBM measurements,
the first 1,000 then full 9,558-case framework corpus, preserved OLD LR binary
comparison, Windows candidate integration and public-package qualification are
separate gates. Do not infer measured build gains from skipping native stages.
