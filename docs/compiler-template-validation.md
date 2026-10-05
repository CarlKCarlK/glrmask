# Compiler template validation

Normal compilation builds the same native template automata without running
reference language products and compiler quotient/signature self-checks.
GLRMASK_ENABLE_COMMIT_TEMPLATE_DFAS selects construction; it no longer enables
compiler self-validation.

Set GLRMASK_VALIDATE_COMPILER_TEMPLATES=1 for integration tests, CI and explicit
qualification. Parser-DWA unit tests always enable these checks, even if the
environment switch is 0. The existing specific GLRMASK_VALIDATE_TEMPLATE_QUOTIENT,
GLRMASK_VALIDATE_CHARACTERIZATION_QUOTIENT and GLRMASK_VALIDATE_SPARSE_ACTION_SIGNATURES
switches remain explicit opt-ins for their checks. The template quotient switch
also enables the full POP/READ/PUSH split-language comparison.

The normal split constructor still rejects cycles and invalid phase transitions.
Scalar DEFAULT dispatch certificates, alphabet/target bounds, selected-domain
coverage, artifact input validation and runtime admission checks remain required.
No runtime or serialized LR table is allowed. Validation failures still panic or
reject compilation as before when validation is enabled.

Runtime performance measurements must unset all compiler validation flags;
record the exact settings. Validation-on qualification is measured separately.
