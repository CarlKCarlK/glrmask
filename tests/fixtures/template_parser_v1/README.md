# Legacy parser fixtures

Generated once before TPR2 was introduced, by commit7ecc35bac plus the temporary exporter test recorded in the session notes, using Rust1.95.0 on aarch64-apple-darwin. Grammar and vocabulary match `GRAMMARS[1]` and `vocab()` in `tests/template_parser_artifact.rs`. The exporter was removed after generation to prevent accidentally refreshing old-version fixtures with the current writer.

These are small synthetic grammar artifacts, not user data or production models. Tests must keep reading them with the ordinary LR, self-contained template and external-vocabulary template readers after the compact parser writer changes.

SHA256:
- `static-v30-lr.bin`:08ce2bfc1a17f3edc0d682c8c3b9276f3e7b6693934ee9afe4aafd150e38bcc3
- `static-v31-tpr1.bin`:e4b46bcc778f16432151ebc2759b4b3b7ec570e669bad3a9a0e295e150068d49
- `o2-v21-tpr1.bin`:d9aa6777084ff994d78b0b6348c0af809984805bece91a2d97ab6150b02b0999
- `o2-transfer-v14-tpx1.bin`:006321e3e77685b59c05380c804e42bb99b0d1b77f3e33af5df93a197d9843b6
