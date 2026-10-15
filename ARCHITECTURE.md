# Project structure notes

- `compress.py` is a small repository-level launcher; it adds `src/` to Python's import path and calls `llm_compress.compression.main()`.
- `src/llm_compress/` contains the Python implementation. `llm_encode.py`, `finetune.py`, and `traditional_encode.py` are kept as separate modules to preserve the current workflow.
- `src/llm_compress/utils/` contains resource monitoring and fine-tuning helpers.
- `compress.sh` and `Tawa-0.7/` remain at the repository root because the legacy Tawa shell workflow relies on those relative paths. The pipeline changes to the repository root before invoking it.
- The current algorithms and model settings have intentionally not been redesigned. This refactor focuses on organization, launch behavior, dependency metadata, documentation, and keeping credentials out of source.

The vendored Tawa toolkit includes native binaries and legacy scripts. Test it on the target OS before relying on it; binary compatibility and upstream licensing should be reviewed separately.
