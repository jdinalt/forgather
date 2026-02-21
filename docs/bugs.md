# Known Bugs

This file documents known bugs found in the `src/forgather/` top-level modules.
Bugs are documented here but not yet fixed; unit tests in `tests/unit/forgather/`
verify the buggy behavior (marked with `pytest.mark.xfail` where appropriate).

---

## Pyright: `latent.py` analysis times out

**File**: `src/forgather/latent.py`
**Severity**: Low (type-checking only; no runtime impact)

### Description

Running `pyright src/forgather/latent.py` consistently times out regardless of
timeout budget. The file is type-correct — a standalone equivalent passes Pyright
in under 30 seconds. The issue is at the package-analysis level: when Pyright
analyzes `latent.py` as part of the `forgather` package it loads `__init__.py`,
which imports `project.py`, which imports `latent.py`, forming a circular
analysis dependency that causes excessive re-analysis work.

All other top-level files in `src/forgather/` pass Pyright with 0 errors.

### Workaround

Analyze the other files explicitly, omitting `latent.py` (and `codegen.py` which
imports it):

```bash
pyright src/forgather/utils.py src/forgather/dynamic.py src/forgather/config.py \
        src/forgather/graph_encoder.py src/forgather/yaml_utils.py \
        src/forgather/yaml_encoder.py src/forgather/trainer_control.py \
        src/forgather/template_utils.py src/forgather/meta_config.py \
        src/forgather/preprocess.py src/forgather/dotdict.py \
        src/forgather/__init__.py src/forgather/codegen.py
```

### Fix

Break the circular package-analysis chain. Options:
- Move the `project.py` import in `__init__.py` to a lazy/deferred import.
- Import `latent.py` symbols inside function bodies in `project.py` rather than
  at module level.

---

## Summary Table

| # | File | Severity | Description |
|---|------|----------|-------------|
| - | `latent.py` | Low | Pyright analysis times out due to circular package dependency |
