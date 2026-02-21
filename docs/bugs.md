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

## `LengthSyncCallback` passes dataset as `debug` parameter

**File**: `src/forgather/ml/datasets/dataloader_utils.py`
**Severity**: Medium (silent logic error at runtime)

### Description

`LengthSyncCallback.on_step_end()` and `on_evaluate()` call
`sync_dataset_state_from_dataloader(self.dataloader, self.dataset)`, but the
function signature is `sync_dataset_state_from_dataloader(dataloader, debug=False)`.
The `self.dataset` argument is passed as the `debug` parameter, which evaluates
as truthy (being a non-None object), causing debug output to be printed on every
sync. The dataset argument itself is unused.

### Fix

Remove the second argument from both call sites:

```python
sync_dataset_state_from_dataloader(self.dataloader)
```

The function already accesses `dataloader.dataset` internally, so the separate
`dataset` parameter on `LengthSyncCallback.__init__` and the `create_length_sync_callback`
factory are unnecessary. They could be kept for API documentation purposes but
should not be passed to the sync function.

---

## `model_conversion.__init__` exports undefined symbols

**File**: `src/forgather/ml/model_conversion/__init__.py`
**Severity**: Low (import error if accessed)

### Description

The `__all__` list includes `validate_tp_plan`, `validate_pp_plan`,
`validate_vllm_plans`, and `print_model_structure`, but none of these functions
are defined or imported anywhere in the codebase. Accessing them via
`from forgather.ml.model_conversion import validate_vllm_plans` raises
`ImportError`.

### Fix

Either implement the functions or remove them from `__all__`. The functions are
referenced in `docs/inference/vllm_integration.md` as planned features.

---

## `JsonLogger` uses deprecated `datetime.datetime.utcnow()`

**File**: `src/forgather/ml/trainer/callbacks/json_logger.py`
**Severity**: Low (deprecation warning, scheduled for removal in future Python)

### Description

Line 73 uses `datetime.datetime.utcnow().timestamp()` which is deprecated since
Python 3.12. The deprecation warning appears during testing.

### Fix

Replace with timezone-aware equivalent:

```python
timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
```

---

## Summary Table

| # | File | Severity | Description |
|---|------|----------|-------------|
| - | `latent.py` | Low | Pyright analysis times out due to circular package dependency |
| 1 | `dataloader_utils.py` | Medium | LengthSyncCallback passes dataset as debug parameter |
| 2 | `model_conversion/__init__.py` | Low | __all__ exports undefined symbols |
| 3 | `json_logger.py` | Low | Uses deprecated `datetime.utcnow()` |
