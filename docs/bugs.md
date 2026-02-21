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

## `TensorTracker.track_tensor` weakref holds reference, preventing GC

**File:** `src/forgather/ml/memory_monitor.py:65`
**Severity:** Medium (memory monitoring never detects tensor deallocation)

### Description

The `track_tensor` method creates a weakref callback that captures the `tensor`
variable by reference in a lambda closure. This reference keeps the tensor alive,
defeating the purpose of using a weakref for garbage collection detection.

```python
def track_tensor(self, tensor, creation_info=""):
    self.register_tensor(tensor, creation_info)
    weakref.ref(tensor, lambda ref: self.tensor_finalizer(id(tensor)))
    #                                                     ^^^^^^^^
    #  This lambda captures `tensor` itself, preventing GC
```

### Fix

Capture the tensor ID before the lambda:

```python
def track_tensor(self, tensor, creation_info=""):
    self.register_tensor(tensor, creation_info)
    tensor_id = id(tensor)
    weakref.ref(tensor, lambda ref: self.tensor_finalizer(tensor_id))
```

### Test

`tests/unit/ml/test_memory_monitor.py::TestTensorTrackerTrack::test_track_cleanup_on_gc`

---

## `normalize_range` returns `None` when `select_range=None`

**File:** `src/forgather/ml/tokenizer.py:60`
**Severity:** Medium (callers must handle None; docstring says range(0, length))

### Description

The docstring states that `normalize_range(1000, None)` should return
`range(0, 1000)`, but the code returns `None` because the `None` case falls
through to `return select_range`.

```python
if select_range is None or isinstance(select_range, range):
    return select_range  # Returns None when select_range is None
```

### Fix

```python
if select_range is None:
    return range(length)
if isinstance(select_range, range):
    return select_range
```

### Test

`tests/unit/ml/test_tokenizer.py::TestNormalizeRangeNoneAndRange::test_none_returns_none_not_range`

---

## `normalize_range` negative int handling is inverted

**File:** `src/forgather/ml/tokenizer.py:51`
**Severity:** Medium (negative indexing produces wrong result)

### Description

Negative integer values are computed as `length - value` where `value` is already
negative. For example, with `value=-10` and `length=1000`, this produces
`1000 - (-10) = 1010` (clamped to 1000). The intended Python-style negative
indexing behavior would be `length + value = 1000 + (-10) = 990`.

```python
if value < 0:
    value = length - value  # Bug: double negation. -(-10) = +10
```

### Fix

```python
if value < 0:
    value = length + value  # Python-style: 1000 + (-10) = 990
```

### Test

`tests/unit/ml/test_tokenizer.py::TestNormalizeRangeNegativeValues::test_negative_int_current_behavior`

---

## `normalize_range` crashes on empty sequence

**File:** `src/forgather/ml/tokenizer.py:65`
**Severity:** Low (unlikely input)

### Description

Passing an empty list `[]` raises `TypeError` because `range(*())` requires at
least one argument. No guard against empty sequences.

```python
elif isinstance(select_range, Sequence):
    return range(*tuple(normalize_value(value) for value in select_range))
    # When select_range=[], this becomes range() which raises TypeError
```

### Fix

```python
elif isinstance(select_range, Sequence):
    if len(select_range) == 0:
        return range(length)  # or raise ValueError
    return range(*tuple(normalize_value(value) for value in select_range))
```

### Test

`tests/unit/ml/test_tokenizer.py::TestNormalizeRangeSequenceInput::test_empty_sequence`

---

## `DataCollatorForCausalLM.__repr__` has misplaced closing parenthesis

**File:** `src/forgather/ml/data_collator.py:228`
**Severity:** Low (cosmetic, repr output only)

### Description

The f-string has a closing parenthesis `)` placed after the `ignore_index` field
instead of at the end of the string. There is also a trailing space before `}`.
This produces output like:

```
DataCollatorForCausalLM(..., ignore_index=-100 ), pad_kwargs={}
```

instead of:

```
DataCollatorForCausalLM(..., ignore_index=-100, pad_kwargs={})
```

```python
f"ignore_index={self.ignore_index }), pad_kwargs={self.pad_kwargs}"
#                                ^              ^
#                    extra space -+   misplaced )-+
```

### Fix

```python
f"ignore_index={self.ignore_index}, pad_kwargs={self.pad_kwargs})"
```

### Test

`tests/unit/ml/test_data_collator.py::TestDataCollatorRepr::test_repr_format`

---

## `test_build_sync_distributed.py` typo: `AssertionError`

**File:** `tests/unit/ml/test_build_sync_distributed.py:254`
**Severity:** Medium (exception handler is dead code)

### Description

The exception handler catches `AssertionError` (missing second 'i') instead of
`AssertionError`. This means assertion failures in the distributed tests will raise
an uncaught `NameError` instead of being handled gracefully.

```python
except AssertionError as e:  # NameError at runtime - should be AssertionError
```

### Fix

```python
except AssertionError as e:
```

---

## `_pos_ids_from_boundaries` crashes when boundary values exceed sequence length

**File:** `src/forgather/ml/data_collator.py:19-28`
**Severity:** Medium (runtime crash with truncation + packed sequences)

### Description

When `document_starts` contains boundary values >= `T` (the sequence length),
`_pos_ids_from_boundaries` computes `end = starts[i+1]` without clamping to `T`.
This causes `torch.arange(doc_length)` to produce more elements than the slice
`pos_ids[batch_idx, start:end]` can hold, resulting in a `RuntimeError`.

This is triggered in practice when `truncation=True` and `packed_sequences=True`
are used together: truncation slices `document_starts` by index position but does
not filter out boundary *values* that exceed the new (truncated) sequence length.

```python
end = starts[i + 1] if i + 1 < len(starts) else T
doc_length = end - start
pos_ids[batch_idx, start:end] = torch.arange(doc_length, device=device)
# If start or end > T, arange produces more elements than the slice accepts
```

### Fix

Clamp `start` and `end` to `T`:

```python
start_clamped = min(int(start), T)
end = min(int(starts[i + 1]), T) if i + 1 < len(starts) else T
doc_length = end - start_clamped
if doc_length > 0:
    pos_ids[batch_idx, start_clamped:end] = torch.arange(doc_length, device=device)
```

### Test

`tests/unit/ml/test_data_collator.py::TestDataCollatorPackedSequences::test_truncation_with_out_of_bounds_document_starts_bug`

---

## Summary Table

| # | File | Severity | Description |
|---|------|----------|-------------|
| - | `latent.py` | Low | Pyright analysis times out due to circular package dependency |
| 1 | `memory_monitor.py:65` | Medium | `track_tensor` weakref lambda captures tensor, preventing GC |
| 2 | `tokenizer.py:60` | Medium | `normalize_range(n, None)` returns None, not range(0, n) |
| 3 | `tokenizer.py:51` | Medium | Negative int handling: `length - value` should be `length + value` |
| 4 | `tokenizer.py:65` | Low | Empty sequence `[]` raises TypeError |
| 5 | `data_collator.py:228` | Low | `__repr__` has misplaced `)` and extra space |
| 6 | `test_build_sync_distributed.py:254` | Medium | `AssertionError` typo makes except clause dead code |
| 7 | `data_collator.py:19-28` | Medium | `_pos_ids_from_boundaries` crashes when boundaries exceed sequence length |
