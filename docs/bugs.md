# Known Bugs

This file documents known bugs found in the `src/forgather/` top-level modules.
Bugs are documented here but not yet fixed; unit tests in `tests/unit/forgather/`
verify the buggy behavior (marked with `pytest.mark.xfail` where appropriate).

---

## Bug 1: `utils.py` — `indent_block()` references undefined variable

**File**: `src/forgather/utils.py`
**Line**: 147–150
**Severity**: High (function is completely broken)

### Description

The top-level `indent_block()` function references `indent_level`, which is not
defined anywhere in its scope:

```python
def indent_block(block):
    indent = " " * indent_level   # NameError: name 'indent_level' is not defined
    s = "".join(map(lambda s: indent + s + "\n", block.split("\n")))
    return s[:-1]
```

Any call to `indent_block()` will raise `NameError`.

### Note

`config.py` defines a **local** `indent_block()` inside `fconfig()` that works
correctly (it captures `indent_level` from the enclosing scope). The broken
top-level version in `utils.py` appears to be a leftover from an earlier
refactoring where the function was moved but the `indent_level` parameter was
not added.

### Fix

Add `indent_level` as a parameter (with a default) or remove the function
entirely since the local version in `config.py` already exists:

```python
def indent_block(block, indent_level=4):
    indent = " " * indent_level
    s = "".join(map(lambda s: indent + s + "\n", block.split("\n")))
    return s[:-1]
```

---

## Bug 2: `utils.py` / `config.py` — `add_exception_notes()` overwrites `.note` instead of appending

**File**: `src/forgather/utils.py`
**Line**: 61
**Severity**: Medium (note is silently lost)

### Description

In `add_exception_notes()`, when an exception has a `.note` attribute, the code
attempts to detect whether the note is a string using `isinstance(error, str)`.
This check tests the **exception object** (always an `Exception`, never a `str`),
not the note:

```python
if hasattr(error, "note"):
    if isinstance(error, str):      # BUG: should be isinstance(error.note, str)
        error.note += note          # never reached
    else:
        error.note = note           # always overwrites existing note
    return error
```

Because `isinstance(error, str)` is always `False` for an Exception, `error.note`
is always **overwritten** instead of **appended to**. Any pre-existing note is
silently lost.

### Fix

Change `isinstance(error, str)` to `isinstance(error.note, str)`:

```python
if hasattr(error, "note"):
    if isinstance(error.note, str):
        error.note += note
    else:
        error.note = note
    return error
```

---

## Bug 3: `codegen.py` — Duplicate `_getitem()` method definition

**File**: `src/forgather/codegen.py`
**Lines**: 207–210 (first definition), 245–249 (second definition)
**Severity**: Medium (first definition is dead code; behavior differs)

### Description

`PyEncoder` defines `_getitem()` twice. In Python, the second definition
silently shadows the first:

```python
# Line 207 — first definition (DEAD CODE, never called)
def _getitem(self, obj):
    mapping = obj.args[0]
    key = obj.args[1]
    s = self._encode(mapping) + f"[{repr(key)}]"  # uses repr(key)
    return s

# ... intervening methods ...

# Line 245 — second definition (this is the one actually used)
def _getitem(self, obj):
    o = obj.args[0]
    key = obj.args[1]
    s = self._encode(o) + "[" + self._encode(key) + "]"  # uses self._encode(key)
    return s
```

The two implementations differ in how they encode the key:
- First uses `repr(key)` — always produces a Python literal.
- Second uses `self._encode(key)` — recursively encodes, which is more correct
  for node graph keys but produces different output for simple string/int keys.

### Fix

Remove the first (dead) definition. Verify that `self._encode(key)` is the
intended behavior for all key types.

---

## Bug 4: `meta_config.py` — Missing `f` prefix on f-string

**File**: `src/forgather/meta_config.py`
**Line**: 145
**Severity**: Low (wrong error message, but error is still raised)

### Description

An error message is constructed as a plain string instead of an f-string:

```python
raise ValueError("fThe directory, '{project_directory}', does not exist.")
```

The `f` prefix is missing from the string. The literal character `f` appears at
the start of the message, and `{project_directory}` is not interpolated. The
error message is always the static text `"fThe directory, '{project_directory}',
does not exist."` regardless of the actual directory.

Compare the corrected line below it (line 147):
```python
raise ValueError(f"The directory, '{project_directory}', does not exist.")
```

### Fix

```python
raise ValueError(f"The directory, '{project_directory}', does not exist.")
```

---

## Bug 5: `yaml_utils.py` — Empty `!tuple` uses `named_list` instead of `named_tuple`

**File**: `src/forgather/yaml_utils.py`
**Line**: 144
**Severity**: Medium (silent type error — empty tuple becomes a list)

### Description

In `tuple_constructor()`, the empty scalar case creates a `SingletonNode` with
`"named_list"` instead of `"named_tuple"`:

```python
def tuple_constructor(loader, tag_suffix, node):
    constructor, identity = split_tag_idenity(tag_suffix)
    if isinstance(node, yaml.SequenceNode):
        return SingletonNode(
            "named_tuple", loader.construct_sequence(node), _identity=identity
        )
    elif isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
        if not isinstance(value, str) or value != "":
            raise TypeError(...)
        return SingletonNode("named_list", _identity=identity)  # BUG: should be "named_tuple"
```

An empty `!tuple` YAML tag (written as `!tuple ""` or `!tuple`) will materialize
as an empty `list` instead of an empty `tuple`.

### Fix

```python
return SingletonNode("named_tuple", _identity=identity)
```

---

## Bug 6: `yaml_utils.py` — Scalar YAML node in `CallableConstructor` unpacks string as args

**File**: `src/forgather/yaml_utils.py`
**Lines**: 76–80
**Severity**: Medium (runtime error or incorrect behavior for scalar constructor args)

### Description

In `CallableConstructor.__call__()`, when the YAML node is a scalar (e.g.,
`!singleton:list simple_string`), the scalar value is assigned directly to
`args`:

```python
else:
    args = loader.construct_scalar(node)  # Returns a string (or other scalar)
    kwargs = {}
...
return self.node_type(constructor, *args, _identity=identity, **kwargs)
```

If `args` is a string like `"hello"`, then `*args` unpacks it into individual
characters: `('h', 'e', 'l', 'l', 'o')`. This is almost certainly not intended.

### Fix

Wrap the scalar in a tuple or list to treat it as a single argument:

```python
else:
    scalar = loader.construct_scalar(node)
    args = (scalar,) if scalar else ()
    kwargs = {}
```

Alternatively, if the scalar should be treated as no argument (like an empty
sequence), assign `args = []`.

---

## Bug 7: `tests/conftest.py` — Incompatible with Python 3.11

**File**: `tests/conftest.py` (indirect — via `forgather.ml.trainer`)
**Severity**: Medium (blocks running any test suite on Python 3.11)

### Description

The root conftest imports `TrainingArguments` from `forgather.ml.trainer`,
which imports `override` from Python's `typing` module. `typing.override` was
added in Python 3.12 (PEP 698), so Python 3.11 raises:

```
ImportError: cannot import name 'override' from 'typing'
```

### Workaround (Python 3.11)

Run the forgather unit tests with `--noconftest` or `--confcutdir`:

```bash
python -m pytest tests/unit/forgather/ --noconftest
# or
python -m pytest tests/unit/forgather/ --confcutdir=tests/unit/forgather
```

### Fix

Guard the import in `src/forgather/ml/trainer/trainer.py`:

```python
try:
    from typing import override
except ImportError:
    from typing_extensions import override
```

Or declare `python_requires = ">=3.12"` in `pyproject.toml`.

---

## Missing Packages

The following packages were missing and have been installed:

| Package | Reason |
|---------|--------|
| `pytest-mock` | Required by test suite (listed in `pyproject.toml` optional-dependencies) |
| `pytest-cov` | Required by test suite (listed in `pyproject.toml` optional-dependencies) |

Install with: `pip install pytest-mock pytest-cov`

---

## Summary Table

| # | File | Line | Severity | Description |
|---|------|------|----------|-------------|
| 1 | `utils.py` | 147 | High | `indent_block()` references undefined `indent_level` |
| 2 | `utils.py` | 61 | Medium | `isinstance(error, str)` should be `isinstance(error.note, str)` |
| 3 | `codegen.py` | 207, 245 | Medium | Duplicate `_getitem()` — first definition is dead code |
| 4 | `meta_config.py` | 145 | Low | Missing `f` prefix on f-string in error message |
| 5 | `yaml_utils.py` | 144 | Medium | Empty `!tuple` yields `named_list` instead of `named_tuple` |
| 6 | `yaml_utils.py` | 76 | Medium | Scalar YAML node unpacked as string characters in `*args` |
| 7 | `tests/conftest.py` | 14 | Medium | `typing.override` requires Python 3.12+ |
