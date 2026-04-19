# Dynamic LM

A Jupyter notebook demonstrating how Forgather's dynamic model construction works at the low level — the API that the `Project` abstraction is built on top of.

The notebook walks through how a configuration graph is preprocessed, parsed, and compiled into Python code that constructs a live model object. It is the best available documentation for the internals of the code-generation pipeline.

Note: this notebook uses Forgather's original low-level API. It has been kept because the low-level mechanics it demonstrates are not covered elsewhere, and the code still runs without modification.

## Contents

- **[dynamic_lm.ipynb](./dynamic_lm.ipynb)** - Interactive notebook: configuration preprocessing, node graph construction, code generation, and model instantiation.
