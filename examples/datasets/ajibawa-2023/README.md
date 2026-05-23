# ajibawa-2023 Datasets

A collection of large-scale code corpora and a synthetic stories collection from [ajibawa-2023](https://huggingface.co/ajibawa-2023), wired up for Forgather training.

## Configurations

### Python-Code-Large

https://huggingface.co/datasets/ajibawa-2023/Python-Code-Large

> A large-scale corpus of 2M+ rows of Python source code designed to support research in large language model pretraining, code intelligence, and software engineering automation.

- [Python-Code-Large.yaml](./templatelib/configs/Python-Code-Large.yaml) Python Code Large, with best-fit sequence packing

### C-Code-Large

https://huggingface.co/datasets/ajibawa-2023/C-Code-Large

> A large-scale corpus of 4M+ C source code samples in JSONL format, covering procedural programming, manual memory management, and low-level abstractions.

- [C-Code-Large.yaml](./templatelib/configs/C-Code-Large.yaml) C Code Large, with best-fit sequence packing

### Cpp-Code-Large

https://huggingface.co/datasets/ajibawa-2023/Cpp-Code-Large

> A large-scale corpus of 5M+ lines of C++ source code spanning systems software, embedded systems, scientific computing, and modern C++ (C++11/14/17/20) paradigms.

- [Cpp-Code-Large.yaml](./templatelib/configs/Cpp-Code-Large.yaml) C++ Code Large, with best-fit sequence packing

### JavaScript-Code-Large

https://huggingface.co/datasets/ajibawa-2023/JavaScript-Code-Large

> A large-scale corpus of 5M+ JavaScript source files including modern ES6+ features, async patterns, and frontend/backend framework components.

- [JavaScript-Code-Large.yaml](./templatelib/configs/JavaScript-Code-Large.yaml) JavaScript Code Large, with best-fit sequence packing

### General-Stories-Collection

https://huggingface.co/datasets/ajibawa-2023/General-Stories-Collection

> A synthetic dataset of approximately 1.3M diverse stories with prompt-text pairs and token length information, suitable for training language models on text generation.

- [General-Stories-Collection.yaml](./templatelib/configs/General-Stories-Collection.yaml) General Stories Collection

### Code Interleaved

Interleaves all four ajibawa-2023 code datasets (Python, C++, C, JavaScript) using `interleave_datasets` with balanced remaining-examples weighting.

- [code-interleaved.yaml](./templatelib/configs/code-interleaved.yaml) Code Interleaved (all four code datasets, packed)
