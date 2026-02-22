# Testing Guide

This document covers how to run and work with the Forgather test suite. It is intended for developers contributing to the project.

## Prerequisites

Install test dependencies:

```bash
pip install -e ".[test]"
```

This pulls in `pytest`, `pytest-cov`, and `pytest-mock`. You also need a working PyTorch installation. Some tests additionally require:

- **CUDA** -- a GPU with CUDA support (tests that need it are skipped automatically when unavailable)
- **torchrun** -- PyTorch's distributed launcher, included with PyTorch
- **torchdata** -- the `torchdata.stateful_dataloader` package (needed by `test_fast_hf_loader.py`; tests are skipped if missing)

## Quick Reference

```bash
# Run the full unit test suite
pytest tests/unit/

# Run a specific test group
pytest tests/unit/forgather/
pytest tests/unit/ml/
pytest tests/unit/ml/diloco/

# Run a single test file
pytest tests/unit/ml/test_checkpoints.py

# Run with coverage
pytest tests/unit/ --cov=forgather --cov-report=term-missing

# Run distributed tests (require torchrun)
torchrun --nproc_per_node=2 tests/test_checkpoint_integration.py
./tests/run_dataloader_dispatcher_tests.sh
```

## Test Organization

```
tests/
├── conftest.py                        # Shared fixtures (temp_dir, mock_model, etc.)
├── CHECKPOINT_TESTING.md              # Checkpoint integration test documentation
├── run_dataloader_dispatcher_tests.sh # Shell driver for distributed dataloader tests
│
├── unit/                              # Standard pytest unit tests
│   ├── forgather/                     # Core framework tests
│   └── ml/                            # ML subsystem tests
│       ├── datasets/                  # Dataset loading and processing
│       └── diloco/                    # Distributed local-compute optimization
│
├── integration/                       # Integration tests (placeholder)
├── pipeline_split/                    # Pipeline parallelism tests (torchrun)
├── fixtures/                          # Test fixtures (placeholder)
├── utils/                             # Test utilities (placeholder)
│
└── [root-level test files]            # Standalone tests, benchmarks, profiling scripts
```

### Test Markers

The following pytest markers are defined in `pytest.ini`:

| Marker | Meaning |
|---|---|
| `unit` | Unit tests |
| `integration` | Integration tests |
| `slow` | Slow-running tests |

Use `-m` to filter by marker:

```bash
pytest tests/ -m "unit"
pytest tests/ -m "not slow"
```

## Unit Tests (`tests/unit/`)

Unit tests are the primary test suite. They run with plain `pytest`, require no GPUs, and mock distributed state where needed. This is what you should run most often during development.

### Core Framework (`tests/unit/forgather/`)

Tests for the configuration, template, and code generation systems:

| Area | Files | What They Test |
|---|---|---|
| Configuration | `test_config.py` | `ConfigText`, `Config`, `ConfigDict` parsing and handling |
| Preprocessing | `test_preprocess.py` | Jinja2 preprocessing with custom line statement syntax |
| Code generation | `test_codegen.py` | `PyEncoder` -- generating Python code from configuration objects |
| YAML | `test_yaml_encoder.py`, `test_yaml_utils.py` | YAML serialization with custom tags (`!partial`, `!singleton`, `!factory`, `!var`) |
| Templates | `test_template_utils.py` | Template inheritance and inclusion utilities |
| Utilities | `test_dotdict.py`, `test_dynamic.py`, `test_utils.py` | Dynamic objects, nested dict access, general utilities |
| Graph | `test_graph_encoder.py`, `test_latent.py` | Graph representation and encoding |

**When to run:** After modifying anything in the core `forgather` package -- configuration parsing, template resolution, code generation, or YAML handling.

### ML Subsystem (`tests/unit/ml/`)

Tests for the machine learning infrastructure. These cover training, checkpointing, optimization, model construction, and distributed coordination:

| Area | Files | What They Test |
|---|---|---|
| Checkpointing | `test_checkpoints.py`, `test_checkpoint_types.py`, `test_sharded_checkpoint.py` | Checkpoint saving/loading, state sharing patterns (replicated, per_rank, global), distributed checkpoint coordination, manifest generation |
| Trainer | `test_trainer_components.py`, `test_training_script.py` | `PeriodicFunction`, `AMPContext`, `JsonLogger`, `TrainerState`, `TrainerControl`, training script generation |
| Distributed | `test_distributed.py`, `test_replication_validation.py` | Rank/world-size utilities, DDP replication validation (uses mocks, no actual distributed init) |
| Model | `test_construct.py`, `test_model_conversion.py`, `test_resize_embeddings.py`, `test_no_init_weights.py`, `test_remap_params.py` | Model construction, vLLM conversion, embedding resizing, weight remapping |
| Optimization | `test_optim_components.py` | Optimizer construction, scheduling, parameter groups |
| Loss | `test_loss.py` | `CausalLoss` and `ChunkedCausalLoss` for memory-efficient large-vocabulary training |
| Data | `test_data_collator.py`, `test_tokenizer.py` | Data collation for padded sequences, tokenizer utilities |
| Analysis | `test_analysis.py` | Training log parsing and summary statistics |
| Infrastructure | `test_file_locking.py`, `test_memory_monitor.py`, `test_utils.py` | File locking for multi-GPU saves, memory tracking, ML utilities |

**When to run:** After modifying trainers, checkpointing, model construction, optimizers, loss functions, or any ML infrastructure code.

**Note on CUDA:** Most tests in this group run on CPU. A few tests in `test_trainer_components.py` (mixed-precision / AMP tests), `test_optim_components.py`, and `test_loss.py` are gated behind `torch.cuda.is_available()` and skip automatically on CPU-only machines.

### Datasets (`tests/unit/ml/datasets/`)

| File | What It Tests |
|---|---|
| `test_dataset_utils.py` | Dataset utility functions |
| `test_soft_sequential.py` | Soft sequential dataset interleaving probabilities |
| `test_fast_hf_loader.py` | Fast HuggingFace dataset loader (requires `torchdata.stateful_dataloader`) |
| `test_interleaved_checkpoint_bug.py` | Checkpoint state restoration for interleaved datasets |

**When to run:** After modifying dataset loading, interleaving, or the HuggingFace loader.

### DiLoCo (`tests/unit/ml/diloco/`)

Tests for the Distributed Local-Compute Optimization (DiLoCo) system:

| File | What It Tests |
|---|---|
| `test_worker.py` | Pseudo-gradient computation and synchronization logic |
| `test_server.py` | Outer optimizer (SGD with Nesterov), state serialization |
| `test_server_client.py` | Client-server communication |
| `test_streaming.py` | Streaming and buffering during sync |
| `test_async.py` | Asynchronous operations |
| `test_dashboard.py` | Monitoring dashboard |
| `test_fault_tolerance.py` | Failure recovery mechanisms |
| `test_diloco_callback.py` | DiLoCo training callback integration |

**When to run:** After modifying anything in the DiLoCo subsystem.

## Root-Level Tests (`tests/`)

Root-level test files are a mix of standalone integration tests, benchmarks, and profiling scripts. Some of these run via `pytest`, but several are standalone scripts that require `torchrun` or manual execution.

### Data Processing and Packing

These test the sequence packing and data pipeline logic. They run with `pytest`:

```bash
pytest tests/test_bin_packing.py
pytest tests/test_packing_comparison.py
pytest tests/test_packing_batch_behavior.py
pytest tests/test_shuffle_output.py
pytest tests/test_qwen3_packing.py
pytest tests/test_document_boundaries.py
```

| File | What It Tests |
|---|---|
| `test_bin_packing.py` | Bin-packing algorithm for fitting documents into fixed-size containers |
| `test_packing_comparison.py` | Greedy vs. optimized packing strategy efficiency |
| `test_packing_batch_behavior.py` | Batch-level packing behavior |
| `test_shuffle_output.py` | Shuffling correctness in packed sequences |
| `test_qwen3_packing.py` | Packing with the Qwen3 tokenizer (151k vocabulary) |
| `test_document_boundaries.py` | Document boundary tracking in packed sequences |

**When to run:** After modifying sequence packing, bin packing, or the data collation pipeline.

### Other Standalone Tests (pytest-compatible)

```bash
pytest tests/test_divergence_detection.py
pytest tests/test_model_equivalence.py
pytest tests/test_optimizer_state_dict.py
```

| File | What It Tests |
|---|---|
| `test_divergence_detection.py` | Loss divergence detection via dual exponential moving average windows |
| `test_model_equivalence.py` | HuggingFace vs. Forgather model output comparison and weight transfer |
| `test_optimizer_state_dict.py` | Optimizer state serialization and deserialization |

## Tests Requiring `torchrun` (Distributed)

These tests launch multiple processes and require `torchrun`. They cannot be run with plain `pytest`.

### Checkpoint Integration Test

```bash
# Single process (no torchrun needed)
python tests/test_checkpoint_integration.py

# Multi-process DDP simulation
torchrun --nproc_per_node=2 tests/test_checkpoint_integration.py
torchrun --nproc_per_node=4 tests/test_checkpoint_integration.py

# Run specific scenario
python tests/test_checkpoint_integration.py --scenario basic
python tests/test_checkpoint_integration.py --scenario spike
python tests/test_checkpoint_integration.py --scenario all
```

Tests checkpoint preservation, race-condition fixes, divergence detection, and DDP coordination with synthetic metrics (no actual training). See `tests/CHECKPOINT_TESTING.md` for full details.

### Dataloader Dispatcher Tests

```bash
# Run full suite with gloo backend (CPU-only, no GPUs needed)
./tests/run_dataloader_dispatcher_tests.sh

# Run with nccl backend (requires GPUs)
./tests/run_dataloader_dispatcher_tests.sh nccl
```

Tests multi-dimensional batch dispatching across data-parallel and model-parallel dimensions. The shell script runs multiple configurations: 1D pure DP, 1D pure MP, single-rank edge cases, and 2D hybrid meshes.

### Distributed Pipeline Test

```bash
# Requires CUDA -- uses nccl backend
torchrun --nnodes 1 --nproc_per_node 2 tests/pipeline_split/test_distributed_pipeline.py
torchrun --nnodes 1 --nproc_per_node 4 tests/pipeline_split/test_distributed_pipeline.py
```

Tests pipeline parallelism using PyTorch's `PipelineStage`, including `BlockMask` attention transport between stages. **Requires CUDA.**

### Distributed build_sync Test

```bash
# Uses gloo backend (CPU-only)
torchrun --nproc_per_node 4 --standalone tests/unit/ml/test_build_sync_distributed.py

# Test with local process group
torchrun --nproc_per_node 4 --standalone tests/unit/ml/test_build_sync_distributed.py --local
```

Tests the `build_sync` context manager's barrier-based synchronization across ranks.

## Tests Requiring CUDA

The following tests or test subsets require a CUDA-capable GPU. When CUDA is unavailable, they are skipped automatically via `@pytest.mark.skipif` or `@unittest.skipUnless` decorators.

**Selectively skipped tests within pytest-compatible files:**

- `tests/unit/ml/test_trainer_components.py` -- 8 tests for mixed-precision (`AMPContext`, `FP16State`, gradient scaling)
- `tests/unit/ml/test_optim_components.py` -- 4 tests for CUDA-specific optimizer behavior
- `tests/unit/ml/test_loss.py` -- 2 tests (memory profiling, chunked cross-entropy on GPU)

**Standalone scripts that require CUDA:**

- `tests/pipeline_split/test_distributed_pipeline.py` -- uses `nccl` backend and `torch.cuda.set_device()`
- `tests/benchmark_adafactor_triton.py` -- Triton kernel benchmarks (GPU-only)
- `tests/test_adafactor_triton.py` -- AdaFactor Triton kernel functional tests
- `tests/profile_large_vocab_memory.py` -- CUDA memory profiling for large-vocabulary models

## Benchmarks and Profiling Scripts

These are not part of the regular test suite. Run them manually when profiling performance:

```bash
# AdaFactor Triton kernel benchmark (requires CUDA)
python tests/benchmark_adafactor_triton.py

# Large vocabulary memory profiling (requires CUDA)
python tests/profile_large_vocab_memory.py

# AdaFactor Triton functional tests (requires CUDA)
python tests/test_adafactor_triton.py
```

## Shared Fixtures

`tests/conftest.py` provides fixtures available to all pytest-based tests:

| Fixture | Description |
|---|---|
| `temp_dir` | Creates and cleans up a temporary directory |
| `training_args` | Default `TrainingArguments` with sensible test defaults |
| `mock_model` | A `SimpleMockModel` (single `nn.Linear` layer) |
| `mock_dataset` | A mock dataset with `__len__` returning 10 |
| `mock_optimizer` | Factory function that creates an Adam optimizer |
| `mock_scheduler` | Factory function that creates a StepLR scheduler |

## Recommended Workflows

### Before Submitting a PR

Run the full unit test suite to catch regressions:

```bash
pytest tests/unit/
```

If your changes touch data loading or packing:

```bash
pytest tests/test_bin_packing.py tests/test_packing_comparison.py tests/test_document_boundaries.py
```

If your changes touch checkpointing:

```bash
pytest tests/unit/ml/test_checkpoints.py tests/unit/ml/test_checkpoint_types.py
python tests/test_checkpoint_integration.py --scenario all
```

### With GPU Access

Run the full suite including CUDA-gated tests:

```bash
pytest tests/unit/

# Distributed tests
torchrun --nproc_per_node=2 tests/test_checkpoint_integration.py --scenario all
./tests/run_dataloader_dispatcher_tests.sh
torchrun --nproc_per_node 4 --standalone tests/unit/ml/test_build_sync_distributed.py
```

If you have multiple GPUs, also run the pipeline test:

```bash
torchrun --nnodes 1 --nproc_per_node 2 tests/pipeline_split/test_distributed_pipeline.py
```

### Focused Development

Run only the tests relevant to what you changed:

```bash
# Configuration / templates
pytest tests/unit/forgather/

# Trainers and training loop
pytest tests/unit/ml/test_trainer_components.py tests/unit/ml/test_training_script.py

# DiLoCo
pytest tests/unit/ml/diloco/

# Datasets
pytest tests/unit/ml/datasets/

# Optimizers
pytest tests/unit/ml/test_optim_components.py

# Loss functions
pytest tests/unit/ml/test_loss.py
```

## Configuration

Pytest configuration lives in `pytest.ini` at the project root:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
```

`--strict-markers` means any undeclared marker will cause an error. If you add a new marker, register it in `pytest.ini`.
