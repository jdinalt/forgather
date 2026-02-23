# Integration Testing

This document describes the spec-driven integration test framework for Forgather. Integration tests automate what was previously a manual pre-merge workflow: running training projects end-to-end, checking for errors, validating loss convergence, and verifying that trained models produce coherent inference output.

## Running Integration Tests

```bash
# Smoke test -- single fastest project (~40s)
pytest tests/integration/ -m smoke

# All single-GPU training tests (~75s)
pytest tests/integration/ -m "integration and not slow"

# Full suite including inference + perplexity scoring (~2 min)
pytest tests/integration/ -m integration

# Collect without running (verify discovery)
pytest tests/integration/ --collect-only
```

All integration tests require at least one CUDA GPU. Tests that require more GPUs than available are skipped automatically.

## How It Works

### Architecture

Each integration test is defined as a **YAML spec file** in `tests/integration/specs/`. A pytest runner discovers these specs, executes `forgather train` as a subprocess (testing the real CLI path including `torchrun`), and validates results against the assertions defined in the spec.

```
tests/integration/
├── specs/                     # YAML test specifications
│   ├── tiny_experiment.yaml   # Smoke test
│   ├── tiny_llama.yaml        # Standard training test
│   └── tiny_llama_inference.yaml  # Training + inference + perplexity
├── conftest.py                # Fixtures, hooks, spec discovery
├── spec.py                    # Spec dataclass schema + YAML loader
├── runner.py                  # Subprocess runner for forgather train
├── assertions.py              # Assertion helpers (exit code, logs, stderr)
├── perplexity.py              # GPT-2 perplexity scorer
├── test_training.py           # Parametrized training tests
└── test_inference.py          # Inference + perplexity tests
```

### Execution Flow

1. `conftest.py:pytest_generate_tests` discovers all `*.yaml` files in `specs/` and parametrizes test functions with them.
2. `conftest.py:pytest_collection_modifyitems` applies pytest markers from each spec (e.g. `smoke`, `slow`) and skips tests that require more GPUs than available.
3. The `output_dir` fixture creates a temporary directory for training output.
4. `runner.py:run_forgather_train` builds and executes `forgather -p <project> -t <config> train --output-dir <tmp> <dynamic_args>` as a subprocess. Projects run in-place from their real locations so all template paths resolve naturally.
5. After training completes, the runner discovers `trainer_logs.json` in the output directory and parses it with `TrainingLog.from_file()`.
6. `assertions.py` validates: exit code, forbidden stderr patterns, expected output files, training log step count, and loss bounds.
7. For specs with an `inference` section, `test_inference.py` additionally starts the inference server, sends a completion request, and scores the output using GPT-2 perplexity.

### Output Isolation

Tests use the `--output-dir` CLI flag to redirect all training output (generated model code, logs, checkpoints) to a temporary directory provided by pytest's `tmp_path`. This means:

- Projects run in-place from their real locations in the repository, so relative template paths, workspace resolution, and cross-project references all work correctly.
- No files are written to `output_models/` or any other directory inside the repository.
- Each test gets its own isolated output directory that is automatically cleaned up by pytest.

### Perplexity Scoring

Inference tests evaluate the quality of generated text using GPT-2 as a reference model:

1. The trained model is loaded by the inference server using `--from-checkpoint` (the spec sets `save_strategy: "steps"` so a final checkpoint is saved).
2. A completion request is sent (e.g. prompt: "Once upon a time").
3. The prompt + generated text is scored using GPT-2's cross-entropy loss: `perplexity = exp(loss)`.
4. The test asserts that perplexity is below a threshold defined in the spec.

Lower perplexity means more coherent text. Typical ranges for GPT-2 scoring:

| Text Quality | Perplexity Range |
|---|---|
| Well-written English | 20--60 |
| Acceptable model output | 50--200 |
| Poorly trained model | 200--1000 |
| Random tokens | 1000+ |

The GPT-2 model (124M parameters) is loaded once per process via `@lru_cache` and is already present in the HuggingFace cache from other tests in this repository.

## Spec Reference

A spec is a YAML file with the following fields:

### Required Fields

| Field | Type | Description |
|---|---|---|
| `test_id` | string | Unique identifier, used as the pytest parameter ID |
| `project_dir` | string | Path to the Forgather project, relative to the repo root |
| `config` | string | Template config name (e.g. `train_tiny_llama.yaml`) |

### Optional Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `dynamic_args` | dict | `{}` | CLI overrides passed to `forgather train` (e.g. `max_steps`, `save_strategy`) |
| `loss.final_max` | float | none | Maximum allowed final training loss |
| `loss.final_min` | float | none | Minimum expected final training loss (sanity check) |
| `loss.no_nan` | bool | `true` | Fail if any training step has NaN loss |
| `stderr.forbidden_patterns` | list[string] | `[]` | Fail if any of these strings appear in stderr |
| `stderr.warn_patterns` | list[string] | `[]` | Log a warning (but don't fail) if these appear in stderr |
| `expected_files` | list[string] | `["trainer_logs.json"]` | Files that must exist in the training run directory |
| `min_steps_logged` | int | `1` | Minimum number of training step entries in `trainer_logs.json` |
| `gpu_requirement` | int | `1` | Number of GPUs required; test is skipped if fewer are available |
| `timeout` | int | `300` | Subprocess timeout in seconds |
| `markers` | list[string] | `["integration"]` | Pytest markers to apply (e.g. `smoke`, `slow`) |

### Inference Section (optional)

Include an `inference` section to add an inference smoke test after training:

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.prompt` | string | `"Once upon a time"` | Text completion prompt |
| `inference.max_tokens` | int | `50` | Maximum tokens to generate |
| `inference.temperature` | float | `0.7` | Sampling temperature |
| `inference.perplexity_max` | float | `500.0` | Maximum GPT-2 perplexity for the generated text |
| `inference.server_timeout` | int | `60` | Seconds to wait for the inference server to start |

When `inference` is present, the spec must set `save_strategy` to something other than `"no"` (e.g. `"steps"`) so that model weights are saved for the inference server to load.

### Routing: Training vs Inference Tests

Specs are automatically routed to the correct test function:
- Specs **without** an `inference` section are collected by `test_training.py:test_training_project`.
- Specs **with** an `inference` section are collected by `test_inference.py:test_inference_with_perplexity`.

## Creating a New Test

### Step 1: Identify the Project and Config

Determine which Forgather project and config template to test. You can list available configs with:

```bash
forgather -p examples/tutorials/tiny_llama ls
```

### Step 2: Determine Dynamic Args

Training specs should override `max_steps` to keep test duration short. Set it high enough for at least one training log entry (typically `max_steps >= logging_steps` from the project config). Check the project's `logging_steps` with:

```bash
forgather -p <project_dir> -t <config> pp | grep logging_steps
```

Use `save_strategy: "no"` for training-only tests. Use `save_strategy: "steps"` if the spec includes inference (to produce a checkpoint).

### Step 3: Establish Loss Bounds

Run the training manually to determine a reasonable `final_max` loss:

```bash
forgather -p <project_dir> -t <config> train --max-steps <N> --save-strategy no
```

Check the final loss in `trainer_logs.json` and set `final_max` with headroom (e.g. 20--30% above the observed value). Loss values vary with random seeds, so leave enough margin to avoid flaky tests.

### Step 4: Write the Spec File

Create a new YAML file in `tests/integration/specs/`. Example for a training-only test:

```yaml
test_id: my_new_project
project_dir: examples/my_category/my_project
config: my_config.yaml
dynamic_args:
  max_steps: 200
  save_strategy: "no"
loss:
  final_max: 8.0
  no_nan: true
stderr:
  forbidden_patterns:
    - "RuntimeError"
    - "CUDA error"
    - "Traceback"
  warn_patterns: []
expected_files:
  - trainer_logs.json
min_steps_logged: 2
timeout: 300
gpu_requirement: 1
markers:
  - integration
```

Example for a test with inference and perplexity scoring:

```yaml
test_id: my_project_inference
project_dir: examples/my_category/my_project
config: my_config.yaml
dynamic_args:
  max_steps: 500
  save_strategy: "steps"
loss:
  final_max: 5.0
  no_nan: true
stderr:
  forbidden_patterns:
    - "RuntimeError"
    - "CUDA error"
    - "Traceback"
  warn_patterns: []
expected_files:
  - trainer_logs.json
min_steps_logged: 5
inference:
  prompt: "Once upon a time"
  max_tokens: 50
  temperature: 0.7
  perplexity_max: 500.0
  server_timeout: 60
timeout: 900
gpu_requirement: 1
markers:
  - integration
  - slow
```

### Step 5: Verify

```bash
# Check that the spec is discovered
pytest tests/integration/ --collect-only

# Run the new test
pytest tests/integration/ -k my_new_project -v

# Verify no repo pollution
git status examples/
```

## Test Tiers

| Tier | Marker Filter | Tests | Typical Time | When to Run |
|---|---|---|---|---|
| Smoke | `-m smoke` | Fastest single project | ~40s | Every change |
| Standard | `-m "integration and not slow"` | All single-GPU training | ~75s | Before merge |
| Full | `-m integration` | Training + inference + perplexity | ~2 min | Pre-release, nightly |

## Existing Specs

| Spec | Project | Steps | What It Tests |
|---|---|---|---|
| `tiny_experiment.yaml` | `examples/tiny_experiments/tiny_experiment` | 150 | Smoke test: basic causal LM training (4M params, TinyStories) |
| `tiny_llama.yaml` | `examples/tutorials/tiny_llama` | 200 | Standard: Llama-architecture training (4M params, TinyStories) |
| `tiny_llama_inference.yaml` | `examples/tutorials/tiny_llama` | 500 | Full: training + checkpoint + inference server + GPT-2 perplexity |

## Troubleshooting

### HuggingFace Model Code Cache

Forgather models use `trust_remote_code=True` and HuggingFace caches the model's Python code in `~/.cache/huggingface/modules/transformers_modules/<model_name>/`. If model source code changes (e.g. in `modelsrc/transformer/`), the cached version may be stale. Clear the relevant cache entry:

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/<model_name>
```

### `max_steps` vs `logging_steps`

If `assert_log_metrics` fails with "Expected at least N training log entries, got 0", the spec's `max_steps` is likely smaller than the project's `logging_steps`. Training step entries in `trainer_logs.json` are only written every `logging_steps` intervals. Increase `max_steps` to be at least `logging_steps` + 1.

### Inference "No model.safetensors found"

The inference server loads model weights via `from_pretrained()`. Training with `save_strategy: "no"` does not save weights. For inference specs, use `save_strategy: "steps"` to ensure a final checkpoint is created, and the test will pass `--from-checkpoint` to the server.

## Code Reference

| File | Purpose |
|---|---|
| `tests/integration/spec.py` | `IntegrationSpec`, `LossBounds`, `StderrAssertions`, `InferenceSpec` dataclasses; `load_all_specs()` YAML loader |
| `tests/integration/runner.py` | `TrainingResult` dataclass; `run_forgather_train()` subprocess runner |
| `tests/integration/assertions.py` | `assert_exit_code`, `assert_no_forbidden_stderr`, `assert_expected_files`, `assert_log_metrics`, `check_warn_patterns` |
| `tests/integration/perplexity.py` | `compute_perplexity()` using GPT-2 with `@lru_cache` model loading |
| `tests/integration/conftest.py` | `output_dir` fixture; `pytest_generate_tests` (spec discovery); `pytest_collection_modifyitems` (marker application, GPU skip) |
| `tests/integration/test_training.py` | `test_training_project` -- parametrized by specs without `inference` section |
| `tests/integration/test_inference.py` | `test_inference_with_perplexity` -- parametrized by specs with `inference` section |

The runner reuses `forgather.ml.analysis.log_parser.TrainingLog` from the main codebase for log parsing.
