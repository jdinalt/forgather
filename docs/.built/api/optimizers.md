# Optimizers

Forgather ships several optimizers and learning rate schedulers, available as configuration templates or directly via the Python API.

**Related documentation:**

- [Adafactor Triton Performance](../trainers/adafactor-triton-performance.md) — performance analysis and benchmarks for the Triton-optimized Adafactor kernel

## Optimizers

### `AdamW` {#forgather-ml-optim-adamw-adamw}

`forgather.ml.optim.adamw.AdamW`

```python
class AdamW(params: Iterable[nn.parameter.Parameter], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-06, weight_decay: float = 0.01, torch_compile: bool = True, bf16_stochastic_round: bool = True)
```

AdamW optimizer with optional stochastic rounding for pure-bf16 training.

Implements decoupled weight-decay regularization (Loshchilov & Hutter,
arXiv:1711.05101) on top of the Adam update rule (Kingma & Ba,
arXiv:1412.6980).  The distinguishing feature of this implementation is
first-class support for *pure bf16 training* — parameters, gradients, and
optimizer states all stay in bf16, with stochastic rounding (SR) used for
every write-back to avoid systematic truncation bias.  This eliminates the
need for fp32 master-weight copies while retaining most of the numerical
quality of mixed-precision training.

Prefer this optimizer over standard ``torch.optim.AdamW`` when:

* Training on hardware with fast bf16 throughput and limited memory.
* Running pure-bf16 experiments where fp32 master weights are undesirable.
* Using FSDP2 (DTensor-backed parameters are handled transparently).

> **Note**
>
> Stochastic rounding is seeded from a dedicated ``torch.Generator``
> initialised with a fixed seed (5489) so that all DDP ranks make identical
> rounding decisions and parameters stay in sync without extra communication.

> The inner ``_adam`` kernel is optionally compiled with
> ``torch.compile(..., fullgraph=True)`` for improved throughput.

> **References**
>
> Kingma, D. & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
> arXiv:1412.6980.

> Loshchilov, I. & Hutter, F. (2017). Decoupled Weight Decay Regularization.
> arXiv:1711.05101.

**Attributes**

- `compile`

**Methods**

#### `add_param_group` {#forgather-ml-optim-adamw-adamw-add_param_group}

```python
def add_param_group(param_group: dict)
```

_No documentation._

#### `step` {#forgather-ml-optim-adamw-adamw-step}

```python
def step(closure: Callable = None)
```

_No documentation._

#### `state_dict` {#forgather-ml-optim-adamw-adamw-state_dict}

```python
def state_dict()
```

Return optimizer state with structure validation.

#### `load_state_dict` {#forgather-ml-optim-adamw-adamw-load_state_dict}

```python
def load_state_dict(state_dict)
```

Load optimizer state with validation.

---

### `Adafactor` {#forgather-ml-optim-adafactor-adafactor}

`forgather.ml.optim.adafactor.Adafactor`

```python
class Adafactor(params: Iterable[nn.parameter.Parameter], lr: float = 0.001, decay_rate: float = -0.8, clip_threshold: float = 1.0, betas: Tuple[float, float] = (0.9, 0.999), eps: Tuple[float, float] = (1e-30, 0.001), weight_decay: float = 0.01, relative_step: bool = False, torch_compile: bool = True, bf16_stochastic_round: bool = True, use_triton: bool = False)
```

Memory-efficient adaptive optimizer with factored second-moment estimation.

Implements the Adafactor algorithm (Shazeer & Stern, arXiv:1804.04235).
For matrices, the second-moment accumulator is factored into outer-product
row and column vectors, reducing per-parameter memory from O(n*m) to
O(n+m).  For vectors and scalars the full accumulator is retained.

Like `AdamW`, this implementation supports *pure bf16 training* via
stochastic rounding on all write-backs, and handles FSDP2 DTensor
parameters transparently.  An optional Triton kernel path is available
for higher GPU throughput on CUDA devices.

Prefer Adafactor over AdamW when:

* Memory is the primary constraint (large models, small accelerators).
* Training transformers with large embedding or projection matrices where
  the factored approximation is a good fit.

> **Note**
>
> ``decay_rate`` controls how the effective ``beta2`` grows with step count:
> ``beta2t = clamp(1 - step^decay_rate, max=beta2)``.  The default of
> ``-0.8`` replicates the schedule from the paper.

> The Triton kernel path (``use_triton=True``) does not support
> ``relative_step=True``.

> **References**
>
> Shazeer, N. & Stern, M. (2018). Adafactor: Adaptive Learning Rates with
> Sublinear Memory Cost. arXiv:1804.04235.

> Loshchilov, I. & Hutter, F. (2017). Decoupled Weight Decay Regularization.
> arXiv:1711.05101.

**Attributes**

- `compile`
- `use_triton`
- `triton_module`

**Methods**

#### `add_param_group` {#forgather-ml-optim-adafactor-adafactor-add_param_group}

```python
def add_param_group(param_group: dict)
```

_No documentation._

#### `step` {#forgather-ml-optim-adafactor-adafactor-step}

```python
def step(closure: Callable = None)
```

_No documentation._

#### `state_dict` {#forgather-ml-optim-adafactor-adafactor-state_dict}

```python
def state_dict()
```

Return optimizer state handling conditional col=None.

#### `load_state_dict` {#forgather-ml-optim-adafactor-adafactor-load_state_dict}

```python
def load_state_dict(state_dict)
```

Load optimizer state handling conditional col=None.

---

### `Apollo` {#forgather-ml-optim-apollo-apollo}

`forgather.ml.optim.apollo.Apollo`

```python
class Apollo(params: Iterable[nn.parameter.Parameter], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-06, weight_decay: float = 0.0, rank: int = 1, scale: float = 1.0, scale_front: bool = False, update_steps: int = 10, mini: bool = False, projector_factory: Callable = None)
```

Low-rank gradient-projection optimizer with AdamW-level performance.

Implements the Apollo algorithm (Zhu et al., arXiv:2412.05270).  Rather
than maintaining full-size first and second moment buffers, Apollo projects
gradients into a low-rank subspace (controlled by ``rank``), runs the
Adam update there, and uses the resulting per-column scaling signal to
scale the full-rank gradient.  Moment buffer memory scales as
``O(rank * max(n, m))`` instead of ``O(n * m)``.

Also applies the Norm-Growth Limiter from Fira (arXiv:2410.01623) to
prevent destructive gradient updates.

Prefer Apollo over AdamW when:

* Memory is constrained and Adafactor's factored approximation is too
  aggressive (Apollo retains the full gradient for the parameter update).
* ``rank=1`` (Apollo-Mini) is desired for maximum memory savings while
  still outperforming SGD.

> **Note**
>
> The ``projector_factory`` callable is not serialisable and is therefore
> stripped from checkpoints.  It must be supplied again via the constructor
> when resuming from a checkpoint.

> **References**
>
> Zhu, W. et al. (2024). APOLLO: SGD-like Memory, AdamW-level Performance.
> arXiv:2412.05270.

> Chen, Y. et al. (2024). Fira: Can We Achieve Full-Rank Training of LLMs
> Under Low-Rank Constraint? arXiv:2410.01623.

**Methods**

#### `step` {#forgather-ml-optim-apollo-apollo-step}

```python
def step(closure: Callable = None)
```

_No documentation._

#### `state_dict` {#forgather-ml-optim-apollo-apollo-state_dict}

```python
def state_dict()
```

Return optimizer state with serialized projector objects.

Projector objects are converted to dicts containing only tensors and primitives
to ensure proper checkpoint serialization.

Note: The projector_factory in param_groups is removed since it's a
non-serializable function. On load_state_dict, it must be provided
via the optimizer constructor.

#### `load_state_dict` {#forgather-ml-optim-apollo-apollo-load_state_dict}

```python
def load_state_dict(state_dict)
```

Load optimizer state and reconstruct projector objects.

Deserializes projector dicts back into projector objects.

---

### `SGD` {#forgather-ml-optim-sgd-sgd}

`forgather.ml.optim.sgd.SGD`

```python
class SGD(params: Iterable[nn.parameter.Parameter], lr: float = 0.001)
```

Minimal vanilla SGD optimizer.

Applies the plain stochastic gradient descent update rule:

    ``p = p - lr * grad``

No momentum, weight decay, or gradient clipping.  Intended as a minimal
reference implementation and starting point for custom optimizers.  For
production training, prefer `AdamW` or `Adafactor`.

**Parameters**

- `params` (iterable of Parameter) — Model parameters to optimize.
- `lr` (float) — Learning rate.  Default is 1e-3.

**Methods**

#### `step` {#forgather-ml-optim-sgd-sgd-step}

```python
def step(closure: Callable = None)
```

_No documentation._

## Schedulers

### `InfiniteLRScheduler` {#forgather-ml-optim-infinite_lr_scheduler-infinitelrscheduler}

`forgather.ml.optim.infinite_lr_scheduler.InfiniteLRScheduler`

```python
class InfiniteLRScheduler(optimizer: Optimizer, warmup_steps: int = 0, cooldown_steps: int = 0, constant_lr: float = 3.75e-05, min_lr: float = 1e-08, tau: float = 10000.0, checkpoint_step: int = -1, start_annealing: bool = False, annealing_type: str = 'exponential', annealing_steps: int = 0, last_epoch: int = -1)
```

Learning rate scheduler for continual pre-training without a fixed budget.

Implements the Infinite Cosine Schedule (arXiv:2503.02844).  The key idea
is a permanent *constant phase* that can run indefinitely, enabling
continual pre-training without committing to a total step count up front.
Annealing is triggered on demand — typically by resuming from a checkpoint
with ``start_annealing=True`` — so multiple annealed checkpoints can be
derived from a single long training run.

The schedule has four sequential phases:

1. **Warmup** — linear ramp from 0 to ``base_lr`` over ``warmup_steps``.
2. **Cooldown** — cosine decay from ``base_lr`` to ``constant_lr`` over
   ``cooldown_steps``.
3. **Constant** — holds ``constant_lr`` indefinitely (the "infinite"
   phase).
4. **Annealing** — decays from ``constant_lr`` toward ``min_lr``,
   triggered at ``checkpoint_step``.  Two decay curves are supported:

   * ``"exponential"`` (default) — original paper formula; exponential
     decay controlled by ``tau``.
   * ``"rsqrt"`` — harmonic/rational decay from the WSD-S paper
     (arXiv:2410.05192); drops quickly at first then slows.

> **Note**
>
> ``start_annealing``, ``annealing_type``, ``annealing_steps``, and
> ``min_lr`` are *config-only* keys: they are taken from the constructor
> arguments and are not saved to or loaded from checkpoints.  This ensures
> backward compatibility and allows the annealing policy to be changed when
> resuming.

> **References**
>
> Zhu, Y. et al. (2025). Beyond Cosine Decay: On the effectiveness of
> Infinite Learning Rate Schedule for Continual Pre-training.
> arXiv:2503.02844.

> Hu, S. et al. (2024). Understanding Warmup-Stable-Decay Learning Rates:
> A River Valley Loss Landscape Perspective. arXiv:2410.05192.

**Attributes**

- `warmup_steps`
- `cooldown_steps`
- `constant_lr`
- `checkpoint_step`
- `min_lr`
- `tau`
- `start_annealing`
- `annealing_type`
- `annealing_steps`

**Methods**

#### `get_lr` {#forgather-ml-optim-infinite_lr_scheduler-infinitelrscheduler-get_lr}

```python
def get_lr()
```

Compute learning rate for the current step.

#### `state_dict` {#forgather-ml-optim-infinite_lr_scheduler-infinitelrscheduler-state_dict}

```python
def state_dict()
```

Return state dict excluding config-only parameters.

Config-only parameters (start_annealing, annealing_type,
annealing_steps) are always determined by the constructor
arguments, not by checkpoint state. This ensures backward
compatibility with checkpoints saved before these parameters
existed.

#### `load_state_dict` {#forgather-ml-optim-infinite_lr_scheduler-infinitelrscheduler-load_state_dict}

```python
def load_state_dict(state_dict)
```

Load state dict, preserving config-only parameters.

Config-only parameters are always taken from the constructor,
never from the checkpoint. The start_annealing flag controls
how checkpoint_step is resolved after loading:

- start_annealing=True with loaded checkpoint_step < 0:
  Begin annealing at the current step (last_epoch).
- start_annealing=True with loaded checkpoint_step >= 0:
  Resume annealing from where it left off.
- start_annealing=False:
  Restore checkpoint_step from the constructor value,
  ignoring whatever was saved in the checkpoint.

---

### `CosineLRScheduler` {#forgather-ml-optim-cosine_lr_scheduler-cosinelrscheduler}

`forgather.ml.optim.cosine_lr_scheduler.CosineLRScheduler`

```python
class CosineLRScheduler(optimizer: Optimizer, total_steps: int, warmup_steps: int = 0, min_lr: float = 0.0, last_epoch: int = -1)
```

Cosine decay learning rate scheduler with optional linear warmup.

Linearly warms the learning rate from 0 to ``base_lr`` over
``warmup_steps``, then applies a half-cosine decay from ``base_lr`` to
``min_lr`` over the remaining ``total_steps - warmup_steps`` steps.

This is the standard schedule for fixed-budget training runs.  For
continual pre-training without a predetermined budget, prefer
`InfiniteLRScheduler` or `WSDScheduler`.

**Parameters**

- `optimizer` (Optimizer) — Wrapped optimizer whose ``param_groups`` LRs will be managed.
- `total_steps` (int) — Total number of training steps (warmup + decay combined).
- `warmup_steps` (int) — Number of linear warmup steps before cosine decay begins.
Default is 0.
- `min_lr` (float) — Minimum learning rate at the end of cosine decay.  Default is 0.0.
- `last_epoch` (int) — Index of the last epoch, used when resuming.  Default is -1.

**Attributes**

- `total_steps`
- `warmup_steps`
- `decay_steps`
- `min_lr`

**Methods**

#### `get_lr` {#forgather-ml-optim-cosine_lr_scheduler-cosinelrscheduler-get_lr}

```python
def get_lr()
```

_No documentation._

---

### `WSDScheduler` {#forgather-ml-optim-wsd_scheduler-wsdscheduler}

`forgather.ml.optim.wsd_scheduler.WSDScheduler`

```python
class WSDScheduler(optimizer: Optimizer, warmup_steps: int = 0, min_lr: float = 1e-08, decay_steps: int = 1, decay_start_step: int = -1, start_decay: bool = False, last_epoch: int = -1)
```

Warmup-Stable-Decay learning rate scheduler.

Implements the WSD-S protocol (Hu et al., arXiv:2410.05192).  The stable
phase holds ``base_lr`` indefinitely, enabling training without a fixed
step budget.  Decay is triggered on demand — by setting
``decay_start_step`` ahead of time or retroactively via ``start_decay=True``
when resuming from a checkpoint — so multiple decayed checkpoints can be
produced from a single stable-phase run.

The schedule has three sequential phases:

1. **Warmup** — linear ramp from 0 to ``base_lr`` over ``warmup_steps``.
2. **Stable** — holds ``base_lr`` indefinitely until decay is triggered.
3. **Decay** — harmonic/rational decay from ``base_lr`` to ``min_lr``
   over ``decay_steps`` using linear interpolation of inverse LR.  The
   curve drops quickly at first then slows (convex shape).

> **Note**
>
> ``start_decay``, ``min_lr``, and ``decay_steps`` are *config-only* keys:
> they are taken from the constructor arguments and are not saved to or
> loaded from checkpoints.  This ensures backward compatibility and allows
> the decay policy to be changed when resuming.

> **References**
>
> Hu, S. et al. (2024). Understanding Warmup-Stable-Decay Learning Rates:
> A River Valley Loss Landscape Perspective. arXiv:2410.05192.

**Attributes**

- `warmup_steps`
- `min_lr`
- `decay_steps`
- `decay_start_step`
- `start_decay`

**Methods**

#### `get_lr` {#forgather-ml-optim-wsd_scheduler-wsdscheduler-get_lr}

```python
def get_lr()
```

Compute learning rate for the current step.

#### `state_dict` {#forgather-ml-optim-wsd_scheduler-wsdscheduler-state_dict}

```python
def state_dict()
```

Return state dict excluding config-only parameters.

#### `load_state_dict` {#forgather-ml-optim-wsd_scheduler-wsdscheduler-load_state_dict}

```python
def load_state_dict(state_dict)
```

Load state dict, preserving config-only parameters.

The start_decay flag controls how decay_start_step is resolved:

- start_decay=True with loaded decay_start_step < 0:
  Begin decay at the current step (last_epoch).
- start_decay=True with loaded decay_start_step >= 0:
  Resume decay from where it left off.
- start_decay=False:
  Restore decay_start_step from the constructor value,
  ignoring whatever was saved in the checkpoint.
