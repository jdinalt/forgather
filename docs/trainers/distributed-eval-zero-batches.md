# Eval Loop: "produced zero batches" / "did not yield any examples"

This guide diagnoses three related errors from the trainer's eval path:

```
RuntimeError: Distributed evaluation produced zero batches across all N ranks.
```

raised from `DDPTrainer` when every shard is empty under sharded eval
(`dispatch_eval_batches=False`);

```
RuntimeError: Distributed evaluation produced zero batches on K of N
ranks (empty ranks: ...).
```

raised from `DDPTrainer`'s pre-flight check when *some* shards are empty
(this is the common case at world_size 5 or 6 with a small eval split:
the asymmetric `self.model(...)` calls would deadlock the DDP collectives,
so the trainer fails fast before the loop instead of hanging); and

```
RuntimeError: The eval dataloader did not yield any examples.
```

raised from the base `Trainer._eval_loop` (and reached by `DDPTrainer` when
`dispatch_eval_batches=True`, which routes eval through the same path).

All three errors share the same root cause: the eval pipeline produced
zero batches on at least one rank. The accompanying message lists the
relevant settings, including which combination of `dispatch_batches`,
`dispatch_eval_batches`, `dataloader_drop_last`, and
`per_device_eval_batch_size` produced the failure, and (for the partial
case) the exact list of empty ranks.

## What the error means

The eval split is too small to produce a batch under the current settings.
Two paths reach this state:

**Sharded eval (`dispatch_eval_batches=False`).** Each rank receives its
own shard of the eval dataset. With `world_size=4` a 200-example eval split
becomes ~50 examples per rank; with sequence packing the *number of packed
sequences per shard* can be much smaller still - for example, ~6 packed
sequences per shard at `seq_len=4096`. If any shard contains fewer than
`per_device_eval_batch_size` examples and `dataloader_drop_last=True`, the
dataloader yields nothing for that shard.

`DDPTrainer._eval_loop_all_shards` runs a pre-flight `next()` on every
rank's iterator before any model forward pass; an `all_reduce(SUM)` over
per-rank "has at least one batch" flags then surfaces the empty ranks.
- If every rank is empty: the loop is skipped and the trainer raises
  the "across all N ranks" message.
- If only some ranks are empty: the trainer raises the "on K of N ranks"
  message, listing the empty ranks. Without this pre-flight the loop
  would hang because the non-empty ranks would call `self.model(...)`
  while the empty ranks skipped it - the wrapped DDP module participates
  in collectives even under `torch.no_grad()`, so asymmetric forward
  calls deadlock the process group.

**Centralised eval (`dispatch_eval_batches=True`).** Rank 0 loads the eval
dataloader and broadcasts batches to the other ranks via
`DataloaderDispatcher`. The dispatcher needs to assemble at least one batch
per rank before any rank can step. If rank 0 exhausts the dataloader before
that point - for example, the eval split has fewer than
`per_device_eval_batch_size * world_size` examples and
`dataloader_drop_last=True` drops the only partial batch - every rank
reports zero steps and the base `Trainer._eval_loop` raises. Lowering
`dataloader_drop_last` or `per_device_eval_batch_size` does not help here
unless the eval split also has enough total examples to populate one batch
per rank.

## Why it happens

Two settings interact to cause this failure:

1. **The eval split is small or sharded.** A 200-example eval split
   becomes ~50 examples per rank with `world_size=4`. With sequence packing
   (`packed=True`, e.g. via `openorca-packed.yaml`) the *number of packed
   sequences per shard* can be much smaller still - for example,
   ~6 packed sequences per shard at `seq_len=4096`.

2. **`dataloader_drop_last=True` drops incomplete batches.** This is the
   Forgather default (set unconditionally in
   `templatelib/examples/projects/lm_training_project.yaml`) because some
   training paths require all batches to share the same shape - notably
   `torch.compile` with static shapes and pipeline parallel. Dropping the
   final, partial batch keeps shapes uniform across iterations.

When the *effective* per-rank example count is fewer than
`per_device_eval_batch_size`, `drop_last=True` discards the only available
batch. With sharded eval that's a per-shard count; with centralised eval
the dispatcher needs `per_device_eval_batch_size * world_size` examples on
rank 0 to populate one batch per rank.

`_eval_loop_all_shards` is intentional: it lets each rank process all of
its local eval data even when shard sizes are uneven (so a short shard
does not truncate longer shards via the SynchronizedDataLoader's MIN
reduction). But it cannot recover data that the dataloader silently
dropped before iteration began.

## Why `drop_last=True` is the default

Setting `dataloader_drop_last=False` causes the *last* batch of an epoch to
have a smaller-than-usual shape. Configurations that depend on uniform batch
shapes (compiled graphs, pipeline schedules, fixed-shape kernels) will then
fail right at the end of an epoch, **after** training has run successfully
for hours and **before** an end-of-epoch checkpoint is saved. That is a
worse user experience than failing fast at eval time, so the global default
errs on the side of `drop_last=True`.

## Remedies

Pick whichever applies to your situation:

### 1. Increase the eval split size

The simplest fix when the eval split was made artificially small. If
the source dataset has more validation examples available, widen the
split:

```yaml
[validation_dataset_split]
    == super()
    split: "train[:1000]"   # was train[:200]
```

Aim for at least `per_device_eval_batch_size * world_size` *post-packing*
sequences. With sequence packing this is hard to predict exactly; the
`forgather dataset --target eval_dataset --shard-index ... --num-shards N`
command will show how many packed sequences each shard actually produces.

### 2. Disable eval sharding (`shard_eval=False`)

Defined in `templatelib/base/datasets/load_dataset.yaml`. When set, the eval
dataset is loaded in full on every rank instead of being sharded:

```bash
forgather -t my_config.yaml train -d 0,1,2,3 -p 'shard_eval=False'
```

or in YAML:

```yaml
[config_metadata]
    == super()
    -- set ns.shard_eval = False
```

Each rank then evaluates the same eval data, which is wasteful (every rank
recomputes the same loss) but always correct.

### 3. Centralise eval on rank 0 (`dispatch_eval_batches=True`)

`DDPTrainingArguments.dispatch_eval_batches` is an eval-only override for
`dispatch_batches`. Set it to `True` to make eval run through
`DataloaderDispatcher`: rank 0 iterates the eval dataloader and broadcasts
each batch to a different rank, so each example is processed exactly once
across the world.

This still requires the eval split to contain at least
`per_device_eval_batch_size * world_size` examples (post-packing) when
`dataloader_drop_last=True` - otherwise rank 0 cannot assemble one full
batch per rank and the base `_eval_loop` raises. If the underlying split is
that small, prefer remedy 1 (enlarge the split) or remedy 4 (lower the
eval batch size).

```yaml
[trainer_args]
    == super()
    dispatch_eval_batches: True
```

This requires the eval dataset to **not** be sharded across ranks (otherwise
rank 0 would only dispatch its own 1/N shard). Combine with `shard_eval=False`,
or run a config where `dispatch_batches=True` for eval to keep training
sharded:

```yaml
[trainer_args]
    == super()
    dispatch_batches: False         # train is sharded
    dispatch_eval_batches: True     # eval centralises on rank 0

[config_metadata]
    == super()
    -- set ns.shard_eval = False
```

This combination makes training data-parallel (rank-local shards, no
broadcasts) while letting eval run on the full split on every rank.

### 4. Lower `per_device_eval_batch_size`

If the eval split simply cannot be enlarged, shrinking the eval batch size
makes it more likely that each shard yields at least one batch. The default
in `lm_training_project.yaml` is `per_device_train_batch_size * 2`; override
it in `[trainer_args]`:

```yaml
[trainer_args]
    == super()
    per_device_eval_batch_size: 1
```

This is a workaround, not a real fix - if the eval split is too small to
support data-parallel eval at all, prefer remedy 1, 2, or 3.

### 5. Disable `dataloader_drop_last` (last resort)

Setting `dataloader_drop_last=False` lets the dataloader yield the final
incomplete batch instead of dropping it. This will resolve the zero-batches
error, but it can cause the **train** path to fail near end-of-epoch on
configurations that require uniform batch shapes (compiled models, pipeline
parallel). Only use this if you understand and accept that risk.

```yaml
[trainer_args]
    == super()
    dataloader_drop_last: False
```

## Diagnostic tips

To see how many packed sequences each rank actually receives for a given
configuration, run:

```bash
for i in $(seq 0 $((N-1))); do
  forgather -t my_config.yaml dataset \
      --target eval_dataset \
      --shard-index $i --num-shards N \
      --tokenized -n 4 --truncate 100
done
```

Compare the per-shard count to `per_device_eval_batch_size`: if any shard
count is smaller than the eval batch size **and**
`dataloader_drop_last=True`, that rank will yield zero batches.

## Related options

- [`dispatch_batches`](trainer_options.md): controls the train and eval
  dataloader-dispatch strategy (eval falls back to this when
  `dispatch_eval_batches` is `None`).
- [`dispatch_eval_batches`](trainer_options.md): eval-only override for
  `dispatch_batches`. Default `None` (follow `dispatch_batches`).
- [`dataloader_drop_last`](trainer_options.md): drop incomplete batches.
  Defaults to `True` in Forgather's bundled project templates.
- `shard_eval` (template variable in
  `templatelib/base/datasets/load_dataset.yaml`): when `False`, the eval
  dataset is not sharded across ranks.
