# Checkpoints

Forgather's checkpointing system saves and restores all trainer state — model weights,
optimizer, LR scheduler, dataset position, per-rank RNG, and training progress. Model
weights are written as standard HuggingFace Safetensors shards, readable by any
HF-compatible tool without a Forgather dependency.

**Related documentation:**

- [Checkpointing Overview](../checkpointing/README.md) — concepts and basic usage
- [User Guide](../checkpointing/user_guide.md) — practical patterns and troubleshooting
- [Distributed Abstraction](../checkpointing/distributed_checkpoint_abstraction.md) — state-sharing patterns (GLOBAL, PER_RANK, REPLICATED, etc.)
- [Migration Guide](../checkpointing/migration_guide.md) — implementing custom trainers
- [Sharded Checkpoint API](../checkpointing/sharded_checkpoint_api.md) — low-level shard API reference

---

## Checkpoint Management

::: forgather.ml.sharded_checkpoint.CheckpointMeta

---

::: forgather.ml.sharded_checkpoint.save_checkpoint

---

::: forgather.ml.sharded_checkpoint.load_checkpoint

---

::: forgather.ml.sharded_checkpoint.find_latest_checkpoint

---

::: forgather.ml.sharded_checkpoint.next_checkpoint_path

---

::: forgather.ml.sharded_checkpoint.validate_checkpoint

---

## Protocols

Protocols that trainer components implement to participate in the checkpoint system.

::: forgather.ml.trainer.trainer_types.CheckpointInterface

---

::: forgather.ml.trainer.trainer_types.StatefulProvider
