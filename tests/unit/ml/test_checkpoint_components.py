"""Tests for configurable checkpoint state components (PR 4).

`TrainingArguments.checkpoint_components` selects which state components a run
actually saves/loads, filtered at the single chokepoint
`BaseTrainer.get_active_state_components()`. Excluding `"model"` means the
CheckpointManager skips model-weight save/load and the trainer treats the
weights as externally supplied (DiLoCo). Covers:

* the filter (None = all; subset; unknown-key guard),
* `_model_weights_external()` derived from the component set,
* CheckpointManager skipping model save/load when `"model"` is excluded,
  while still persisting the non-model training state.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from forgather.ml.distributed import StaticDistributedEnvironment
from forgather.ml.trainer.base_trainer import BaseTrainer
from forgather.ml.trainer.checkpoint_manager import CheckpointConfig, CheckpointManager
from forgather.ml.trainer.checkpoint_types import SharingPattern, StateComponent
from forgather.ml.trainer.trainer import Trainer, TrainingArguments
from forgather.ml.trainer.trainer_types import TrainerState

# ---------------------------------------------------------------------------
# get_active_state_components — the filter
# ---------------------------------------------------------------------------


def _mk(key):
    return StateComponent(key=key, stateful=None, sharing_pattern=SharingPattern.GLOBAL)


def _filter(components, checkpoint_components):
    stub = SimpleNamespace(
        get_state_components=lambda: components,
        args=SimpleNamespace(checkpoint_components=checkpoint_components),
        KNOWN_CHECKPOINT_COMPONENTS=BaseTrainer.KNOWN_CHECKPOINT_COMPONENTS,
    )
    return [c.key for c in BaseTrainer.get_active_state_components(stub)]


class TestGetActiveStateComponents(unittest.TestCase):
    def setUp(self):
        self.components = [
            _mk("model"),
            _mk("optimizer"),
            _mk("scheduler"),
            _mk("trainer"),
            _mk("rng"),
        ]

    def test_none_returns_all(self):
        self.assertEqual(
            _filter(self.components, None),
            ["model", "optimizer", "scheduler", "trainer", "rng"],
        )

    def test_subset_excludes_model(self):
        self.assertEqual(
            _filter(self.components, ["optimizer", "scheduler", "trainer", "rng"]),
            ["optimizer", "scheduler", "trainer", "rng"],
        )

    def test_unknown_key_raises(self):
        # A key outside the known vocabulary is a typo -> fail loud, so a
        # misspelled "model" can't silently turn a normal run into a
        # weights-external one (no-silent-fallback).
        with self.assertRaises(ValueError):
            _filter(self.components, ["optimizer", "modle"])

    def test_known_but_unproduced_key_is_ignored(self):
        # "dataset" is a valid key but absent from this component set (e.g. a
        # non-stateful loader) -> allowed, simply no effect.
        self.assertEqual(
            _filter(self.components, ["optimizer", "dataset"]), ["optimizer"]
        )

    def test_order_preserved_from_get_state_components(self):
        self.assertEqual(_filter(self.components, ["rng", "model"]), ["model", "rng"])


# ---------------------------------------------------------------------------
# _has_event_handler — external-load handler presence
# ---------------------------------------------------------------------------


class TestHasEventHandler(unittest.TestCase):
    def _has(self, callbacks):
        stub = SimpleNamespace(callbacks=callbacks)
        return BaseTrainer._has_event_handler(stub, "on_load_model_weights")

    def test_true_when_a_callback_implements_it(self):
        class _Loader:
            def on_load_model_weights(self, *a, **k):
                pass

        self.assertTrue(self._has([_Loader()]))

    def test_false_when_none_implement_it(self):
        class _Other:
            def on_train_begin(self, *a, **k):
                pass

        self.assertFalse(self._has([_Other()]))
        self.assertFalse(self._has([]))


# ---------------------------------------------------------------------------
# _verify_external_weights_loaded — the persistent weights were actually loaded
# ---------------------------------------------------------------------------


class _ModelWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        # Non-persistent buffer (RoPE-like): excluded from state_dict, so the
        # verification must NOT require it to be flagged.
        self.register_buffer("rope", torch.zeros(4), persistent=False)


class TestVerifyExternalWeightsLoaded(unittest.TestCase):
    def _verify(self, model):
        stub = SimpleNamespace(_materialized_modules=lambda: [model])
        Trainer._verify_external_weights_loaded(stub)

    def test_raises_when_persistent_weights_unflagged(self):
        # Nothing flagged -> the external source loaded nothing -> raise.
        with self.assertRaises(RuntimeError):
            self._verify(_ModelWithBuffer())

    def test_passes_when_persistent_weights_flagged(self):
        model = _ModelWithBuffer()
        # Flag the persistent params (what the server would supply); leave the
        # non-persistent 'rope' buffer unflagged (init recomputes it).
        for p in model.linear.parameters():
            p._is_hf_initialized = True
        self._verify(model)  # no raise

    def test_raises_when_only_some_flagged(self):
        model = _ModelWithBuffer()
        model.linear.weight._is_hf_initialized = True  # bias left unflagged
        with self.assertRaises(RuntimeError):
            self._verify(model)


# ---------------------------------------------------------------------------
# _model_weights_external — derived signal
# ---------------------------------------------------------------------------


class TestModelWeightsExternal(unittest.TestCase):
    def _ext(self, cc):
        stub = SimpleNamespace(args=SimpleNamespace(checkpoint_components=cc))
        return Trainer._model_weights_external(stub)

    def test_none_is_internal(self):
        self.assertFalse(self._ext(None))

    def test_model_present_is_internal(self):
        self.assertFalse(self._ext(["model", "optimizer"]))

    def test_model_absent_is_external(self):
        self.assertTrue(self._ext(["optimizer", "trainer", "rng"]))

    def test_empty_list_is_external(self):
        self.assertTrue(self._ext([]))


# ---------------------------------------------------------------------------
# CheckpointManager — skip model save/load when "model" is excluded
# ---------------------------------------------------------------------------


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)


class _Dataset(torch.utils.data.IterableDataset, Stateful):
    def __init__(self, n=8):
        self.n, self.i = n, 0

    def load_state_dict(self, sd):
        self.i = sd["index"]

    def state_dict(self):
        return {"index": self.i}

    def __iter__(self):
        return iter(range(self.n))


class _MockTrainer(BaseTrainer):
    """Minimal BaseTrainer exercising the checkpoint path with a real model,
    optimizer, scheduler, dataloader, and RNG — but no training loop."""

    def __init__(self, args):
        self.optimizer = None
        self.lr_scheduler = None
        super().__init__(args=args, model=_MockModel())

    def _post_init(self):
        self.args.device = "cpu"

    def _prepare(self, train_dataset, eval_dataset):
        self.state = TrainerState(
            logging_steps=10,
            eval_steps=10,
            train_batch_size=8,
            max_steps=100,
            max_eval_steps=-1,
            save_steps=20,
            best_metric=None,
            best_model_checkpoint=None,
            num_train_epochs=1,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10
        )
        self.train_dataloader = StatefulDataLoader(dataset=_Dataset())
        self.checkpoint_manager = CheckpointManager(
            config=CheckpointConfig(
                output_dir=self.args.output_dir, save_total_limit=3
            ),
            dist=StaticDistributedEnvironment(),
            stateful_provider=self,
            model=self.model,
        )

    def _train_loop(self):
        return Mock()

    def _eval_loop(self):
        return {}

    def load_state_dict(self, state_dict):
        self.state.global_step = state_dict["global_step"]

    def state_dict(self):
        return {"global_step": self.state.global_step}


_MODEL_FILE_HINTS = ("model.safetensors", "pytorch_model", ".bin")


def _has_model_weight_file(path):
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".safetensors") or "pytorch_model" in f or f.endswith(".bin"):
                return True
    return False


class TestCheckpointManagerModelSkip(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _trainer(self, checkpoint_components):
        args = TrainingArguments(
            output_dir=self.test_dir,
            save_total_limit=3,
            checkpoint_components=checkpoint_components,
        )
        t = _MockTrainer(args)
        t._prepare(train_dataset=Mock(), eval_dataset=None)
        return t

    def test_model_included_by_default_saves_weights(self):
        t = self._trainer(None)
        self.assertIsNotNone(t.checkpoint_manager.model_state_component)
        ckpt = t.checkpoint_manager.save_checkpoint(checkpoint_id="checkpoint-1")
        self.assertTrue(
            _has_model_weight_file(ckpt), "model weights should be saved by default"
        )

    def test_model_excluded_skips_weight_save(self):
        from forgather.ml.sharded_checkpoint import (
            MODEL_EXCLUDED_MARKER,
            validate_checkpoint,
        )

        t = self._trainer(["optimizer", "scheduler", "trainer", "dataset", "rng"])
        # Model component filtered out -> manager has no model component.
        self.assertIsNone(t.checkpoint_manager.model_state_component)
        ckpt = t.checkpoint_manager.save_checkpoint(checkpoint_id="checkpoint-1")
        self.assertFalse(
            _has_model_weight_file(ckpt),
            "model weights must NOT be saved when 'model' is excluded",
        )
        # Non-model state IS persisted (the checkpoint dir is non-empty).
        self.assertTrue(os.listdir(ckpt), "non-model state should still be written")
        # The model-less marker is present, so the checkpoint validates...
        self.assertTrue(os.path.exists(os.path.join(ckpt, MODEL_EXCLUDED_MARKER)))
        self.assertTrue(validate_checkpoint(ckpt))
        # ...but a model-less checkpoint WITHOUT the marker (a partial/corrupt
        # normal checkpoint) is rejected.
        os.remove(os.path.join(ckpt, MODEL_EXCLUDED_MARKER))
        self.assertFalse(validate_checkpoint(ckpt))

    def test_model_excluded_load_does_not_touch_weights(self):
        # Save without model weights, then mutate live weights and load — the
        # load must not overwrite them (model load is skipped).
        t = self._trainer(["optimizer", "trainer", "rng"])
        ckpt = t.checkpoint_manager.save_checkpoint(checkpoint_id="checkpoint-1")
        with torch.no_grad():
            t.model.linear.weight.fill_(3.14)
        t.checkpoint_manager.load_checkpoint(ckpt)
        self.assertTrue(
            torch.all(t.model.linear.weight == 3.14),
            "model weights must be untouched by load when 'model' is excluded",
        )


if __name__ == "__main__":
    unittest.main()
