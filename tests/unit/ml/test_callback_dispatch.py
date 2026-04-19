"""Tests for the indexed callback dispatch system in BaseTrainer."""

import torch

from forgather.ml.trainer.base_trainer import BaseTrainer, BaseTrainingArguments
from forgather.ml.trainer.trainer_types import TrainerCallback

# ---------------------------------------------------------------------------
# Minimal concrete trainer subclass for testing dispatch in isolation
# ---------------------------------------------------------------------------


class _StubTrainer(BaseTrainer):
    """Minimal BaseTrainer subclass that stubs out abstract methods."""

    def _prepare(self, train_dataset=None, eval_dataset=None):
        pass

    def _train_loop(self):
        pass

    def _eval_loop(self):
        return {}

    def get_state_components(self):
        return []


def _make_trainer(**overrides):
    """Create a _StubTrainer with sensible defaults for dispatch testing."""
    model = torch.nn.Linear(2, 2)
    args = BaseTrainingArguments(
        output_dir="/tmp/test_dispatch",
        max_eval_steps=-1,
    )
    return _StubTrainer(args, model, callbacks=overrides.get("callbacks", []))


# ---------------------------------------------------------------------------
# Lightweight callback fixtures
# ---------------------------------------------------------------------------


class StepOnlyCallback:
    """Callback that only handles on_step_end."""

    def __init__(self):
        self.calls = []

    def on_step_end(self, args, state, control, **kwargs):
        self.calls.append("on_step_end")


class LogOnlyCallback:
    """Callback that only handles on_log."""

    def __init__(self):
        self.calls = []

    def on_log(self, args, state, control, **kwargs):
        self.calls.append("on_log")


class MultiEventCallback:
    """Callback that handles both on_step_end and on_log."""

    def __init__(self):
        self.calls = []

    def on_step_end(self, args, state, control, **kwargs):
        self.calls.append("on_step_end")

    def on_log(self, args, state, control, **kwargs):
        self.calls.append("on_log")


class ControlModifyingCallback:
    """Callback that sets should_save on on_step_end."""

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control


class InheritingCallback(TrainerCallback):
    """Callback that inherits from TrainerCallback (marker class)."""

    def __init__(self):
        self.calls = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.calls.append("on_train_begin")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIndexBuilding:
    """Verify the lazy event index is built correctly."""

    def test_only_relevant_callbacks_dispatched(self):
        step_cb = StepOnlyCallback()
        log_cb = LogOnlyCallback()
        trainer = _make_trainer(callbacks=[step_cb, log_cb])

        trainer._dispatch_event("on_step_end")

        assert step_cb.calls == ["on_step_end"]
        assert log_cb.calls == []

    def test_index_cached_after_first_dispatch(self):
        step_cb = StepOnlyCallback()
        trainer = _make_trainer(callbacks=[step_cb])

        trainer._dispatch_event("on_step_end")
        assert "on_step_end" in trainer._event_index

        # Second dispatch should reuse the cached index
        trainer._dispatch_event("on_step_end")
        assert step_cb.calls == ["on_step_end", "on_step_end"]

    def test_multiple_callbacks_same_event(self):
        cb1 = StepOnlyCallback()
        cb2 = StepOnlyCallback()
        trainer = _make_trainer(callbacks=[cb1, cb2])

        trainer._dispatch_event("on_step_end")

        assert cb1.calls == ["on_step_end"]
        assert cb2.calls == ["on_step_end"]

    def test_multi_event_callback_indexed_for_both(self):
        cb = MultiEventCallback()
        trainer = _make_trainer(callbacks=[cb])

        trainer._dispatch_event("on_step_end")
        trainer._dispatch_event("on_log")

        assert cb.calls == ["on_step_end", "on_log"]


class TestNoHandlers:
    """Dispatch of events with no handlers should be a no-op."""

    def test_no_callbacks_registered(self):
        trainer = _make_trainer(callbacks=[])
        control = trainer._dispatch_event("on_step_end")
        assert not control.should_save

    def test_no_callback_handles_event(self):
        step_cb = StepOnlyCallback()
        trainer = _make_trainer(callbacks=[step_cb])

        control = trainer._dispatch_event("on_epoch_begin")
        assert step_cb.calls == []
        assert not control.should_save

    def test_unknown_event_name(self):
        step_cb = StepOnlyCallback()
        trainer = _make_trainer(callbacks=[step_cb])

        # Novel event that no callback defines
        control = trainer._dispatch_event("on_custom_event")
        assert not control.should_save


class TestDynamicAddRemove:
    """Adding or removing callbacks invalidates the index."""

    def test_add_callback_invalidates_index(self):
        cb1 = StepOnlyCallback()
        trainer = _make_trainer(callbacks=[cb1])

        trainer._dispatch_event("on_step_end")
        assert len(trainer._event_index["on_step_end"]) == 1

        cb2 = StepOnlyCallback()
        trainer.add_callback(cb2)
        # Index should be cleared
        assert trainer._event_index == {}

        trainer._dispatch_event("on_step_end")
        assert cb2.calls == ["on_step_end"]
        assert len(trainer._event_index["on_step_end"]) == 2

    def test_remove_callback_invalidates_index(self):
        cb1 = StepOnlyCallback()
        cb2 = StepOnlyCallback()
        trainer = _make_trainer(callbacks=[cb1, cb2])

        trainer._dispatch_event("on_step_end")
        assert len(trainer._event_index["on_step_end"]) == 2

        trainer.remove_callback(cb1)
        assert trainer._event_index == {}

        trainer._dispatch_event("on_step_end")
        assert len(trainer._event_index["on_step_end"]) == 1

    def test_pop_callback_invalidates_index(self):
        cb = StepOnlyCallback()
        trainer = _make_trainer(callbacks=[cb])

        trainer._dispatch_event("on_step_end")
        popped = trainer.pop_callback(cb)
        assert popped is cb
        assert trainer._event_index == {}

    def test_pop_nonexistent_callback_preserves_index(self):
        cb = StepOnlyCallback()
        trainer = _make_trainer(callbacks=[cb])

        trainer._dispatch_event("on_step_end")
        original_index = dict(trainer._event_index)

        other = LogOnlyCallback()
        result = trainer.pop_callback(other)
        assert result is None
        # Index not cleared because nothing was removed
        assert trainer._event_index == original_index

    def test_add_callback_type(self):
        trainer = _make_trainer(callbacks=[])
        trainer.add_callback(StepOnlyCallback)
        assert len(trainer.callbacks) == 1
        assert isinstance(trainer.callbacks[0], StepOnlyCallback)


class TestControlPropagation:
    """Callbacks can modify TrainerControl."""

    def test_control_returned_and_applied(self):
        cb = ControlModifyingCallback()
        trainer = _make_trainer(callbacks=[cb])

        control = trainer._dispatch_event("on_step_end")
        assert control.should_save is True
        assert trainer.control.should_save is True

    def test_last_control_wins(self):
        class SetSave:
            def on_step_end(self, args, state, control, **kwargs):
                control.should_save = True
                return control

        class ClearSave:
            def on_step_end(self, args, state, control, **kwargs):
                control.should_save = False
                return control

        trainer = _make_trainer(callbacks=[SetSave(), ClearSave()])
        control = trainer._dispatch_event("on_step_end")
        assert control.should_save is False

    def test_none_return_preserves_control(self):
        """Callback returning None should not replace the control object."""
        cb = StepOnlyCallback()  # returns None
        trainer = _make_trainer(callbacks=[cb])
        trainer.control.should_save = True

        control = trainer._dispatch_event("on_step_end")
        assert control.should_save is True


class TestDuckTyping:
    """Callbacks don't need to inherit from TrainerCallback."""

    def test_plain_class_callback(self):
        cb = StepOnlyCallback()  # does not inherit TrainerCallback
        trainer = _make_trainer(callbacks=[cb])

        trainer._dispatch_event("on_step_end")
        assert cb.calls == ["on_step_end"]

    def test_inheriting_callback(self):
        cb = InheritingCallback()
        trainer = _make_trainer(callbacks=[cb])

        trainer._dispatch_event("on_train_begin")
        assert cb.calls == ["on_train_begin"]

        # Events not defined on InheritingCallback should not dispatch
        trainer._dispatch_event("on_step_end")
        assert cb.calls == ["on_train_begin"]  # unchanged


class TestUnknownEvents:
    """Novel event names only dispatch to callbacks that define them."""

    def test_custom_event_dispatched(self):
        class CustomCallback:
            def __init__(self):
                self.called = False

            def on_custom_event(self, args, state, control, **kwargs):
                self.called = True

        cb = CustomCallback()
        trainer = _make_trainer(callbacks=[cb])

        trainer._dispatch_event("on_custom_event")
        assert cb.called is True

    def test_custom_event_skips_unrelated(self):
        step_cb = StepOnlyCallback()

        class CustomCallback:
            def __init__(self):
                self.called = False

            def on_custom_event(self, args, state, control, **kwargs):
                self.called = True

        custom_cb = CustomCallback()
        trainer = _make_trainer(callbacks=[step_cb, custom_cb])

        trainer._dispatch_event("on_custom_event")
        assert custom_cb.called is True
        assert step_cb.calls == []


class TestCallbackOrdering:
    """Callbacks are dispatched in insertion order."""

    def test_dispatch_order(self):
        order = []

        class First:
            def on_step_end(self, args, state, control, **kwargs):
                order.append("first")

        class Second:
            def on_step_end(self, args, state, control, **kwargs):
                order.append("second")

        trainer = _make_trainer(callbacks=[First(), Second()])
        trainer._dispatch_event("on_step_end")
        assert order == ["first", "second"]
