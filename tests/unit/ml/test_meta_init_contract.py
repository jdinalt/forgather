"""Tests for the HF-v5 'initialize only the missing tensors' contract used
when loading checkpoints onto meta-constructed models.

Covers:
* the ``_is_hf_initialized`` flag polarity in the model-side init functions
  (``modelsrc/transformer/init_weights.py``) — unflagged tensors get
  initialized, flagged (loaded) ones are skipped;
* ``flag_loaded_tensors`` marking only the keys a loader actually filled;
* ``initialize_missing_weights`` dispatching to ``model._init_weights``
  (apply path) or the safe non-persistent-buffer reset fallback.
"""

import sys
from functools import partial

import torch
import torch.nn as nn

# modelsrc/transformer is not a package; add to path for direct import
# (mirrors tests/unit/ml/test_transformer_modules.py).
sys.path.insert(0, "modelsrc/transformer")

from init_weights import (  # noqa: E402
    init_weights_by_regex,
    module_is_initialized,
    simple_weight_init,
)

from forgather.ml.sharded_checkpoint import (  # noqa: E402
    flag_loaded_tensors,
    initialize_missing_weights,
)

# ---------------------------------------------------------------------------
# module_is_initialized — flag polarity
# ---------------------------------------------------------------------------


def test_module_is_initialized_default_false():
    """An unflagged module with local state is NOT considered initialized
    (HF v5: unmarked == needs init)."""
    lin = nn.Linear(4, 4)
    assert module_is_initialized(lin) is False


def test_module_is_initialized_true_when_all_flagged():
    lin = nn.Linear(4, 4)
    for t in lin.parameters(recurse=False):
        t._is_hf_initialized = True
    assert module_is_initialized(lin) is True


def test_module_is_initialized_false_when_any_unflagged():
    """A module with a loaded weight but an unflagged (non-persistent)
    buffer still needs initialization for that buffer."""
    lin = nn.Linear(4, 4)
    lin.register_buffer("aux", torch.zeros(4), persistent=False)
    for p in lin.parameters(recurse=False):
        p._is_hf_initialized = True
    # aux buffer left unflagged
    assert module_is_initialized(lin) is False


def test_module_is_initialized_false_for_stateless():
    assert module_is_initialized(nn.Sequential()) is False


# ---------------------------------------------------------------------------
# simple_weight_init / init_weights_by_regex — skip flagged, init unflagged
# ---------------------------------------------------------------------------


def test_simple_weight_init_skips_flagged():
    lin = nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.fill_(1.5)
        lin.bias.fill_(2.5)
    lin.weight._is_hf_initialized = True
    lin.bias._is_hf_initialized = True
    simple_weight_init(lin)
    assert torch.all(lin.weight == 1.5)
    assert torch.all(lin.bias == 2.5)


def test_simple_weight_init_runs_when_unflagged():
    lin = nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.fill_(1.5)
    simple_weight_init(lin)  # no flags -> reset_parameters runs
    assert not torch.all(lin.weight == 1.5)


def test_init_weights_by_regex_skips_flagged():
    lin = nn.Linear(4, 4)
    setattr(lin, "init_prefix", "ff.linear1")
    with torch.no_grad():
        lin.weight.fill_(3.0)
        lin.bias.fill_(0.0)
    lin.weight._is_hf_initialized = True
    lin.bias._is_hf_initialized = True
    regex = [
        ("weight", partial(nn.init.constant_, val=9.0)),
        ("bias", partial(nn.init.constant_, val=9.0)),
    ]
    init_weights_by_regex(lin, regex)
    assert torch.all(lin.weight == 3.0)  # untouched (flagged/loaded)
    assert torch.all(lin.bias == 0.0)


def test_init_weights_by_regex_inits_unflagged():
    lin = nn.Linear(4, 4)
    setattr(lin, "init_prefix", "ff.linear1")
    regex = [
        ("weight", partial(nn.init.constant_, val=9.0)),
        ("bias", partial(nn.init.constant_, val=9.0)),
    ]
    init_weights_by_regex(lin, regex)  # default-False -> initialize
    assert torch.all(lin.weight == 9.0)
    assert torch.all(lin.bias == 9.0)


# ---------------------------------------------------------------------------
# flag_loaded_tensors
# ---------------------------------------------------------------------------


def test_flag_loaded_tensors_marks_only_loaded_keys():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    flag_loaded_tensors(model, {"0.weight"})
    assert getattr(model[0].weight, "_is_hf_initialized", False) is True
    # bias of layer 0 and all of layer 1 were not in the loaded set.
    assert getattr(model[0].bias, "_is_hf_initialized", False) is False
    assert getattr(model[1].weight, "_is_hf_initialized", False) is False


def test_flag_loaded_tensors_marks_buffers():
    mod = nn.Module()
    mod.register_buffer("buf", torch.zeros(2))  # persistent -> in state_dict
    flag_loaded_tensors(mod, {"buf"})
    assert getattr(mod.buf, "_is_hf_initialized", False) is True


# ---------------------------------------------------------------------------
# initialize_missing_weights
# ---------------------------------------------------------------------------


def test_initialize_missing_weights_apply_path():
    """When the model exposes _init_weights, it is applied to every module."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.fill_(2.0)

    m = M()
    with torch.no_grad():
        m.lin.weight.fill_(0.0)
    initialize_missing_weights(m)
    assert torch.all(m.lin.weight == 2.0)


def test_initialize_missing_weights_fallback_resets_only_buffers():
    """A module without _init_weights falls back to resetting ONLY modules
    that own a non-persistent buffer (never in the checkpoint). Persistent,
    loaded params are left untouched even without flags."""

    class RoPEish(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.zeros(4), persistent=False)

        def reset_parameters(self):
            with torch.no_grad():
                self.inv_freq.copy_(torch.arange(4.0))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.rope = RoPEish()

    net = Net()
    with torch.no_grad():
        net.lin.weight.fill_(5.0)
    initialize_missing_weights(net)  # no _init_weights -> fallback
    assert torch.all(net.lin.weight == 5.0)  # persistent param untouched
    assert torch.all(net.rope.inv_freq == torch.arange(4.0))  # buffer recomputed
