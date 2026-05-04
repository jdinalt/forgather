#!/usr/bin/env python3
"""
Unit tests for the forgather ML optim module components.

Tests cover:
- opt_utils: build_parameter_groups, make_grouped_optimizer
- rounding_utils: fp32_to_bf16_stochastic_round
- cosine_lr_scheduler: CosineLRScheduler (warmup, cosine decay)
- infinite_lr_scheduler: InfiniteLRScheduler (warmup, cooldown, constant, annealing)
- sequential_lr_factory: sequential_lr_factory
- subspace_proj: OnlinePCAProjector, RandProjector
- adamw: AdamW optimizer
- sgd: SGD optimizer
"""

import math
from functools import partial
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, StepLR

from forgather.ml.optim.adamw import AdamW
from forgather.ml.optim.cosine_lr_scheduler import CosineLRScheduler
from forgather.ml.optim.infinite_lr_scheduler import InfiniteLRScheduler
from forgather.ml.optim.multiopt import Multiopt
from forgather.ml.optim.opt_utils import (
    build_optimizer_buckets,
    build_parameter_groups,
    make_grouped_optimizer,
)
from forgather.ml.optim.rounding_utils import fp32_to_bf16_stochastic_round
from forgather.ml.optim.sequential_lr_factory import sequential_lr_factory
from forgather.ml.optim.sgd import SGD
from forgather.ml.optim.subspace_proj import (
    OnlinePCAProjector,
    RandProjector,
    SubspaceProjector,
)
from forgather.ml.optim.wsd_scheduler import WSDScheduler

# ---------------------------------------------------------------------------
# Helper: simple multi-layer model for optimizer grouping tests
# ---------------------------------------------------------------------------


class TwoLayerModel(nn.Module):
    """A simple model with distinct weight and bias parameters for grouping tests."""

    def __init__(self, in_features=8, hidden_features=16, out_features=4):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


# ===========================================================================
# Tests for opt_utils.py
# ===========================================================================


class TestBuildParameterGroups:
    """Tests for build_parameter_groups."""

    def _make_model(self):
        return TwoLayerModel()

    def test_basic_weight_bias_grouping(self):
        """Parameters are correctly split into weight and bias groups."""
        model = self._make_model()
        optimizer_groups = {
            "weight_group": {
                "regex": r"weight",
                "config": {"lr": 1e-3, "weight_decay": 0.01},
            },
            "bias_group": {
                "regex": r"bias",
                "config": {"lr": 1e-4, "weight_decay": 0.0},
            },
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        assert len(param_groups) == 2

        # Weight group
        weight_group = param_groups[0]
        assert weight_group["lr"] == 1e-3
        assert weight_group["weight_decay"] == 0.01
        weight_params: list[Any] = weight_group["params"]  # type: ignore[assignment]
        weight_names = [name for name, _ in weight_params]
        assert all("weight" in n for n in weight_names)
        assert len(weight_names) == 2  # linear1.weight, linear2.weight

        # Bias group
        bias_group = param_groups[1]
        assert bias_group["lr"] == 1e-4
        assert bias_group["weight_decay"] == 0.0
        bias_params: list[Any] = bias_group["params"]  # type: ignore[assignment]
        bias_names = [name for name, _ in bias_params]
        assert all("bias" in n for n in bias_names)
        assert len(bias_names) == 2  # linear1.bias, linear2.bias

    def test_first_match_wins(self):
        """A parameter matches the first regex that applies."""
        model = self._make_model()
        # "linear1.weight" contains both "linear1" and "weight". With "linear1"
        # defined first, it should capture both weight and bias of linear1,
        # leaving linear2's parameters for the fall-through default group.
        optimizer_groups = {
            "first_layer": {"regex": r"linear1", "config": {"lr": 1e-2}},
            "rest": {"regex": r"linear2", "config": {"lr": 1e-3}},
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        first_params: list[Any] = param_groups[0]["params"]  # type: ignore[assignment]
        rest_params: list[Any] = param_groups[1]["params"]  # type: ignore[assignment]
        first_names = [n for n, _ in first_params]
        rest_names = [n for n, _ in rest_params]

        assert len(first_names) == 2  # linear1.weight, linear1.bias
        assert all("linear1" in n for n in first_names)
        assert len(rest_names) == 2  # linear2.weight, linear2.bias
        assert all("linear2" in n for n in rest_names)

    def test_unmatched_parameters_go_to_default(self):
        """Parameters that match no user regex fall through to the default group."""
        model = self._make_model()
        # Only match linear1 parameters — linear2.* should land in default.
        optimizer_groups = {
            "group_a": {"regex": r"linear1", "config": {"lr": 1e-3}},
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        assert len(param_groups) == 2  # group_a + default fall-through
        # First group is group_a with user-specified lr
        assert param_groups[0]["lr"] == 1e-3
        group_a_names = [n for n, _ in param_groups[0]["params"]]
        assert all("linear1" in n for n in group_a_names)
        assert len(group_a_names) == 2

        # Second group is the default fall-through, no lr override
        assert "lr" not in param_groups[1]
        default_names = [n for n, _ in param_groups[1]["params"]]
        assert all("linear2" in n for n in default_names)
        assert len(default_names) == 2

    def test_empty_optimizer_groups_puts_all_in_default(self):
        """Passing an empty mapping sends every parameter to the default group."""
        model = self._make_model()
        param_groups = build_parameter_groups(model.named_parameters(), {})
        # Only the default group is produced
        assert len(param_groups) == 1
        names = [n for n, _ in param_groups[0]["params"]]
        assert set(names) == {
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
        }

    def test_empty_named_parameters_produces_no_groups(self):
        """An empty named_parameters iterator yields no groups (empty ones filtered)."""
        optimizer_groups = {"w": {"regex": r"weight", "config": {"lr": 0.1}}}
        param_groups = build_parameter_groups(iter([]), optimizer_groups)
        assert param_groups == []

    def test_reserved_default_name_rejected(self):
        """Using the reserved fall-through group name raises a clear error."""
        from forgather.ml.optim.opt_utils import _DEFAULT_GROUP

        model = self._make_model()
        optimizer_groups = {
            _DEFAULT_GROUP: {"regex": r".*", "config": {"lr": 0.1}},
        }
        with pytest.raises(ValueError, match="reserved"):
            build_parameter_groups(model.named_parameters(), optimizer_groups)

    def test_config_omitted_or_none(self):
        """A group spec with no 'config' key (or config=None) is valid."""
        model = self._make_model()
        optimizer_groups = {
            "weights_no_config": {"regex": r"weight"},
            "biases_null_config": {"regex": r"bias", "config": None},
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        assert len(param_groups) == 2
        for pg in param_groups:
            assert list(pg.keys()) == ["params"]

    def test_none_spec_removes_group(self):
        """A spec value of None removes the group — used to clear inherited entries."""
        model = self._make_model()
        # Parent would define 'no_decay'; child sets it to null to cancel it.
        optimizer_groups = {
            "no_decay": None,
            "weights": {"regex": r"weight", "config": {"lr": 1e-3}},
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        # Only 'weights' + default fall-through; 'no_decay' is gone.
        assert len(param_groups) == 2
        # First group is weights with lr override
        assert param_groups[0]["lr"] == 1e-3
        weight_names = [n for n, _ in param_groups[0]["params"]]
        assert all("weight" in n for n in weight_names)

        # Second is the default fall-through (biases), no overrides
        assert "lr" not in param_groups[1]
        default_names = [n for n, _ in param_groups[1]["params"]]
        assert all("bias" in n for n in default_names)

    def test_none_spec_can_clear_all_groups(self):
        """Setting every entry to None is equivalent to no groups at all."""
        model = self._make_model()
        optimizer_groups = {"group1": None, "group2": None}

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        # Nothing left but the default fall-through containing everything.
        assert len(param_groups) == 1
        names = {n for n, _ in param_groups[0]["params"]}
        assert len(names) == 4

    def test_missing_regex_raises(self):
        """A spec without a 'regex' key is rejected with a clear error."""
        model = self._make_model()
        with pytest.raises(ValueError, match="regex"):
            build_parameter_groups(
                model.named_parameters(),
                {"bad": {"config": {"lr": 0.1}}},  # type: ignore[typeddict-item]
            )

    def test_non_dict_config_rejected(self):
        """A spec with a non-dict config value is rejected."""
        model = self._make_model()
        with pytest.raises(ValueError, match="config"):
            build_parameter_groups(
                model.named_parameters(),
                {"bad": {"regex": r".*", "config": "not a dict"}},  # type: ignore[dict-item]
            )

    def test_empty_groups_are_filtered_out(self):
        """Groups that end up with no parameters are removed from the output."""
        model = self._make_model()
        # 'nomatch' won't be picked by any parameter name.
        optimizer_groups = {
            "nomatch": {
                "regex": r"this_substring_is_not_in_any_param_name",
                "config": {"lr": 99.0},
            },
            "all_weights": {"regex": r"weight", "config": {"lr": 1e-3}},
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        # nomatch dropped; all_weights + default fall-through for biases
        assert len(param_groups) == 2
        lrs = {pg.get("lr") for pg in param_groups}
        assert 1e-3 in lrs
        assert 99.0 not in lrs

    def test_debug_flag_emits_log_messages(self):
        """Setting debug=True should not raise and should emit log entries."""
        # The forgather.ml.optim.opt_utils logger has propagate=False and its
        # own stderr handler (set up by prefix_logger_rank), so caplog cannot
        # see its output. Attach a temporary handler and explicitly set the
        # level to INFO so debug messages reach it.
        import logging

        opt_logger = logging.getLogger("forgather.ml.optim.opt_utils")
        captured: list[str] = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        previous_level = opt_logger.level
        handler = _ListHandler(level=logging.INFO)
        opt_logger.setLevel(logging.INFO)
        opt_logger.addHandler(handler)
        try:
            model = self._make_model()
            optimizer_groups = {"all": {"regex": r".*", "config": {"lr": 0.1}}}
            build_parameter_groups(
                model.named_parameters(), optimizer_groups, debug=True
            )
        finally:
            opt_logger.removeHandler(handler)
            opt_logger.setLevel(previous_level)

        assert any("param group:" in m for m in captured)

    def test_group_config_hyperparams_forwarded(self):
        """All hyperparameters from the group config appear in the returned dicts."""
        model = self._make_model()
        optimizer_groups = {
            "all": {
                "regex": r".*",
                "config": {
                    "lr": 0.1,
                    "weight_decay": 0.05,
                    "betas": (0.9, 0.98),
                    "eps": 1e-8,
                },
            },
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        pg = param_groups[0]
        assert pg["lr"] == 0.1
        assert pg["weight_decay"] == 0.05
        assert pg["betas"] == (0.9, 0.98)
        assert pg["eps"] == 1e-8

    def test_no_decay_convention(self):
        """The typical 'no weight-decay for norms/biases/embeddings' idiom works."""

        class LMLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.norm = nn.LayerNorm(8)
                self.linear = nn.Linear(8, 8)
                self.lm_head = nn.Linear(8, 16, bias=False)

            def forward(self, x):
                return self.lm_head(self.linear(self.norm(self.embed(x))))

        model = LMLike()
        optimizer_groups = {
            "no_decay": {
                "regex": r"norm|bias|embed|lm_head",
                "config": {"weight_decay": 0.0},
            },
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        # no_decay group + default fall-through
        assert len(param_groups) == 2
        no_decay_group, default_group = param_groups

        assert no_decay_group["weight_decay"] == 0.0
        no_decay_names = {n for n, _ in no_decay_group["params"]}
        assert "embed.weight" in no_decay_names
        assert "norm.weight" in no_decay_names
        assert "norm.bias" in no_decay_names
        assert "linear.bias" in no_decay_names
        assert "lm_head.weight" in no_decay_names

        # Remaining parameters (just linear.weight here) live in the default group
        # with no weight-decay override.
        assert "weight_decay" not in default_group
        default_names = {n for n, _ in default_group["params"]}
        assert default_names == {"linear.weight"}

    def test_insertion_order_preserved(self):
        """Resulting param_groups preserve the insertion order of the input mapping."""
        model = self._make_model()
        optimizer_groups = {
            "biases": {"regex": r"bias", "config": {"lr": 1e-5}},
            "weights": {"regex": r"weight", "config": {"lr": 1e-3}},
        }

        param_groups = build_parameter_groups(
            model.named_parameters(), optimizer_groups
        )

        # biases appears before weights in the input, so must come first.
        assert param_groups[0]["lr"] == 1e-5
        assert param_groups[1]["lr"] == 1e-3


class TestMakeGroupedOptimizer:
    """Tests for make_grouped_optimizer."""

    def test_creates_optimizer_with_groups(self):
        """make_grouped_optimizer returns a working optimizer instance."""
        model = TwoLayerModel()
        optimizer_groups = {
            "weights": {
                "regex": r"weight",
                "config": {"lr": 1e-3, "weight_decay": 0.01},
            },
            "biases": {
                "regex": r"bias",
                "config": {"lr": 1e-4, "weight_decay": 0.0},
            },
        }

        optimizer = make_grouped_optimizer(
            model.named_parameters(),
            optimizer_groups=optimizer_groups,
            optimizer_factory=torch.optim.SGD,
        )

        assert isinstance(optimizer, torch.optim.SGD)
        assert len(optimizer.param_groups) == 2

    def test_optimizer_factory_is_called_with_param_groups(self):
        """The optimizer factory receives the computed param groups."""
        model = TwoLayerModel()
        optimizer_groups = {
            "all": {"regex": r".*", "config": {"lr": 1e-3}},
        }

        # partial binding extra kwargs into the factory is the standard pattern.
        factory = partial(torch.optim.SGD, momentum=0.9)

        optimizer = make_grouped_optimizer(
            model.named_parameters(),
            optimizer_groups=optimizer_groups,
            optimizer_factory=factory,
        )

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.9
        assert optimizer.param_groups[0]["lr"] == 1e-3


class TestBuildOptimizerBuckets:
    """Tests for build_optimizer_buckets."""

    def test_single_default_factory_one_bucket(self):
        """All groups resolve onto the default factory → 1 bucket, multiple param groups."""
        model = TwoLayerModel()
        default_factory = partial(torch.optim.SGD, lr=0.0)
        optimizer_groups = {
            "weight_group": {
                "regex": r"weight",
                "config": {"lr": 1e-3},
            },
            "bias_group": {
                "regex": r"bias",
                "config": {"lr": 1e-4},
            },
        }

        buckets = build_optimizer_buckets(
            model.named_parameters(),
            optimizer_groups,
            default_factory=default_factory,
        )

        assert len(buckets) == 1
        factory, param_groups = buckets[0]
        assert factory is default_factory
        assert len(param_groups) == 2
        assert param_groups[0]["lr"] == 1e-3
        assert param_groups[1]["lr"] == 1e-4

    def test_distinct_per_group_factories(self):
        """Two groups with different `factory` entries → 2 buckets in declared order."""
        model = TwoLayerModel()
        default_factory = partial(torch.optim.SGD, lr=0.0)
        weight_factory = partial(torch.optim.SGD, lr=0.5)
        bias_factory = partial(torch.optim.Adam, lr=0.25)
        optimizer_groups = {
            "weight_group": {"regex": r"weight", "factory": weight_factory},
            "bias_group": {"regex": r"bias", "factory": bias_factory},
        }

        buckets = build_optimizer_buckets(
            model.named_parameters(),
            optimizer_groups,
            default_factory=default_factory,
        )

        assert [factory for factory, _ in buckets] == [weight_factory, bias_factory]
        for _factory, param_groups in buckets:
            assert len(param_groups) == 1

    def test_mixed_explicit_and_default_factory(self):
        """A group without `factory` falls back to default; bucket order is declaration order."""
        model = TwoLayerModel()
        default_factory = partial(torch.optim.SGD, lr=0.0)
        weight_factory = partial(torch.optim.Adam, lr=0.5)
        optimizer_groups = {
            "weight_group": {"regex": r"weight", "factory": weight_factory},
            "bias_group": {"regex": r"bias", "config": {"lr": 1e-4}},
        }

        buckets = build_optimizer_buckets(
            model.named_parameters(),
            optimizer_groups,
            default_factory=default_factory,
        )

        assert [factory for factory, _ in buckets] == [weight_factory, default_factory]
        # Default-factory bucket has bias_group with its config override.
        assert buckets[1][1][0]["lr"] == 1e-4

    def test_unmatched_params_join_default_bucket(self):
        """Implicit fall-through groups merge into the default factory bucket."""
        model = TwoLayerModel()
        default_factory = partial(torch.optim.SGD, lr=0.0)
        weight_factory = partial(torch.optim.Adam, lr=0.5)
        optimizer_groups = {
            "weight_group": {"regex": r"weight", "factory": weight_factory},
        }

        buckets = build_optimizer_buckets(
            model.named_parameters(),
            optimizer_groups,
            default_factory=default_factory,
        )

        assert len(buckets) == 2
        # Default bucket only contains the implicit fall-through group (biases).
        default_bucket = buckets[1]
        assert default_bucket[0] is default_factory
        assert len(default_bucket[1]) == 1
        names = [n for n, _ in default_bucket[1][0]["params"]]
        assert all("bias" in n for n in names)

    def test_same_factory_object_merges_buckets(self):
        """Same factory reused across two groups → 1 bucket with both param groups."""
        model = TwoLayerModel()
        default_factory = partial(torch.optim.SGD, lr=0.0)
        shared_factory = partial(torch.optim.Adam, lr=0.5)
        optimizer_groups = {
            "weight_group": {
                "regex": r"weight",
                "config": {"lr": 1e-3},
                "factory": shared_factory,
            },
            "bias_group": {
                "regex": r"bias",
                "config": {"lr": 1e-4},
                "factory": shared_factory,
            },
        }

        buckets = build_optimizer_buckets(
            model.named_parameters(),
            optimizer_groups,
            default_factory=default_factory,
        )

        assert len(buckets) == 1
        factory, param_groups = buckets[0]
        assert factory is shared_factory
        assert len(param_groups) == 2

    def test_empty_groups_filtered(self):
        """Groups whose regex matches nothing are not emitted as buckets."""
        model = TwoLayerModel()
        default_factory = partial(torch.optim.SGD, lr=0.0)
        explicit_factory = partial(torch.optim.Adam, lr=0.5)
        optimizer_groups = {
            "matches_nothing": {
                "regex": r"this_will_not_match_anything",
                "factory": explicit_factory,
            },
        }

        buckets = build_optimizer_buckets(
            model.named_parameters(),
            optimizer_groups,
            default_factory=default_factory,
        )

        # Only the implicit default-factory bucket survives.
        assert len(buckets) == 1
        assert buckets[0][0] is default_factory

    def test_bad_factory_raises(self):
        """A non-callable `factory` value is rejected at validation time."""
        model = TwoLayerModel()
        default_factory = partial(torch.optim.SGD, lr=0.0)
        optimizer_groups = {
            "broken": {"regex": r"weight", "factory": "not a callable"},
        }

        with pytest.raises(ValueError, match="factory.*callable"):
            build_optimizer_buckets(
                model.named_parameters(),
                optimizer_groups,
                default_factory=default_factory,
            )

    def test_buckets_drive_multiopt_state_dict_round_trip(self):
        """End-to-end: build buckets, instantiate Multiopt, round-trip state_dict."""
        model = TwoLayerModel()
        default_factory = partial(torch.optim.SGD, lr=0.1)
        adam_factory = partial(torch.optim.Adam, lr=0.01)
        optimizer_groups = {
            "weights": {"regex": r"weight", "factory": adam_factory},
            "biases": {"regex": r"bias"},  # default factory
        }

        buckets = build_optimizer_buckets(
            model.named_parameters(),
            optimizer_groups,
            default_factory=default_factory,
        )
        opt = Multiopt([factory(pg) for factory, pg in buckets])

        # Take a step so the wrapped optimizers accumulate state.
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # Round-trip the state dict into a freshly-built clone.
        clone_buckets = build_optimizer_buckets(
            model.named_parameters(),
            optimizer_groups,
            default_factory=default_factory,
        )
        clone = Multiopt([factory(pg) for factory, pg in clone_buckets])
        clone.load_state_dict(opt.state_dict())

        assert len(clone.optimizers) == len(opt.optimizers)
        for src, dst in zip(opt.optimizers, clone.optimizers):
            assert type(src) is type(dst)


# ===========================================================================
# Tests for rounding_utils.py
# ===========================================================================


class TestFp32ToBf16StochasticRound:
    """Tests for fp32_to_bf16_stochastic_round."""

    def test_output_dtype_is_bfloat16(self):
        """Returned tensor must be bfloat16."""
        x = torch.randn(64, dtype=torch.float32)
        result = fp32_to_bf16_stochastic_round(x)
        assert result.dtype == torch.bfloat16

    def test_output_shape_matches_input(self):
        """Output tensor shape must match the input."""
        for shape in [(10,), (4, 8), (2, 3, 5)]:
            x = torch.randn(shape, dtype=torch.float32)
            result = fp32_to_bf16_stochastic_round(x)
            assert result.shape == x.shape, f"Shape mismatch for input shape {shape}"

    def test_close_to_input_values(self):
        """Stochastic rounding should produce values within bf16 precision of input."""
        torch.manual_seed(0)
        x = torch.randn(1000, dtype=torch.float32)
        result = fp32_to_bf16_stochastic_round(x)

        # Compare in float32 space
        diff = (x - result.float()).abs()
        # bf16 has 7-8 bit mantissa, so max rounding error per value is bounded
        # by the ULP (unit in the last place) of bfloat16, roughly |x| * 2^-7
        # For standard normal values most are < 4, so max error ~ 0.03
        assert (
            diff.max().item() < 0.1
        ), f"Maximum rounding error {diff.max().item()} exceeds tolerance"

    def test_determinism_with_seeded_generator(self):
        """Using the same generator seed should produce identical results."""
        x = torch.randn(256, dtype=torch.float32)

        gen1 = torch.Generator()
        gen1.manual_seed(42)
        result1 = fp32_to_bf16_stochastic_round(x, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(42)
        result2 = fp32_to_bf16_stochastic_round(x, generator=gen2)

        assert torch.equal(
            result1, result2
        ), "Results should be identical with same seed"

    def test_different_seeds_differ(self):
        """Different generator seeds should (very likely) produce different results."""
        x = torch.randn(1024, dtype=torch.float32)

        gen1 = torch.Generator()
        gen1.manual_seed(0)
        result1 = fp32_to_bf16_stochastic_round(x, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(999)
        result2 = fp32_to_bf16_stochastic_round(x, generator=gen2)

        # Not all values will differ (many fp32 values round exactly to bf16)
        # but with 1024 elements some should differ.
        assert not torch.equal(
            result1, result2
        ), "Results with different seeds should differ for a large enough tensor"

    def test_zero_input(self):
        """Zero should round to zero."""
        x = torch.zeros(16, dtype=torch.float32)
        result = fp32_to_bf16_stochastic_round(x)
        assert torch.equal(result, torch.zeros(16, dtype=torch.bfloat16))

    def test_exact_bf16_values_unchanged(self):
        """Values that are exactly representable in bf16 should remain unchanged."""
        # Powers of two and small integers are exact in bf16
        x = torch.tensor([1.0, 2.0, -4.0, 0.5, 0.25], dtype=torch.float32)
        result = fp32_to_bf16_stochastic_round(x)
        expected = x.bfloat16()
        assert torch.equal(
            result, expected
        ), f"Exact bf16 values should be unchanged: got {result}, expected {expected}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensor(self):
        """Stochastic rounding should work on CUDA tensors."""
        x = torch.randn(128, dtype=torch.float32, device="cuda")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(42)
        result = fp32_to_bf16_stochastic_round(x, generator=gen)
        assert result.dtype == torch.bfloat16
        assert result.device.type == "cuda"
        assert result.shape == x.shape

    def test_unbiased_rounding_mean(self):
        """Stochastic rounding should be unbiased: mean of many rounds should converge."""
        torch.manual_seed(123)
        # Use a value that is NOT exactly representable in bf16
        # 1.001953125 is exact in bf16; pick something between two bf16 values.
        # bf16 near 1.0 has step size 2^-7 = 0.0078125
        # 1.0 + 0.003 is between bf16 values 1.0 and 1.0078125
        val = 1.003
        x = torch.full((1,), val, dtype=torch.float32)

        results = []
        for _ in range(5000):
            r = fp32_to_bf16_stochastic_round(x)
            results.append(r.float().item())

        mean_rounded = sum(results) / len(results)
        # Mean should be close to the original value (unbiased)
        assert (
            abs(mean_rounded - val) < 0.005
        ), f"Mean of stochastic rounding {mean_rounded} too far from true value {val}"

    def test_rand_bits_path_matches_generator_path(self):
        """Pre-generated rand_bits should produce identical results to generator path."""
        x = torch.randn(256, dtype=torch.float32)

        # Generator path
        gen = torch.Generator()
        gen.manual_seed(42)
        result_gen = fp32_to_bf16_stochastic_round(x, generator=gen)

        # rand_bits path: generate identical noise manually
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        rand_bits = torch.randint(
            0, 1 << 16, x.shape, dtype=torch.int32, generator=gen2
        )
        result_bits = fp32_to_bf16_stochastic_round(x, rand_bits=rand_bits)

        assert torch.equal(
            result_gen, result_bits
        ), "rand_bits path must produce identical results to generator path"

    def test_rand_bits_ignores_generator(self):
        """When rand_bits is provided, generator should be ignored."""
        x = torch.randn(64, dtype=torch.float32)

        gen = torch.Generator()
        gen.manual_seed(42)
        rand_bits = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, generator=gen)

        # Pass a differently-seeded generator alongside rand_bits
        gen_different = torch.Generator()
        gen_different.manual_seed(999)
        result = fp32_to_bf16_stochastic_round(
            x, generator=gen_different, rand_bits=rand_bits
        )

        # Should match the rand_bits, not the generator
        result_expected = fp32_to_bf16_stochastic_round(x, rand_bits=rand_bits)
        assert torch.equal(result, result_expected)


class TestStochasticRoundingCompile:
    """Tests that stochastic rounding is compatible with torch.compile."""

    def test_generator_causes_graph_break(self):
        """Verify that passing a generator to SR causes fullgraph=True to fail.

        This is the regression test: if this ever stops failing, it means
        PyTorch added native generator support in dynamo, and the rand_bits
        workaround can be removed.
        """

        def fn(x, gen):
            return fp32_to_bf16_stochastic_round(x, generator=gen)

        x = torch.randn(64, dtype=torch.float32)
        gen = torch.Generator()
        gen.manual_seed(42)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        with pytest.raises(torch._dynamo.exc.Unsupported):
            compiled(x, gen)

    def test_rand_bits_compiles_fullgraph(self):
        """SR with pre-generated rand_bits compiles with fullgraph=True."""

        def fn(x, rand_bits):
            return fp32_to_bf16_stochastic_round(x, rand_bits=rand_bits)

        x = torch.randn(64, dtype=torch.float32)
        rand_bits = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result = compiled(x, rand_bits)
        assert result.dtype == torch.bfloat16
        assert result.shape == x.shape

    def test_compiled_sr_matches_eager(self):
        """Compiled SR produces identical output to eager mode."""
        x = torch.randn(128, dtype=torch.float32)
        rand_bits = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)

        eager = fp32_to_bf16_stochastic_round(x, rand_bits=rand_bits)

        def fn(x, rand_bits):
            return fp32_to_bf16_stochastic_round(x, rand_bits=rand_bits)

        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        compiled_result = compiled(x, rand_bits)

        assert torch.equal(
            eager, compiled_result
        ), "Compiled SR must be bitwise identical to eager"

    def test_adafactor_compiled_with_bf16_sr(self):
        """Adafactor with torch_compile=True and bf16_stochastic_round=True runs without error."""
        from forgather.ml.optim.adafactor import Adafactor

        model = nn.Linear(16, 8)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        model.bias.data = model.bias.data.to(torch.bfloat16)

        opt = Adafactor(
            model.parameters(),
            lr=1e-3,
            torch_compile=True,
            bf16_stochastic_round=True,
        )

        # Run a few steps to exercise compile caching
        for _ in range(3):
            x = torch.randn(2, 16, dtype=torch.bfloat16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_adamw_compiled_with_bf16_sr(self):
        """AdamW with torch_compile=True and bf16_stochastic_round=True runs without error."""
        model = nn.Linear(16, 8)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        model.bias.data = model.bias.data.to(torch.bfloat16)

        opt = AdamW(
            model.parameters(),
            lr=1e-3,
            torch_compile=True,
            bf16_stochastic_round=True,
        )

        for _ in range(3):
            x = torch.randn(2, 16, dtype=torch.bfloat16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_adafactor_compile_sr_reduces_loss(self):
        """Adafactor with compile + SR should actually train (loss decreases)."""
        from forgather.ml.optim.adafactor import Adafactor

        torch.manual_seed(0)
        model = nn.Linear(16, 4)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        model.bias.data = model.bias.data.to(torch.bfloat16)

        opt = Adafactor(
            model.parameters(),
            lr=1e-2,
            torch_compile=True,
            bf16_stochastic_round=True,
        )

        x = torch.randn(8, 16, dtype=torch.bfloat16)
        y = torch.randn(8, 4, dtype=torch.bfloat16)

        initial_loss = nn.functional.mse_loss(model(x), y).item()
        for _ in range(50):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert (
            final_loss < initial_loss
        ), f"Loss should decrease: {initial_loss} -> {final_loss}"

    def test_adafactor_compile_sr_1d_param(self):
        """Adafactor compile + SR works for 1D parameters (bias, norms)."""
        from forgather.ml.optim.adafactor import Adafactor

        model = nn.Linear(16, 8, bias=True)
        model.weight.data = model.weight.data.to(torch.bfloat16)
        model.bias.data = model.bias.data.to(torch.bfloat16)

        opt = Adafactor(
            model.parameters(),
            lr=1e-3,
            torch_compile=True,
            bf16_stochastic_round=True,
        )

        x = torch.randn(2, 16, dtype=torch.bfloat16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # Verify both 2D (weight) and 1D (bias) were updated
        assert model.weight.grad is not None
        assert model.bias.grad is not None


# ===========================================================================
# Tests for infinite_lr_scheduler.py
# ===========================================================================


class TestInfiniteLRScheduler:
    """Tests for InfiniteLRScheduler with 4 phases: warmup, cooldown, constant, annealing."""

    def _make_optimizer(self, lr=1.0):
        """Create a simple optimizer with a single parameter group."""
        param = nn.Parameter(torch.randn(4))
        return torch.optim.SGD([param], lr=lr)

    # --- warmup phase ---

    def test_warmup_starts_at_zero(self):
        """At step 0, LR should be 0 during warmup phase."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt, warmup_steps=10, cooldown_steps=0, constant_lr=0.5
        )

        # After __init__, last_epoch=0 and get_lr has been called once
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.0), f"Expected LR=0 at warmup start, got {lr}"

    def test_warmup_linear_ramp(self):
        """LR should increase linearly during warmup."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 10
        sched = InfiniteLRScheduler(
            opt, warmup_steps=warmup_steps, cooldown_steps=0, constant_lr=0.5
        )

        lrs = [sched.get_last_lr()[0]]
        for _ in range(warmup_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        # LR at step i = base_lr * i / warmup_steps
        for i in range(warmup_steps + 1):
            expected = 1.0 * i / warmup_steps
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Step {i}: expected LR={expected}, got {lrs[i]}"

    def test_warmup_end_matches_base_lr(self):
        """At the last warmup step, LR should equal base_lr."""
        opt = self._make_optimizer(lr=0.5)
        warmup_steps = 20
        sched = InfiniteLRScheduler(
            opt, warmup_steps=warmup_steps, cooldown_steps=0, constant_lr=0.1
        )

        for _ in range(warmup_steps):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(
            0.5, abs=1e-7
        ), f"LR at end of warmup should equal base_lr=0.5, got {lr}"

    # --- cooldown phase ---

    def test_cooldown_starts_at_base_lr(self):
        """At the beginning of cooldown, LR should be close to base_lr."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 5
        cooldown_steps = 20
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=0.1,
        )

        # Advance through warmup
        for _ in range(warmup_steps):
            sched.step()

        lr = sched.get_last_lr()[0]
        # At step=warmup_steps, cosine term: cos(0) = 1, so lr = constant_lr + (base_lr - constant_lr)/2 * 2 = base_lr
        assert lr == pytest.approx(
            1.0, abs=1e-6
        ), f"LR at start of cooldown should be base_lr=1.0, got {lr}"

    def test_cooldown_ends_at_constant_lr(self):
        """At the end of cooldown, LR should be close to constant_lr."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 5
        cooldown_steps = 20
        constant_lr = 0.2
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=constant_lr,
        )

        # Advance through warmup + cooldown
        for _ in range(warmup_steps + cooldown_steps):
            sched.step()

        lr = sched.get_last_lr()[0]
        # At the last cooldown step (step = warmup + cooldown - 1):
        # cosine arg = pi * (cooldown_steps - 1) / cooldown_steps ~ pi
        # So cos ~ -1, LR ~ constant_lr + 0 ~ constant_lr (approximately)
        assert lr == pytest.approx(
            constant_lr, abs=0.05
        ), f"LR at end of cooldown should approach constant_lr={constant_lr}, got {lr}"

    def test_cooldown_cosine_shape(self):
        """Cooldown LR follows cosine decay from base_lr to constant_lr."""
        opt = self._make_optimizer(lr=2.0)
        warmup_steps = 0
        cooldown_steps = 100
        constant_lr = 0.5
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=constant_lr,
        )

        lrs = [sched.get_last_lr()[0]]
        for _ in range(cooldown_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        # Verify a midpoint value: at step=cooldown_steps/2, cosine=cos(pi/2)=0
        mid = cooldown_steps // 2
        expected_mid = constant_lr + (2.0 - constant_lr) / 2.0 * (1.0 + 0.0)
        assert lrs[mid] == pytest.approx(
            expected_mid, abs=0.05
        ), f"Midpoint LR should be ~{expected_mid}, got {lrs[mid]}"

    # --- constant phase ---

    def test_constant_phase_with_cooldown(self):
        """After cooldown, LR should remain at constant_lr."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 5
        cooldown_steps = 10
        constant_lr = 0.3
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=constant_lr,
        )

        # Advance past warmup + cooldown
        for _ in range(warmup_steps + cooldown_steps):
            sched.step()

        # Constant phase: LR should stay at constant_lr
        for step_i in range(20):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(
                constant_lr, abs=1e-7
            ), f"Constant phase step {step_i}: expected {constant_lr}, got {lr}"

    def test_constant_phase_no_cooldown(self):
        """With cooldown_steps=0, constant phase should use base_lr instead."""
        opt = self._make_optimizer(lr=0.7)
        warmup_steps = 5
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=0,
            constant_lr=0.1,  # Should be ignored when cooldown_steps=0
        )

        # Advance past warmup
        for _ in range(warmup_steps):
            sched.step()

        # Constant phase with no cooldown: LR = base_lr
        for _ in range(20):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(
                0.7, abs=1e-7
            ), f"Expected base_lr=0.7 in constant phase without cooldown, got {lr}"

    # --- annealing phase ---

    def test_annealing_exponential_decay(self):
        """After checkpoint_step, LR should decay per the paper's formula (Eq. 1).

        eta(n) = eta_const * (eta_min / eta_const) ^ ((n - N_d) / (t_a + N_d))
        """
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 5
        cooldown_steps = 5
        constant_lr = 0.5
        min_lr = 1e-6
        tau = 100.0
        checkpoint_step = 20
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=constant_lr,
            min_lr=min_lr,
            tau=tau,
            checkpoint_step=checkpoint_step,
        )

        # Advance to just past checkpoint_step
        for _ in range(checkpoint_step):
            sched.step()

        # Now in annealing phase
        prev_lr = sched.get_last_lr()[0]
        for i in range(1, 50):
            sched.step()
            lr = sched.get_last_lr()[0]
            # LR should be decreasing
            assert (
                lr < prev_lr
            ), f"Annealing step {i}: LR should decrease, prev={prev_lr}, curr={lr}"
            # Verify paper formula: eta_const * (eta_min / eta_const) ^ (t / (tau + checkpoint_step))
            t = i
            exponent = t / (tau + checkpoint_step)
            expected = constant_lr * (min_lr / constant_lr) ** exponent
            assert lr == pytest.approx(
                expected, abs=1e-10
            ), f"Annealing step {i}: expected {expected}, got {lr}"
            prev_lr = lr

    def test_annealing_reaches_min_lr(self):
        """LR should reach exactly min_lr at n - N_d = tau + checkpoint_step."""
        opt = self._make_optimizer(lr=1.0)
        constant_lr = 0.5
        min_lr = 1e-5
        tau = 10.0
        checkpoint_step = 5
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=constant_lr,
            min_lr=min_lr,
            tau=tau,
            checkpoint_step=checkpoint_step,
        )

        # Advance to exactly checkpoint_step + tau + checkpoint_step
        # At this point exponent = (tau + checkpoint_step) / (tau + checkpoint_step) = 1
        # so LR = constant_lr * (min_lr / constant_lr) ^ 1 = min_lr
        target_step = checkpoint_step + int(tau) + checkpoint_step
        for _ in range(target_step):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(
            min_lr, rel=1e-7
        ), f"At exponent=1, LR should be {min_lr}, got {lr}"

    # --- no warmup / no cooldown ---

    def test_no_warmup_no_cooldown(self):
        """With warmup=0 and cooldown=0, scheduler goes straight to constant (base_lr)."""
        opt = self._make_optimizer(lr=0.3)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.1,
        )

        # First LR should already be base_lr (no warmup, no cooldown, cooldown_steps==0 means constant=base_lr)
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.3, abs=1e-7)

        for _ in range(10):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(0.3, abs=1e-7)

    # --- state dict round-trip ---

    def test_state_dict_round_trip(self):
        """state_dict / load_state_dict preserves scheduler state across save/load."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=10,
            cooldown_steps=10,
            constant_lr=0.5,
            min_lr=1e-6,
            tau=50.0,
            checkpoint_step=30,
        )

        # Advance scheduler partway through
        for _ in range(15):
            sched.step()

        saved_state = sched.state_dict()
        lr_before = sched.get_last_lr()[0]

        # Create a new scheduler and load state
        opt2 = self._make_optimizer(lr=1.0)
        sched2 = InfiniteLRScheduler(
            opt2,
            warmup_steps=10,
            cooldown_steps=10,
            constant_lr=0.5,
            min_lr=1e-6,
            tau=50.0,
            checkpoint_step=30,
        )
        sched2.load_state_dict(saved_state)

        lr_after = sched2.get_last_lr()[0]
        assert lr_before == pytest.approx(
            lr_after, abs=1e-10
        ), f"LR mismatch after state_dict round-trip: {lr_before} vs {lr_after}"

        # Step both and verify they remain in sync
        for _ in range(20):
            sched.step()
            sched2.step()
            assert sched.get_last_lr()[0] == pytest.approx(
                sched2.get_last_lr()[0], abs=1e-10
            )

    # --- multi-parameter-group ---

    def test_multiple_param_groups(self):
        """Scheduler should handle optimizers with multiple parameter groups."""
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(4))
        opt = torch.optim.SGD(
            [
                {"params": [p1], "lr": 1.0},
                {"params": [p2], "lr": 0.5},
            ]
        )

        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=10,
            cooldown_steps=0,
            constant_lr=0.1,
        )

        # At step 0 both LRs should be 0 (warmup start)
        lrs = sched.get_last_lr()
        assert lrs[0] == pytest.approx(0.0)
        assert lrs[1] == pytest.approx(0.0)

        # Advance to step 5
        for _ in range(5):
            sched.step()

        lrs = sched.get_last_lr()
        assert lrs[0] == pytest.approx(1.0 * 5 / 10, abs=1e-7)
        assert lrs[1] == pytest.approx(0.5 * 5 / 10, abs=1e-7)

    # --- validation ---

    def test_invalid_checkpoint_step(self):
        """checkpoint_step < warmup + cooldown should raise AssertionError."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            InfiniteLRScheduler(
                opt,
                warmup_steps=10,
                cooldown_steps=10,
                constant_lr=0.5,
                checkpoint_step=15,  # < 10 + 10 = 20
            )

    def test_negative_warmup_raises(self):
        """Negative warmup_steps should raise."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            InfiniteLRScheduler(opt, warmup_steps=-1, cooldown_steps=0, constant_lr=0.1)

    # --- start_annealing flag ---

    def test_start_annealing_triggers_from_constant_phase(self):
        """start_annealing=True should begin annealing at the loaded step."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            checkpoint_step=-1,
        )

        # Advance 50 steps into constant phase
        for _ in range(50):
            sched.step()

        saved_state = sched.state_dict()
        lr_before = sched.get_last_lr()[0]

        # Create new scheduler with start_annealing=True
        opt2 = self._make_optimizer(lr=1.0)
        sched2 = InfiniteLRScheduler(
            opt2,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            min_lr=1e-6,
            tau=100.0,
            checkpoint_step=-1,
            start_annealing=True,
        )
        sched2.load_state_dict(saved_state)

        assert sched2.checkpoint_step == 50
        # Step forward and verify LR is decaying
        sched2.step()
        lr_after = sched2.get_last_lr()[0]
        assert (
            lr_after < lr_before
        ), f"LR should decay after start_annealing, got {lr_after} >= {lr_before}"

    def test_start_annealing_resumes_existing_annealing(self):
        """start_annealing=True preserves checkpoint_step if already annealing."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            min_lr=1e-6,
            tau=100.0,
            checkpoint_step=20,
        )

        # Advance to step 30 (in annealing)
        for _ in range(30):
            sched.step()

        saved_state = sched.state_dict()
        lr_at_30 = sched.get_last_lr()[0]

        # Load with start_annealing=True
        opt2 = self._make_optimizer(lr=1.0)
        sched2 = InfiniteLRScheduler(
            opt2,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            min_lr=1e-6,
            tau=100.0,
            checkpoint_step=-1,
            start_annealing=True,
        )
        sched2.load_state_dict(saved_state)

        assert (
            sched2.checkpoint_step == 20
        ), f"Should keep loaded checkpoint_step=20, got {sched2.checkpoint_step}"
        assert sched2.get_last_lr()[0] == pytest.approx(lr_at_30, abs=1e-10)

    def test_start_annealing_false_restores_constructor_checkpoint_step(self):
        """start_annealing=False restores constructor checkpoint_step."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            min_lr=1e-6,
            tau=100.0,
            checkpoint_step=20,
        )

        # Advance to step 30 (annealing in progress)
        for _ in range(30):
            sched.step()

        saved_state = sched.state_dict()

        # Load with start_annealing=False and constructor checkpoint_step=-1
        opt2 = self._make_optimizer(lr=1.0)
        sched2 = InfiniteLRScheduler(
            opt2,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            checkpoint_step=-1,
            start_annealing=False,
        )
        sched2.load_state_dict(saved_state)

        assert (
            sched2.checkpoint_step == -1
        ), f"Should restore constructor checkpoint_step=-1, got {sched2.checkpoint_step}"
        # Should be in constant phase now (base_lr since cooldown_steps=0)
        sched2.step()
        lr = sched2.get_last_lr()[0]
        assert lr == pytest.approx(
            1.0, abs=1e-7
        ), f"Should be back in constant phase at base_lr=1.0, got {lr}"

    def test_start_annealing_not_in_state_dict(self):
        """Config-only params should not appear in state_dict."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            start_annealing=True,
            annealing_type="rsqrt",
            annealing_steps=100,
        )

        state = sched.state_dict()
        assert "start_annealing" not in state
        assert "annealing_type" not in state
        assert "annealing_steps" not in state

    # --- rsqrt annealing ---

    def test_rsqrt_annealing_decay(self):
        """rsqrt annealing should follow harmonic interpolation formula."""
        opt = self._make_optimizer(lr=1.0)
        constant_lr = 0.5
        min_lr = 1e-6
        annealing_steps = 100
        checkpoint_step = 10
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=constant_lr,
            min_lr=min_lr,
            checkpoint_step=checkpoint_step,
            annealing_type="rsqrt",
            annealing_steps=annealing_steps,
        )

        # Advance to checkpoint_step
        for _ in range(checkpoint_step):
            sched.step()

        # Now in annealing phase, verify formula at each step
        prev_lr = sched.get_last_lr()[0]
        for i in range(1, 50):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr < prev_lr, f"rsqrt annealing step {i}: LR should decrease"
            t = i / annealing_steps
            expected = 1.0 / (t / min_lr + (1.0 - t) / constant_lr)
            assert lr == pytest.approx(
                expected, abs=1e-10
            ), f"rsqrt annealing step {i}: expected {expected}, got {lr}"
            prev_lr = lr

    def test_rsqrt_annealing_reaches_min_lr(self):
        """rsqrt annealing should reach exactly min_lr at annealing_steps."""
        opt = self._make_optimizer(lr=1.0)
        constant_lr = 0.5
        min_lr = 1e-5
        annealing_steps = 50
        checkpoint_step = 5
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=constant_lr,
            min_lr=min_lr,
            checkpoint_step=checkpoint_step,
            annealing_type="rsqrt",
            annealing_steps=annealing_steps,
        )

        # Advance to exactly checkpoint_step + annealing_steps
        for _ in range(checkpoint_step + annealing_steps):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(
            min_lr, rel=1e-7
        ), f"At annealing_steps, LR should be {min_lr}, got {lr}"

    def test_rsqrt_annealing_starts_at_constant_lr(self):
        """At checkpoint_step, rsqrt annealing LR should be constant_lr."""
        opt = self._make_optimizer(lr=1.0)
        constant_lr = 0.3
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=constant_lr,
            min_lr=1e-6,
            checkpoint_step=10,
            annealing_type="rsqrt",
            annealing_steps=100,
        )

        # Advance to checkpoint_step
        for _ in range(10):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(
            constant_lr, abs=1e-10
        ), f"At checkpoint_step, LR should be constant_lr={constant_lr}, got {lr}"

    def test_rsqrt_clamps_beyond_annealing_steps(self):
        """Past annealing_steps, rsqrt LR should stay at min_lr."""
        opt = self._make_optimizer(lr=1.0)
        min_lr = 1e-5
        annealing_steps = 20
        checkpoint_step = 5
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            min_lr=min_lr,
            checkpoint_step=checkpoint_step,
            annealing_type="rsqrt",
            annealing_steps=annealing_steps,
        )

        # Advance past annealing_steps
        for _ in range(checkpoint_step + annealing_steps + 10):
            sched.step()

        for _ in range(5):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(
                min_lr, rel=1e-7
            ), f"Past annealing_steps, LR should be {min_lr}, got {lr}"

    # --- config-only keys and backward compatibility ---

    def test_load_old_checkpoint_without_new_keys(self):
        """Old checkpoints (without new keys) should load correctly."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            checkpoint_step=-1,
        )
        for _ in range(20):
            sched.step()

        # Simulate an old checkpoint (no new keys)
        old_state = sched.state_dict()
        assert "start_annealing" not in old_state
        assert "annealing_type" not in old_state

        # Load into a new scheduler with new config
        opt2 = self._make_optimizer(lr=1.0)
        sched2 = InfiniteLRScheduler(
            opt2,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            start_annealing=True,
            annealing_type="rsqrt",
            annealing_steps=50,
        )
        sched2.load_state_dict(old_state)

        # Config-only params should be preserved from constructor
        assert sched2.start_annealing is True
        assert sched2.annealing_type == "rsqrt"
        assert sched2.annealing_steps == 50
        # start_annealing=True + loaded checkpoint_step=-1 -> checkpoint_step=last_epoch
        assert sched2.checkpoint_step == 20

    # --- start_annealing + rsqrt combined ---

    def test_start_annealing_with_rsqrt(self):
        """start_annealing + rsqrt should work together."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            min_lr=1e-6,
            checkpoint_step=-1,
        )

        for _ in range(50):
            sched.step()

        saved_state = sched.state_dict()

        # Resume with rsqrt annealing triggered by start_annealing
        opt2 = self._make_optimizer(lr=1.0)
        annealing_steps = 100
        sched2 = InfiniteLRScheduler(
            opt2,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.5,
            min_lr=1e-6,
            checkpoint_step=-1,
            start_annealing=True,
            annealing_type="rsqrt",
            annealing_steps=annealing_steps,
        )
        sched2.load_state_dict(saved_state)

        assert sched2.checkpoint_step == 50

        # Step and verify rsqrt formula
        sched2.step()
        lr = sched2.get_last_lr()[0]
        t = 1 / annealing_steps
        expected = 1.0 / (t / 1e-6 + (1.0 - t) / 0.5)
        assert lr == pytest.approx(expected, abs=1e-10)

    # --- validation for new params ---

    def test_invalid_annealing_type_raises(self):
        """Invalid annealing_type should raise."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            InfiniteLRScheduler(
                opt,
                warmup_steps=0,
                cooldown_steps=0,
                constant_lr=0.5,
                annealing_type="linear",
            )

    def test_rsqrt_requires_positive_annealing_steps(self):
        """rsqrt with annealing_steps=0 should raise."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            InfiniteLRScheduler(
                opt,
                warmup_steps=0,
                cooldown_steps=0,
                constant_lr=0.5,
                annealing_type="rsqrt",
                annealing_steps=0,
            )


# ===========================================================================
# Tests for wsd_scheduler.py
# ===========================================================================


class TestWSDScheduler:
    """Tests for WSDScheduler with 3 phases: warmup, stable, decay."""

    def _make_optimizer(self, lr=1.0):
        """Create a simple optimizer with a single parameter group."""
        param = nn.Parameter(torch.randn(4))
        return torch.optim.SGD([param], lr=lr)

    # --- warmup phase ---

    def test_warmup_starts_at_zero(self):
        """At step 0, LR should be 0 during warmup."""
        opt = self._make_optimizer(lr=1.0)
        sched = WSDScheduler(opt, warmup_steps=10)

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.0), f"Expected LR=0 at warmup start, got {lr}"

    def test_warmup_linear_ramp(self):
        """LR should increase linearly during warmup."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 10
        sched = WSDScheduler(opt, warmup_steps=warmup_steps)

        lrs = [sched.get_last_lr()[0]]
        for _ in range(warmup_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        for i in range(warmup_steps + 1):
            expected = 1.0 * i / warmup_steps
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Step {i}: expected LR={expected}, got {lrs[i]}"

    def test_warmup_end_matches_base_lr(self):
        """At the last warmup step, LR should equal base_lr."""
        opt = self._make_optimizer(lr=0.5)
        sched = WSDScheduler(opt, warmup_steps=20)

        for _ in range(20):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.5, abs=1e-7)

    # --- stable phase ---

    def test_stable_phase(self):
        """After warmup, LR should remain at base_lr."""
        opt = self._make_optimizer(lr=0.3)
        sched = WSDScheduler(opt, warmup_steps=5)

        for _ in range(5):
            sched.step()

        for _ in range(20):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(0.3, abs=1e-7)

    def test_no_warmup_starts_at_base_lr(self):
        """With warmup=0, scheduler starts at base_lr."""
        opt = self._make_optimizer(lr=0.7)
        sched = WSDScheduler(opt, warmup_steps=0)

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.7, abs=1e-7)

    # --- decay phase ---

    def test_decay_harmonic_formula(self):
        """Decay should follow harmonic interpolation formula."""
        opt = self._make_optimizer(lr=1.0)
        min_lr = 1e-6
        decay_steps = 100
        decay_start_step = 10
        sched = WSDScheduler(
            opt,
            warmup_steps=0,
            min_lr=min_lr,
            decay_steps=decay_steps,
            decay_start_step=decay_start_step,
        )

        for _ in range(decay_start_step):
            sched.step()

        prev_lr = sched.get_last_lr()[0]
        for i in range(1, 50):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr < prev_lr, f"Decay step {i}: LR should decrease"
            t = i / decay_steps
            expected = 1.0 / (t / min_lr + (1.0 - t) / 1.0)
            assert lr == pytest.approx(
                expected, abs=1e-10
            ), f"Decay step {i}: expected {expected}, got {lr}"
            prev_lr = lr

    def test_decay_reaches_min_lr(self):
        """Decay should reach exactly min_lr at decay_steps."""
        opt = self._make_optimizer(lr=1.0)
        min_lr = 1e-5
        decay_steps = 50
        decay_start_step = 5
        sched = WSDScheduler(
            opt,
            warmup_steps=0,
            min_lr=min_lr,
            decay_steps=decay_steps,
            decay_start_step=decay_start_step,
        )

        for _ in range(decay_start_step + decay_steps):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(min_lr, rel=1e-7)

    def test_decay_starts_at_base_lr(self):
        """At decay_start_step, LR should still be base_lr."""
        opt = self._make_optimizer(lr=0.3)
        sched = WSDScheduler(
            opt,
            warmup_steps=0,
            min_lr=1e-6,
            decay_steps=100,
            decay_start_step=10,
        )

        for _ in range(10):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.3, abs=1e-10)

    def test_decay_clamps_beyond_decay_steps(self):
        """Past decay_steps, LR should stay at min_lr."""
        opt = self._make_optimizer(lr=1.0)
        min_lr = 1e-5
        sched = WSDScheduler(
            opt,
            warmup_steps=0,
            min_lr=min_lr,
            decay_steps=20,
            decay_start_step=5,
        )

        for _ in range(5 + 20 + 10):
            sched.step()

        for _ in range(5):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(min_lr, rel=1e-7)

    # --- start_decay flag ---

    def test_start_decay_triggers_from_stable(self):
        """start_decay=True should begin decay at the loaded step."""
        opt = self._make_optimizer(lr=1.0)
        sched = WSDScheduler(opt, warmup_steps=0, decay_start_step=-1)

        for _ in range(50):
            sched.step()

        saved_state = sched.state_dict()

        opt2 = self._make_optimizer(lr=1.0)
        sched2 = WSDScheduler(
            opt2,
            warmup_steps=0,
            min_lr=1e-6,
            decay_steps=100,
            decay_start_step=-1,
            start_decay=True,
        )
        sched2.load_state_dict(saved_state)

        assert sched2.decay_start_step == 50

        sched2.step()
        lr = sched2.get_last_lr()[0]
        assert lr < 1.0, f"LR should decay, got {lr}"

    def test_start_decay_resumes_existing(self):
        """start_decay=True preserves decay_start_step if already decaying."""
        opt = self._make_optimizer(lr=1.0)
        sched = WSDScheduler(
            opt,
            warmup_steps=0,
            min_lr=1e-6,
            decay_steps=100,
            decay_start_step=20,
        )

        for _ in range(30):
            sched.step()

        saved_state = sched.state_dict()
        lr_at_30 = sched.get_last_lr()[0]

        opt2 = self._make_optimizer(lr=1.0)
        sched2 = WSDScheduler(
            opt2,
            warmup_steps=0,
            min_lr=1e-6,
            decay_steps=100,
            decay_start_step=-1,
            start_decay=True,
        )
        sched2.load_state_dict(saved_state)

        assert sched2.decay_start_step == 20
        assert sched2.get_last_lr()[0] == pytest.approx(lr_at_30, abs=1e-10)

    def test_start_decay_false_restores_constructor(self):
        """start_decay=False restores constructor decay_start_step."""
        opt = self._make_optimizer(lr=1.0)
        sched = WSDScheduler(
            opt,
            warmup_steps=0,
            min_lr=1e-6,
            decay_steps=100,
            decay_start_step=20,
        )

        for _ in range(30):
            sched.step()

        saved_state = sched.state_dict()

        opt2 = self._make_optimizer(lr=1.0)
        sched2 = WSDScheduler(
            opt2,
            warmup_steps=0,
            decay_start_step=-1,
            start_decay=False,
        )
        sched2.load_state_dict(saved_state)

        assert sched2.decay_start_step == -1
        sched2.step()
        lr = sched2.get_last_lr()[0]
        assert lr == pytest.approx(1.0, abs=1e-7)

    # --- config-only keys ---

    def test_config_only_keys_excluded_from_state_dict(self):
        """Config-only params should not appear in state_dict."""
        opt = self._make_optimizer(lr=1.0)
        sched = WSDScheduler(
            opt,
            warmup_steps=0,
            min_lr=1e-6,
            decay_steps=100,
            start_decay=True,
        )

        state = sched.state_dict()
        assert "start_decay" not in state
        assert "min_lr" not in state
        assert "decay_steps" not in state

    # --- state dict round-trip ---

    def test_state_dict_round_trip(self):
        """state_dict / load_state_dict preserves scheduler state."""
        opt = self._make_optimizer(lr=1.0)
        sched = WSDScheduler(
            opt,
            warmup_steps=10,
            min_lr=1e-6,
            decay_steps=50,
            decay_start_step=30,
        )

        for _ in range(15):
            sched.step()

        saved_state = sched.state_dict()
        lr_before = sched.get_last_lr()[0]

        opt2 = self._make_optimizer(lr=1.0)
        sched2 = WSDScheduler(
            opt2,
            warmup_steps=10,
            min_lr=1e-6,
            decay_steps=50,
            decay_start_step=30,
        )
        sched2.load_state_dict(saved_state)

        lr_after = sched2.get_last_lr()[0]
        assert lr_before == pytest.approx(lr_after, abs=1e-10)

        for _ in range(20):
            sched.step()
            sched2.step()
            assert sched.get_last_lr()[0] == pytest.approx(
                sched2.get_last_lr()[0], abs=1e-10
            )

    # --- multiple param groups ---

    def test_multiple_param_groups(self):
        """Scheduler should handle optimizers with multiple parameter groups."""
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(4))
        opt = torch.optim.SGD(
            [
                {"params": [p1], "lr": 1.0},
                {"params": [p2], "lr": 0.5},
            ]
        )

        sched = WSDScheduler(opt, warmup_steps=10)

        # At step 0 both should be 0 (warmup start)
        lrs = sched.get_last_lr()
        assert lrs[0] == pytest.approx(0.0)
        assert lrs[1] == pytest.approx(0.0)

        # At step 5
        for _ in range(5):
            sched.step()

        lrs = sched.get_last_lr()
        assert lrs[0] == pytest.approx(1.0 * 5 / 10, abs=1e-7)
        assert lrs[1] == pytest.approx(0.5 * 5 / 10, abs=1e-7)

    def test_multiple_param_groups_decay(self):
        """Decay should use each group's base_lr independently."""
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(4))
        opt = torch.optim.SGD(
            [
                {"params": [p1], "lr": 1.0},
                {"params": [p2], "lr": 0.5},
            ]
        )

        min_lr = 1e-6
        decay_steps = 100
        sched = WSDScheduler(
            opt,
            warmup_steps=0,
            min_lr=min_lr,
            decay_steps=decay_steps,
            decay_start_step=0,
        )

        for _ in range(10):
            sched.step()

        lrs = sched.get_last_lr()
        t = 10 / decay_steps
        expected_0 = 1.0 / (t / min_lr + (1.0 - t) / 1.0)
        expected_1 = 1.0 / (t / min_lr + (1.0 - t) / 0.5)
        assert lrs[0] == pytest.approx(expected_0, abs=1e-10)
        assert lrs[1] == pytest.approx(expected_1, abs=1e-10)

    # --- validation ---

    def test_invalid_decay_start_step(self):
        """decay_start_step < warmup_steps should raise."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            WSDScheduler(opt, warmup_steps=10, decay_start_step=5)

    def test_negative_warmup_raises(self):
        """Negative warmup_steps should raise."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            WSDScheduler(opt, warmup_steps=-1)

    def test_zero_decay_steps_raises(self):
        """decay_steps=0 should raise."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            WSDScheduler(opt, decay_steps=0)


# ===========================================================================
# Tests for sequential_lr_factory.py
# ===========================================================================


class TestSequentialLRFactory:
    """Tests for sequential_lr_factory."""

    def _make_optimizer(self, lr=0.1):
        param = nn.Parameter(torch.randn(4))
        return torch.optim.SGD([param], lr=lr)

    def test_returns_sequential_lr(self):
        """Factory should return a SequentialLR instance."""
        opt = self._make_optimizer(lr=0.1)

        scheduler_factories = [
            partial(StepLR, step_size=5, gamma=0.5),
            partial(StepLR, step_size=10, gamma=0.1),
        ]
        milestones = [5]

        result = sequential_lr_factory(opt, scheduler_factories, milestones)
        assert isinstance(result, SequentialLR)

    def test_optimizer_forwarded_to_each_scheduler(self):
        """Each scheduler factory receives the optimizer."""
        opt = self._make_optimizer(lr=0.1)

        received_optimizers = []

        def capturing_factory(optimizer, step_size=5):
            received_optimizers.append(optimizer)
            return StepLR(optimizer, step_size=step_size)

        factories = [
            partial(capturing_factory, step_size=5),
            partial(capturing_factory, step_size=10),
        ]
        milestones = [3]

        sequential_lr_factory(opt, factories, milestones)

        assert len(received_optimizers) == 2
        assert all(o is opt for o in received_optimizers)

    def test_sequential_transition(self):
        """LR should transition between schedulers at the milestone."""
        opt = self._make_optimizer(lr=1.0)

        # Phase 1: constant (StepLR with gamma=1.0 never decays)
        # Phase 2: StepLR that decays every step
        factories = [
            partial(StepLR, step_size=1, gamma=1.0),  # No decay
            partial(StepLR, step_size=1, gamma=0.5),  # Halve each step
        ]
        milestones = [5]

        sched = sequential_lr_factory(opt, factories, milestones)

        # Steps 0-4: constant at 1.0
        for step in range(5):
            sched.step()

        # After milestone, second scheduler kicks in
        lr_before_decay = opt.param_groups[0]["lr"]
        sched.step()
        lr_after_decay = opt.param_groups[0]["lr"]

        # Second scheduler (gamma=0.5) should start decaying
        assert lr_after_decay < lr_before_decay or lr_after_decay == pytest.approx(
            lr_before_decay * 0.5, abs=1e-7
        )

    def test_last_epoch_parameter(self):
        """last_epoch parameter should be forwarded to SequentialLR."""
        opt = self._make_optimizer(lr=0.1)
        factories = [
            partial(StepLR, step_size=5, gamma=0.5),
        ]
        milestones = []

        # Should not raise
        sched = sequential_lr_factory(opt, factories, milestones, last_epoch=-1)
        assert isinstance(sched, SequentialLR)


# ===========================================================================
# Tests for subspace_proj.py
# ===========================================================================


class TestSubspaceProjector:
    """Tests for the SubspaceProjector base class."""

    def test_invalid_proj_type(self):
        """Unknown proj_type should raise."""
        with pytest.raises(Exception, match="Unknow projection type"):
            SubspaceProjector(rank=4, dim=16, proj_type="invalid", update_steps=1)

    def test_left_projection_attributes(self):
        """Left projector should set correct dimensions and einsum strings."""
        proj = SubspaceProjector(rank=4, dim=16, proj_type="left", update_steps=1)
        assert proj.proj_shape == (16, 4)
        assert proj.dim == 16

    def test_right_projection_attributes(self):
        """Right projector should set correct dimensions and einsum strings."""
        proj = SubspaceProjector(rank=4, dim=16, proj_type="right", update_steps=1)
        assert proj.proj_shape == (4, 16)
        assert proj.dim == 16

    def test_scale_computation(self):
        """scale should be sqrt(dim) / sqrt(rank)."""
        proj = SubspaceProjector(rank=4, dim=16, proj_type="left", update_steps=1)
        expected_scale = math.sqrt(16) / math.sqrt(4)
        assert proj.scale == pytest.approx(expected_scale)


class TestOnlinePCAProjector:
    """Tests for OnlinePCAProjector."""

    def test_init_left(self):
        """Left PCA projector initializes correctly."""
        proj = OnlinePCAProjector(rank=4, dim=16, proj_type="left", update_steps=1)
        assert proj.rank == 4
        assert proj.dim == 16
        assert proj.A is None  # Lazy initialization

    def test_init_right(self):
        """Right PCA projector initializes correctly."""
        proj = OnlinePCAProjector(rank=4, dim=16, proj_type="right", update_steps=1)
        assert proj.rank == 4
        assert proj.dim == 16
        assert proj.A is None

    def test_down_projection_shape_left(self):
        """Left down projection: (out, in) -> (rank, in)."""
        rank, dim = 4, 16
        proj = OnlinePCAProjector(rank=rank, dim=dim, proj_type="left", update_steps=1)
        x = torch.randn(dim, 8)  # (dim, other)
        proj.step(x)
        result = proj.down(x)
        assert result.shape == (rank, 8)

    def test_up_projection_shape_left(self):
        """Left up projection: (rank, in) -> (dim, in)."""
        rank, dim = 4, 16
        proj = OnlinePCAProjector(rank=rank, dim=dim, proj_type="left", update_steps=1)
        x = torch.randn(dim, 8)
        proj.step(x)
        down = proj.down(x)
        result = proj.up(down)
        assert result.shape == x.shape

    def test_down_projection_shape_right(self):
        """Right down projection: (out, dim) -> (rank, out)."""
        rank, dim = 4, 16
        proj = OnlinePCAProjector(rank=rank, dim=dim, proj_type="right", update_steps=1)
        x = torch.randn(8, dim)  # (other, dim)
        proj.step(x)
        result = proj.down(x)
        assert result.shape == (rank, 8)

    def test_up_projection_shape_right(self):
        """Right up projection: (rank, out) -> (out, dim)."""
        rank, dim = 4, 16
        proj = OnlinePCAProjector(rank=rank, dim=dim, proj_type="right", update_steps=1)
        x = torch.randn(8, dim)
        proj.step(x)
        down = proj.down(x)
        result = proj.up(down)
        assert result.shape == x.shape

    def test_step_initializes_projection(self):
        """After the first step, the projection matrix A should be set."""
        proj = OnlinePCAProjector(rank=4, dim=16, proj_type="left", update_steps=1)
        assert proj.A is None
        x = torch.randn(16, 8)
        proj.step(x)
        assert proj.A is not None
        assert proj.A.shape == (16, 4)

    def test_step_update_frequency(self):
        """_update should only be called every update_steps."""
        rank, dim = 4, 16
        update_steps = 3
        proj = OnlinePCAProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=update_steps
        )
        x = torch.randn(dim, 8)

        # First step (step 0 % 3 == 0): initializes A
        proj.step(x)
        assert proj.A is not None
        A_after_first = proj.A.clone()

        # Steps 1, 2: no update expected
        proj.step(x)
        assert proj.A is not None
        assert torch.equal(proj.A, A_after_first)
        proj.step(x)
        assert proj.A is not None
        assert torch.equal(proj.A, A_after_first)

        # Step 3 (3 % 3 == 0): update expected
        proj.step(x)
        # After update, A may change (depends on fitting)
        # We just verify it does not crash
        assert proj.A is not None

    def test_reconstruction_error_decreases_with_rank(self):
        """Higher rank should yield lower reconstruction error."""
        torch.manual_seed(42)
        dim = 32
        x = torch.randn(dim, 64)

        errors = []
        for rank in [2, 8, 16]:
            proj = OnlinePCAProjector(
                rank=rank, dim=dim, proj_type="left", update_steps=1
            )
            proj.step(x)
            recon = proj.up(proj.down(x))
            error = (x - recon).square().mean().item()
            errors.append(error)

        # Errors should decrease as rank increases
        assert (
            errors[0] > errors[1] > errors[2]
        ), f"Reconstruction error should decrease with rank: {errors}"

    def test_orthag_qr_left(self):
        """QR orthogonalization should produce orthonormal columns for left proj."""
        proj = OnlinePCAProjector(
            rank=4, dim=16, proj_type="left", update_steps=1, orthag="qr"
        )
        x = torch.randn(16, 8)
        proj.step(x)

        # A columns should be approximately orthonormal
        assert proj.A is not None
        AtA = proj.A.T @ proj.A
        eye = torch.eye(4)
        assert torch.allclose(
            AtA, eye, atol=1e-5
        ), "QR orthogonalized projection should have orthonormal columns"

    def test_invalid_orthag_raises(self):
        """Unknown orthagonalization method should raise."""
        with pytest.raises(Exception, match="Unknow orthagonalization"):
            OnlinePCAProjector(
                rank=4, dim=16, proj_type="left", update_steps=1, orthag="bad"
            )


class TestRandProjector:
    """Tests for RandProjector."""

    def test_init_lazy(self):
        """Lazy RandProjector should not allocate A until needed."""
        proj = RandProjector(
            rank=4, dim=16, proj_type="left", update_steps=1, lazy=True
        )
        assert proj.A is None

    def test_init_not_lazy(self):
        """Non-lazy RandProjector should allocate A on first step."""
        proj = RandProjector(
            rank=4, dim=16, proj_type="left", update_steps=1, lazy=False
        )
        x = torch.randn(16, 8)
        proj.step(x)
        assert proj.A is not None
        assert proj.A.shape == (16, 4)

    def test_down_projection_shape_left(self):
        """Left down projection shape check for RandProjector."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=42
        )
        x = torch.randn(dim, 8)
        proj.step(x)
        result = proj.down(x)
        assert result.shape == (rank, 8)

    def test_down_projection_shape_right(self):
        """Right down projection shape check for RandProjector."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="right", update_steps=1, lazy=True, seed=42
        )
        x = torch.randn(8, dim)
        proj.step(x)
        result = proj.down(x)
        assert result.shape == (rank, 8)

    def test_up_projection_shape_left(self):
        """Left up projection shape check for RandProjector."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=42
        )
        x = torch.randn(dim, 8)
        proj.step(x)
        down = proj.down(x)
        result = proj.up(down)
        assert result.shape == x.shape

    def test_up_projection_shape_right(self):
        """Right up projection shape check for RandProjector."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="right", update_steps=1, lazy=True, seed=42
        )
        x = torch.randn(8, dim)
        proj.step(x)
        down = proj.down(x)
        result = proj.up(down)
        assert result.shape == x.shape

    def test_determinism_with_seed_lazy(self):
        """Two lazy RandProjectors with the same seed should produce identical projections."""
        rank, dim = 4, 16
        x = torch.randn(dim, 8)

        proj1 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=123
        )
        proj1.step(x)
        result1 = proj1.down(x)

        proj2 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=123
        )
        proj2.step(x)
        result2 = proj2.down(x)

        assert torch.allclose(
            result1, result2, atol=1e-6
        ), "Identically seeded RandProjectors should produce identical results"

    def test_determinism_with_seed_not_lazy(self):
        """Two non-lazy RandProjectors with the same seed should produce identical matrices."""
        rank, dim = 4, 16
        x = torch.randn(dim, 8)

        proj1 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=False, seed=99
        )
        proj1.step(x)

        proj2 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=False, seed=99
        )
        proj2.step(x)

        assert proj1.A is not None
        assert proj2.A is not None
        assert torch.allclose(proj1.A, proj2.A, atol=1e-6)

    def test_different_seeds_differ(self):
        """Different seeds should produce different projections."""
        rank, dim = 4, 16
        x = torch.randn(dim, 8)

        proj1 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=1
        )
        proj1.step(x)
        result1 = proj1.down(x)

        proj2 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=2
        )
        proj2.step(x)
        result2 = proj2.down(x)

        assert not torch.allclose(
            result1, result2, atol=1e-6
        ), "Different seeds should produce different projections"

    def test_orthogonal_init(self):
        """RandProjector with 'orthogonal' init should work."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank,
            dim=dim,
            proj_type="left",
            update_steps=1,
            lazy=False,
            init="orthogonal",
            seed=42,
        )
        x = torch.randn(dim, 8)
        proj.step(x)
        assert proj.A is not None
        assert proj.A.shape == (dim, rank)

    def test_invalid_init_raises(self):
        """Unknown init method should raise."""
        proj = RandProjector(
            rank=4, dim=16, proj_type="left", update_steps=1, lazy=False, init="bad"
        )
        x = torch.randn(16, 8)
        with pytest.raises(Exception):
            proj.step(x)

    def test_update_frequency(self):
        """Projection should only be regenerated every update_steps steps."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=3, lazy=False, seed=42
        )
        x = torch.randn(dim, 8)

        # Step 0: initialize
        proj.step(x)
        assert proj.A is not None
        A_initial = proj.A.clone()

        # Steps 1, 2: no regeneration
        proj.step(x)
        assert proj.A is not None
        assert torch.equal(proj.A, A_initial)
        proj.step(x)
        assert proj.A is not None
        assert torch.equal(proj.A, A_initial)

        # Step 3: regenerate
        proj.step(x)
        # A should change because the generator state advanced
        # (or at minimum, _update was called)
        assert proj.A is not None


# ===========================================================================
# Tests for adamw.py
# ===========================================================================


class TestAdamW:
    """Tests for the custom AdamW optimizer."""

    def _make_simple_problem(self):
        """Create a simple linear regression problem."""
        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=False)
        x = torch.randn(32, 8)
        y = torch.randn(32, 1)
        return model, x, y

    def test_loss_decreases(self):
        """A few steps of AdamW should reduce loss on a simple problem."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        initial_loss = nn.functional.mse_loss(model(x), y).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert (
            final_loss < initial_loss
        ), f"Loss should decrease: initial={initial_loss}, final={final_loss}"

    def test_weight_decay_applied(self):
        """With weight_decay > 0, weights should shrink compared to no weight decay."""
        # Create identical models and shared data
        torch.manual_seed(42)
        model_wd = nn.Linear(8, 1, bias=False)
        x = torch.randn(32, 8)
        y = torch.randn(32, 1)

        torch.manual_seed(42)
        model_no_wd = nn.Linear(8, 1, bias=False)

        # Verify models start identical
        assert torch.equal(model_wd.weight.data, model_no_wd.weight.data)

        opt_wd = AdamW(model_wd.parameters(), lr=1e-2, weight_decay=0.5)
        opt_no_wd = AdamW(model_no_wd.parameters(), lr=1e-2, weight_decay=0.0)

        for _ in range(100):
            opt_wd.zero_grad()
            loss_wd = nn.functional.mse_loss(model_wd(x), y)
            loss_wd.backward()
            opt_wd.step()

            opt_no_wd.zero_grad()
            loss_no_wd = nn.functional.mse_loss(model_no_wd(x), y)
            loss_no_wd.backward()
            opt_no_wd.step()

        norm_wd = model_wd.weight.data.norm().item()
        norm_no_wd = model_no_wd.weight.data.norm().item()
        assert (
            norm_wd < norm_no_wd
        ), f"Weight decay should reduce weight norm: with_wd={norm_wd}, without_wd={norm_no_wd}"

    def test_state_dict_structure(self):
        """state_dict should contain expected keys: step, m, v."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        # Take one step to initialize state
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        assert "state" in state_dict
        assert "param_groups" in state_dict

        # Each param should have step, m, v
        for param_id, param_state in state_dict["state"].items():
            assert "step" in param_state
            assert "m" in param_state
            assert "v" in param_state

    def test_state_dict_load_round_trip(self):
        """state_dict should survive save/load round-trip."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        for _ in range(5):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        state_dict = optimizer.state_dict()

        # Create a new optimizer and load state
        model2, _, _ = self._make_simple_problem()
        optimizer2 = AdamW(model2.parameters(), lr=1e-2)
        # Need to take a step first to init state, then overwrite
        optimizer2.zero_grad()
        loss2 = nn.functional.mse_loss(model2(x), y)
        loss2.backward()
        optimizer2.step()

        optimizer2.load_state_dict(state_dict)

        # Verify state was loaded
        loaded_state = optimizer2.state_dict()
        for pid in state_dict["state"]:
            orig_step = state_dict["state"][pid]["step"]
            loaded_step = loaded_state["state"][pid]["step"]
            assert torch.equal(orig_step, loaded_step)

    def test_state_dict_validation_missing_key(self):
        """state_dict with missing keys should raise ValueError."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        # Remove a required key
        for pid in state_dict["state"]:
            del state_dict["state"][pid]["m"]
            break

        with pytest.raises(ValueError, match="missing keys"):
            optimizer.load_state_dict(state_dict)

    def test_closure_returns_loss(self):
        """AdamW step should call the closure and return its value."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        # Compute gradients before calling step (outside no_grad context)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Closure just returns a pre-computed loss value
        loss_val = loss.detach()

        def closure():
            return loss_val

        result = optimizer.step(closure)
        assert result is not None
        assert result.item() > 0

    def test_no_grad_parameters_skipped(self):
        """Parameters without gradients should be skipped without error."""
        model = nn.Sequential(
            nn.Linear(8, 4),
            nn.Linear(4, 1),
        )
        optimizer = AdamW(model.parameters(), lr=1e-2)

        # Only compute grad for one layer
        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Manually set one param's grad to None
        model[0].weight.grad = None  # type: ignore[assignment]

        # Should not raise
        optimizer.step()

    def test_different_betas(self):
        """AdamW should accept custom beta values."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2, betas=(0.8, 0.99))

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        # Verify betas in param_groups
        assert optimizer.param_groups[0]["betas"] == (0.8, 0.99)


# ===========================================================================
# Tests for sgd.py
# ===========================================================================


class TestSGD:
    """Tests for the custom SGD optimizer."""

    def _make_simple_problem(self):
        """Create a simple linear regression problem."""
        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=False)
        x = torch.randn(32, 8)
        y = torch.randn(32, 1)
        return model, x, y

    def test_loss_decreases(self):
        """A few steps of SGD should reduce loss on a simple problem."""
        model, x, y = self._make_simple_problem()
        optimizer = SGD(model.parameters(), lr=1e-2)

        initial_loss = nn.functional.mse_loss(model(x), y).item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert (
            final_loss < initial_loss
        ), f"Loss should decrease: initial={initial_loss}, final={final_loss}"

    def test_update_direction_matches_negative_gradient(self):
        """SGD update should be in the direction of negative gradient."""
        torch.manual_seed(42)
        model = nn.Linear(4, 1, bias=False)
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)

        lr = 0.01
        optimizer = SGD(model.parameters(), lr=lr)

        # Record weights before step
        w_before = model.weight.data.clone()

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        assert model.weight.grad is not None
        grad = model.weight.grad.clone()

        optimizer.step()

        w_after = model.weight.data.clone()

        # Update = w_after - w_before should equal -lr * grad
        update = w_after - w_before
        expected_update = -lr * grad

        assert torch.allclose(update, expected_update, atol=1e-6), (
            f"SGD update should be -lr * grad.\n"
            f"Actual update: {update}\n"
            f"Expected: {expected_update}"
        )

    def test_closure_returns_loss(self):
        """SGD step should call the closure and return its value."""
        model, x, y = self._make_simple_problem()
        optimizer = SGD(model.parameters(), lr=1e-2)

        # Compute gradients before calling step (outside no_grad context)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        loss_val = loss.detach()

        def closure():
            return loss_val

        result = optimizer.step(closure)
        assert result is not None
        assert result.item() > 0

    def test_no_grad_parameters_skipped(self):
        """Parameters without gradients should be skipped."""
        layer0 = nn.Linear(4, 4)
        layer1 = nn.Linear(4, 1)
        model = nn.Sequential(layer0, layer1)
        optimizer = SGD(model.parameters(), lr=1e-2)

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Remove grad from one parameter
        layer0.weight.grad = None  # type: ignore[assignment]

        # Record the parameter that has no grad
        w_before = layer0.weight.data.clone()

        optimizer.step()

        # Parameter without grad should be unchanged
        assert torch.equal(
            layer0.weight.data, w_before
        ), "Parameter without gradient should not be modified"

    def test_multiple_param_groups(self):
        """SGD should handle multiple parameter groups with different LRs."""
        layer0 = nn.Linear(4, 4)
        layer1 = nn.Linear(4, 1)
        model = nn.Sequential(layer0, layer1)

        optimizer = SGD(
            [  # type: ignore[arg-type]
                {"params": layer0.parameters(), "lr": 0.1},
                {"params": layer1.parameters(), "lr": 0.01},
            ],
            lr=0.001,  # default LR
        )

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Record weights before step
        w0_before = layer0.weight.data.clone()
        w1_before = layer1.weight.data.clone()

        optimizer.step()

        # Both should be updated
        assert not torch.equal(layer0.weight.data, w0_before)
        assert not torch.equal(layer1.weight.data, w1_before)

        # Verify correct LR was used
        assert optimizer.param_groups[0]["lr"] == 0.1
        assert optimizer.param_groups[1]["lr"] == 0.01

    def test_zero_lr_no_update(self):
        """With lr=0, parameters should not change."""
        torch.manual_seed(42)
        model = nn.Linear(4, 1, bias=False)
        optimizer = SGD(model.parameters(), lr=0.0)

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        w_before = model.weight.data.clone()
        optimizer.step()

        assert torch.equal(
            model.weight.data, w_before
        ), "With lr=0, weights should remain unchanged"

    def test_large_lr_large_update(self):
        """A larger LR should produce larger parameter updates."""
        torch.manual_seed(42)
        model_small = nn.Linear(4, 1, bias=False)
        torch.manual_seed(42)
        model_large = nn.Linear(4, 1, bias=False)

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)

        opt_small = SGD(model_small.parameters(), lr=0.001)
        opt_large = SGD(model_large.parameters(), lr=0.1)

        # Same forward/backward
        opt_small.zero_grad()
        loss_s = nn.functional.mse_loss(model_small(x), y)
        loss_s.backward()
        w_before_small = model_small.weight.data.clone()
        opt_small.step()

        opt_large.zero_grad()
        loss_l = nn.functional.mse_loss(model_large(x), y)
        loss_l.backward()
        w_before_large = model_large.weight.data.clone()
        opt_large.step()

        update_small = (model_small.weight.data - w_before_small).norm().item()
        update_large = (model_large.weight.data - w_before_large).norm().item()

        assert update_large > update_small, (
            f"Larger LR should produce larger update: "
            f"small={update_small}, large={update_large}"
        )


# ===========================================================================
# CUDA-specific tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAdamWCUDA:
    """CUDA-specific tests for AdamW."""

    def test_loss_decreases_on_cuda(self):
        """AdamW should reduce loss when model is on CUDA."""
        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=False).cuda()
        x = torch.randn(32, 8, device="cuda")
        y = torch.randn(32, 1, device="cuda")

        optimizer = AdamW(model.parameters(), lr=1e-2)

        initial_loss = nn.functional.mse_loss(model(x), y).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert final_loss < initial_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSGDCUDA:
    """CUDA-specific tests for SGD."""

    def test_loss_decreases_on_cuda(self):
        """SGD should reduce loss when model is on CUDA."""
        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=False).cuda()
        x = torch.randn(32, 8, device="cuda")
        y = torch.randn(32, 1, device="cuda")

        optimizer = SGD(model.parameters(), lr=1e-2)

        initial_loss = nn.functional.mse_loss(model(x), y).item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert final_loss < initial_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRoundingUtilsCUDA:
    """CUDA-specific stochastic rounding tests."""

    def test_stochastic_rounding_deterministic_on_cuda(self):
        """Seeded stochastic rounding on CUDA should be deterministic."""
        x = torch.randn(256, dtype=torch.float32, device="cuda")

        gen1 = torch.Generator(device="cuda")
        gen1.manual_seed(42)
        result1 = fp32_to_bf16_stochastic_round(x, generator=gen1)

        gen2 = torch.Generator(device="cuda")
        gen2.manual_seed(42)
        result2 = fp32_to_bf16_stochastic_round(x, generator=gen2)

        assert torch.equal(result1, result2)


# ---------------------------------------------------------------------------
# CosineLRScheduler tests
# ---------------------------------------------------------------------------


class TestCosineLRScheduler:
    """Tests for CosineLRScheduler with optional warmup and cosine decay."""

    def _make_optimizer(self, lr=1.0):
        param = nn.Parameter(torch.randn(4))
        return torch.optim.SGD([param], lr=lr)

    # --- warmup ---

    def test_warmup_starts_at_zero(self):
        opt = self._make_optimizer(lr=1.0)
        sched = CosineLRScheduler(opt, total_steps=100, warmup_steps=10)

        assert sched.get_last_lr()[0] == pytest.approx(0.0)

    def test_warmup_linear_ramp(self):
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 10
        sched = CosineLRScheduler(opt, total_steps=100, warmup_steps=warmup_steps)

        lrs = [sched.get_last_lr()[0]]
        for _ in range(warmup_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        for i in range(warmup_steps + 1):
            expected = 1.0 * i / warmup_steps
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Step {i}: expected {expected}, got {lrs[i]}"

    def test_warmup_end_matches_base_lr(self):
        opt = self._make_optimizer(lr=0.5)
        sched = CosineLRScheduler(opt, total_steps=100, warmup_steps=20)

        for _ in range(20):
            sched.step()

        assert sched.get_last_lr()[0] == pytest.approx(0.5, abs=1e-7)

    # --- cosine decay ---

    def test_no_warmup_starts_at_base_lr(self):
        opt = self._make_optimizer(lr=1.0)
        sched = CosineLRScheduler(opt, total_steps=100, warmup_steps=0)

        assert sched.get_last_lr()[0] == pytest.approx(1.0)

    def test_decay_reaches_zero_at_total_steps(self):
        opt = self._make_optimizer(lr=1.0)
        total_steps = 100
        sched = CosineLRScheduler(opt, total_steps=total_steps, warmup_steps=0)

        for _ in range(total_steps):
            sched.step()

        assert sched.get_last_lr()[0] == pytest.approx(0.0, abs=1e-7)

    def test_decay_reaches_min_lr_at_total_steps(self):
        opt = self._make_optimizer(lr=1.0)
        total_steps = 100
        sched = CosineLRScheduler(
            opt, total_steps=total_steps, warmup_steps=0, min_lr=0.1
        )

        for _ in range(total_steps):
            sched.step()

        assert sched.get_last_lr()[0] == pytest.approx(0.1, abs=1e-7)

    def test_decay_midpoint_is_half(self):
        """At the midpoint of cosine decay, LR should be 0.5 * base_lr."""
        opt = self._make_optimizer(lr=2.0)
        total_steps = 100
        sched = CosineLRScheduler(opt, total_steps=total_steps, warmup_steps=0)

        for _ in range(total_steps // 2):
            sched.step()

        assert sched.get_last_lr()[0] == pytest.approx(1.0, abs=1e-7)

    def test_decay_midpoint_with_min_lr(self):
        """At the midpoint, LR should be midway between base_lr and min_lr."""
        opt = self._make_optimizer(lr=1.0)
        total_steps = 100
        sched = CosineLRScheduler(
            opt, total_steps=total_steps, warmup_steps=0, min_lr=0.2
        )

        for _ in range(total_steps // 2):
            sched.step()

        # min_lr + (base_lr - min_lr) * 0.5 = 0.2 + 0.8 * 0.5 = 0.6
        assert sched.get_last_lr()[0] == pytest.approx(0.6, abs=1e-7)

    def test_decay_cosine_shape(self):
        """Verify LR follows cosine curve during decay."""
        opt = self._make_optimizer(lr=1.0)
        total_steps = 200
        warmup_steps = 0
        decay_steps = total_steps
        sched = CosineLRScheduler(
            opt, total_steps=total_steps, warmup_steps=warmup_steps
        )

        lrs = [sched.get_last_lr()[0]]
        for _ in range(total_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        for i in range(total_steps + 1):
            progress = i / decay_steps
            expected = 0.5 * (1.0 + math.cos(math.pi * progress))
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Step {i}: expected {expected}, got {lrs[i]}"

    def test_decay_cosine_shape_with_min_lr(self):
        """Verify LR follows cosine curve with min_lr floor."""
        opt = self._make_optimizer(lr=1.0)
        total_steps = 200
        min_lr = 0.1
        sched = CosineLRScheduler(
            opt, total_steps=total_steps, warmup_steps=0, min_lr=min_lr
        )

        lrs = [sched.get_last_lr()[0]]
        for _ in range(total_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        for i in range(total_steps + 1):
            progress = i / total_steps
            expected = min_lr + (1.0 - min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Step {i}: expected {expected}, got {lrs[i]}"

    # --- warmup + decay combined ---

    def test_warmup_then_decay(self):
        """Full schedule: warmup to base_lr, then cosine decay to 0."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 20
        total_steps = 120
        decay_steps = total_steps - warmup_steps
        sched = CosineLRScheduler(
            opt, total_steps=total_steps, warmup_steps=warmup_steps
        )

        # Collect all LRs
        lrs = [sched.get_last_lr()[0]]
        for _ in range(total_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        # Warmup: linear from 0 to 1
        for i in range(warmup_steps + 1):
            expected = i / warmup_steps
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Warmup step {i}: expected {expected}, got {lrs[i]}"

        # Decay: cosine from 1 to 0
        for i in range(warmup_steps, total_steps + 1):
            progress = (i - warmup_steps) / decay_steps
            expected = 0.5 * (1.0 + math.cos(math.pi * progress))
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Decay step {i}: expected {expected}, got {lrs[i]}"

    def test_warmup_then_decay_with_min_lr(self):
        """Full schedule: warmup to base_lr, then cosine decay to min_lr."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 20
        total_steps = 120
        decay_steps = total_steps - warmup_steps
        min_lr = 0.05
        sched = CosineLRScheduler(
            opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=min_lr
        )

        lrs = [sched.get_last_lr()[0]]
        for _ in range(total_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        # Warmup: linear from 0 to 1 (unaffected by min_lr)
        for i in range(warmup_steps + 1):
            expected = i / warmup_steps
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Warmup step {i}: expected {expected}, got {lrs[i]}"

        # Decay: cosine from 1 to min_lr
        for i in range(warmup_steps, total_steps + 1):
            progress = (i - warmup_steps) / decay_steps
            expected = min_lr + (1.0 - min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            assert lrs[i] == pytest.approx(
                expected, abs=1e-7
            ), f"Decay step {i}: expected {expected}, got {lrs[i]}"

    # --- multiple param groups ---

    def test_multiple_param_groups(self):
        """Each param group should use its own base_lr."""
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(4))
        opt = torch.optim.SGD(
            [{"params": [p1], "lr": 1.0}, {"params": [p2], "lr": 0.1}]
        )
        sched = CosineLRScheduler(opt, total_steps=100, warmup_steps=10)

        for _ in range(10):
            sched.step()

        lrs = sched.get_last_lr()
        assert lrs[0] == pytest.approx(1.0, abs=1e-7)
        assert lrs[1] == pytest.approx(0.1, abs=1e-7)

    # --- validation ---

    def test_total_steps_must_be_positive(self):
        opt = self._make_optimizer()
        with pytest.raises(AssertionError):
            CosineLRScheduler(opt, total_steps=0)

    def test_warmup_must_be_less_than_total(self):
        opt = self._make_optimizer()
        with pytest.raises(AssertionError):
            CosineLRScheduler(opt, total_steps=10, warmup_steps=10)

    def test_negative_warmup_rejected(self):
        opt = self._make_optimizer()
        with pytest.raises(AssertionError):
            CosineLRScheduler(opt, total_steps=10, warmup_steps=-1)

    def test_negative_min_lr_rejected(self):
        opt = self._make_optimizer()
        with pytest.raises(AssertionError):
            CosineLRScheduler(opt, total_steps=10, min_lr=-0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
