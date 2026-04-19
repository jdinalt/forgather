"""Unit tests for forgather.eval_config (TestConfig + EvalResult)."""

import dataclasses

import pytest

from forgather.eval_config import EvalResult, TestConfig

REQUIRED = dict(
    eval_name="x",
    name="X",
    description="desc",
    dataset_proj="/p",
    dataset_config="c.yaml",
    dataset_target="test_dataset",
)

RUNTIME = dict(
    model_path="/m",
    checkpoint_path=None,
    batch_size=4,
    max_length=1024,
    stride=0,
    dtype="bfloat16",
    attn_implementation="sdpa",
    trainer="simple",
    world_size=1,
)


class TestTestConfigDefaults:
    def test_library_defaults(self):
        cfg = TestConfig(**REQUIRED)
        assert cfg.default_batch_size == 8
        assert cfg.default_max_length == 4096
        assert cfg.default_stride == 0

    def test_overrides_are_preserved(self):
        cfg = TestConfig(
            **REQUIRED,
            default_batch_size=32,
            default_max_length=1024,
            default_stride=128,
        )
        assert cfg.default_batch_size == 32
        assert cfg.default_max_length == 1024
        assert cfg.default_stride == 128

    def test_required_field_missing_raises(self):
        required = dict(REQUIRED)
        required.pop("dataset_proj")
        with pytest.raises(TypeError):
            TestConfig(**required)

    def test_is_a_dataclass(self):
        # Fail fast if someone accidentally refactors it into a plain class —
        # the CLI relies on ``TestConfig(**dict)`` construction.
        assert dataclasses.is_dataclass(TestConfig)


class TestEvalResultFromConfig:
    def test_identity_fields_are_copied_from_config(self):
        cfg = TestConfig(**REQUIRED)
        result = EvalResult.from_config(cfg, **RUNTIME)
        assert result.eval_name == "x"
        assert result.config_name == "X"  # TestConfig.name → EvalResult.config_name
        assert result.description == "desc"
        assert result.dataset_proj == "/p"
        assert result.dataset_config == "c.yaml"
        assert result.dataset_target == "test_dataset"

    def test_runtime_fields_are_preserved(self):
        cfg = TestConfig(**REQUIRED)
        result = EvalResult.from_config(cfg, **RUNTIME)
        for key, value in RUNTIME.items():
            assert getattr(result, key) == value

    def test_outcomes_start_as_none(self):
        cfg = TestConfig(**REQUIRED)
        result = EvalResult.from_config(cfg, **RUNTIME)
        assert result.eval_loss is None
        assert result.perplexity is None
        assert result.wall_time_s is None
        assert result.timestamp is None

    def test_asdict_round_trip_is_json_serializable(self):
        import json

        cfg = TestConfig(**REQUIRED)
        result = EvalResult.from_config(cfg, **RUNTIME)
        result.eval_loss = 1.23
        result.perplexity = 3.42
        result.wall_time_s = 4.5
        result.timestamp = "2026-04-18T00:00:00Z"
        # Must serialize cleanly — no non-primitive values leak in.
        payload = json.dumps(dataclasses.asdict(result))
        assert json.loads(payload)["eval_loss"] == pytest.approx(1.23)

    def test_is_a_dataclass(self):
        assert dataclasses.is_dataclass(EvalResult)
