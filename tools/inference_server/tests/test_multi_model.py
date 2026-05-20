"""
Tests for multi-model registry, acquire/swap protocol, and CLI parsing.

These tests stub the actual model load (it's expensive) and exercise the
registry/lock/swap logic and the input-validation layer.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# conftest.py adds tools/inference_server to sys.path; add tools/ too so
# absolute "inference_server.X" imports work whether the package layout
# carries an __init__.py or not.
_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from inference_server.config import (  # noqa: E402
    _parse_model_arg,
    _parse_yaml_model_entry,
    merge_model_entries,
)
from inference_server.service import (  # noqa: E402
    InferenceService,
    ModelEntry,
    resolve_dtype,
)


def _fake_entry(name: str, path: str = "/fake/path") -> ModelEntry:
    """Build a ModelEntry with stubbed load/to_device so tests don't hit torch."""
    e = ModelEntry(name=name, model_path=path, dtype=resolve_dtype("float32"))

    def fake_load(device, from_checkpoint, default_chat_template_factory):
        # Populate the attrs the service properties read; mark loaded.
        e.tokenizer = MagicMock()
        e.model = MagicMock()
        e.default_generation_config = MagicMock()
        e.chat_template = "{{ messages }}"
        e.stop_token_ids = set()
        e.stop_processor = MagicMock()
        e.finish_detector = MagicMock()
        e.tokenizer_wrapper = MagicMock()
        e.state = "gpu" if str(device).startswith("cuda") else "cpu"

    def fake_to_device(device):
        e.state = "gpu" if str(device).startswith("cuda") else "cpu"

    e.load = fake_load  # type: ignore[assignment]
    e.to_device = fake_to_device  # type: ignore[assignment]
    return e


def _make_service(entries, device="cuda:0", **kwargs):
    return InferenceService(entries=entries, device=device, **kwargs)


class TestParseModelArg:
    def test_bare_path_derives_name(self):
        spec = _parse_model_arg("/path/to/my_model")
        assert spec == {"name": "my_model", "path": "/path/to/my_model"}

    def test_name_equals_path(self):
        spec = _parse_model_arg("alpha=/path/to/beta")
        assert spec == {"name": "alpha", "path": "/path/to/beta"}

    def test_trailing_slash_normalized(self):
        spec = _parse_model_arg("/path/to/my_model/")
        assert spec["name"] == "my_model"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="empty name"):
            _parse_model_arg("=/path/to/x")

    def test_empty_path_rejected(self):
        with pytest.raises(ValueError, match="empty path"):
            _parse_model_arg("name=")


class TestParseYamlModelEntry:
    def test_explicit_name_path(self):
        spec = _parse_yaml_model_entry(0, {"name": "a", "path": "/x"})
        assert spec["name"] == "a"
        assert spec["path"] == "/x"

    def test_model_path_alias(self):
        spec = _parse_yaml_model_entry(0, {"name": "a", "model_path": "/x"})
        assert spec["path"] == "/x"

    def test_derives_name_from_path(self):
        spec = _parse_yaml_model_entry(0, {"path": "/foo/bar"})
        assert spec["name"] == "bar"

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown keys"):
            _parse_yaml_model_entry(0, {"name": "a", "path": "/x", "typo": 1})

    def test_per_model_overrides_kept(self):
        spec = _parse_yaml_model_entry(
            0,
            {
                "name": "a",
                "path": "/x",
                "dtype": "float16",
                "stop_sequences": ["</s>"],
                "chat_template": "/tmpl",
            },
        )
        assert spec["dtype"] == "float16"
        assert spec["stop_sequences"] == ["</s>"]
        assert spec["chat_template"] == "/tmpl"


class TestMergeModelEntries:
    def test_cli_wins_over_yaml_models(self):
        cli = ["a=/x", "b=/y"]
        cfg = {"models": [{"name": "ignored", "path": "/z"}]}
        specs = merge_model_entries(cli, cfg)
        assert [s["name"] for s in specs] == ["a", "b"]

    def test_yaml_models_used_when_no_cli(self):
        cfg = {"models": [{"name": "a", "path": "/x"}, {"name": "b", "path": "/y"}]}
        specs = merge_model_entries(None, cfg)
        assert [s["name"] for s in specs] == ["a", "b"]

    def test_empty_returns_empty(self):
        assert merge_model_entries(None, {}) == []
        assert merge_model_entries([], {}) == []

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            merge_model_entries(["a=/x", "a=/y"], {})

    def test_yaml_models_must_be_list(self):
        with pytest.raises(ValueError, match="expected a list"):
            merge_model_entries(None, {"models": "/path/to/model"})


class TestInferenceServiceConstruction:
    def test_empty_entries_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            InferenceService(entries=[], device="cpu")

    def test_duplicate_names_rejected(self):
        a = _fake_entry("dup")
        b = _fake_entry("dup")
        with pytest.raises(ValueError, match="Duplicate"):
            InferenceService(entries=[a, b], device="cuda:0")

    def test_specific_checkpoint_rejected_with_multi_model(self):
        a = _fake_entry("a")
        b = _fake_entry("b")
        with pytest.raises(ValueError, match="checkpoint"):
            InferenceService(
                entries=[a, b],
                device="cuda:0",
                from_checkpoint="/specific/ckpt",
            )

    def test_auto_device_rejected_with_multi_model(self):
        a = _fake_entry("a")
        b = _fake_entry("b")
        with pytest.raises(ValueError, match="auto"):
            InferenceService(entries=[a, b], device="auto")

    def test_single_model_loads_eagerly(self):
        a = _fake_entry("a")
        svc = _make_service([a])
        assert a.state != "unloaded"
        assert svc.active is a

    def test_multi_model_lazy(self):
        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b])
        assert a.state == "unloaded"
        assert b.state == "unloaded"
        assert svc.active is None


class TestResolveEntry:
    def test_named_match(self):
        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b])
        assert svc._resolve_entry("a") is a
        assert svc._resolve_entry("b") is b

    def test_empty_name_single_model_returns_sole_entry(self):
        a = _fake_entry("a")
        svc = _make_service([a])
        assert svc._resolve_entry("") is a
        assert svc._resolve_entry(None) is a

    def test_unknown_name_multi_model_raises_404(self):
        from fastapi import HTTPException

        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b])
        with pytest.raises(HTTPException) as exc:
            svc._resolve_entry("nonexistent")
        assert exc.value.status_code == 404

    def test_empty_name_multi_model_raises_404(self):
        from fastapi import HTTPException

        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b])
        with pytest.raises(HTTPException) as exc:
            svc._resolve_entry(None)
        assert exc.value.status_code == 404


class TestAcquireSwap:
    def test_lazy_load_then_reuse(self):
        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b])

        async def run():
            async with svc.acquire("a"):
                assert a.state == "gpu"
                assert svc.active is a
            async with svc.acquire("a"):
                assert a.state == "gpu"
                assert svc.active is a

        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.cuda.empty_cache"
        ):
            asyncio.run(run())

    def test_swap_moves_previous_to_cpu(self):
        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b])

        async def run():
            async with svc.acquire("a"):
                assert a.state == "gpu"
            async with svc.acquire("b"):
                assert a.state == "cpu"
                assert b.state == "gpu"
                assert svc.active is b
            async with svc.acquire("a"):
                assert b.state == "cpu"
                assert a.state == "gpu"

        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.cuda.empty_cache"
        ):
            asyncio.run(run())

    def test_unknown_name_raises_before_lock_held(self):
        from fastapi import HTTPException

        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b])

        async def run():
            async with svc.acquire("does-not-exist"):
                pass  # pragma: no cover

        with pytest.raises(HTTPException) as exc:
            asyncio.run(run())
        assert exc.value.status_code == 404


class TestKeepOnGpu:
    """``keep_on_gpu=True``: inactive models stay on GPU instead of
    swapping to CPU. ``self.active`` still tracks the current request
    target (strategies need it for routing) but no demotion happens."""

    def test_no_demotion_on_swap(self):
        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b], keep_on_gpu=True)

        async def run():
            async with svc.acquire("a"):
                assert a.state == "gpu"
            # 'b' is unloaded → lazy-load, but 'a' stays on GPU because
            # keep_on_gpu is set.
            async with svc.acquire("b"):
                assert a.state == "gpu"
                assert b.state == "gpu"
                assert svc.active is b
            # Back to 'a' — no swap to CPU happens anywhere.
            async with svc.acquire("a"):
                assert a.state == "gpu"
                assert b.state == "gpu"
                assert svc.active is a

        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.cuda.empty_cache"
        ):
            asyncio.run(run())

    def test_default_off_still_swaps(self):
        # Sanity: leaving keep_on_gpu unset should preserve the
        # swap-to-CPU behavior covered by TestAcquireSwap.
        a = _fake_entry("a")
        b = _fake_entry("b")
        svc = _make_service([a, b])  # keep_on_gpu defaults to False
        assert svc.keep_on_gpu is False
