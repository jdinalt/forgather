#!/usr/bin/env python3
"""
Unit tests for uncovered functions in forgather.ml.sharded_checkpoint.

Covers: id_to_fqn, get_all_fqns, make_cannonical_names, map_cannonical_names,
create_sharing_metadata, index_file_name, make_shard_index,
_intersect_weight_map, _make_shard_dictionaries, get_checkpoint_metadata,
next_checkpoint_path, save_checkpoint_metrics, load_checkpoint_metrics,
maybe_delete_oldest_checkpoint.

Functions already covered in test_checkpoints.py (find_latest_checkpoint,
validate_checkpoint) are NOT duplicated here.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime

import torch
import torch.nn as nn

from forgather.ml.sharded_checkpoint import (
    CHECKPOINT_MANIFEST_FILENAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    CheckpointMeta,
    _get_checkpoint_timestamp,
    _intersect_weight_map,
    _make_shard_dictionaries,
    _resolve_state_dict,
    create_sharing_metadata,
    get_all_fqns,
    get_checkpoint_metadata,
    id_to_fqn,
    index_file_name,
    load_checkpoint,
    load_checkpoint_metrics,
    load_sharded_checkpoint,
    make_cannonical_names,
    make_shard_index,
    map_cannonical_names,
    maybe_delete_oldest_checkpoint,
    next_checkpoint_path,
    save_checkpoint,
    save_checkpoint_metrics,
    save_sharded_checkpoint,
)


class SimpleModel(nn.Module):
    """Simple model for testing parameter-related functions."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)


class TiedModel(nn.Module):
    """Model with tied (shared) parameters, like embedding/lm_head weight tying."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.lm_head = nn.Linear(32, 100, bias=False)
        # Tie the weights
        self.lm_head.weight = self.embedding.weight


class ModelWithBuffer(nn.Module):
    """Model with both parameters and buffers."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.register_buffer("my_buffer", torch.zeros(4))


class TestIdToFqn(unittest.TestCase):
    """Test id_to_fqn function."""

    def test_simple_model(self):
        """Each parameter should map to its FQN."""
        model = SimpleModel()
        mapping = id_to_fqn(model)

        # Collect all FQN names from the mapping
        all_fqns = set()
        for fqn_set in mapping.values():
            all_fqns.update(fqn_set)

        # Should contain all parameter names
        self.assertIn("linear1.weight", all_fqns)
        self.assertIn("linear1.bias", all_fqns)
        self.assertIn("linear2.weight", all_fqns)
        self.assertIn("linear2.bias", all_fqns)

    def test_tied_parameters(self):
        """Tied parameters should share the same id and appear in the same set."""
        model = TiedModel()
        mapping = id_to_fqn(model)

        # Find the set that contains the embedding weight
        found_tied = False
        for param_id, fqn_set in mapping.items():
            if "embedding.weight" in fqn_set:
                # lm_head.weight should be in the same set since they are tied
                self.assertIn("lm_head.weight", fqn_set)
                found_tied = True
                break
        self.assertTrue(found_tied, "Should find tied parameters sharing the same id")

    def test_model_with_buffer(self):
        """Buffers should also appear in the mapping."""
        model = ModelWithBuffer()
        mapping = id_to_fqn(model)

        all_fqns = set()
        for fqn_set in mapping.values():
            all_fqns.update(fqn_set)

        self.assertIn("my_buffer", all_fqns)
        self.assertIn("linear.weight", all_fqns)
        self.assertIn("linear.bias", all_fqns)


class TestGetAllFqns(unittest.TestCase):
    """Test get_all_fqns function."""

    def test_simple_model(self):
        """Should return all parameter FQNs."""
        model = SimpleModel()
        fqns = get_all_fqns(model)

        self.assertIn("linear1.weight", fqns)
        self.assertIn("linear1.bias", fqns)
        self.assertIn("linear2.weight", fqns)
        self.assertIn("linear2.bias", fqns)
        self.assertEqual(len(fqns), 4)

    def test_model_with_buffer(self):
        """Should include both parameters and buffers."""
        model = ModelWithBuffer()
        fqns = get_all_fqns(model)

        self.assertIn("linear.weight", fqns)
        self.assertIn("linear.bias", fqns)
        self.assertIn("my_buffer", fqns)
        self.assertEqual(len(fqns), 3)

    def test_tied_model_includes_duplicates(self):
        """With remove_duplicate=False, tied params appear under both FQNs."""
        model = TiedModel()
        fqns = get_all_fqns(model)

        self.assertIn("embedding.weight", fqns)
        self.assertIn("lm_head.weight", fqns)

    def test_empty_model(self):
        """A model with no parameters or buffers should return an empty set."""

        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()

        model = EmptyModel()
        fqns = get_all_fqns(model)
        self.assertEqual(len(fqns), 0)


class TestMakeCannonicalNames(unittest.TestCase):
    """Test make_cannonical_names function."""

    def test_basic_sharing(self):
        """Should map the first name in the intersection to aliases."""
        fqns = {"embedding.weight", "lm_head.weight", "linear.weight"}
        sharing_metadata = [["embedding.weight", "lm_head.weight"]]

        cnames = make_cannonical_names(fqns, sharing_metadata)

        # The canonical name is the first element after intersection
        # which is either "embedding.weight" or "lm_head.weight" (set intersection order)
        self.assertEqual(len(cnames), 1)
        canonical = list(cnames.keys())[0]
        aliases = cnames[canonical]

        # Together they should form the full set
        all_names = {canonical} | set(aliases)
        self.assertEqual(all_names, {"embedding.weight", "lm_head.weight"})

    def test_no_overlap_with_fqns(self):
        """When sharing metadata names are not in fqns, should return empty."""
        fqns = {"linear.weight", "linear.bias"}
        sharing_metadata = [["embedding.weight", "lm_head.weight"]]

        cnames = make_cannonical_names(fqns, sharing_metadata)
        self.assertEqual(len(cnames), 0)

    def test_partial_overlap(self):
        """When only one name from sharing group is in fqns, single entry with no aliases."""
        fqns = {"embedding.weight", "linear.weight"}
        sharing_metadata = [["embedding.weight", "lm_head.weight"]]

        cnames = make_cannonical_names(fqns, sharing_metadata)

        # Only embedding.weight is in fqns, so it becomes canonical with no aliases
        self.assertEqual(len(cnames), 1)
        self.assertIn("embedding.weight", cnames)
        self.assertEqual(cnames["embedding.weight"], [])

    def test_empty_sharing_metadata(self):
        """Empty sharing metadata should return empty dict."""
        fqns = {"linear.weight", "linear.bias"}
        cnames = make_cannonical_names(fqns, [])
        self.assertEqual(len(cnames), 0)

    def test_multiple_sharing_groups(self):
        """Should handle multiple sharing groups."""
        fqns = {"a.weight", "b.weight", "c.weight", "d.weight"}
        sharing_metadata = [
            ["a.weight", "b.weight"],
            ["c.weight", "d.weight"],
        ]

        cnames = make_cannonical_names(fqns, sharing_metadata)
        self.assertEqual(len(cnames), 2)


class TestMapCannonicalNames(unittest.TestCase):
    """Test map_cannonical_names function."""

    def test_basic_inversion(self):
        """Should invert the canonical name mapping."""
        cnames = {
            "embedding.weight": ["lm_head.weight"],
        }
        cname_map = map_cannonical_names(cnames)

        self.assertEqual(cname_map, {"lm_head.weight": "embedding.weight"})

    def test_multiple_aliases(self):
        """Should map all aliases to their canonical name."""
        cnames = {
            "a.weight": ["b.weight", "c.weight"],
        }
        cname_map = map_cannonical_names(cnames)

        self.assertEqual(cname_map["b.weight"], "a.weight")
        self.assertEqual(cname_map["c.weight"], "a.weight")
        self.assertEqual(len(cname_map), 2)

    def test_no_aliases(self):
        """When there are no aliases, should return empty dict."""
        cnames = {"a.weight": []}
        cname_map = map_cannonical_names(cnames)
        self.assertEqual(len(cname_map), 0)

    def test_multiple_canonical_names(self):
        """Should handle multiple canonical name groups."""
        cnames = {
            "a.weight": ["b.weight"],
            "c.weight": ["d.weight"],
        }
        cname_map = map_cannonical_names(cnames)

        self.assertEqual(cname_map["b.weight"], "a.weight")
        self.assertEqual(cname_map["d.weight"], "c.weight")
        self.assertEqual(len(cname_map), 2)

    def test_empty_input(self):
        """Empty input should return empty dict."""
        cname_map = map_cannonical_names({})
        self.assertEqual(len(cname_map), 0)


class TestCreateSharingMetadata(unittest.TestCase):
    """Test create_sharing_metadata function."""

    def test_no_tied_parameters(self):
        """Model with no tied parameters should return empty list."""
        model = SimpleModel()
        metadata = create_sharing_metadata(model)
        self.assertEqual(metadata, [])

    def test_tied_parameters(self):
        """Model with tied parameters should return sharing groups."""
        model = TiedModel()
        metadata = create_sharing_metadata(model)

        self.assertEqual(len(metadata), 1)
        # The sharing group should contain both embedding.weight and lm_head.weight
        sharing_group = set(metadata[0])
        self.assertIn("embedding.weight", sharing_group)
        self.assertIn("lm_head.weight", sharing_group)

    def test_model_with_buffer_no_sharing(self):
        """Model with buffers but no sharing should return empty list."""
        model = ModelWithBuffer()
        metadata = create_sharing_metadata(model)
        self.assertEqual(metadata, [])


class TestIndexFileName(unittest.TestCase):
    """Test index_file_name function."""

    def test_safetensors_true(self):
        """Should return safetensors index file name."""
        result = index_file_name(safetensors=True)
        self.assertEqual(result, SAFE_WEIGHTS_INDEX_NAME)
        self.assertEqual(result, "model.safetensors.index.json")

    def test_safetensors_false(self):
        """Should return pytorch index file name."""
        result = index_file_name(safetensors=False)
        self.assertEqual(result, WEIGHTS_INDEX_NAME)
        self.assertEqual(result, "pytorch_model.bin.index.json")


class TestMakeShardIndex(unittest.TestCase):
    """Test make_shard_index function."""

    def test_single_state_dict(self):
        """Single state dict should produce a single shard."""
        state_dict = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        }
        index = make_shard_index([state_dict])

        self.assertIn("metadata", index)
        self.assertIn("weight_map", index)
        self.assertIn("total_size", index["metadata"])

        weight_map = index["weight_map"]
        self.assertIn("layer1.weight", weight_map)
        self.assertIn("layer1.bias", weight_map)

        # Both should be in the same shard (pytorch naming by default)
        shard_files = set(weight_map.values())
        self.assertEqual(len(shard_files), 1)
        self.assertTrue(list(shard_files)[0].startswith("pytorch_model-"))

    def test_safetensors_naming(self):
        """With safetensors=True, shard names should use safetensors convention."""
        state_dict = {"w": torch.randn(2, 2)}
        index = make_shard_index([state_dict], safetensors=True)

        weight_map = index["weight_map"]
        shard_file = list(weight_map.values())[0]
        self.assertTrue(shard_file.endswith(".safetensors"))

    def test_multiple_state_dicts(self):
        """Multiple state dicts should produce separate shards."""
        sd1 = {"layer1.weight": torch.randn(5, 5)}
        sd2 = {"layer2.weight": torch.randn(5, 5)}

        index = make_shard_index([sd1, sd2])
        weight_map = index["weight_map"]

        # Each state dict should map to a different shard file
        shard_for_l1 = weight_map["layer1.weight"]
        shard_for_l2 = weight_map["layer2.weight"]
        self.assertNotEqual(shard_for_l1, shard_for_l2)

    def test_max_shard_size_splits(self):
        """When a state dict exceeds max_shard_size, it should be split."""
        # Create a tensor that is at least a few bytes
        big_tensor = torch.randn(100, 100)  # ~40KB in float32
        small_tensor = torch.randn(2, 2)

        state_dict = {"big": big_tensor, "small": small_tensor}

        # Set a very small max_shard_size to force splitting
        index = make_shard_index([state_dict], max_shard_size=1)

        weight_map = index["weight_map"]
        shard_files = set(weight_map.values())
        # Each parameter should be in its own shard since max_shard_size=1
        self.assertEqual(len(shard_files), 2)

    def test_metadata_passthrough(self):
        """Custom metadata should be included in the index."""
        state_dict = {"w": torch.randn(2, 2)}
        metadata = {"dtype": "bfloat16", "custom_key": "custom_value"}

        index = make_shard_index([state_dict], metadata=metadata)

        self.assertEqual(index["metadata"]["dtype"], "bfloat16")
        self.assertEqual(index["metadata"]["custom_key"], "custom_value")
        self.assertIn("total_size", index["metadata"])

    def test_param_sharing_metadata(self):
        """Param sharing metadata should be included when provided."""
        state_dict = {"w": torch.randn(2, 2)}
        sharing = [["embedding.weight", "lm_head.weight"]]

        index = make_shard_index([state_dict], param_sharing_metadata=sharing)

        self.assertEqual(index["metadata"]["param_sharing"], sharing)

    def test_no_param_sharing_metadata(self):
        """When no sharing metadata is provided, it should not be in the index."""
        state_dict = {"w": torch.randn(2, 2)}
        index = make_shard_index([state_dict])
        self.assertNotIn("param_sharing", index["metadata"])

    def test_empty_state_dict(self):
        """An empty state dict should produce no shard entries."""
        index = make_shard_index([{}])
        self.assertEqual(len(index["weight_map"]), 0)
        self.assertEqual(index["metadata"]["total_size"], 0)

    def test_shard_numbering_format(self):
        """Shard files should be numbered with 5-digit zero-padded format."""
        sd1 = {"a": torch.randn(2)}
        sd2 = {"b": torch.randn(2)}
        sd3 = {"c": torch.randn(2)}

        index = make_shard_index([sd1, sd2, sd3])
        weight_map = index["weight_map"]

        shard_files = sorted(set(weight_map.values()))
        self.assertEqual(len(shard_files), 3)
        self.assertIn("00001-of-00003", shard_files[0])
        self.assertIn("00002-of-00003", shard_files[1])
        self.assertIn("00003-of-00003", shard_files[2])


class TestIntersectWeightMap(unittest.TestCase):
    """Test _intersect_weight_map function."""

    def test_basic_intersection(self):
        """Should return keys that exist in both weight_map and state_dict."""
        weight_map = {
            "layer1.weight": "shard1.bin",
            "layer2.weight": "shard2.bin",
            "layer3.weight": "shard2.bin",
        }
        state_dict = {
            "layer1.weight": torch.randn(5),
            "layer2.weight": torch.randn(5),
            "other.weight": torch.randn(5),
        }

        result = _intersect_weight_map(weight_map, state_dict)
        self.assertEqual(result, {"layer1.weight", "layer2.weight"})

    def test_no_intersection(self):
        """When there is no overlap, should return empty set."""
        weight_map = {"a": "shard1.bin"}
        state_dict = {"b": torch.randn(2)}

        result = _intersect_weight_map(weight_map, state_dict)
        self.assertEqual(result, set())

    def test_full_overlap(self):
        """When all keys overlap, should return all keys."""
        weight_map = {"a": "shard1.bin", "b": "shard1.bin"}
        state_dict = {"a": torch.randn(2), "b": torch.randn(2)}

        result = _intersect_weight_map(weight_map, state_dict)
        self.assertEqual(result, {"a", "b"})

    def test_empty_inputs(self):
        """Empty inputs should return empty set."""
        self.assertEqual(_intersect_weight_map({}, {}), set())
        self.assertEqual(_intersect_weight_map({"a": "s"}, {}), set())
        self.assertEqual(_intersect_weight_map({}, {"a": torch.randn(1)}), set())


class TestMakeShardDictionaries(unittest.TestCase):
    """Test _make_shard_dictionaries function."""

    def test_basic_grouping(self):
        """Should group tensors by their shard file."""
        t1 = torch.randn(3)
        t2 = torch.randn(4)
        t3 = torch.randn(5)

        weight_map = {
            "layer1.weight": "shard-00001.bin",
            "layer2.weight": "shard-00001.bin",
            "layer3.weight": "shard-00002.bin",
        }
        state_dict = {
            "layer1.weight": t1,
            "layer2.weight": t2,
            "layer3.weight": t3,
        }

        result = _make_shard_dictionaries(weight_map, state_dict)

        self.assertIn("shard-00001.bin", result)
        self.assertIn("shard-00002.bin", result)
        self.assertEqual(len(result["shard-00001.bin"]), 2)
        self.assertEqual(len(result["shard-00002.bin"]), 1)
        self.assertIs(result["shard-00001.bin"]["layer1.weight"], t1)
        self.assertIs(result["shard-00002.bin"]["layer3.weight"], t3)

    def test_partial_state_dict(self):
        """When state_dict has fewer keys than weight_map, only matching keys are grouped."""
        weight_map = {
            "layer1.weight": "shard1.bin",
            "layer2.weight": "shard2.bin",
        }
        t1 = torch.randn(3)
        state_dict = {"layer1.weight": t1}

        result = _make_shard_dictionaries(weight_map, state_dict)

        self.assertIn("shard1.bin", result)
        self.assertNotIn("shard2.bin", result)

    def test_empty_state_dict(self):
        """Empty state_dict should return empty dict."""
        weight_map = {"a": "shard1.bin"}
        result = _make_shard_dictionaries(weight_map, {})
        self.assertEqual(result, {})


class TestGetCheckpointMetadata(unittest.TestCase):
    """Test get_checkpoint_metadata function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_safetensors_index(self):
        """Should detect safetensors sharded checkpoint."""
        index_path = os.path.join(self.tmpdir, SAFE_WEIGHTS_INDEX_NAME)
        with open(index_path, "w") as f:
            json.dump({"metadata": {}, "weight_map": {}}, f)

        meta = get_checkpoint_metadata(self.tmpdir)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.file_name, SAFE_WEIGHTS_INDEX_NAME)
        self.assertTrue(meta.is_index)
        self.assertTrue(meta.safetensors)

    def test_torch_index(self):
        """Should detect pytorch sharded checkpoint."""
        index_path = os.path.join(self.tmpdir, WEIGHTS_INDEX_NAME)
        with open(index_path, "w") as f:
            json.dump({"metadata": {}, "weight_map": {}}, f)

        meta = get_checkpoint_metadata(self.tmpdir)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.file_name, WEIGHTS_INDEX_NAME)
        self.assertTrue(meta.is_index)
        self.assertFalse(meta.safetensors)

    def test_single_pytorch_file(self):
        """Should detect single pytorch model file."""
        weights_path = os.path.join(self.tmpdir, WEIGHTS_NAME)
        torch.save({"w": torch.randn(2)}, weights_path)

        meta = get_checkpoint_metadata(self.tmpdir)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.file_name, WEIGHTS_NAME)
        self.assertFalse(meta.is_index)
        self.assertFalse(meta.safetensors)

    def test_single_safetensors_file(self):
        """Should detect single safetensors model file."""
        from safetensors.torch import save_file

        weights_path = os.path.join(self.tmpdir, SAFE_WEIGHTS_NAME)
        save_file({"w": torch.randn(2)}, weights_path)

        meta = get_checkpoint_metadata(self.tmpdir)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.file_name, SAFE_WEIGHTS_NAME)
        self.assertFalse(meta.is_index)
        self.assertTrue(meta.safetensors)

    def test_no_checkpoint(self):
        """Should return None when no checkpoint files are found."""
        meta = get_checkpoint_metadata(self.tmpdir)
        self.assertIsNone(meta)

    def test_priority_safetensors_index_over_torch(self):
        """Safetensors index should be preferred over pytorch index."""
        # Create both
        with open(os.path.join(self.tmpdir, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
            json.dump({}, f)
        with open(os.path.join(self.tmpdir, WEIGHTS_INDEX_NAME), "w") as f:
            json.dump({}, f)

        meta = get_checkpoint_metadata(self.tmpdir)
        self.assertEqual(meta.file_name, SAFE_WEIGHTS_INDEX_NAME)

    def test_priority_torch_index_over_single_files(self):
        """Torch index should be preferred over single weight files."""
        with open(os.path.join(self.tmpdir, WEIGHTS_INDEX_NAME), "w") as f:
            json.dump({}, f)
        torch.save({"w": torch.randn(2)}, os.path.join(self.tmpdir, WEIGHTS_NAME))

        meta = get_checkpoint_metadata(self.tmpdir)
        self.assertTrue(meta.is_index)


class TestNextCheckpointPath(unittest.TestCase):
    """Test next_checkpoint_path function."""

    def test_integer_checkpoint_id(self):
        """Should construct path with integer checkpoint id."""
        path = next_checkpoint_path("/models/my_model", 1000)
        self.assertEqual(path, "/models/my_model/checkpoints/checkpoint-1000")

    def test_string_checkpoint_id(self):
        """Should construct path with string checkpoint id."""
        path = next_checkpoint_path("/models/my_model", "step_500")
        self.assertEqual(path, "/models/my_model/checkpoints/checkpoint-step_500")

    def test_zero_checkpoint_id(self):
        """Should handle zero checkpoint id."""
        path = next_checkpoint_path("/models/my_model", 0)
        self.assertEqual(path, "/models/my_model/checkpoints/checkpoint-0")


class TestSaveAndLoadCheckpointMetrics(unittest.TestCase):
    """Test save_checkpoint_metrics and load_checkpoint_metrics functions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip(self):
        """Should save and load metrics correctly."""
        metrics = {"loss": 0.5, "accuracy": 0.95, "perplexity": 12.3}
        save_checkpoint_metrics(self.tmpdir, metrics)
        loaded = load_checkpoint_metrics(self.tmpdir)

        self.assertEqual(loaded, metrics)

    def test_load_missing_file_returns_none(self):
        """Should return None when metrics file does not exist."""
        loaded = load_checkpoint_metrics(self.tmpdir)
        self.assertIsNone(loaded)

    def test_creates_directory(self):
        """Should create checkpoint directory if it does not exist."""
        new_dir = os.path.join(self.tmpdir, "new_checkpoint")
        metrics = {"loss": 1.0}
        save_checkpoint_metrics(new_dir, metrics)

        self.assertTrue(os.path.isdir(new_dir))
        self.assertTrue(os.path.isfile(os.path.join(new_dir, "eval_metrics.json")))

    def test_empty_metrics(self):
        """Should handle empty metrics dict."""
        save_checkpoint_metrics(self.tmpdir, {})
        loaded = load_checkpoint_metrics(self.tmpdir)
        self.assertEqual(loaded, {})

    def test_overwrites_existing_metrics(self):
        """Should overwrite existing metrics file."""
        save_checkpoint_metrics(self.tmpdir, {"loss": 1.0})
        save_checkpoint_metrics(self.tmpdir, {"loss": 0.5})
        loaded = load_checkpoint_metrics(self.tmpdir)
        self.assertEqual(loaded["loss"], 0.5)

    def test_metrics_file_is_valid_json(self):
        """Saved file should be valid JSON with proper formatting."""
        metrics = {"loss": 0.123}
        save_checkpoint_metrics(self.tmpdir, metrics)

        metrics_path = os.path.join(self.tmpdir, "eval_metrics.json")
        with open(metrics_path, "r") as f:
            content = f.read()
        # Should be indented (indent=2)
        self.assertIn("\n", content)
        parsed = json.loads(content)
        self.assertEqual(parsed, metrics)


class TestMaybeDeleteOldestCheckpoint(unittest.TestCase):
    """Test maybe_delete_oldest_checkpoint function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoints_dir = os.path.join(self.tmpdir, "checkpoints")
        os.makedirs(self.checkpoints_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_checkpoint(self, step, delay=0.05):
        """Helper to create a mock checkpoint directory."""
        if delay > 0:
            time.sleep(delay)
        path = os.path.join(self.checkpoints_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        # Write a file so we can verify deletion
        with open(os.path.join(path, "data.txt"), "w") as f:
            f.write(f"step-{step}")
        return path

    def test_no_deletion_under_limit(self):
        """When checkpoint count <= max, nothing should be deleted."""
        cp1 = self._create_checkpoint(100)
        cp2 = self._create_checkpoint(200)

        maybe_delete_oldest_checkpoint(self.tmpdir, max_checkpoints=3)

        self.assertTrue(os.path.exists(cp1))
        self.assertTrue(os.path.exists(cp2))

    def test_no_deletion_at_limit(self):
        """When checkpoint count == max, nothing should be deleted."""
        cp1 = self._create_checkpoint(100)
        cp2 = self._create_checkpoint(200)
        cp3 = self._create_checkpoint(300)

        maybe_delete_oldest_checkpoint(self.tmpdir, max_checkpoints=3)

        self.assertTrue(os.path.exists(cp1))
        self.assertTrue(os.path.exists(cp2))
        self.assertTrue(os.path.exists(cp3))

    def test_deletes_oldest_by_mtime(self):
        """Should delete the oldest checkpoint by modification time."""
        cp1 = self._create_checkpoint(100)  # Oldest
        cp2 = self._create_checkpoint(200)
        cp3 = self._create_checkpoint(300)
        cp4 = self._create_checkpoint(400)  # Newest

        maybe_delete_oldest_checkpoint(self.tmpdir, max_checkpoints=3)

        # cp1 (oldest by mtime) should be deleted
        self.assertFalse(os.path.exists(cp1))
        self.assertTrue(os.path.exists(cp2))
        self.assertTrue(os.path.exists(cp3))
        self.assertTrue(os.path.exists(cp4))

    def test_preserves_best_checkpoint(self):
        """Should not delete the best checkpoint even if it is the oldest."""
        cp1 = self._create_checkpoint(100)  # Oldest, but best
        cp2 = self._create_checkpoint(200)
        cp3 = self._create_checkpoint(300)
        cp4 = self._create_checkpoint(400)

        maybe_delete_oldest_checkpoint(
            self.tmpdir, max_checkpoints=3, best_checkpoint=cp1
        )

        # cp1 should be preserved (it is the best)
        self.assertTrue(os.path.exists(cp1))
        # cp2 should be deleted instead (next oldest)
        self.assertFalse(os.path.exists(cp2))
        self.assertTrue(os.path.exists(cp3))
        self.assertTrue(os.path.exists(cp4))

    def test_preserves_multiple_checkpoints(self):
        """Should preserve all checkpoints in preserved_checkpoints list.

        With 5 checkpoints and max=3, num_to_delete = 5-3 = 2.
        Non-preserved candidates are cp3, cp4, cp5 (sorted by mtime).
        The 2 oldest non-preserved (cp3, cp4) are deleted.
        """
        cp1 = self._create_checkpoint(100)  # Oldest, preserved
        cp2 = self._create_checkpoint(200)  # Second oldest, preserved
        cp3 = self._create_checkpoint(300)
        cp4 = self._create_checkpoint(400)
        cp5 = self._create_checkpoint(500)

        maybe_delete_oldest_checkpoint(
            self.tmpdir, max_checkpoints=3, preserved_checkpoints=[cp1, cp2]
        )

        # cp1 and cp2 are preserved
        self.assertTrue(os.path.exists(cp1))
        self.assertTrue(os.path.exists(cp2))
        # cp3 and cp4 are the two oldest non-preserved, so both are deleted
        self.assertFalse(os.path.exists(cp3))
        self.assertFalse(os.path.exists(cp4))
        self.assertTrue(os.path.exists(cp5))

    def test_no_checkpoints_dir(self):
        """Should handle missing checkpoints directory gracefully."""
        empty_dir = tempfile.mkdtemp()
        try:
            # Should not raise
            maybe_delete_oldest_checkpoint(empty_dir, max_checkpoints=2)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_deletes_multiple_when_far_over_limit(self):
        """Should delete multiple checkpoints when far over the limit."""
        cp1 = self._create_checkpoint(100)
        cp2 = self._create_checkpoint(200)
        cp3 = self._create_checkpoint(300)
        cp4 = self._create_checkpoint(400)
        cp5 = self._create_checkpoint(500)

        maybe_delete_oldest_checkpoint(self.tmpdir, max_checkpoints=2)

        # Oldest 3 should be deleted
        self.assertFalse(os.path.exists(cp1))
        self.assertFalse(os.path.exists(cp2))
        self.assertFalse(os.path.exists(cp3))
        self.assertTrue(os.path.exists(cp4))
        self.assertTrue(os.path.exists(cp5))

    def test_best_checkpoint_deprecated_param(self):
        """best_checkpoint (deprecated) should still work alongside preserved_checkpoints.

        With 4 checkpoints and max=2, num_to_delete = 4-2 = 2.
        Non-preserved candidates are cp3, cp4 (sorted by mtime).
        Both are deleted since num_to_delete=2.
        """
        cp1 = self._create_checkpoint(100)
        cp2 = self._create_checkpoint(200)
        cp3 = self._create_checkpoint(300)
        cp4 = self._create_checkpoint(400)

        maybe_delete_oldest_checkpoint(
            self.tmpdir,
            max_checkpoints=2,
            best_checkpoint=cp1,
            preserved_checkpoints=[cp2],
        )

        # Both cp1 (best_checkpoint) and cp2 (preserved) should survive
        self.assertTrue(os.path.exists(cp1))
        self.assertTrue(os.path.exists(cp2))
        # Both non-preserved checkpoints are deleted (num_to_delete=2)
        self.assertFalse(os.path.exists(cp3))
        self.assertFalse(os.path.exists(cp4))

    def _create_checkpoint_with_manifest(self, step, timestamp, delay=0.05):
        """Create a checkpoint with a manifest containing the given timestamp."""
        path = self._create_checkpoint(step, delay=delay)
        manifest = {"timestamp": timestamp.isoformat()}
        with open(os.path.join(path, CHECKPOINT_MANIFEST_FILENAME), "w") as f:
            json.dump(manifest, f)
        return path

    def test_deletes_oldest_by_manifest_timestamp(self):
        """Should delete based on manifest timestamp, not mtime."""
        # Create checkpoints in filesystem order 100, 200, 300, 400
        # but with manifest timestamps that reverse 200 and 300
        t1 = datetime(2026, 1, 1, 10, 0, 0)
        t2 = datetime(2026, 1, 1, 12, 0, 0)  # Newer than t3 by manifest
        t3 = datetime(2026, 1, 1, 11, 0, 0)  # Older than t2 by manifest
        t4 = datetime(2026, 1, 1, 13, 0, 0)

        cp1 = self._create_checkpoint_with_manifest(100, t1)
        cp2 = self._create_checkpoint_with_manifest(200, t2)
        cp3 = self._create_checkpoint_with_manifest(300, t3)
        cp4 = self._create_checkpoint_with_manifest(400, t4)

        maybe_delete_oldest_checkpoint(self.tmpdir, max_checkpoints=3)

        # cp1 has the oldest manifest timestamp, should be deleted
        self.assertFalse(os.path.exists(cp1))
        self.assertTrue(os.path.exists(cp2))
        self.assertTrue(os.path.exists(cp3))
        self.assertTrue(os.path.exists(cp4))

    def test_mixed_manifest_and_legacy_checkpoints(self):
        """Checkpoints with and without manifests should be orderable together."""
        # cp1: manifest with the oldest timestamp
        cp1 = self._create_checkpoint_with_manifest(100, datetime(2020, 1, 1, 10, 0, 0))
        # cp2: legacy (no manifest), uses mtime (current time, between the manifest times)
        cp2 = self._create_checkpoint(200)
        # cp3: manifest with the newest timestamp (far in the future)
        cp3 = self._create_checkpoint_with_manifest(300, datetime(2099, 1, 1, 12, 0, 0))
        # cp4: legacy (no manifest), uses mtime (current time, slightly after cp2)
        cp4 = self._create_checkpoint(400)

        maybe_delete_oldest_checkpoint(self.tmpdir, max_checkpoints=3)

        # cp1 has the oldest effective timestamp (2020 manifest), should be deleted
        self.assertFalse(os.path.exists(cp1))
        self.assertTrue(os.path.exists(cp2))
        self.assertTrue(os.path.exists(cp3))
        self.assertTrue(os.path.exists(cp4))


class TestGetCheckpointTimestamp(unittest.TestCase):
    """Test _get_checkpoint_timestamp helper function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.tmpdir, "checkpoint-100")
        os.makedirs(self.checkpoint_path)
        # Write a file so the directory has content
        with open(os.path.join(self.checkpoint_path, "data.txt"), "w") as f:
            f.write("test")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_manifest(self, data):
        manifest_path = os.path.join(self.checkpoint_path, CHECKPOINT_MANIFEST_FILENAME)
        with open(manifest_path, "w") as f:
            json.dump(data, f)

    def test_timestamp_from_manifest(self):
        """Should return manifest timestamp when available."""
        ts = datetime(2026, 3, 15, 14, 30, 0)
        self._write_manifest({"timestamp": ts.isoformat()})

        result = _get_checkpoint_timestamp(self.checkpoint_path)
        self.assertAlmostEqual(result, ts.timestamp(), places=0)

    def test_timestamp_fallback_to_mtime(self):
        """Should return mtime when no manifest exists."""
        result = _get_checkpoint_timestamp(self.checkpoint_path)
        expected = os.path.getmtime(self.checkpoint_path)
        self.assertEqual(result, expected)

    def test_timestamp_fallback_on_corrupt_json(self):
        """Should fall back to mtime when manifest JSON is corrupt."""
        manifest_path = os.path.join(self.checkpoint_path, CHECKPOINT_MANIFEST_FILENAME)
        with open(manifest_path, "w") as f:
            f.write("not valid json{{{")

        result = _get_checkpoint_timestamp(self.checkpoint_path)
        expected = os.path.getmtime(self.checkpoint_path)
        self.assertEqual(result, expected)

    def test_timestamp_fallback_on_missing_key(self):
        """Should fall back to mtime when manifest lacks timestamp key."""
        self._write_manifest({"world_size": 4})

        result = _get_checkpoint_timestamp(self.checkpoint_path)
        expected = os.path.getmtime(self.checkpoint_path)
        self.assertEqual(result, expected)

    def test_timestamp_fallback_on_bad_timestamp_format(self):
        """Should fall back to mtime when timestamp string is unparseable."""
        self._write_manifest({"timestamp": "not-a-date"})

        result = _get_checkpoint_timestamp(self.checkpoint_path)
        expected = os.path.getmtime(self.checkpoint_path)
        self.assertEqual(result, expected)

    def test_manifest_timestamp_stable_after_copy(self):
        """Core regression test: copied checkpoint should have same effective timestamp."""
        ts = datetime(2026, 3, 15, 14, 30, 0)
        self._write_manifest({"timestamp": ts.isoformat()})

        # Copy the checkpoint directory
        copy_path = os.path.join(self.tmpdir, "checkpoint-100-copy")
        shutil.copytree(self.checkpoint_path, copy_path)

        # Manifest timestamps should be identical
        orig_ts = _get_checkpoint_timestamp(self.checkpoint_path)
        copy_ts = _get_checkpoint_timestamp(copy_path)
        self.assertEqual(orig_ts, copy_ts)

        # But mtime should differ (copy is newer)
        orig_mtime = os.path.getmtime(self.checkpoint_path)
        copy_mtime = os.path.getmtime(copy_path)
        self.assertGreaterEqual(copy_mtime, orig_mtime)


class TestResolveStateDict(unittest.TestCase):
    """Test _resolve_state_dict helper."""

    def test_module_input(self):
        """Should return state_dict from nn.Module."""
        model = SimpleModel()
        sd = _resolve_state_dict(model)
        self.assertIsInstance(sd, dict)
        self.assertIn("linear1.weight", sd)

    def test_dict_input(self):
        """Should return the dict as-is."""
        d = {"w": torch.randn(3)}
        result = _resolve_state_dict(d)
        self.assertIs(result, d)


class TestSaveLoadCheckpointWithDict(unittest.TestCase):
    """Test save_checkpoint and load_checkpoint with raw state dicts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_state_dict(self):
        return {
            "layer1.weight": torch.randn(4, 3),
            "layer1.bias": torch.randn(4),
            "layer2.weight": torch.randn(2, 4),
            "layer2.bias": torch.randn(2),
        }

    def test_save_dict_load_dict_pytorch(self):
        """Round-trip: save raw dict, load as dict (pytorch format)."""
        original = self._make_state_dict()
        save_checkpoint(self.tmpdir, original, safetensors=False)
        loaded = load_checkpoint(self.tmpdir, module=None, device="cpu")

        self.assertIsInstance(loaded, dict)
        for key in original:
            self.assertIn(key, loaded)
            torch.testing.assert_close(loaded[key], original[key])

    def test_save_dict_load_dict_safetensors(self):
        """Round-trip: save raw dict, load as dict (safetensors format)."""
        original = self._make_state_dict()
        save_checkpoint(self.tmpdir, original, safetensors=True)
        loaded = load_checkpoint(self.tmpdir, module=None, device="cpu")

        self.assertIsInstance(loaded, dict)
        for key in original:
            self.assertIn(key, loaded)
            torch.testing.assert_close(loaded[key], original[key])

    def test_save_dict_load_module(self):
        """Save raw dict, load into nn.Module."""
        model = SimpleModel()
        original_sd = model.state_dict()

        # Save the state dict directly
        save_checkpoint(self.tmpdir, original_sd, safetensors=False)

        # Create fresh model and load
        model2 = SimpleModel()
        load_checkpoint(self.tmpdir, model2, device="cpu")

        for key in original_sd:
            torch.testing.assert_close(model2.state_dict()[key], original_sd[key])

    def test_save_module_load_dict(self):
        """Save nn.Module, load as raw dict."""
        model = SimpleModel()
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        save_checkpoint(self.tmpdir, model, safetensors=False)
        loaded = load_checkpoint(self.tmpdir, module=None, device="cpu")

        self.assertIsInstance(loaded, dict)
        for key in original_sd:
            torch.testing.assert_close(loaded[key], original_sd[key])

    def test_load_dict_with_keys_filter(self):
        """Load only specific keys when module=None."""
        original = self._make_state_dict()
        save_checkpoint(self.tmpdir, original, safetensors=False)

        wanted_keys = {"layer1.weight", "layer1.bias"}
        loaded = load_checkpoint(
            self.tmpdir, module=None, device="cpu", keys=wanted_keys
        )

        self.assertEqual(set(loaded.keys()), wanted_keys)
        for key in wanted_keys:
            torch.testing.assert_close(loaded[key], original[key])

    def test_save_dict_no_param_sharing_autodetect(self):
        """When saving a dict, param sharing auto-detection is skipped."""
        sd = self._make_state_dict()
        # Should not raise even though dict has no parameter sharing info
        save_checkpoint(self.tmpdir, sd, safetensors=False)
        loaded = load_checkpoint(self.tmpdir, module=None, device="cpu")
        self.assertEqual(set(loaded.keys()), set(sd.keys()))

    def test_save_dict_with_explicit_param_sharing(self):
        """Can provide explicit param_sharing_metadata with a dict."""
        sd = {"embed.weight": torch.randn(10, 4), "head.weight": torch.randn(10, 4)}
        sharing = [["embed.weight", "head.weight"]]
        save_checkpoint(
            self.tmpdir, sd, safetensors=False, param_sharing_metadata=sharing
        )
        loaded = load_checkpoint(self.tmpdir, module=None, device="cpu")
        self.assertIn("embed.weight", loaded)
        self.assertIn("head.weight", loaded)


class TestSaveShardedCheckpointWithDict(unittest.TestCase):
    """Test save_sharded_checkpoint with raw state dicts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_dict_pytorch(self):
        """save_sharded_checkpoint with raw dict (pytorch format)."""
        sd = {"a.weight": torch.randn(5, 3), "b.weight": torch.randn(2, 5)}
        shard_index = make_shard_index([sd], safetensors=False)
        save_sharded_checkpoint(self.tmpdir, shard_index, sd, safetensors=False)

        # Verify shard files were written
        weight_map = shard_index["weight_map"]
        for shard_file in set(weight_map.values()):
            self.assertTrue(os.path.exists(os.path.join(self.tmpdir, shard_file)))

    def test_save_dict_safetensors(self):
        """save_sharded_checkpoint with raw dict (safetensors format)."""
        sd = {"a.weight": torch.randn(5, 3), "b.weight": torch.randn(2, 5)}
        shard_index = make_shard_index([sd], safetensors=True)
        save_sharded_checkpoint(self.tmpdir, shard_index, sd, safetensors=True)

        weight_map = shard_index["weight_map"]
        for shard_file in set(weight_map.values()):
            self.assertTrue(os.path.exists(os.path.join(self.tmpdir, shard_file)))


class TestLoadShardedCheckpointWithDict(unittest.TestCase):
    """Test load_sharded_checkpoint with module=None."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _save_sharded(self, sd, safetensors=False):
        shard_index = make_shard_index([sd], safetensors=safetensors)
        from forgather.ml.sharded_checkpoint import index_file_name, save_shard_index

        save_shard_index(
            shard_index,
            self.tmpdir,
            index_file_name(safetensors),
        )
        save_sharded_checkpoint(self.tmpdir, shard_index, sd, safetensors=safetensors)
        return shard_index

    def test_load_all_keys_pytorch(self):
        """load_sharded_checkpoint with module=None returns all tensors (pytorch)."""
        original = {"x": torch.randn(3, 2), "y": torch.randn(4)}
        shard_index = self._save_sharded(original, safetensors=False)

        loaded = load_sharded_checkpoint(
            self.tmpdir, shard_index, module=None, device="cpu"
        )

        self.assertIsInstance(loaded, dict)
        for key in original:
            torch.testing.assert_close(loaded[key], original[key])

    def test_load_all_keys_safetensors(self):
        """load_sharded_checkpoint with module=None returns all tensors (safetensors)."""
        original = {"x": torch.randn(3, 2), "y": torch.randn(4)}
        shard_index = self._save_sharded(original, safetensors=True)

        loaded = load_sharded_checkpoint(
            self.tmpdir, shard_index, module=None, device="cpu", safetensors=True
        )

        self.assertIsInstance(loaded, dict)
        for key in original:
            torch.testing.assert_close(loaded[key], original[key])

    def test_load_with_keys_filter(self):
        """load_sharded_checkpoint with module=None and keys filters results."""
        original = {"a": torch.randn(3), "b": torch.randn(4), "c": torch.randn(5)}
        shard_index = self._save_sharded(original, safetensors=False)

        wanted = {"a", "c"}
        loaded = load_sharded_checkpoint(
            self.tmpdir, shard_index, module=None, device="cpu", keys=wanted
        )

        self.assertEqual(set(loaded.keys()), wanted)
        for key in wanted:
            torch.testing.assert_close(loaded[key], original[key])

    def test_load_module_still_works(self):
        """Existing module-based loading still works after refactor."""
        model = SimpleModel()
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        shard_index = self._save_sharded(original_sd, safetensors=False)

        model2 = SimpleModel()
        unloaded = load_sharded_checkpoint(
            self.tmpdir, shard_index, module=model2, device="cpu"
        )

        self.assertIsInstance(unloaded, set)
        self.assertEqual(len(unloaded), 0)
        for key in original_sd:
            torch.testing.assert_close(model2.state_dict()[key], original_sd[key])


class TestLoadCheckpointNonShardedDict(unittest.TestCase):
    """Test load_checkpoint with module=None for non-sharded checkpoints."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_single_pytorch_file(self):
        """load_checkpoint with module=None for single pytorch file."""
        original = {"w": torch.randn(3, 2), "b": torch.randn(3)}
        torch.save(original, os.path.join(self.tmpdir, "pytorch_model.bin"))

        loaded = load_checkpoint(self.tmpdir, module=None, device="cpu")
        self.assertIsInstance(loaded, dict)
        for key in original:
            torch.testing.assert_close(loaded[key], original[key])

    def test_single_safetensors_file(self):
        """load_checkpoint with module=None for single safetensors file."""
        from safetensors.torch import save_file

        original = {"w": torch.randn(3, 2), "b": torch.randn(3)}
        save_file(original, os.path.join(self.tmpdir, "model.safetensors"))

        loaded = load_checkpoint(self.tmpdir, module=None, device="cpu")
        self.assertIsInstance(loaded, dict)
        for key in original:
            torch.testing.assert_close(loaded[key], original[key])

    def test_single_file_with_keys_filter(self):
        """load_checkpoint with module=None and keys for single file."""
        original = {"w": torch.randn(3, 2), "b": torch.randn(3)}
        torch.save(original, os.path.join(self.tmpdir, "pytorch_model.bin"))

        loaded = load_checkpoint(self.tmpdir, module=None, device="cpu", keys={"w"})
        self.assertEqual(set(loaded.keys()), {"w"})
        torch.testing.assert_close(loaded["w"], original["w"])


class TestSafetensorsTiedWeights(unittest.TestCase):
    """Tests for safetensors behaviour with tied (shared) weights."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_safetensors_rejects_tied_weights(self):
        """safetensors_save raises RuntimeError when tensors share storage."""
        from safetensors.torch import save_file

        model = TiedModel()
        state_dict = model.state_dict()
        path = os.path.join(self.tmpdir, "model.safetensors")
        with self.assertRaises(RuntimeError):
            save_file(state_dict, path)

    def test_save_checkpoint_safetensors_tied_weights(self):
        """save_checkpoint with safetensors=True and tied weights should fail."""
        model = TiedModel()
        with self.assertRaises(RuntimeError):
            save_checkpoint(self.tmpdir, model, safetensors=True)

    def test_round_trip_pytorch_preserves_ties(self):
        """Round-trip through PyTorch format preserves tied weight identity."""
        model = TiedModel()
        save_checkpoint(self.tmpdir, model, safetensors=False)

        model2 = TiedModel()
        load_checkpoint(self.tmpdir, model2, device="cpu")

        # After loading with assign=False into a model that already has ties,
        # the parameters should still share the same object.
        self.assertIs(model2.lm_head.weight, model2.embedding.weight)

    def test_safetensors_dedup_round_trip_with_tie_weights(self):
        """Loading deduplicated safetensors and calling tie_weights() restores ties.

        This simulates loading an HF checkpoint where tied weights were
        deduplicated before saving to safetensors.
        """
        from safetensors.torch import save_file

        model = TiedModel()

        # Simulate HF-style deduplication: save only one copy
        state_dict = model.state_dict()
        deduped = {k: v for k, v in state_dict.items() if k != "lm_head.weight"}
        path = os.path.join(self.tmpdir, "model.safetensors")
        save_file(deduped, path)

        # Build a fresh model and load the deduped checkpoint (non-strict
        # because lm_head.weight is missing from the file).
        model2 = TiedModel()
        from safetensors.torch import load_file

        loaded = load_file(path, device="cpu")
        model2.load_state_dict(loaded, strict=False)

        # Before tie_weights: the lm_head.weight was not updated from the
        # checkpoint (it retains its random init value from construction), so
        # it may differ from embedding.weight.

        # If the model is a PreTrainedModel, tie_weights() would fix this.
        # For our plain nn.Module test fixture, manually tie:
        model2.lm_head.weight = model2.embedding.weight
        self.assertIs(model2.lm_head.weight, model2.embedding.weight)
        torch.testing.assert_close(model2.embedding.weight, model.embedding.weight)


class TestLoadCheckpointWithTiedWeights(unittest.TestCase):
    """load_checkpoint must accept HF-style deduplicated tied weights.

    Regression: Gemma3 and other HF models with tied embedding/lm_head store
    only ``model.embed_tokens.weight`` in safetensors; ``lm_head.weight`` is
    omitted. Strict-mode load_state_dict would reject this without the
    tied-alias synthesis in ``load_checkpoint``.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_deduped_safetensors(self, model, omit_key):
        from safetensors.torch import save_file

        state_dict = {k: v for k, v in model.state_dict().items() if k != omit_key}
        path = os.path.join(self.tmpdir, SAFE_WEIGHTS_NAME)
        save_file(state_dict, path)

    def test_single_file_safetensors_strict_load(self):
        """Single-file safetensors with deduped tied weight loads under strict=True."""
        source = TiedModel()
        with torch.no_grad():
            source.embedding.weight.normal_()
        self._write_deduped_safetensors(source, omit_key="lm_head.weight")

        target = TiedModel()
        # Sanity: target's lm_head.weight starts different from source.
        self.assertFalse(
            torch.allclose(target.embedding.weight, source.embedding.weight)
        )

        load_checkpoint(self.tmpdir, target, device="cpu", strict=True)

        torch.testing.assert_close(target.embedding.weight, source.embedding.weight)
        # Tied storage is preserved through copy_-mode load_state_dict.
        self.assertIs(target.lm_head.weight, target.embedding.weight)

    def test_sharded_safetensors_strict_load(self):
        """Sharded safetensors with deduped tied weight loads under strict=True."""
        from safetensors.torch import save_file

        source = TiedModel()
        with torch.no_grad():
            source.embedding.weight.normal_()

        # Write an HF-style sharded index where lm_head.weight is omitted.
        deduped = {k: v for k, v in source.state_dict().items() if k != "lm_head.weight"}
        index = make_shard_index([deduped], safetensors=True)
        with open(os.path.join(self.tmpdir, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
            json.dump(index, f)
        shard_file = next(iter(set(index["weight_map"].values())))
        save_file(deduped, os.path.join(self.tmpdir, shard_file))

        target = TiedModel()
        load_checkpoint(self.tmpdir, target, device="cpu", strict=True)

        torch.testing.assert_close(target.embedding.weight, source.embedding.weight)
        self.assertIs(target.lm_head.weight, target.embedding.weight)

    def test_genuinely_missing_key_still_fails(self):
        """Untied keys missing from the checkpoint must still trigger strict-mode failure."""
        from safetensors.torch import save_file

        # Save a checkpoint with only one of SimpleModel's four params.
        sd = {"linear1.weight": torch.randn(5, 10)}
        save_file(sd, os.path.join(self.tmpdir, SAFE_WEIGHTS_NAME))

        target = SimpleModel()
        with self.assertRaises(Exception):
            load_checkpoint(self.tmpdir, target, device="cpu", strict=True)

    def test_sharded_genuinely_missing_key_still_fails(self):
        """Sharded variant: untied missing keys must still raise under strict."""
        from safetensors.torch import save_file

        sd = {"linear1.weight": torch.randn(5, 10)}
        index = make_shard_index([sd], safetensors=True)
        with open(os.path.join(self.tmpdir, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
            json.dump(index, f)
        shard_file = next(iter(set(index["weight_map"].values())))
        save_file(sd, os.path.join(self.tmpdir, shard_file))

        target = SimpleModel()
        with self.assertRaises(Exception):
            load_checkpoint(self.tmpdir, target, device="cpu", strict=True)

    def test_assign_true_breaks_identity_recoverable_by_retie(self):
        """assign=True load: aliases share storage but lose ``is``-identity; manual retie restores it.

        Documents the contract: callers that use ``assign=True`` against a
        tied-embedding checkpoint must invoke ``tie_weights()`` afterward
        before any code path that depends on Parameter identity (e.g.
        save_checkpoint with safetensors=True).
        """
        from safetensors.torch import save_file

        source = TiedModel()
        with torch.no_grad():
            source.embedding.weight.normal_()
        self._write_deduped_safetensors(source, omit_key="lm_head.weight")

        target = TiedModel()
        load_checkpoint(self.tmpdir, target, device="cpu", strict=True, assign=True)

        # Data is correct via shared storage.
        torch.testing.assert_close(target.embedding.weight, source.embedding.weight)
        torch.testing.assert_close(target.lm_head.weight, source.embedding.weight)
        self.assertEqual(
            target.lm_head.weight.data_ptr(),
            target.embedding.weight.data_ptr(),
            "assign=True synthesis aliases must share underlying storage",
        )
        # But Parameter identity is broken (the assign=True contract).
        self.assertIsNot(target.lm_head.weight, target.embedding.weight)

        # Manual re-tie (what the trainer does post-load) restores identity.
        target.lm_head.weight = target.embedding.weight
        self.assertIs(target.lm_head.weight, target.embedding.weight)


if __name__ == "__main__":
    unittest.main()
