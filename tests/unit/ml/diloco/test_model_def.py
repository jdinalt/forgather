"""Tests for the DiLoCo model-definition bundle policy and pack/extract.

Covers the include/exclude rules (weights and server state out; config,
custom code, and tokenizer in), deterministic hashing, the two-file custom
model case (split config + modeling .py), and traversal-safe extraction.
"""

import io
import os
import tarfile

import pytest

from forgather.ml.diloco import model_def as md


def _write(path, content="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _make_checkpoint(root):
    """A self-contained model dir with a SPLIT two-file custom definition."""
    _write(os.path.join(root, "config.json"), '{"hidden_size": 8}')
    _write(os.path.join(root, "generation_config.json"), "{}")
    _write(os.path.join(root, "configuration_demo.py"), "class Cfg: pass")
    _write(os.path.join(root, "modeling_demo.py"), "class Model: pass")
    _write(os.path.join(root, "tokenizer.json"), "{}")
    _write(os.path.join(root, "tokenizer_config.json"), "{}")
    _write(os.path.join(root, "special_tokens_map.json"), "{}")
    # Excluded artifacts:
    _write(os.path.join(root, "model.safetensors"), "WEIGHTS")
    _write(os.path.join(root, "pytorch_model.bin"), "WEIGHTS")
    _write(os.path.join(root, "model.safetensors.index.json"), "{}")
    _write(os.path.join(root, "server_state.pt"), "STATE")
    _write(os.path.join(root, "diloco_audit.log"), "audit")
    # Excluded nested rollout checkpoint:
    _write(os.path.join(root, "checkpoint-5", "config.json"), "nested")
    _write(os.path.join(root, "checkpoint-5", "model.safetensors"), "W")


def test_enumerate_includes_definition_excludes_weights(tmp_path):
    root = str(tmp_path)
    _make_checkpoint(root)
    arcnames = [arc for _, arc in md.enumerate_model_def_files(root)]
    assert arcnames == [
        "config.json",
        "configuration_demo.py",
        "generation_config.json",
        "modeling_demo.py",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    # Both halves of a split custom definition ride along.
    assert "configuration_demo.py" in arcnames
    assert "modeling_demo.py" in arcnames
    # Weights, indices, server state, audit, and nested checkpoints excluded.
    for bad in (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "server_state.pt",
        "diloco_audit.log",
    ):
        assert bad not in arcnames
    assert not any(a.startswith("checkpoint-5") for a in arcnames)


def test_enumerate_skips_symlinks(tmp_path):
    root = str(tmp_path)
    _make_checkpoint(root)
    outside = tmp_path.parent / "secret.py"
    outside.write_text("SECRET")
    os.symlink(str(outside), os.path.join(root, "linked.py"))
    arcnames = [arc for _, arc in md.enumerate_model_def_files(root)]
    assert "linked.py" not in arcnames


def test_bundle_hash_deterministic_and_content_sensitive(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_checkpoint(str(a))
    _make_checkpoint(str(b))
    assert md.compute_bundle_hash(str(a)) == md.compute_bundle_hash(str(b))
    # Editing an included file changes the hash...
    _write(os.path.join(str(b), "modeling_demo.py"), "class Model: x = 1")
    assert md.compute_bundle_hash(str(a)) != md.compute_bundle_hash(str(b))
    # ...but editing an EXCLUDED file (weights) does not.
    before = md.compute_bundle_hash(str(a))
    _write(os.path.join(str(a), "model.safetensors"), "DIFFERENT WEIGHTS")
    assert md.compute_bundle_hash(str(a)) == before


def test_pack_is_byte_stable(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_checkpoint(str(a))
    _make_checkpoint(str(b))
    # Same definition packs to identical bytes regardless of on-disk mtimes.
    assert md.pack_model_def(str(a)) == md.pack_model_def(str(b))


def test_pack_extract_roundtrip(tmp_path):
    root = tmp_path / "ckpt"
    _make_checkpoint(str(root))
    data = md.pack_model_def(str(root))
    dest = tmp_path / "staged"
    md.extract_model_def(data, str(dest))
    got = sorted(os.listdir(dest))
    assert got == [
        "config.json",
        "configuration_demo.py",
        "generation_config.json",
        "modeling_demo.py",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert (dest / "config.json").read_text() == '{"hidden_size": 8}'


def test_extract_rejects_traversal(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        info = tarfile.TarInfo("../escape.py")
        info.size = 3
        t.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="out-of-tree|absolute"):
        md.extract_model_def(buf.getvalue(), str(tmp_path / "dest"))


def test_extract_rejects_absolute(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        info = tarfile.TarInfo("/etc/evil.py")
        info.size = 3
        t.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="absolute|out-of-tree"):
        md.extract_model_def(buf.getvalue(), str(tmp_path / "dest"))


def test_extract_rejects_symlink_member(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        info = tarfile.TarInfo("evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        t.addfile(info)
    with pytest.raises(ValueError, match="non-regular"):
        md.extract_model_def(buf.getvalue(), str(tmp_path / "dest"))
