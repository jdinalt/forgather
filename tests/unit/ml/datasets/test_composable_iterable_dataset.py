"""
Tests for ComposableIterableDataset and the abstract backend interface.

The wrapper is exercised against an InMemoryBackend so the tests are
fast and have no external dependencies. The same wrapper (returned
directly by `fast_load_iterable_dataset`) is also exercised against
the real Arrow backend when cache prerequisites are available, so we
know the abstraction works for the production backend too.
"""

from __future__ import annotations

import pytest

from forgather.ml.datasets import (
    ComposableIterableDataset,
    InMemoryBackend,
    IterableDatasetBackend,
    fast_load_iterable_dataset,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_examples(n: int):
    return [{"id": i, "text": f"example_{i}"} for i in range(n)]


def _make_wrapper(n: int = 100) -> ComposableIterableDataset:
    return ComposableIterableDataset(InMemoryBackend(_make_examples(n)))


# ---------------------------------------------------------------------
# Backend interface conformance — InMemoryBackend
# ---------------------------------------------------------------------


class TestInMemoryBackend:
    def test_implements_interface(self):
        be = InMemoryBackend(_make_examples(5))
        assert isinstance(be, IterableDatasetBackend)

    def test_basic_iteration(self):
        be = InMemoryBackend(_make_examples(5))
        out = list(be)
        assert [ex["id"] for ex in out] == [0, 1, 2, 3, 4]
        assert len(be) == 5

    def test_position_updates_during_iter(self):
        be = InMemoryBackend(_make_examples(5))
        assert be.position() == 0
        it = iter(be)
        next(it)
        assert be.position() == 1
        next(it)
        assert be.position() == 2

    def test_seek_returns_new_instance_at_position(self):
        be = InMemoryBackend(_make_examples(10))
        be2 = be.seek(7)
        assert be2 is not be
        assert be2.position() == 7
        assert be.position() == 0  # original unchanged
        assert [ex["id"] for ex in be2] == [7, 8, 9]

    def test_seek_past_end_clamps(self):
        be = InMemoryBackend(_make_examples(3))
        be2 = be.seek(100)
        assert list(be2) == []

    def test_seek_negative_raises(self):
        be = InMemoryBackend(_make_examples(3))
        with pytest.raises(ValueError):
            be.seek(-1)

    def test_shuffle_returns_new_permutation(self):
        be = InMemoryBackend(_make_examples(20))
        be_a = be.shuffle(seed=42)
        be_b = be.shuffle(seed=42)
        be_c = be.shuffle(seed=99)
        assert [e["id"] for e in be_a] == [e["id"] for e in be_b]  # determinism
        assert [e["id"] for e in be_a] != list(range(20))  # actually permuted
        assert [e["id"] for e in be_a] != [e["id"] for e in be_c]  # diff seed
        assert [e["id"] for e in be] == list(range(20))  # original unchanged

    def test_shuffle_then_seek(self):
        be = InMemoryBackend(_make_examples(20))
        shuffled = be.shuffle(seed=42)
        full = [e["id"] for e in shuffled]
        from_5 = [e["id"] for e in shuffled.seek(5)]
        assert from_5 == full[5:]


# ---------------------------------------------------------------------
# ComposableIterableDataset — basic ops on InMemoryBackend
# ---------------------------------------------------------------------


class TestWrapperBasic:
    def test_passthrough_iteration(self):
        ds = _make_wrapper(10)
        assert [ex["id"] for ex in ds] == list(range(10))
        assert len(ds) == 10

    def test_column_names_forwarded(self):
        ds = _make_wrapper(3)
        assert ds.column_names == ["id", "text"]

    def test_n_shards_default(self):
        ds = _make_wrapper(3)
        assert ds.n_shards == 1


class TestWrapperSlice:
    def test_slice_int(self):
        ds = _make_wrapper(20).slice(5, 15)
        ids = [ex["id"] for ex in ds]
        assert ids == list(range(5, 15))
        assert len(ds) == 10

    def test_slice_percentage_string(self):
        ds = _make_wrapper(100).slice("80%", None)
        ids = [ex["id"] for ex in ds]
        assert ids == list(range(80, 100))

    def test_slice_float(self):
        ds = _make_wrapper(100).slice(None, 0.2)
        ids = [ex["id"] for ex in ds]
        assert ids == list(range(0, 20))

    def test_slice_negative_int(self):
        ds = _make_wrapper(20).slice(-5, None)
        ids = [ex["id"] for ex in ds]
        assert ids == list(range(15, 20))

    def test_slice_composition(self):
        # slice([0, 50)) then slice([20, 30)) → absolute [20, 30)
        ds = _make_wrapper(100).slice(0, 50).slice(20, 30)
        ids = [ex["id"] for ex in ds]
        assert ids == list(range(20, 30))

    def test_select_contiguous(self):
        ds = _make_wrapper(20).select(range(5, 12))
        ids = [ex["id"] for ex in ds]
        assert ids == list(range(5, 12))

    def test_select_non_contiguous_raises(self):
        ds = _make_wrapper(20)
        with pytest.raises(NotImplementedError):
            ds.select([0, 2, 4])


class TestWrapperShard:
    def test_shard_no_mode_param(self):
        # The wrapper signature must NOT accept a `mode` kwarg.
        ds = _make_wrapper(10)
        with pytest.raises(TypeError):
            ds.shard(num_shards=2, index=0, mode="auto")  # type: ignore[call-arg]

    def test_shard_partitions_disjoint_and_complete(self):
        ds = _make_wrapper(20)
        all_ids = set()
        for i in range(4):
            shard_ids = [ex["id"] for ex in ds.shard(num_shards=4, index=i)]
            assert len(shard_ids) == 5  # evenly divisible
            assert set(shard_ids).isdisjoint(all_ids)
            all_ids |= set(shard_ids)
        assert all_ids == set(range(20))

    def test_shard_with_remainder(self):
        # 23 examples into 4 shards: sizes (6, 6, 6, 5).
        ds = _make_wrapper(23)
        sizes = [len(list(ds.shard(num_shards=4, index=i))) for i in range(4)]
        assert sizes == [6, 6, 6, 5]
        assert sum(sizes) == 23

    def test_shard_after_slice(self):
        # Slice [10, 30) then shard into 2 should give [10, 20) and [20, 30).
        sliced = _make_wrapper(50).slice(10, 30)
        s0 = [ex["id"] for ex in sliced.shard(num_shards=2, index=0)]
        s1 = [ex["id"] for ex in sliced.shard(num_shards=2, index=1)]
        assert s0 == list(range(10, 20))
        assert s1 == list(range(20, 30))


class TestWrapperShuffle:
    def test_shuffle_changes_order(self):
        ds = _make_wrapper(50)
        ids_unshuffled = [ex["id"] for ex in ds]
        ids_shuffled = [ex["id"] for ex in ds.shuffle(seed=7, buffer_size=0)]
        assert sorted(ids_shuffled) == ids_unshuffled
        assert ids_shuffled != ids_unshuffled

    def test_shuffle_deterministic_with_seed(self):
        ds = _make_wrapper(50)
        a = [ex["id"] for ex in ds.shuffle(seed=42, buffer_size=0)]
        b = [ex["id"] for ex in ds.shuffle(seed=42, buffer_size=0)]
        assert a == b

    def test_shuffle_buffer_zero_pure_backend_order(self):
        ds = _make_wrapper(50).shuffle(seed=42, buffer_size=0)
        ids = [ex["id"] for ex in ds]
        # With buffer disabled, order is exactly the backend permutation.
        backend_order = [
            ex["id"] for ex in InMemoryBackend(_make_examples(50)).shuffle(seed=42)
        ]
        assert ids == backend_order

    def test_shuffle_buffer_keeps_full_set(self):
        ds = _make_wrapper(50).shuffle(seed=42, buffer_size=10)
        ids = [ex["id"] for ex in ds]
        assert sorted(ids) == list(range(50))

    def test_set_epoch_changes_order(self):
        ds = _make_wrapper(50).shuffle(seed=42, buffer_size=0)
        epoch0 = [ex["id"] for ex in ds]
        ds.set_epoch(1)
        epoch1 = [ex["id"] for ex in ds]
        assert sorted(epoch0) == sorted(epoch1)
        assert epoch0 != epoch1


class TestWrapperMap:
    def test_map_single(self):
        def add_one(ex):
            return {"id_plus_1": ex["id"] + 1}

        ds = _make_wrapper(5).map(add_one)
        out = list(ds)
        assert [ex["id_plus_1"] for ex in out] == [1, 2, 3, 4, 5]
        # Original column preserved.
        assert [ex["id"] for ex in out] == [0, 1, 2, 3, 4]

    def test_map_remove_columns(self):
        def add(ex):
            return {"text_len": len(ex["text"])}

        ds = _make_wrapper(3).map(add, remove_columns=["text"])
        out = list(ds)
        assert all("text" not in ex for ex in out)
        assert all("text_len" in ex for ex in out)

    def test_filter_via_map_returning_none(self):
        def keep_even(ex):
            return ex if ex["id"] % 2 == 0 else None

        ds = _make_wrapper(10).map(keep_even)
        out = list(ds)
        assert [ex["id"] for ex in out] == [0, 2, 4, 6, 8]

    def test_filter_method(self):
        ds = _make_wrapper(10).filter(lambda ex: ex["id"] % 3 == 0)
        out = [ex["id"] for ex in ds]
        assert out == [0, 3, 6, 9]

    def test_map_chain_composes(self):
        def add_a(ex):
            return {"a": ex["id"] * 2}

        def add_b(ex):
            return {"b": ex["a"] + 1}

        ds = _make_wrapper(3).map(add_a).map(add_b)
        out = list(ds)
        assert [(ex["a"], ex["b"]) for ex in out] == [(0, 1), (2, 3), (4, 5)]

    def test_map_with_indices(self):
        def with_idx(ex, idx):
            return {"global_idx": idx}

        ds = _make_wrapper(5).slice(2, 5).map(with_idx, with_indices=True)
        out = list(ds)
        # The slice starts at backend index 2; map indices begin there.
        assert [ex["global_idx"] for ex in out] == [2, 3, 4]

    def test_batched_map_n_to_n(self):
        def upper(batch):
            return {"text_upper": [t.upper() for t in batch["text"]]}

        ds = _make_wrapper(5).map(upper, batched=True, batch_size=2)
        out = list(ds)
        assert [ex["text_upper"] for ex in out] == [f"EXAMPLE_{i}" for i in range(5)]

    def test_batched_map_n_to_m(self):
        def duplicate(batch):
            new_ids = []
            for i in batch["id"]:
                new_ids.extend([i, i])
            return {"id": new_ids}

        ds = _make_wrapper(3).map(duplicate, batched=True, batch_size=3)
        out = list(ds)
        assert [ex["id"] for ex in out] == [0, 0, 1, 1, 2, 2]

    def test_mixed_batched_chain_raises(self):
        ds = _make_wrapper(3).map(lambda ex: ex)
        with pytest.raises(ValueError):
            ds.map(lambda b: b, batched=True)


class TestWrapperCheckpoint:
    def test_state_dict_roundtrip_resumes(self):
        ds = _make_wrapper(20)
        out_partial = []
        it = iter(ds)
        for _ in range(7):
            out_partial.append(next(it))
        state = ds.state_dict()

        # Reconstruct fresh wrapper and resume.
        ds2 = _make_wrapper(20)
        ds2.load_state_dict(state)
        out_rest = list(ds2)
        full = [ex["id"] for ex in out_partial + out_rest]
        assert full == list(range(20))

    def test_checkpoint_with_slice(self):
        ds = _make_wrapper(50).slice(10, 40)
        it = iter(ds)
        out_partial = [next(it) for _ in range(5)]
        state = ds.state_dict()

        ds2 = _make_wrapper(50).slice(10, 40)
        ds2.load_state_dict(state)
        out_rest = list(ds2)
        full = [ex["id"] for ex in out_partial + out_rest]
        assert full == list(range(10, 40))

    def test_checkpoint_with_shuffle(self):
        ds = _make_wrapper(30).shuffle(seed=42, buffer_size=0)
        full_seq = [ex["id"] for ex in ds]

        ds2 = _make_wrapper(30).shuffle(seed=42, buffer_size=0)
        it = iter(ds2)
        partial = [next(it)["id"] for _ in range(8)]
        state = ds2.state_dict()

        ds3 = _make_wrapper(30).shuffle(seed=42, buffer_size=0)
        ds3.load_state_dict(state)
        rest = [ex["id"] for ex in ds3]
        assert partial + rest == full_seq

    def test_checkpoint_map_chain_length_mismatch_raises(self):
        ds = _make_wrapper(10).map(lambda ex: {"a": 1})
        state = ds.state_dict()
        ds2 = _make_wrapper(10)  # No map — length mismatch.
        with pytest.raises(ValueError, match="length mismatch"):
            ds2.load_state_dict(state)

    def test_checkpoint_map_batched_mismatch_raises(self):
        ds = _make_wrapper(10).map(lambda b: b, batched=True, batch_size=2)
        state = ds.state_dict()
        ds2 = _make_wrapper(10).map(lambda ex: ex, batched=False)
        with pytest.raises(ValueError, match="batched-mode mismatch"):
            ds2.load_state_dict(state)


class TestWrapperShardSliceCompose:
    def test_shuffle_then_shard_disjoint(self):
        ds = _make_wrapper(40).shuffle(seed=11, buffer_size=0)
        all_ids = set()
        for i in range(4):
            shard = ds.shard(num_shards=4, index=i)
            shard_ids = [ex["id"] for ex in shard]
            assert set(shard_ids).isdisjoint(all_ids)
            all_ids |= set(shard_ids)
        assert all_ids == set(range(40))

    def test_shuffle_then_slice_window_covers_subset(self):
        # shuffle() re-permutes the whole backend; slice() then takes a
        # contiguous window of that shuffled order. The set of visible
        # ids depends on the shuffle, but its size must equal the slice
        # length and all ids come from the source.
        ds = _make_wrapper(50).shuffle(seed=3, buffer_size=0).slice(10, 30)
        ids = [ex["id"] for ex in ds]
        assert len(ids) == 20
        assert all(0 <= i < 50 for i in ids)
        assert len(set(ids)) == 20  # no duplicates


# ---------------------------------------------------------------------
# Wrapper over the real Arrow backend
# ---------------------------------------------------------------------


# Use the same small dataset the existing fast_hf_loader tests rely on.
DATASET_PATH = "Skylion007/openwebtext"
SPLIT = "train[:100]"


def _try_arrow_wrapper():
    try:
        ds = fast_load_iterable_dataset(
            "wikitext", name="wikitext-2-raw-v1", split="train[:200]"
        )
    except Exception as exc:  # pragma: no cover - network/cache absent
        pytest.skip(f"Arrow backend unavailable: {exc}")
    # The loader now returns a ComposableIterableDataset directly.
    assert isinstance(ds, ComposableIterableDataset)
    return ds


class TestWrapperOverArrowBackend:
    """Smoke tests that the wrapper works over the real Arrow backend."""

    def test_iteration(self):
        ds = _try_arrow_wrapper()
        out = list(ds)
        assert len(out) == 200
        assert "text" in out[0]

    def test_slice(self):
        ds = _try_arrow_wrapper().slice(50, 100)
        out = list(ds)
        assert len(out) == 50

    def test_shard_partitions_complete(self):
        # wikitext has many duplicate/empty rows so set-membership
        # checks aren't reliable; check counts and total instead.
        ds = _try_arrow_wrapper()
        sizes = [len(list(ds.shard(num_shards=4, index=i))) for i in range(4)]
        assert sum(sizes) == 200
        assert all(s == 50 for s in sizes)

    def test_map(self):
        ds = _try_arrow_wrapper().map(lambda ex: {"length": len(ex["text"])})
        out = list(ds)
        assert all("length" in ex for ex in out)
        assert all(isinstance(ex["length"], int) for ex in out)

    def test_checkpoint_roundtrip(self):
        ds = _try_arrow_wrapper()
        it = iter(ds)
        partial = [next(it) for _ in range(30)]
        state = ds.state_dict()
        # Fresh wrapper around a fresh backend.
        ds2 = _try_arrow_wrapper()
        ds2.load_state_dict(state)
        rest = list(ds2)
        assert len(partial) + len(rest) == 200
