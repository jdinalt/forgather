#!/usr/bin/env python3
"""
Unit tests for data_collator module in Forgather.

Tests the position ID generation functions and the DataCollatorForCausalLM class,
covering:
- _pos_ids_from_boundaries: position reset at explicit document boundaries
- _pos_ids_from_tokens: position reset at special token boundaries (legacy)
- get_pos_ids_for_packed_sequence: dispatch logic
- DataCollatorForCausalLM: collation, truncation, padding, packed sequences, repr
"""

import copy
import unittest
from unittest.mock import Mock, MagicMock, patch

import torch
from torch import Tensor

from forgather.ml.data_collator import (
    _pos_ids_from_boundaries,
    _pos_ids_from_tokens,
    get_pos_ids_for_packed_sequence,
    DataCollatorForCausalLM,
)


# ---------------------------------------------------------------------------
# Mock tokenizer for DataCollatorForCausalLM tests
# ---------------------------------------------------------------------------

class MockTokenizer:
    """
    Lightweight mock tokenizer that implements enough of the HuggingFace
    tokenizer interface for DataCollatorForCausalLM to work.
    """

    def __init__(
        self,
        vocab_size=100,
        pad_token_id=0,
        eos_token_id=2,
        model_max_length=512,
        padding_side="right",
        truncation_side="right",
    ):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.pad_token_id = pad_token_id
        self.eos_token = "</s>"
        self.eos_token_id = eos_token_id
        self.model_max_length = model_max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.deprecation_warnings = {}

    def pad(self, features, return_tensors="pt", **kwargs):
        """
        Minimal pad implementation that pads 'input_ids' (and optionally
        'labels' / 'attention_mask') to the same length with right-padding.
        Only supports return_tensors='pt'.
        """
        if not features:
            return {}

        # Collect all keys present in the features
        keys = list(features[0].keys())
        result = {}

        for key in keys:
            values = [f[key] for f in features]
            # Convert to tensors if needed
            tensors = []
            for v in values:
                if isinstance(v, Tensor):
                    tensors.append(v)
                elif isinstance(v, list):
                    tensors.append(torch.tensor(v, dtype=torch.long))
                else:
                    tensors.append(v)

            max_len = max(t.shape[0] for t in tensors)

            # Determine padding value
            if key == "input_ids":
                pad_value = self.pad_token_id
            elif key == "labels":
                pad_value = -100
            elif key == "attention_mask":
                pad_value = 0
            else:
                pad_value = 0

            padded = []
            for t in tensors:
                if t.shape[0] < max_len:
                    padding = torch.full(
                        (max_len - t.shape[0],), pad_value, dtype=t.dtype
                    )
                    padded.append(torch.cat([t, padding]))
                else:
                    padded.append(t)

            result[key] = torch.stack(padded)

        return result

    def __repr__(self):
        return f"MockTokenizer(vocab_size={self.vocab_size})"

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new


# ===========================================================================
# Tests for _pos_ids_from_boundaries
# ===========================================================================


class TestPosIdsFromBoundaries(unittest.TestCase):
    """Tests for _pos_ids_from_boundaries()."""

    def test_single_document(self):
        """A single document starting at 0 should give sequential position IDs."""
        x = torch.tensor([[10, 20, 30, 40, 50]])
        doc_starts = torch.tensor([[0]])
        pos_ids = _pos_ids_from_boundaries(x, doc_starts)
        expected = torch.tensor([[0, 1, 2, 3, 4]])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_two_documents(self):
        """Two documents should reset position at the second boundary."""
        x = torch.tensor([[10, 20, 30, 40, 50, 60]])
        doc_starts = torch.tensor([[0, 3]])
        pos_ids = _pos_ids_from_boundaries(x, doc_starts)
        # doc1: positions 0,1,2 at indices 0,1,2
        # doc2: positions 0,1,2 at indices 3,4,5
        expected = torch.tensor([[0, 1, 2, 0, 1, 2]])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_three_documents(self):
        """Three documents packed together."""
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        doc_starts = torch.tensor([[0, 2, 5]])
        pos_ids = _pos_ids_from_boundaries(x, doc_starts)
        expected = torch.tensor([[0, 1, 0, 1, 2, 0, 1, 2]])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_padded_doc_starts_with_neg1(self):
        """Document starts padded with -1 should be filtered out."""
        # Batch of 2: first has 3 docs, second has 1 doc, padded with -1
        x = torch.tensor([
            [10, 20, 30, 40, 50, 60],
            [70, 80, 90, 10, 20, 30],
        ])
        doc_starts = torch.tensor([
            [0, 2, 4],
            [0, -1, -1],
        ])
        pos_ids = _pos_ids_from_boundaries(x, doc_starts)
        expected = torch.tensor([
            [0, 1, 0, 1, 0, 1],
            [0, 1, 2, 3, 4, 5],
        ])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_all_neg1_boundaries(self):
        """If all document starts are -1, fall back to sequential positions."""
        x = torch.tensor([[10, 20, 30, 40]])
        doc_starts = torch.tensor([[-1, -1, -1]])
        pos_ids = _pos_ids_from_boundaries(x, doc_starts)
        expected = torch.tensor([[0, 1, 2, 3]])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_empty_boundaries_tensor(self):
        """An empty boundaries tensor (no columns) should give sequential positions."""
        x = torch.tensor([[10, 20, 30]])
        # Shape (1, 0) -- no document boundaries at all
        doc_starts = torch.zeros((1, 0), dtype=torch.long)
        pos_ids = _pos_ids_from_boundaries(x, doc_starts)
        expected = torch.tensor([[0, 1, 2]])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_batch_dimension(self):
        """Test with a batch of multiple sequences."""
        x = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ])
        doc_starts = torch.tensor([
            [0, 2],
            [0, -1],
            [0, 1],
        ])
        pos_ids = _pos_ids_from_boundaries(x, doc_starts)
        expected = torch.tensor([
            [0, 1, 0, 1],  # 2 docs of length 2
            [0, 1, 2, 3],  # 1 doc
            [0, 0, 1, 2],  # doc of length 1, then doc of length 3
        ])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_document_starting_not_at_zero(self):
        """When first document starts at a position > 0, positions before it remain 0."""
        # NOTE: This tests the actual behavior of the function. When the first
        # document starts at position > 0, positions 0..start-1 remain as
        # initialized (zeros) since the function only fills ranges between
        # boundary starts. This may or may not be intentional behavior.
        x = torch.tensor([[10, 20, 30, 40, 50]])
        doc_starts = torch.tensor([[2]])  # First doc starts at position 2
        pos_ids = _pos_ids_from_boundaries(x, doc_starts)
        # Positions 0,1 remain at 0 (initialized zeros), positions 2,3,4 get 0,1,2
        expected = torch.tensor([[0, 0, 0, 1, 2]])
        self.assertTrue(torch.equal(pos_ids, expected))


# ===========================================================================
# Tests for _pos_ids_from_tokens
# ===========================================================================


class TestPosIdsFromTokens(unittest.TestCase):
    """Tests for _pos_ids_from_tokens()."""

    def test_single_boundary_eos_true(self):
        """Single EOS token with eos=True: reset AFTER the EOS token."""
        # Sequence: [1, 2, EOS, 4, 5]
        # With eos=True, positions reset after EOS:
        #   positions: [0, 1, 2, 0, 1]
        x = torch.tensor([[1, 2, 3, 4, 5]])
        token_id = 3  # Use 3 as the boundary token
        pos_ids = _pos_ids_from_tokens(x, token_id, eos=True)
        expected = torch.tensor([[0, 1, 2, 0, 1]])
        self.assertTrue(
            torch.equal(pos_ids, expected),
            f"Expected {expected}, got {pos_ids}",
        )

    def test_single_boundary_eos_false(self):
        """Single boundary token with eos=False: reset AT (before) the token."""
        # Sequence: [1, 2, BOS, 4, 5]
        # With eos=False, positions reset at the boundary token:
        #   positions: [0, 1, 0, 1, 2]
        x = torch.tensor([[1, 2, 3, 4, 5]])
        token_id = 3
        pos_ids = _pos_ids_from_tokens(x, token_id, eos=False)
        expected = torch.tensor([[0, 1, 0, 1, 2]])
        self.assertTrue(
            torch.equal(pos_ids, expected),
            f"Expected {expected}, got {pos_ids}",
        )

    def test_no_boundary_tokens(self):
        """No boundary tokens present: sequential positions 0..T-1."""
        x = torch.tensor([[10, 20, 30, 40]])
        pos_ids = _pos_ids_from_tokens(x, token_id=99, eos=True)
        expected = torch.tensor([[0, 1, 2, 3]])
        self.assertTrue(
            torch.equal(pos_ids, expected),
            f"Expected {expected}, got {pos_ids}",
        )

    def test_multiple_boundary_tokens(self):
        """Multiple boundary tokens in one sequence."""
        # [1, EOS, 3, EOS, 5] with EOS=2, eos=True
        x = torch.tensor([[1, 2, 3, 2, 5]])
        pos_ids = _pos_ids_from_tokens(x, token_id=2, eos=True)
        expected = torch.tensor([[0, 1, 0, 1, 0]])
        self.assertTrue(
            torch.equal(pos_ids, expected),
            f"Expected {expected}, got {pos_ids}",
        )

    def test_batch_of_sequences(self):
        """Batch of two sequences, each with boundary tokens."""
        # Batch item 0: [1, 2, 3, 4]  -- EOS=2 at position 1
        # Batch item 1: [5, 6, 2, 8]  -- EOS=2 at position 2
        x = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 2, 8],
        ])
        pos_ids = _pos_ids_from_tokens(x, token_id=2, eos=True)
        expected_0 = torch.tensor([0, 1, 0, 1])
        expected_1 = torch.tensor([0, 1, 2, 0])
        self.assertTrue(
            torch.equal(pos_ids[0], expected_0),
            f"Batch 0: expected {expected_0}, got {pos_ids[0]}",
        )
        self.assertTrue(
            torch.equal(pos_ids[1], expected_1),
            f"Batch 1: expected {expected_1}, got {pos_ids[1]}",
        )

    def test_boundary_at_end_eos_true(self):
        """Boundary token at the very last position with eos=True."""
        x = torch.tensor([[1, 2, 3]])
        pos_ids = _pos_ids_from_tokens(x, token_id=3, eos=True)
        # With eos=True: positions are [0, 1, 2] -- boundary at end, reset would
        # come after it but there's nothing after.
        expected = torch.tensor([[0, 1, 2]])
        self.assertTrue(
            torch.equal(pos_ids, expected),
            f"Expected {expected}, got {pos_ids}",
        )

    def test_boundary_at_start_eos_false(self):
        """Boundary token at position 0 with eos=False: entire sequence resets."""
        x = torch.tensor([[3, 1, 2, 4]])
        pos_ids = _pos_ids_from_tokens(x, token_id=3, eos=False)
        expected = torch.tensor([[0, 1, 2, 3]])
        self.assertTrue(
            torch.equal(pos_ids, expected),
            f"Expected {expected}, got {pos_ids}",
        )


# ===========================================================================
# Tests for get_pos_ids_for_packed_sequence
# ===========================================================================


class TestGetPosIdsForPackedSequence(unittest.TestCase):
    """Tests for get_pos_ids_for_packed_sequence() dispatch function."""

    def test_dispatch_to_boundaries(self):
        """When document_starts is provided, should use _pos_ids_from_boundaries."""
        x = torch.tensor([[10, 20, 30, 40]])
        doc_starts = torch.tensor([[0, 2]])
        pos_ids = get_pos_ids_for_packed_sequence(x, document_starts=doc_starts)
        expected = torch.tensor([[0, 1, 0, 1]])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_dispatch_to_tokens(self):
        """When token_id is provided (without document_starts), should use _pos_ids_from_tokens."""
        x = torch.tensor([[1, 2, 3, 4]])
        pos_ids = get_pos_ids_for_packed_sequence(x, token_id=2, eos=True)
        expected = torch.tensor([[0, 1, 0, 1]])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_document_starts_takes_precedence(self):
        """When both document_starts and token_id are provided, document_starts wins."""
        x = torch.tensor([[1, 2, 3, 4]])
        doc_starts = torch.tensor([[0]])  # Single doc -> sequential
        pos_ids = get_pos_ids_for_packed_sequence(
            x, token_id=2, document_starts=doc_starts
        )
        expected = torch.tensor([[0, 1, 2, 3]])
        self.assertTrue(torch.equal(pos_ids, expected))

    def test_raises_value_error_if_neither_provided(self):
        """Should raise ValueError when neither document_starts nor token_id is given."""
        x = torch.tensor([[1, 2, 3]])
        with self.assertRaises(ValueError) as ctx:
            get_pos_ids_for_packed_sequence(x)
        self.assertIn("document_starts", str(ctx.exception))
        self.assertIn("token_id", str(ctx.exception))

    def test_eos_parameter_forwarded(self):
        """The eos parameter should be forwarded to token-based detection."""
        x = torch.tensor([[1, 2, 3, 4]])
        # token_id=2, eos=False -> reset AT position of token 2
        pos_ids = get_pos_ids_for_packed_sequence(x, token_id=2, eos=False)
        expected = torch.tensor([[0, 0, 1, 2]])
        self.assertTrue(
            torch.equal(pos_ids, expected),
            f"Expected {expected}, got {pos_ids}",
        )


# ===========================================================================
# Tests for DataCollatorForCausalLM
# ===========================================================================


class TestDataCollatorForCausalLMInit(unittest.TestCase):
    """Tests for DataCollatorForCausalLM initialization."""

    def test_basic_init(self):
        """Basic initialization with default parameters."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer)
        self.assertEqual(collator.max_length, 512)
        self.assertEqual(collator.input_name, "input_ids")
        self.assertEqual(collator.labels_name, "labels")
        self.assertFalse(collator.truncation)
        self.assertEqual(collator.ignore_index, -100)
        self.assertIsNone(collator.packed_sequences)

    def test_tokenizer_is_deepcopied(self):
        """Collator should deep-copy the tokenizer so mutations don't leak."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer)
        # Modify the original tokenizer -- collator's copy should be unaffected
        tokenizer.pad_token_id = 999
        self.assertNotEqual(collator.tokenizer.pad_token_id, 999)

    def test_no_pad_token_falls_back_to_eos(self):
        """If pad_token is None, it should be set to eos_token."""
        tokenizer = MockTokenizer()
        tokenizer.pad_token = None
        tokenizer.pad_token_id = None
        collator = DataCollatorForCausalLM(tokenizer)
        self.assertEqual(collator.tokenizer.pad_token, "</s>")
        self.assertEqual(collator.tokenizer.pad_token_id, 2)

    def test_left_padding_side_changed_to_right(self):
        """Padding side 'left' should be changed to 'right'."""
        tokenizer = MockTokenizer(padding_side="left")
        collator = DataCollatorForCausalLM(tokenizer)
        self.assertEqual(collator.tokenizer.padding_side, "right")

    def test_left_truncation_side_changed_to_right(self):
        """Truncation side 'left' should be changed to 'right'."""
        tokenizer = MockTokenizer(truncation_side="left")
        collator = DataCollatorForCausalLM(tokenizer)
        self.assertEqual(collator.tokenizer.truncation_side, "right")

    def test_max_length_from_pad_kwargs(self):
        """max_length from pad_kwargs should be used."""
        tokenizer = MockTokenizer(model_max_length=512)
        collator = DataCollatorForCausalLM(tokenizer, max_length=256)
        self.assertEqual(collator.max_length, 256)

    def test_max_length_exceeds_model_max_raises(self):
        """max_length > model_max_length should raise ValueError."""
        tokenizer = MockTokenizer(model_max_length=128)
        with self.assertRaises(ValueError):
            DataCollatorForCausalLM(tokenizer, max_length=256)

    def test_max_length_popped_when_not_pad_to_max_length(self):
        """max_length should be removed from pad_kwargs unless padding='max_length'."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, max_length=256)
        self.assertNotIn("max_length", collator.pad_kwargs)

    def test_max_length_kept_when_pad_to_max_length(self):
        """max_length should be kept in pad_kwargs when padding='max_length'."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(
            tokenizer, max_length=256, padding="max_length"
        )
        self.assertIn("max_length", collator.pad_kwargs)
        self.assertEqual(collator.pad_kwargs["max_length"], 256)

    def test_custom_ignore_index(self):
        """Custom ignore_index should be stored."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, ignore_index=-200)
        self.assertEqual(collator.ignore_index, -200)

    def test_custom_input_name(self):
        """Custom input_name should be stored."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, input_name="tokens")
        self.assertEqual(collator.input_name, "tokens")

    def test_packed_sequences_explicit(self):
        """Explicit packed_sequences values should be stored."""
        tokenizer = MockTokenizer()
        collator_true = DataCollatorForCausalLM(tokenizer, packed_sequences=True)
        self.assertTrue(collator_true.packed_sequences)
        collator_false = DataCollatorForCausalLM(tokenizer, packed_sequences=False)
        self.assertFalse(collator_false.packed_sequences)


class TestDataCollatorForCausalLMRepr(unittest.TestCase):
    """Tests for DataCollatorForCausalLM.__repr__."""

    def test_repr_contains_class_name(self):
        """__repr__ should contain the class name."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer)
        r = repr(collator)
        self.assertIn("DataCollatorForCausalLM", r)

    def test_repr_contains_truncation_and_ignore_index(self):
        """__repr__ should include truncation and ignore_index values."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, truncation=True, ignore_index=-200)
        r = repr(collator)
        self.assertIn("truncation=True", r)
        self.assertIn("ignore_index=-200", r)

    def test_repr_contains_pad_kwargs(self):
        """__repr__ should include pad_kwargs."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, padding="max_length", max_length=256)
        r = repr(collator)
        self.assertIn("pad_kwargs=", r)

    def test_repr_mismatched_parenthesis_bug(self):
        """
        BUG: The __repr__ method has a mismatched closing parenthesis.

        The format string is:
            f"...ignore_index={self.ignore_index }), pad_kwargs=..."
        Note the extra ')' after ignore_index -- there is a ')' that closes a
        non-existent opening paren, making the repr syntactically odd. For
        example, the output looks like:
            DataCollatorForCausalLM(..., ignore_index=-100), pad_kwargs={...}

        This is a cosmetic bug. We test the actual (buggy) output here so
        that the test passes against the current code.
        """
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer)
        r = repr(collator)
        # The buggy repr has "), pad_kwargs=" -- the ')' closes the class but
        # then pad_kwargs follows outside.
        self.assertIn("), pad_kwargs=", r)


class TestDataCollatorForCausalLMCall(unittest.TestCase):
    """Tests for DataCollatorForCausalLM.__call__."""

    def _make_collator(self, **kwargs):
        tokenizer = MockTokenizer()
        return DataCollatorForCausalLM(tokenizer, **kwargs)

    def _make_features(self, input_ids_list):
        """Create a list of feature dicts from a list of input_ids lists."""
        return [{"input_ids": ids} for ids in input_ids_list]

    def test_basic_collation_labels_from_input_ids(self):
        """Labels should copy input_ids, with pad tokens replaced by ignore_index."""
        collator = self._make_collator()
        features = self._make_features([
            [1, 2, 3],
            [4, 5, 0],  # 0 is pad_token_id
        ])
        batch = collator(features)
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)

        # Labels: pad positions should be ignore_index (-100)
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Where input_ids == pad_token_id (0), labels should be -100
        pad_mask = input_ids == 0
        self.assertTrue((labels[pad_mask] == -100).all())
        # Where input_ids != pad_token_id, labels should equal input_ids
        non_pad_mask = input_ids != 0
        self.assertTrue((labels[non_pad_mask] == input_ids[non_pad_mask]).all())

    def test_basic_collation_with_unequal_lengths(self):
        """Sequences of different lengths should be padded to the longest."""
        collator = self._make_collator()
        features = self._make_features([
            [1, 2, 3, 4, 5],
            [6, 7, 8],
        ])
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 5))
        # Second sequence should be padded with pad_token_id (0) at positions 3, 4
        self.assertEqual(batch["input_ids"][1, 3].item(), 0)
        self.assertEqual(batch["input_ids"][1, 4].item(), 0)
        # Padded label positions should be ignore_index
        self.assertEqual(batch["labels"][1, 3].item(), -100)
        self.assertEqual(batch["labels"][1, 4].item(), -100)

    def test_truncation_right(self):
        """With truncation=True, sequences should be truncated from the right."""
        collator = self._make_collator(truncation=True, max_length=3)
        features = self._make_features([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ])
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertTrue(torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(batch["input_ids"][1], torch.tensor([6, 7, 8])))

    def test_truncation_with_short_sequences(self):
        """Truncation should not affect sequences shorter than max_length."""
        collator = self._make_collator(truncation=True, max_length=10)
        features = self._make_features([
            [1, 2, 3],
        ])
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, (1, 3))

    def test_labels_name_none_returns_tuple(self):
        """When labels_name=None, __call__ should return (dict, labels) tuple."""
        collator = self._make_collator(labels_name=None)
        features = self._make_features([
            [1, 2, 3],
        ])
        result = collator(features)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        output_dict, labels = result
        self.assertIn("input_ids", output_dict)
        self.assertNotIn("labels", output_dict)
        self.assertIsInstance(labels, Tensor)

    def test_labels_in_features_are_preserved(self):
        """If features already contain 'labels', they should be used directly."""
        collator = self._make_collator()
        features = [
            {"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
            {"input_ids": [4, 5, 6], "labels": [40, 50, 60]},
        ]
        batch = collator(features)
        # Labels from features should be preserved, not overwritten
        expected_labels = torch.tensor([[10, 20, 30], [40, 50, 60]])
        self.assertTrue(torch.equal(batch["labels"], expected_labels))

    def test_custom_input_name(self):
        """Custom input_name should be used as the key in the output dict."""
        collator = self._make_collator(input_name="tokens")
        features = self._make_features([
            [1, 2, 3],
        ])
        batch = collator(features)
        self.assertIn("tokens", batch)
        self.assertNotIn("input_ids", batch)

    def test_custom_ignore_index(self):
        """Custom ignore_index should be used for pad token labels."""
        collator = self._make_collator(ignore_index=-200)
        features = self._make_features([
            [1, 2, 3],
            [4, 0, 0],  # 0 is pad_token_id
        ])
        batch = collator(features)
        # Pad positions in labels should use custom ignore_index
        self.assertEqual(batch["labels"][1, 1].item(), -200)
        self.assertEqual(batch["labels"][1, 2].item(), -200)


class TestDataCollatorPackedSequences(unittest.TestCase):
    """Tests for packed sequence handling in DataCollatorForCausalLM."""

    def _make_collator(self, **kwargs):
        tokenizer = MockTokenizer()
        return DataCollatorForCausalLM(tokenizer, **kwargs)

    def test_packed_sequences_true_with_document_starts(self):
        """packed_sequences=True with document_starts should generate position_ids."""
        collator = self._make_collator(packed_sequences=True)
        features = [
            {"input_ids": [1, 2, 3, 4, 5, 6], "document_starts": [0, 3]},
            {"input_ids": [7, 8, 9, 10, 11, 12], "document_starts": [0, 2, 4]},
        ]
        batch = collator(features)
        self.assertIn("position_ids", batch)
        # First sequence: docs at [0,3] -> [0,1,2, 0,1,2]
        expected_0 = torch.tensor([0, 1, 2, 0, 1, 2])
        self.assertTrue(
            torch.equal(batch["position_ids"][0], expected_0),
            f"Expected {expected_0}, got {batch['position_ids'][0]}",
        )
        # Second sequence: docs at [0,2,4] -> [0,1, 0,1, 0,1]
        expected_1 = torch.tensor([0, 1, 0, 1, 0, 1])
        self.assertTrue(
            torch.equal(batch["position_ids"][1], expected_1),
            f"Expected {expected_1}, got {batch['position_ids'][1]}",
        )

    def test_packed_sequences_true_without_document_starts_falls_back_to_token(self):
        """packed_sequences=True without document_starts uses EOS token detection."""
        collator = self._make_collator(packed_sequences=True)
        # EOS token is 2
        features = [
            {"input_ids": [1, 2, 3, 4]},  # EOS at position 1
        ]
        batch = collator(features)
        self.assertIn("position_ids", batch)
        # With eos=True (default), reset after EOS token (token_id=2 at pos 1):
        # positions: [0, 1, 0, 1]
        expected = torch.tensor([[0, 1, 0, 1]])
        self.assertTrue(
            torch.equal(batch["position_ids"], expected),
            f"Expected {expected}, got {batch['position_ids']}",
        )

    def test_packed_sequences_none_autodetects_from_document_starts(self):
        """packed_sequences=None should auto-detect from document_starts field."""
        collator = self._make_collator(packed_sequences=None)
        features = [
            {"input_ids": [1, 2, 3, 4], "document_starts": [0, 2]},
        ]
        batch = collator(features)
        self.assertIn("position_ids", batch)
        expected = torch.tensor([[0, 1, 0, 1]])
        self.assertTrue(
            torch.equal(batch["position_ids"], expected),
            f"Expected {expected}, got {batch['position_ids']}",
        )

    def test_packed_sequences_none_no_document_starts_no_position_ids(self):
        """packed_sequences=None without document_starts should not produce position_ids."""
        collator = self._make_collator(packed_sequences=None)
        features = [
            {"input_ids": [1, 2, 3, 4]},
        ]
        batch = collator(features)
        self.assertNotIn("position_ids", batch)

    def test_packed_sequences_false_no_position_ids(self):
        """packed_sequences=False should never produce position_ids."""
        collator = self._make_collator(packed_sequences=False)
        features = [
            {"input_ids": [1, 2, 3, 4], "document_starts": [0, 2]},
        ]
        batch = collator(features)
        self.assertNotIn("position_ids", batch)

    def test_document_starts_popped_from_features(self):
        """document_starts should be removed from features before padding."""
        collator = self._make_collator(packed_sequences=True)
        features = [
            {"input_ids": [1, 2, 3], "document_starts": [0, 2]},
        ]
        batch = collator(features)
        # document_starts should not appear in output
        self.assertNotIn("document_starts", batch)


class TestPadDocumentStarts(unittest.TestCase):
    """Tests for DataCollatorForCausalLM._pad_document_starts."""

    def _make_collator(self):
        tokenizer = MockTokenizer()
        return DataCollatorForCausalLM(tokenizer)

    def test_uniform_length(self):
        """All document_starts lists have the same length."""
        collator = self._make_collator()
        starts = [[0, 3], [0, 4]]
        result = collator._pad_document_starts(starts)
        expected = torch.tensor([[0, 3], [0, 4]], dtype=torch.long)
        self.assertTrue(torch.equal(result, expected))

    def test_padding_with_neg1(self):
        """Shorter lists should be padded with -1."""
        collator = self._make_collator()
        starts = [[0, 2, 5], [0]]
        result = collator._pad_document_starts(starts)
        expected = torch.tensor([[0, 2, 5], [0, -1, -1]], dtype=torch.long)
        self.assertTrue(torch.equal(result, expected))

    def test_empty_list(self):
        """Empty input should return an empty tensor."""
        collator = self._make_collator()
        result = collator._pad_document_starts([])
        self.assertEqual(result.numel(), 0)

    def test_single_entry(self):
        """Single document_starts list."""
        collator = self._make_collator()
        result = collator._pad_document_starts([[0, 5, 10]])
        expected = torch.tensor([[0, 5, 10]], dtype=torch.long)
        self.assertTrue(torch.equal(result, expected))


class TestDataCollatorPad(unittest.TestCase):
    """Tests for DataCollatorForCausalLM._pad (deprecation warning suppression)."""

    def test_pad_with_deprecation_warnings(self):
        """When tokenizer has deprecation_warnings, the pad warning should be suppressed."""
        tokenizer = MockTokenizer()
        tokenizer.deprecation_warnings = {"Asking-to-pad-a-fast-tokenizer": False}
        collator = DataCollatorForCausalLM(tokenizer)

        features = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
        result = collator._pad(features)

        self.assertIn("input_ids", result)
        # Deprecation warning state should be restored
        self.assertFalse(
            collator.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"]
        )

    def test_pad_without_deprecation_warnings(self):
        """When tokenizer lacks deprecation_warnings, pad should still work."""
        tokenizer = MockTokenizer()
        del tokenizer.deprecation_warnings
        collator = DataCollatorForCausalLM(tokenizer)

        features = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
        result = collator._pad(features)
        self.assertIn("input_ids", result)

    def test_pad_restores_warning_state_on_exception(self):
        """Deprecation warning state should be restored even if pad() raises."""
        tokenizer = MockTokenizer()
        tokenizer.deprecation_warnings = {"Asking-to-pad-a-fast-tokenizer": False}

        # Make pad raise an exception
        original_pad = tokenizer.pad
        def bad_pad(*args, **kwargs):
            raise RuntimeError("pad failed")
        tokenizer.pad = bad_pad

        collator = DataCollatorForCausalLM(tokenizer)
        # Restore original pad after deepcopy (need to patch collator's copy)
        collator.tokenizer.pad = bad_pad

        with self.assertRaises(RuntimeError):
            collator._pad([{"input_ids": [1, 2]}])

        # Warning state should be restored despite exception
        self.assertFalse(
            collator.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"]
        )


class TestDataCollatorTruncate(unittest.TestCase):
    """Tests for DataCollatorForCausalLM._truncate."""

    def test_right_truncation(self):
        """Right truncation should keep the first max_length tokens."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, truncation=True, max_length=3)
        features = [
            {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]},
        ]
        truncated = collator._truncate(features)
        self.assertEqual(truncated[0]["input_ids"], [1, 2, 3])
        self.assertEqual(truncated[0]["attention_mask"], [1, 1, 1])

    def test_no_truncation_when_shorter(self):
        """Sequences shorter than max_length should not be changed."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, truncation=True, max_length=10)
        features = [{"input_ids": [1, 2, 3]}]
        truncated = collator._truncate(features)
        self.assertEqual(truncated[0]["input_ids"], [1, 2, 3])

    def test_truncation_applies_to_all_keys(self):
        """Truncation should apply to all keys in the feature dict."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, truncation=True, max_length=2)
        features = [
            {"input_ids": [1, 2, 3], "labels": [10, 20, 30], "attention_mask": [1, 1, 1]},
        ]
        truncated = collator._truncate(features)
        self.assertEqual(truncated[0]["input_ids"], [1, 2])
        self.assertEqual(truncated[0]["labels"], [10, 20])
        self.assertEqual(truncated[0]["attention_mask"], [1, 1])

    def test_truncation_with_document_starts(self):
        """Truncation should also apply to document_starts if present."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, truncation=True, max_length=2)
        features = [
            {"input_ids": [1, 2, 3, 4], "document_starts": [0, 2, 3]},
        ]
        truncated = collator._truncate(features)
        self.assertEqual(truncated[0]["input_ids"], [1, 2])
        # document_starts is also sliced: [0, 2, 3][:2] = [0, 2]
        self.assertEqual(truncated[0]["document_starts"], [0, 2])


class TestDataCollatorIntegration(unittest.TestCase):
    """Integration tests for DataCollatorForCausalLM end-to-end behavior."""

    def test_full_pipeline_no_packed(self):
        """End-to-end: basic collation without packed sequences."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer)
        features = [
            {"input_ids": [1, 5, 3, 4]},
            {"input_ids": [6, 7]},
        ]
        batch = collator(features)
        # Shape: (2, 4) -- padded to longest
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["labels"].shape, (2, 4))
        # No position_ids without packed_sequences
        self.assertNotIn("position_ids", batch)

    def test_full_pipeline_with_packed_and_truncation(self):
        """End-to-end: truncation + packed sequences with document_starts.

        NOTE: Truncation slices document_starts with the same max_length as
        input_ids (since _truncate applies the slice to all keys). If the
        original document_starts contain boundary positions that are >= the
        truncated sequence length, _pos_ids_from_boundaries will raise a
        RuntimeError because it does not clamp 'end' to T. This is a latent
        BUG: when truncation is combined with packed sequences, document
        boundary values that exceed the truncated length are not filtered out
        or clamped. The workaround (tested here) is to ensure document
        boundaries are compatible with the truncated length.
        """
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(
            tokenizer, truncation=True, max_length=4, packed_sequences=True
        )
        # document_starts=[0, 2] -- both boundaries valid after truncation to 4
        features = [
            {"input_ids": [1, 2, 3, 4, 5, 6], "document_starts": [0, 2]},
        ]
        batch = collator(features)
        # After truncation to 4: input_ids=[1,2,3,4], document_starts=[0,2]
        self.assertEqual(batch["input_ids"].shape, (1, 4))
        self.assertIn("position_ids", batch)
        # doc starts [0, 2]: positions [0,1, 0,1]
        expected_pos = torch.tensor([[0, 1, 0, 1]])
        self.assertTrue(
            torch.equal(batch["position_ids"], expected_pos),
            f"Expected {expected_pos}, got {batch['position_ids']}",
        )

    def test_truncation_with_out_of_bounds_document_starts_bug(self):
        """
        BUG: When truncation reduces sequence length, document_starts values
        that are >= the new length cause _pos_ids_from_boundaries to fail.

        _pos_ids_from_boundaries computes doc_length = end - start where
        end = starts[i+1], but does not clamp end to T. When a boundary
        position exceeds T, torch.arange(doc_length) produces more elements
        than fit in pos_ids[batch_idx, start:end] (which is clamped by tensor
        bounds), causing a RuntimeError.

        This test documents the bug by verifying the exception is raised.
        """
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(
            tokenizer, truncation=True, max_length=4, packed_sequences=True
        )
        features = [
            {"input_ids": [1, 2, 3, 4, 5, 6], "document_starts": [0, 3, 5]},
        ]
        # After truncation: input_ids=[1,2,3,4] (len 4), document_starts=[0,3,5]
        # (3 elements, all kept by [:4] slice). Boundary 5 > T=4 causes crash.
        with self.assertRaises(RuntimeError):
            collator(features)

    def test_batch_with_mixed_doc_counts(self):
        """Batch where different sequences have different numbers of documents."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer, packed_sequences=True)
        features = [
            {"input_ids": [1, 2, 3, 4], "document_starts": [0, 2]},
            {"input_ids": [5, 6, 7, 8], "document_starts": [0]},
        ]
        batch = collator(features)
        self.assertIn("position_ids", batch)
        # Seq 0: docs at [0, 2] -> [0, 1, 0, 1]
        # Seq 1: doc at [0] -> [0, 1, 2, 3]
        expected = torch.tensor([
            [0, 1, 0, 1],
            [0, 1, 2, 3],
        ])
        self.assertTrue(
            torch.equal(batch["position_ids"], expected),
            f"Expected {expected}, got {batch['position_ids']}",
        )

    def test_output_types(self):
        """Verify output tensor types are correct."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer)
        features = [{"input_ids": [1, 2, 3]}]
        batch = collator(features)
        self.assertIsInstance(batch["input_ids"], Tensor)
        self.assertIsInstance(batch["labels"], Tensor)
        self.assertEqual(batch["input_ids"].dtype, torch.long)

    def test_single_feature(self):
        """Collation with a single feature should work."""
        tokenizer = MockTokenizer()
        collator = DataCollatorForCausalLM(tokenizer)
        features = [{"input_ids": [1, 2, 3]}]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, (1, 3))
        self.assertEqual(batch["labels"].shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
