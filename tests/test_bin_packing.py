"""
Unit tests for bin-packing algorithms in block_tokenizer.
"""

from forgather.ml.datasets.block_tokenizer import (
    Document,
    Bin,
    split_document_optimally,
    pack_sequences_optimized,
    _pack_single_document,
)


def test_bin_basic():
    """Test basic Bin functionality."""
    bin = Bin(max_length=10)
    assert bin.remaining() == 10
    assert bin.length == 0

    doc = Document(input_ids=[1, 2, 3], length=3, original_index=0)
    bin.add_document(doc)

    assert bin.length == 3
    assert bin.remaining() == 7
    assert bin.get_ids() == [1, 2, 3]


def test_bin_with_stride_tokens():
    """Test Bin initialization with stride tokens."""
    bin = Bin(max_length=10, stride_tokens=[999])
    assert bin.length == 1
    assert bin.remaining() == 9
    assert bin.get_ids() == [999]

    doc = Document(input_ids=[1, 2, 3], length=3, original_index=0)
    bin.add_document(doc)
    assert bin.get_ids() == [999, 1, 2, 3]


def test_bin_can_fit():
    """Test Bin.can_fit() method."""
    bin = Bin(max_length=10)
    assert bin.can_fit(5) == True
    assert bin.can_fit(10) == True
    assert bin.can_fit(11) == False

    doc = Document(input_ids=[1, 2, 3], length=3, original_index=0)
    bin.add_document(doc)

    assert bin.can_fit(7) == True
    assert bin.can_fit(8) == False


def test_split_document_no_split_needed():
    """Test split_document_optimally when document fits."""
    doc = Document(input_ids=[1, 2, 3, 4, 5], length=5, original_index=0)
    chunks = split_document_optimally(doc, max_length=10, stride=0)

    assert len(chunks) == 1
    assert chunks[0].input_ids == [1, 2, 3, 4, 5]
    assert chunks[0].length == 5


def test_split_document_simple_split():
    """Test split_document_optimally with simple split."""
    doc = Document(
        input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], length=10, original_index=0
    )
    chunks = split_document_optimally(doc, max_length=5, stride=0)

    assert len(chunks) == 2
    assert chunks[0].input_ids == [1, 2, 3, 4, 5]
    assert chunks[0].length == 5
    assert chunks[1].input_ids == [6, 7, 8, 9, 10]
    assert chunks[1].length == 5


def test_split_document_with_stride():
    """Test split_document_optimally with stride."""
    doc = Document(input_ids=[1, 2, 3, 4, 5, 6, 7, 8], length=8, original_index=0)
    chunks = split_document_optimally(doc, max_length=5, stride=2)

    # First chunk: [1, 2, 3, 4, 5]
    # Second chunk starts at offset 5-2=3: [4, 5, 6, 7, 8]
    # Third chunk starts at offset 3+5-2=6: [7, 8]
    assert len(chunks) == 3
    assert chunks[0].input_ids == [1, 2, 3, 4, 5]
    assert chunks[1].input_ids == [4, 5, 6, 7, 8]
    assert chunks[2].input_ids == [7, 8]


def test_split_document_with_bos():
    """Test split_document_optimally with BOS token."""
    doc = Document(
        input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], length=10, original_index=0
    )
    chunks = split_document_optimally(doc, max_length=5, stride=0, bos_token_id=999)

    # First chunk: [1, 2, 3, 4, 5] (no BOS)
    # Second chunk: [999, 6, 7, 8, 9] (with BOS, leaving room for 4 tokens)
    assert len(chunks) >= 2
    assert chunks[0].input_ids == [1, 2, 3, 4, 5]
    assert chunks[1].input_ids[0] == 999  # BOS token
    assert chunks[1].length == 5


def test_pack_single_document_best_fit():
    """Test _pack_single_document with best_fit strategy."""
    bins = [
        Bin(max_length=10),
        Bin(max_length=10),
    ]
    # First bin has 7 remaining, second has 3 remaining
    bins[0].add_document(Document([1, 2, 3], 3, 0))
    bins[1].add_document(Document([4, 5, 6, 7, 8, 9, 10], 7, 1))

    # Add a document of length 3 - should go to second bin (best fit)
    doc = Document([11, 12, 13], 3, 2)
    _pack_single_document(doc, bins, max_length=10, strategy="best_fit")

    assert bins[1].length == 10  # Second bin is full
    assert bins[0].length == 3  # First bin unchanged


def test_pack_single_document_first_fit():
    """Test _pack_single_document with first_fit strategy."""
    bins = [
        Bin(max_length=10),
        Bin(max_length=10),
    ]
    bins[0].add_document(Document([1, 2, 3], 3, 0))
    bins[1].add_document(Document([4, 5, 6, 7, 8, 9, 10], 7, 1))

    # Add a document of length 3 - should go to first bin (first fit)
    doc = Document([11, 12, 13], 3, 2)
    _pack_single_document(doc, bins, max_length=10, strategy="first_fit")

    assert bins[0].length == 6  # First bin has 6 tokens
    assert bins[1].length == 7  # Second bin unchanged


def test_pack_single_document_create_new_bin():
    """Test that new bin is created when no existing bin fits."""
    bins = [Bin(max_length=10)]
    bins[0].add_document(Document([1, 2, 3, 4, 5, 6, 7, 8], 8, 0))

    # Add a document of length 5 - won't fit, should create new bin
    doc = Document([11, 12, 13, 14, 15], 5, 1)
    _pack_single_document(doc, bins, max_length=10, strategy="best_fit")

    assert len(bins) == 2
    assert bins[1].length == 5


def test_pack_sequences_optimized_simple():
    """Test pack_sequences_optimized with simple documents."""
    documents = [
        Document([1, 2, 3], 3, 0),
        Document([4, 5, 6, 7], 4, 1),
        Document([8, 9, 10], 3, 2),
    ]

    sequences = pack_sequences_optimized(
        documents=documents,
        max_length=10,
        min_len=1,
        stride=0,
        overflow=True,
        strategy="best_fit",
    )

    # Should pack into 1 sequence: [1,2,3,4,5,6,7,8,9,10]
    assert len(sequences) == 1
    assert len(sequences[0]) == 10


def test_pack_sequences_optimized_with_sorting():
    """Test that documents are sorted by length (descending)."""
    documents = [
        Document([1], 1, 0),
        Document([2, 3, 4, 5, 6], 5, 1),
        Document([7, 8], 2, 2),
    ]

    sequences = pack_sequences_optimized(
        documents=documents,
        max_length=10,
        min_len=1,
        stride=0,
        overflow=True,
        strategy="best_fit",
    )

    # Sorting should pack efficiently: [5] + [2] + [1] = [2,3,4,5,6,7,8,1]
    assert len(sequences) == 1
    assert len(sequences[0]) == 8


def test_pack_sequences_optimized_min_len_filter():
    """Test that sequences below min_len are filtered out."""
    documents = [
        Document([1, 2], 2, 0),
        Document([3], 1, 1),
    ]

    sequences = pack_sequences_optimized(
        documents=documents,
        max_length=10,
        min_len=5,  # Require at least 5 tokens
        stride=0,
        overflow=True,
        strategy="best_fit",
    )

    # Total is only 3 tokens, below min_len
    assert len(sequences) == 0


def test_pack_sequences_optimized_overflow_split():
    """Test document splitting with overflow=True."""
    documents = [
        Document([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 12, 0),
    ]

    sequences = pack_sequences_optimized(
        documents=documents,
        max_length=5,
        min_len=1,
        stride=0,
        overflow=True,
        strategy="best_fit",
    )

    # Should split into 3 chunks: [1-5], [6-10], [11-12]
    assert len(sequences) == 3
    assert len(sequences[0]) == 5
    assert len(sequences[1]) == 5
    assert len(sequences[2]) == 2


def test_pack_sequences_optimized_no_overflow_truncate():
    """Test document truncation with overflow=False."""
    documents = [
        Document([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 12, 0),
    ]

    sequences = pack_sequences_optimized(
        documents=documents,
        max_length=5,
        min_len=1,
        stride=0,
        overflow=False,
        strategy="best_fit",
    )

    # Should truncate to max_length
    assert len(sequences) == 1
    assert len(sequences[0]) == 5
    assert sequences[0] == [1, 2, 3, 4, 5]


def test_pack_sequences_best_fit_vs_first_fit():
    """Compare best_fit vs first_fit packing efficiency."""
    # Create documents that demonstrate difference between strategies
    documents = [
        Document([1] * 7, 7, 0),  # 7 tokens
        Document([2] * 3, 3, 1),  # 3 tokens
        Document([3] * 3, 3, 2),  # 3 tokens
        Document([4] * 5, 5, 3),  # 5 tokens
    ]

    best_fit_seqs = pack_sequences_optimized(
        documents=documents,
        max_length=10,
        min_len=1,
        stride=0,
        overflow=True,
        strategy="best_fit",
    )

    first_fit_seqs = pack_sequences_optimized(
        documents=documents,
        max_length=10,
        min_len=1,
        stride=0,
        overflow=True,
        strategy="first_fit",
    )

    # Both should pack into 2 bins, but potentially with different arrangements
    print(
        f"Best fit sequences: {len(best_fit_seqs)}, sizes: {[len(s) for s in best_fit_seqs]}"
    )
    print(
        f"First fit sequences: {len(first_fit_seqs)}, sizes: {[len(s) for s in first_fit_seqs]}"
    )

    # Both should achieve same number of bins for this case
    assert len(best_fit_seqs) == len(first_fit_seqs)


if __name__ == "__main__":
    test_bin_basic()
    test_bin_with_stride_tokens()
    test_bin_can_fit()
    test_split_document_no_split_needed()
    test_split_document_simple_split()
    test_split_document_with_stride()
    test_split_document_with_bos()
    test_pack_single_document_best_fit()
    test_pack_single_document_first_fit()
    test_pack_single_document_create_new_bin()
    test_pack_sequences_optimized_simple()
    test_pack_sequences_optimized_with_sorting()
    test_pack_sequences_optimized_min_len_filter()
    test_pack_sequences_optimized_overflow_split()
    test_pack_sequences_optimized_no_overflow_truncate()
    test_pack_sequences_best_fit_vs_first_fit()
    print("\nAll unit tests passed!")
