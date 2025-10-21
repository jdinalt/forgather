"""
This module provides functionality to tokenize input text into blocks of tokens,
which is useful for processing long sequences that exceed model input limits.
It includes classes for managing input and output token blocks, as well as a function
to perform the tokenization with various options such as overflow handling, stride,
and special token addition.
"""


class InputTokenBlock:
    def __init__(self, input_ids, length):
        self.length = length
        self.input_ids = input_ids
        self.read_index = 0

    def __len__(self):
        return self.length

    def read(self, length):
        length = min(self.length, length)
        ids = self.input_ids[self.read_index : self.read_index + length]
        self.read_index += length
        self.length -= length
        return ids


class OutputTokenBlock:
    def __init__(self, max_length, input_ids=None):
        self.max_length = max_length
        if input_ids is not None:
            self.input_ids = input_ids
            self.length = len(input_ids)
            assert self.length <= max_length
        else:
            self.length = 0
            self.input_ids = []

    def __len__(self):
        return self.length

    def remaining(self):
        return self.max_length - self.length

    def get_ids(self):
        return self.input_ids

    def append(self, input_block):
        length = min((self.max_length - self.length), len(input_block))
        input_ids = input_block.read(length)
        self.input_ids += input_ids
        self.length += length


def block_tokenize_fn(
    element,
    tokenizer,
    feature,
    block_size=32,
    overflow=True,
    stride=0,
    min_len=1,
    max_len=None,
    add_bos=True,
    add_eos=False,
    combine=False,
    truncate_at=None,
):
    """
    Tokenizes the input element into blocks of tokens.

    The typical use case is to tokenize text into blocks of a given size, with options for overflow,
    stride, and special tokens. Useful when examples are too long for the model's maximum input size.

    Args:
        element: The input element to tokenize.
        tokenizer: The tokenizer to use for tokenization.
        feature: The feature in the element to tokenize.
        block_size: The maximum size of each output block.
        overflow: If True, allows overflow of input tokens into multiple outputs.
        stride: Number of tokens to overlap between blocks.
        min_len: Minimum length of the output blocks.
        max_len: Maximum length of the output blocks (optional).
        add_bos: If True, adds a beginning-of-sequence token.
        add_eos: If True, adds an end-of-sequence token.
        combine: If True, combines input into multiple outputs.
        truncate_at: If provided, truncates input at the first match of this regex.
    Returns:
        A dictionary with a single key "input_ids" containing a list of tokenized blocks.
    """
    assert min_len >= 1

    # If given a regex to truncate at, truncate at the first match.
    if truncate_at is not None:
        input_batch = []
        for text in element[feature]:
            match_offset = re.search(truncate_at, text)
            if match_offset is not None:
                text = text[: match_offset.start()]
            input_batch.append(text)
    else:
        input_batch = element[feature]

    outputs = tokenizer(
        input_batch,
        truncation=False,
        return_length=True,
        # Silence warning about exceeding model's max length
        # We are performing the truncation ourselves.
        max_length=9223372036854775807,
    )

    # A list of strings of tokens of maximum size 'block_size'
    output_batch = []

    # A container for accumulating output tokens.
    output_block = OutputTokenBlock(block_size)

    # A container for the input tokens from the current record in the input batch.
    input_block = None

    # Appends the output block to the output_batch
    # - Conditional upon minimum length
    # - Allocates next output block
    # - Transfers 'stride' tokens from end of old block to start of new block.
    def append_output_batch(output_block, output_batch, stride):
        stride_tokens = None
        # If the present output block is empty, just return and keep the current one.
        if not len(output_block):
            return output_block
        # If the output has at least the minimum number of tokens.
        elif len(output_block) >= min_len:
            # Save 'stride' tokens from the end to prefix the next block with
            if add_bos:
                stride_tokens = [tokenizer.bos_token_id]
            else:
                stride_tokens = []

            if stride != 0:
                stride_tokens += output_block.get_ids()[-stride:]

            # Append the block to the list of output blocks
            output_batch.append(output_block.get_ids())
        # else, we discard the output block

        # Allocate a new output block, initialized with 'stride' tokens
        output_block = OutputTokenBlock(block_size, stride_tokens)
        return output_block

    # Get next tokenized input record
    for record_length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        # print(f"record length {record_length}")
        # If we are not allowed to mix inputs in outputs, get a new output.
        if not combine:
            output_block = append_output_batch(output_block, output_batch, 0)

        # If the length of the record is less than the minimum, discard it.
        if record_length < min_len:
            continue

        # If the input record is longer than the maximum, discard it.
        if max_len is not None and record_length > max_len:
            continue

        # If we will be adding the EOS token, add it now.
        if add_eos:
            record_length += 1
            input_ids += [tokenizer.eos_token_id]

        # Encapsulate the input record in an input block
        input_block = InputTokenBlock(input_ids, record_length)

        # While the input block still has data to read...
        while len(input_block):
            # Move as much data from the input block to the output block as will fit.
            # Note: These classes perform bounds checking to prevent overflow/underflow.
            output_block.append(input_block)

            # If we will not being combining the input into multiple outputs.
            if not overflow:
                # Add to outputs and get next input.
                output_block = append_output_batch(output_block, output_batch, 0)
                break

            # If the output block is mostly full, allocate a new output block
            elif output_block.remaining() < min_len:
                # Add to outputs and continue with present input.
                output_block = append_output_batch(output_block, output_batch, stride)

    # Append the last output data.
    append_output_batch(output_block, output_batch, 0)
    return {"input_ids": output_batch}
