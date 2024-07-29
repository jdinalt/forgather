from typing import Optional
import math

import torch
from torch import nn, Tensor


# Converts a torch array of integers into their equivalent binary codes.
def binary_tensor(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def hadamard_walsh_matrix(k: int):
    # k: The dimension of the matrix is 2^k
    assert k > 0

    # Start with Hadamard H2^1 matrix.
    h1 = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)

    # The series of matrices can be computed by recurisvely applying the Kronecker product,
    # starting with h1.
    #
    # This will produce the series of Hadamard-Wlash matrices in natural order.
    w = h1
    for _ in range(k - 1):
        w = torch.kron(h1, w)

    return w


# This positional encoder adds absolute binary positions to the embedding, encoded via
# Hadamard-Walsh matrix.
#   See: https://en.wikipedia.org/wiki/Hadamard_code
# Each bit in the binary code word is encoded via a row the Hadamard-Walsh matrix, with a
# 1 being encoded by the presense of the row and a 0 by its absence. While training, the base
# sequence offset is randomly selected, which appears to allow the model to generalize to
# sequences longer than it was trained on. This is similar to what is described here:
# https://arxiv.org/pdf/2305.16843.pdf
#   I have tried this approach and found that my approach works better for generalization.
#
# Note: Without random shifting, the early performance of this encoder is exceptionally good.
#   The drawback is that the model can't generalize to longer sequences than it was trained on
#   and can't easily accomidate additonal bits later in the training process.
class WalshPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_sequence_length: int,
        gain: float = 0.333,
        shift: bool = False,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        # Gain controls the weight given to the encodings.
        # When a trainable parameter, the value appears to settle at around 0.333.
        self.gain = gain

        # Randomly shirt the base position to facilitate generalization to longer
        # sequence lengths.
        self.shift = shift

        # Hadamard-Walsh k, where the dimension of the matrix is 2^k
        k = math.ceil(math.log2(d_model))

        # The number of bits required to encode max_sequence_length
        bits = math.ceil(math.log2(max_sequence_length))

        assert (
            bits <= d_model
        ), "max_sequence_length exceeds n-bits available for d_model"

        # Generate sequential binary codes for absolute positionals.
        # The implementation originally used Grey codes, which where successive symbols
        # differ by by only one bit. See: https://en.wikipedia.org/wiki/Gray_code
        # This, along with a few other coding schemes were tested, with a simple
        # binary code having the best performance.
        binary_code = binary_tensor(torch.arange(0, max_sequence_length, 1), bits) - 0.5
        self.register_buffer("binary_code", binary_code, persistent=False)

        # Each bit is encoded via a row of a Hadamard-Walsh matrix.
        # We slice off the unused rows and columns -- ideally, d_model should be
        # the same dimension as the matrix.
        walsh = hadamard_walsh_matrix(k)[:bits, :d_model] * self.gain

        # This alternative appears superior to the original.
        # If starting from scratch, this use this.
        # walsh = (hadamard_walsh_matrix(k)[:bits,:d_model] -0.5) * self.gain
        self.register_buffer("walsh", walsh, persistent=False)

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, max_sequence_length={self.max_sequence_length}, "
            f"gain={self.gain}, shift={self.shift}"
        )

    def forward(
        self, seq_length: int, *, position_ids: Optional[torch.LongTensor] = None
    ) -> Tensor:
        # Get sequence of binary codes...
        # We use a random base offset when training.
        # This results in slower initial gains, but appears to allow the model to generalize to
        # the value of max_sequence_length, even if never trained with sequences of this length. I also have
        # a suspicion that this has a regularizing effect on training, similar to dropout. Models with
        # random base offset shifting, despite slower initial improvement, appear to perform better in the long-run.
        # TODO: Setup a controlled experiment to test this hypothesis.
        if self.shift and self.training:
            shift = torch.randint(
                self.max_sequence_length - seq_length + 1, (1,)
            ).item()
            seq = self.binary_code[shift : seq_length + shift, :]

        # When the cache is used for generation, after the first call, we are only passed a single token at a time,
        # with the remaining tokens being in the cache. We need to make sure that the newly injected tokens have the
        # correct relative position by indexing the codes with the position_ids.
        elif position_ids != None:
            seq = self.binary_code[position_ids, :]

        # Disable shifting when not training. This does not appear to change the evaluation loss, but
        # it does makes predictions easier to analyse when the attention weights are not shifting with each step.
        else:
            seq = self.binary_code[:seq_length, :]

        # Encode binary sequence with Hadamard-Walsh codes and apply to embeddings.
        # If nothing else, the Walsh encodings make the positional information exceptionally
        # robust with respect to dropout and other adversities. They can still be easily detected
        # at the final layer.
        return seq @ self.walsh
