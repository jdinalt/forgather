from typing import Optional, Tuple, Callable

from torch import nn, Tensor, LongTensor, FloatTensor

from .rotary_embeddings import precompute_freqs_cis


class CausalRoPELM(nn.Module):
    """
    Causal Language Model with efficient Rotary Position Embeddings (RoPE).
    
    This model computes RoPE frequencies once and passes them through all layers,
    following the TorchTitan approach for memory efficiency in large models.
    """

    def __init__(
        self,
        loss_fn: Callable,
        input_encoder: Callable,
        output_decoder: Callable,
        layer_stack: Callable,
        init_weights: Callable,
        *,
        d_head: int,
        max_sequence_length: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.input_encoder = input_encoder
        self.output_decoder = output_decoder
        self.layer_stack = layer_stack
        self.max_sequence_length = max_sequence_length
        
        # Precompute RoPE frequencies once for the entire model
        # This is more memory efficient than storing frequencies in each attention layer
        freqs_cis = precompute_freqs_cis(d_head, max_sequence_length, rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)
        
        init_weights(self)

    def extra_repr(self):
        return f"loss_fn={self.loss_fn}, max_sequence_length={self.max_sequence_length}"

    def forward(
        self,
        input_ids: LongTensor,
        labels: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        **kwargs,
    ) -> dict[str, FloatTensor]:
        # Convert input_ids to embeddings and add positional information
        hidden_states = self.input_encoder(input_ids, position_ids)
        
        # Get sequence length for RoPE frequency slicing
        seq_len = hidden_states.size(1)
        
        # Pass the RoPE frequencies to all layers via kwargs
        # Each attention layer will use these shared frequencies
        rope_kwargs = {
            'freqs_cis': self.freqs_cis[:seq_len],
            'attention_mask': attention_mask,
            **kwargs
        }

        # Pass the input through each of the layers
        hidden_states = self.layer_stack(hidden_states, **rope_kwargs)

        # Convert embeddings to log-probabilities of next token-id
        logits = self.output_decoder(hidden_states)

        # Compute loss
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return dict(loss=loss, logits=logits)