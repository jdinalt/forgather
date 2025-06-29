from typing import Optional, Tuple, Callable
from torch import nn, Tensor, LongTensor, FloatTensor


class CausalRpeLM(nn.Module):
    def __init__(
        self,
        loss_fn: Callable,
        input_encoder: Callable,
        output_decoder: Callable,
        layer_stack: Callable,
        init_weights: Callable,
        relative_pe: Callable,
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.input_encoder = input_encoder
        self.output_decoder = output_decoder
        self.layer_stack = layer_stack
        self.relative_pe = relative_pe()
        init_weights(self)

    def extra_repr(self):
        return f"loss_fn={self.loss_fn}"

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
        seq_len = hidden_states.shape[1]
        pos_embeddings = self.relative_pe(seq_len)

        # Pass the relative positional embeddings to all layers via kwargs
        # Each attention layer will use these shared frequencies
        rope_kwargs = {
            "pos_emb": pos_embeddings,
            "attention_mask": attention_mask,
            **kwargs,
        }

        # Pass the input through each of the layers
        hidden_states = self.layer_stack(hidden_states, **rope_kwargs)

        # Convert embeddings to log-probabilities of next token-id
        logits = self.output_decoder(hidden_states)

        # Compute loss
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return dict(loss=loss, logits=logits)
