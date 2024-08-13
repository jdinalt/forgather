from typing import Optional, Tuple, Callable

from torch import nn, Tensor, LongTensor, FloatTensor


class CasualLM(nn.Module):

    def __init__(
        self,
        loss_fn_factory: Callable,
        input_encoder_factory: Callable,
        output_decoder_factory: Callable,
        layer_stack_factory: Callable,
        init_weights_factory: Callable,
    ):
        super().__init__()

        self.loss_fn = loss_fn_factory()
        self.input_encoder = input_encoder_factory()
        self.output_decoder = output_decoder_factory()
        self.layer_stack = layer_stack_factory()
        self.init_weights = init_weights_factory()
        self.init_weights(self)

    def extra_repr(self):
        return f"loss_fn={self.loss_fn}, init_weights={self.init_weights}"

    def forward(
        self,
        input_ids: LongTensor,
        labels: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        **kwargs,
    ) -> dict[str, FloatTensor]:
        # Convert input_ids to embeddings and add positional information.
        hidden_states = self.input_encoder(input_ids, position_ids)

        # Pass the input through each of the layers.
        hidden_states = self.layer_stack(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Convert embeddings to log-probabilities of next token-id
        logits = self.output_decoder(hidden_states)

        # Compute loss.
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return dict(loss=loss, logits=logits)
