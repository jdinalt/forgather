from torch.nn.functional import cross_entropy
from torch import FloatTensor, LongTensor


class CausalLoss:
    def __repr__(self):
        return f"{type(self).__name__}()"

    @staticmethod
    def __call__(logits: FloatTensor, labels: LongTensor, **kwargs) -> FloatTensor:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            # labels with this value are ignored when computing loss
            ignore_index=-100,
            reduction="mean",
        )

        return loss
