import torch
from torch import nn, Tensor
import torch.nn.init as init
from torch.nn import functional as F


class BigramLM(nn.Module):
    def __init__(
        self,
        # d_model is the number of features in the model's embeddings, where a feature is a single floating-point scalar and
        # the embeddings are vectors, each of size d_model. This parameter is sometimes referred to a the model's
        # "hidden" dimension.
        d_model,
        # This is the vocabulary size of the model, which should match the size of the model's tokenizer.
        vocab_size,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # The class contains an array of embeddings, with each token-id in the vocabulary corresponding to the element at that index.
        # For example, self.embedding.weight[token_id] would refer to the features at the index 'token_id.'
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        # We will use a linear layer to convert embeddins into a probability distribution accross all of the token-ids, thus
        # it has an input size of d_model and an output size of vocab_size.
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)

    def forward(
        self, input_ids: Tensor, labels: Tensor = None, attention_mask: Tensor = None
    ):
        # input_ids (batch_size, seq_len):
        #    This contains batches of sequences of token-ids, representing the input text.
        # labels (batch_size, seq_len): If given these are the ground-truth targets the model is striving to predict.
        #    For a causal model, these are identical to the input-ids, with a special value of -100
        #    reserved for padding, which are not scored.
        # attention_mask: The Huggingface APIs pass this in, although we don't use it.

        # Convert input_ids to embeddings.
        x = self.embedding(input_ids)

        # Convert embeddings to log-probabilities of next token-id
        # We could convert the logits to probabilities (0.0 to 1.0) with
        # torch.softmax(logits, dim=-1)
        logits = self.output_projection(x)

        # If we are passed labels, we will compute loss and return both loss and logits.
        if labels is not None:
            loss = self.loss(logits, labels)
            return (loss, logits)
        # Otherwise, we only return the logits.
        else:
            return logits

    def loss(self, logits, labels):
        # Shift so that tokens < n predict n
        # To achieve this, we slice off the last prediction, as we don't have a label
        # corresponding to it and slice off the first label, as nothing preceeds it for which
        # a prediction could be made.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # The loss meaures the error between the ground-truth (lables) and what the model predicted (logits).
        # If the model makes a perfect prediction, the loss will be zero, otherwise, it will be a positive
        # log-scaled measure of the error.
        #
        # For each label, the model makes a prediction for every token in the vocabulary, with the logits being
        # a log-probability distribution of the prediction. Cross-entroy-loss compares the model's predicted
        # distribution with a "one-hot" distribution -- that is, a probability distribution with 1.0 where the label
        # is and 0.0 for all other elements in the vocabulary.
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            # labels with this value are ignored when computing loss
            ignore_index=-100,
            reduction="mean",
        )

        # Allowing the model to return NaN can cause problems, so we convert these values to a number.
        return loss.nan_to_num()
