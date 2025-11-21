from typing import Optional
import contextlib
import functools

from torch.nn.functional import cross_entropy
from torch import Tensor, FloatTensor, LongTensor
import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_loss_fn(logits: FloatTensor, labels: LongTensor) -> FloatTensor:
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


def _chunked_causal_loss_fn(
    logits: FloatTensor,
    labels: LongTensor,
    chunk_size: int = 4096,
    ignore_index: int = -100,
) -> FloatTensor:
    """
    Compute causal cross-entropy loss with chunked vocabulary processing.

    This reduces peak memory by processing the vocabulary in chunks rather than
    materializing the full [batch*seq, vocab_size] logits tensor at once.

    Uses the log-sum-exp trick to compute cross-entropy correctly across chunks:
    1. Compute max logit value across all chunks (for numerical stability)
    2. For each chunk: compute exp(logits - max) and accumulate
    3. Compute log(sum_exp) and final cross-entropy

    Args:
        logits: [batch, seq_len, vocab_size] unnormalized logits
        labels: [batch, seq_len] target token indices
        chunk_size: Number of vocabulary elements to process at once
        ignore_index: Label value to ignore in loss computation

    Returns:
        Scalar loss tensor
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten to [N, vocab_size] where N = batch * (seq_len - 1)
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    vocab_size = flat_logits.size(-1)
    n_tokens = flat_logits.size(0)

    # Mask for valid (non-ignored) tokens
    valid_mask = flat_labels != ignore_index
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Step 1: Find global max logit for numerical stability (only over valid tokens)
    max_logit = torch.full(
        (n_tokens,), float("-inf"), device=logits.device, dtype=logits.dtype
    )

    for start_idx in range(0, vocab_size, chunk_size):
        end_idx = min(start_idx + chunk_size, vocab_size)
        chunk_logits = flat_logits[:, start_idx:end_idx]
        chunk_max = chunk_logits.max(dim=-1).values
        max_logit = torch.maximum(max_logit, chunk_max)

    # Step 2: Accumulate exp(logits - max) and compute target logits
    sum_exp = torch.zeros(n_tokens, device=logits.device, dtype=logits.dtype)
    target_logits = torch.zeros(n_tokens, device=logits.device, dtype=logits.dtype)

    for start_idx in range(0, vocab_size, chunk_size):
        end_idx = min(start_idx + chunk_size, vocab_size)
        chunk_logits = flat_logits[:, start_idx:end_idx]

        # Accumulate exp(logit - max) for log-sum-exp
        chunk_exp = torch.exp(chunk_logits - max_logit.unsqueeze(-1))
        sum_exp += chunk_exp.sum(dim=-1)

        # Extract target logits for tokens whose labels fall in this chunk
        chunk_labels = flat_labels - start_idx
        in_chunk = (chunk_labels >= 0) & (chunk_labels < (end_idx - start_idx))

        if in_chunk.any():
            # Gather the logits corresponding to the true labels in this chunk
            indices = chunk_labels[in_chunk].unsqueeze(-1)
            selected_logits = torch.gather(
                chunk_logits[in_chunk], dim=-1, index=indices
            ).squeeze(-1)
            target_logits[in_chunk] = selected_logits

    # Step 3: Compute cross-entropy = -target_logit + log(sum_exp)
    # CE = -log(exp(target_logit - max) / sum_exp)
    #    = -(target_logit - max) + log(sum_exp)
    #    = -target_logit + max + log(sum_exp)
    log_sum_exp = max_logit + torch.log(sum_exp)
    token_losses = -target_logits + log_sum_exp

    # Step 4: Apply mask and reduce
    valid_losses = token_losses[valid_mask]
    loss = valid_losses.mean()

    return loss


class CausalLoss:
    def __init__(self, compile=False):
        super().__init__()
        if compile:
            self.loss_fn = torch.compile(_causal_loss_fn)
        else:
            self.loss_fn = _causal_loss_fn
        self.compile = compile

    def __repr__(self):
        return f"{type(self).__name__}(compile={self.compile})"

    def __call__(self, logits: FloatTensor, labels: LongTensor) -> FloatTensor:
        return self.loss_fn(logits, labels)


class ChunkedCausalLoss:
    """
    Memory-efficient causal loss for large vocabulary models.

    Processes vocabulary in chunks to reduce peak memory usage. Particularly
    beneficial for models with very large vocabularies (e.g., 150K+ tokens)
    where full logits tensor can consume several GB of memory.

    Memory savings example (Qwen3 with vocab_size=151936):
    - Standard loss: ~5GB for logits alone (batch=8, seq=2048, bf16)
    - Chunked loss (chunk_size=4096): ~140MB peak per chunk

    Args:
        chunk_size: Number of vocabulary elements to process at once.
                    Smaller values use less memory but increase computation time.
                    Recommended: 4096-8192 for vocabularies over 100K.
        compile: Whether to use torch.compile() for the loss function.
                 Note: Compilation may reduce chunking benefits.
    """

    def __init__(self, chunk_size: int = 4096, compile: bool = False):
        super().__init__()
        self.chunk_size = chunk_size
        self.compile = compile

        if compile:
            # Note: torch.compile may try to fuse chunks, reducing memory benefit
            self.loss_fn = torch.compile(
                lambda logits, labels: _chunked_causal_loss_fn(
                    logits, labels, chunk_size=self.chunk_size
                )
            )
        else:
            self.loss_fn = lambda logits, labels: _chunked_causal_loss_fn(
                logits, labels, chunk_size=self.chunk_size
            )

    def __repr__(self):
        return f"{type(self).__name__}(chunk_size={self.chunk_size}, compile={self.compile})"

    def __call__(self, logits: FloatTensor, labels: LongTensor) -> FloatTensor:
        return self.loss_fn(logits, labels)


class FusedLinearCrossEntropy(nn.Module):
    """
    Fused linear output layer + chunked cross-entropy loss.

    This is the key to solving memory issues in pipeline parallel training with
    large vocabularies. Instead of:
        1. linear(hidden) → logits [B, S, V] (5GB+ for Qwen3)
        2. loss_fn(logits, labels) → scalar

    This module directly computes:
        1. chunked_linear_cross_entropy(hidden, labels) → scalar (no logits!)

    The logits tensor is NEVER fully materialized, preventing the 20GB memory
    spike on the last pipeline stage.

    Args:
        in_features: Hidden dimension size
        out_features: Vocabulary size
        chunk_size: Number of vocabulary elements to process at once
        bias: Whether to include bias in linear layer
        ignore_index: Label value to ignore in loss computation

    Usage:
        # Replace standard pattern:
        # output_layer = nn.Linear(hidden_dim, vocab_size)
        # loss = loss_fn(output_layer(hidden), labels)

        # With fused version:
        fused_layer = FusedLinearCrossEntropy(hidden_dim, vocab_size)
        loss = fused_layer(hidden, labels)  # Returns loss directly

        # For inference (get logits):
        logits = fused_layer.forward_logits(hidden)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        chunk_size: int = 4096,
        bias: bool = True,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

        # The underlying linear layer (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters same as nn.Linear"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward_logits(self, hidden_states: FloatTensor) -> FloatTensor:
        """
        Standard forward pass that returns logits (for inference).

        Args:
            hidden_states: [batch, seq_len, in_features]

        Returns:
            logits: [batch, seq_len, out_features]
        """
        return F.linear(hidden_states, self.weight, self.bias)

    def forward(
        self,
        hidden_states: FloatTensor,
        labels: Optional[LongTensor] = None,
    ) -> FloatTensor | tuple[FloatTensor, Optional[FloatTensor]]:
        """
        Fused forward pass that computes loss directly without materializing logits.

        Args:
            hidden_states: [batch, seq_len, in_features]
            labels: [batch, seq_len] target token indices

        Returns:
            If labels provided: scalar loss tensor
            If labels not provided: logits tensor (for inference)
        """
        if labels is None:
            # Inference mode: return logits
            return self.forward_logits(hidden_states)

        # Training mode: compute loss without materializing full logits
        return self._fused_forward_loss(hidden_states, labels)

    def _fused_forward_loss(
        self, hidden_states: FloatTensor, labels: LongTensor
    ) -> FloatTensor:
        """
        Compute cross-entropy loss by chunking over vocabulary dimension.

        This is the core optimization: we compute linear(hidden_states) in chunks
        and immediately consume each chunk in the loss calculation, never storing
        the full logits tensor.
        """
        # Shift for causal prediction: tokens < n predict n
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten to [N, in_features] where N = batch * (seq_len - 1)
        flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
        flat_labels = shift_labels.view(-1)

        n_tokens = flat_hidden.size(0)
        vocab_size = self.out_features

        # Mask for valid (non-ignored) tokens
        valid_mask = flat_labels != self.ignore_index
        n_valid = valid_mask.sum()

        if n_valid == 0:
            return torch.tensor(
                0.0, device=hidden_states.device, dtype=hidden_states.dtype
            )

        # Step 1: Find max logit across all vocab chunks (for numerical stability)
        max_logit = torch.full(
            (n_tokens,),
            float("-inf"),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for start_idx in range(0, vocab_size, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, vocab_size)
            # Compute logits for this vocabulary chunk: [n_tokens, chunk_size]
            chunk_weight = self.weight[start_idx:end_idx]
            chunk_bias = self.bias[start_idx:end_idx] if self.bias is not None else None
            chunk_logits = F.linear(flat_hidden, chunk_weight, chunk_bias)

            chunk_max = chunk_logits.max(dim=-1).values
            max_logit = torch.maximum(max_logit, chunk_max)

        # Step 2: Accumulate exp(logits - max) and extract target logits
        sum_exp = torch.zeros(
            n_tokens, device=hidden_states.device, dtype=hidden_states.dtype
        )
        target_logits = torch.zeros(
            n_tokens, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for start_idx in range(0, vocab_size, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, vocab_size)
            # Recompute logits for this chunk (trading compute for memory)
            chunk_weight = self.weight[start_idx:end_idx]
            chunk_bias = self.bias[start_idx:end_idx] if self.bias is not None else None
            chunk_logits = F.linear(flat_hidden, chunk_weight, chunk_bias)

            # Accumulate exp(logit - max) for log-sum-exp
            chunk_exp = torch.exp(chunk_logits - max_logit.unsqueeze(-1))
            sum_exp += chunk_exp.sum(dim=-1)

            # Extract target logits for labels in this chunk
            chunk_labels = flat_labels - start_idx
            in_chunk = (chunk_labels >= 0) & (chunk_labels < (end_idx - start_idx))

            if in_chunk.any():
                indices = chunk_labels[in_chunk].unsqueeze(-1)
                selected_logits = torch.gather(
                    chunk_logits[in_chunk], dim=-1, index=indices
                ).squeeze(-1)
                target_logits[in_chunk] = selected_logits

        # Step 3: Compute cross-entropy
        log_sum_exp = max_logit + torch.log(sum_exp)
        token_losses = -target_logits + log_sum_exp

        # Step 4: Apply mask and reduce
        valid_losses = token_losses[valid_mask]
        loss = valid_losses.mean()

        return loss

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"chunk_size={self.chunk_size}, bias={self.bias is not None}"
        )


class LinearCrossEntropyLoss:
    """
    General wrapper for fused linear + cross-entropy implementations.

    This class provides a unified interface for different fused loss implementations,
    supporting automatic fallback and dependency injection. It solves the memory
    problem for large vocabulary models by computing loss directly from hidden states
    without materializing the full logits tensor.

    Supported implementations:
    - "cce": Apple's Cut Cross-Entropy (Triton kernels, best memory: 83% reduction)
    - "liger": LinkedIn's Liger Kernel (best framework integration, 60% reduction)
    - "pytorch": Pure PyTorch implementation (best compatibility, 43% reduction)
    - "auto": Automatically select best available (tries Liger → CCE → PyTorch)

    Memory savings example (Qwen3 1.7B, vocab=151936, batch=1, seq=4096):
    - Standard approach: 10.5 GB peak
    - Apple CCE: 1.8 GB (83% reduction)
    - Liger: ~5 GB (50% reduction, estimated)
    - PyTorch: 6.0 GB (43% reduction)

    Args:
        output_embeddings: The output layer from model.get_output_embeddings().
                          Can be nn.Linear or any module with .weight and optional .bias
        impl: Implementation to use ("cce", "liger", "pytorch", "auto")
        chunk_size: For pytorch impl, chunk size for vocabulary processing
        ignore_index: Label value to ignore in loss computation
        **kwargs: Additional arguments passed to the underlying implementation

    Usage:
        # Extract output embeddings from model
        output_embeddings = model.get_output_embeddings()

        # Create fused loss with automatic backend selection
        loss_fn = LinearCrossEntropyLoss(
            output_embeddings=output_embeddings,
            impl="auto"  # or "cce", "liger", "pytorch"
        )

        # Use in trainer (trainer detects via hasattr(loss_fn, 'forward_logits'))
        trainer = Trainer(
            model=model,
            compute_loss_func=loss_fn,
            ...
        )

        # Direct usage
        loss = loss_fn(hidden_states, labels)  # Training
        logits = loss_fn.forward_logits(hidden_states)  # Inference

    Integration with Trainer:
        The trainer automatically detects fused loss via hasattr(loss_fn, 'forward_logits').
        When detected:
        1. Model returns hidden states instead of logits
        2. Trainer extracts hidden states from model outputs
        3. Loss is computed directly: loss_fn(hidden_states, labels)
    """

    def __init__(
        self,
        output_embeddings: nn.Module,
        impl: str = "auto",
        chunk_size: int = 4096,
        ignore_index: int = -100,
        compile: bool = False,
        **kwargs,
    ):
        self.output_embeddings = output_embeddings
        self.requested_impl = impl
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self.compile = compile
        self.kwargs = kwargs

        # Extract weight and bias from output embeddings
        self.weight = output_embeddings.weight
        self.bias = getattr(output_embeddings, "bias", None)

        # Select and initialize implementation
        self.actual_impl, self._compute_fn = self._select_implementation(impl)
        if self.compile:
            self._compute_fn = torch.compile(self._compute_fn)

    def _select_implementation(self, impl: str) -> tuple[str, callable]:
        """
        Select and initialize the loss implementation.

        Returns:
            (actual_impl_name, compute_function)
        """
        import logging

        logger = logging.getLogger(__name__)

        if impl == "auto":
            # Try implementations in order: Liger → CCE → PyTorch
            # Liger first because it has better framework integration
            for candidate in ["liger", "cce", "pytorch"]:
                try:
                    actual_impl, compute_fn = self._select_implementation(candidate)
                    logger.info(
                        f"LinearCrossEntropyLoss: auto-selected '{actual_impl}' implementation"
                    )
                    return actual_impl, compute_fn
                except (ImportError, RuntimeError) as e:
                    logger.debug(
                        f"LinearCrossEntropyLoss: {candidate} not available: {e}"
                    )
                    continue

            raise RuntimeError(
                "No fused cross-entropy implementation available. "
                "Install cut-cross-entropy or liger-kernel, or use impl='pytorch'"
            )

        elif impl == "cce":
            try:
                from cut_cross_entropy import linear_cross_entropy

                logger.info("LinearCrossEntropyLoss: using Apple CCE implementation")
                return "cce", self._compute_cce
            except ImportError as e:
                raise ImportError(
                    "cut-cross-entropy not installed. Install with:\n"
                    '  pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"'
                ) from e

        elif impl == "liger":
            try:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

                logger.info("LinearCrossEntropyLoss: using Liger Kernel implementation")

                # Initialize Liger loss function
                self._liger_loss = LigerFusedLinearCrossEntropyLoss(
                    ignore_index=self.ignore_index, **self.kwargs
                )
                return "liger", self._compute_liger
            except ImportError as e:
                raise ImportError(
                    "liger-kernel not installed. Install with:\n"
                    "  pip install liger-kernel"
                ) from e

        elif impl == "pytorch":
            logger.info("LinearCrossEntropyLoss: using pure PyTorch implementation")
            return "pytorch", self._compute_pytorch

        else:
            raise ValueError(
                f"Unknown implementation: {impl}. "
                "Must be one of: 'auto', 'cce', 'liger', 'pytorch'"
            )

    def _compute_cce(
        self, hidden_states: FloatTensor, labels: LongTensor
    ) -> FloatTensor:
        """Use Apple's CCE implementation."""
        from cut_cross_entropy import linear_cross_entropy

        return linear_cross_entropy(
            hidden_states,
            self.weight,
            labels,
            bias=self.bias,
            shift=1,  # Automatic causal shifting
            ignore_index=self.ignore_index,
            impl="cce",  # Use optimized Triton kernels
            reduction="mean",
            **self.kwargs,
        )

    def _compute_liger(
        self, hidden_states: FloatTensor, labels: LongTensor
    ) -> FloatTensor:
        """Use Liger Kernel implementation."""
        # Liger expects: loss_fn(weight, input, target)
        # Shift for causal prediction
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
        flat_labels = shift_labels.view(-1)

        return self._liger_loss(self.weight, flat_hidden, flat_labels)

    def _compute_pytorch(
        self, hidden_states: FloatTensor, labels: LongTensor
    ) -> FloatTensor:
        """Use pure PyTorch chunked implementation."""
        # Shift for causal prediction
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
        flat_labels = shift_labels.view(-1)

        # Use the same chunked cross-entropy logic as FusedLinearCrossEntropy
        return self._fused_linear_cross_entropy_pytorch(
            flat_hidden,
            flat_labels,
            self.weight,
            self.bias,
            chunk_size=self.chunk_size,
            ignore_index=self.ignore_index,
        )

    def _fused_linear_cross_entropy_pytorch(
        self,
        hidden_states: FloatTensor,
        labels: LongTensor,
        weight: Tensor,
        bias: Optional[Tensor],
        chunk_size: int,
        ignore_index: int,
    ) -> FloatTensor:
        """
        Pure PyTorch implementation of fused linear + cross-entropy.

        This is the same algorithm as FusedLinearCrossEntropy._fused_forward_loss,
        but as a standalone function for use with external weight matrices.
        """
        n_tokens = hidden_states.size(0)
        vocab_size = weight.size(0)

        # Ensure weight and bias match hidden_states dtype (for mixed precision training)
        if weight.dtype != hidden_states.dtype:
            weight = weight.to(hidden_states.dtype)
        if bias is not None and bias.dtype != hidden_states.dtype:
            bias = bias.to(hidden_states.dtype)

        # Mask for valid tokens
        valid_mask = labels != ignore_index
        n_valid = valid_mask.sum()

        if n_valid == 0:
            return torch.tensor(
                0.0, device=hidden_states.device, dtype=hidden_states.dtype
            )

        # Step 1: Find max logit across all vocab chunks
        max_logit = torch.full(
            (n_tokens,),
            float("-inf"),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for start_idx in range(0, vocab_size, chunk_size):
            end_idx = min(start_idx + chunk_size, vocab_size)
            chunk_weight = weight[start_idx:end_idx]
            chunk_bias = bias[start_idx:end_idx] if bias is not None else None
            chunk_logits = F.linear(hidden_states, chunk_weight, chunk_bias)

            chunk_max = chunk_logits.max(dim=-1).values
            max_logit = torch.maximum(max_logit, chunk_max)

        # Step 2: Accumulate exp(logits - max) and extract target logits
        sum_exp = torch.zeros(
            n_tokens, device=hidden_states.device, dtype=hidden_states.dtype
        )
        target_logits = torch.zeros(
            n_tokens, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for start_idx in range(0, vocab_size, chunk_size):
            end_idx = min(start_idx + chunk_size, vocab_size)
            chunk_weight = weight[start_idx:end_idx]
            chunk_bias = bias[start_idx:end_idx] if bias is not None else None
            chunk_logits = F.linear(hidden_states, chunk_weight, chunk_bias)

            # Accumulate exp(logit - max)
            chunk_exp = torch.exp(chunk_logits - max_logit.unsqueeze(-1))
            sum_exp += chunk_exp.sum(dim=-1)

            # Extract target logits for labels in this chunk
            chunk_labels = labels - start_idx
            in_chunk = (chunk_labels >= 0) & (chunk_labels < (end_idx - start_idx))

            if in_chunk.any():
                indices = chunk_labels[in_chunk].unsqueeze(-1)
                selected_logits = torch.gather(
                    chunk_logits[in_chunk], dim=-1, index=indices
                ).squeeze(-1)
                target_logits[in_chunk] = selected_logits

        # Step 3: Compute cross-entropy
        log_sum_exp = max_logit + torch.log(sum_exp)
        token_losses = -target_logits + log_sum_exp

        # Step 4: Apply mask and reduce
        valid_losses = token_losses[valid_mask]
        loss = valid_losses.mean()

        return loss

    def __call__(self, hidden_states: FloatTensor, labels: LongTensor) -> FloatTensor:
        """
        Compute fused loss from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            labels: [batch, seq_len] target token indices

        Returns:
            Scalar loss tensor
        """
        return self._compute_fn(hidden_states, labels)

    def forward_logits(self, hidden_states: FloatTensor) -> FloatTensor:
        """
        Inference mode: materialize logits for generation.

        This method is used by the trainer to detect fused loss capability.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return F.linear(hidden_states, self.weight, self.bias)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"impl='{self.actual_impl}', "
            f"vocab_size={self.weight.size(0)}, "
            f"hidden_dim={self.weight.size(1)}, "
            f"chunk_size={self.chunk_size}), "
            f"compile={self.compile})"
        )


class RescaleLoss:
    """Rescales loss by specified factor

    This is adapted from Torch Titan
    """

    def __init__(self, unwrapped_loss_fn, scale_factor):
        self.unwrapped_loss_fn = unwrapped_loss_fn
        self.scale_factor = scale_factor
        self.skip_rescale = False

        functools.update_wrapper(self, unwrapped_loss_fn, updated=tuple())

    def __call__(self, *args, **kwargs):
        loss = self.unwrapped_loss_fn(*args, **kwargs)
        if self.skip_rescale:
            return loss
        return loss * self.scale_factor

    @contextlib.contextmanager
    def no_rescale(self):
        """Context manager for disabling rescaling"""
        previous = self.skip_rescale
        self.skip_rescale = True
        try:
            yield
        finally:
            self.skip_rescale = previous
