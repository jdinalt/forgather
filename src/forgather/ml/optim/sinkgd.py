import math
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch import nn


def sr_sinkhorn(X: Tensor, num_iters: int, eps: float = 1e-8) -> Tensor:
    """
    SR-Sinkhorn normalization (Algorithm 3 from the paper).

    Alternates between row-wise and column-wise l2 normalization. At convergence,
    all row l2-norms equal √n and all column l2-norms equal √m, giving a Frobenius
    norm of √(mn).

    Args:
        X: Input matrix of shape (m, n), float32.
        num_iters: Number of alternating normalization iterations (L).
        eps: Small constant clamped into denominators for numerical stability.

    Returns:
        Normalized matrix with Frobenius norm ≈ √(mn).
    """
    m, n = X.shape
    sqrt_n = math.sqrt(n)
    sqrt_m = math.sqrt(m)
    for _ in range(num_iters):
        # P_{g1}(X) = √n * Q(X)^{-1} * X  (normalize each row by its l2-norm)
        row_norms = X.norm(dim=1, keepdim=True).clamp(min=eps)
        X = (sqrt_n / row_norms) * X
        # P_{g2}(X) = √m * X * R(X)^{-1}  (normalize each column by its l2-norm)
        col_norms = X.norm(dim=0, keepdim=True).clamp(min=eps)
        X = X * (sqrt_m / col_norms)
    return X


class SinkGD(Optimizer):
    """
    SinkGD: Sinkhorn Gradient Descent (Algorithm 4).

    A stateless optimizer that pre-processes gradients via alternating row-wise
    and column-wise l2 normalization (the SR-Sinkhorn procedure, Algorithm 3).
    Requires no optimizer state for the parameters it handles, giving it the
    same memory footprint as SGD.

    From: "Gradient Multi-Normalization for Stateless and Scalable LLM Training"
    Scetbon, Ma, Gong, Meeds -- arXiv:2502.06742

    The SR-Sinkhorn output has Frobenius norm ≈ √(mn). To keep lr on the same
    scale as Adam, the update is divided by √(mn) before the parameter step
    (controlled by `normalize_output`).

    For parameters with fewer than 2 dimensions (biases, layer-norm scales) the
    gradient is l2-normalized instead.

    Typical paper settings (LLaMA pre-training):
    - num_iters = 5
    - Linear layers: SinkGD with lr = 0.05 * global_lr
    - Embeddings / norms / output projection: Adam with global_lr

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate. Roughly matches Adam when normalize_output=True.
        num_iters: SR-Sinkhorn iterations L. Paper uses 5; 1 is cheaper with
            minor loss in quality (see Table 4 in the paper).
        weight_decay: Decoupled weight decay (AdamW-style, scales with lr).
        eps: Denominator clamp for row/column norms.
        normalize_output: Divide SR-Sinkhorn output by √(mn) so that lr is
            comparable to Adam's lr. Default: True.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        num_iters: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        normalize_output: bool = True,
    ):
        defaults = dict(
            lr=lr,
            num_iters=num_iters,
            weight_decay=weight_decay,
            eps=eps,
            normalize_output=normalize_output,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            num_iters = group["num_iters"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            normalize_output = group["normalize_output"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Decoupled weight decay (AdamW-style)
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                grad = p.grad.float()

                if grad.dim() >= 2:
                    orig_shape = grad.shape
                    if grad.dim() > 2:
                        # Fold leading dims into rows, keep last dim as columns.
                        # Matches Adafactor's convention for conv / higher-D weights.
                        grad = grad.reshape(-1, grad.shape[-1])

                    update = sr_sinkhorn(grad, num_iters, eps)

                    if normalize_output:
                        # Rescale to unit Frobenius norm so lr is Adam-comparable.
                        # SR-Sinkhorn output has ||update||_F ≈ √(mn).
                        m, n = update.shape
                        update = update * (1.0 / math.sqrt(m * n))

                    update = update.reshape(orig_shape)
                elif grad.dim() == 1:
                    # Vectors: normalize by l2-norm
                    update = grad / grad.norm().clamp(min=eps)
                else:
                    # Scalar: use gradient directly
                    update = grad

                if p.dtype == update.dtype:
                    p.add_(update, alpha=-lr)
                else:
                    p.copy_(p.float() - lr * update)

        return loss
