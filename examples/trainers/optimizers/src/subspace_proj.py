import torch
import torch.nn.functional as F
from torch import nn

class SubspaceProjector:
    def __init__(self, X, rank, orthag="qr", update_steps=1, proj_type="left"):
        if proj_type == "auto":
            if X.shape[0] > X.shape[1]:
                proj_type = "right"
            else:
                proj_type = "left"
        self.proj_type = proj_type
        
        match orthag:
            case "qr":
                match proj_type:
                    case "left":
                        self.orthonormalize = lambda x: torch.linalg.qr(x).Q
                    case "right":
                        self.orthonormalize = lambda x: torch.linalg.qr(x.T).Q.T
            case "norm":
                match proj_type:
                    case "left":
                        self.orthonormalize = lambda x: x / torch.linalg.norm(x, dim=0)
                    case "right":
                        self.orthonormalize = lambda x: x / torch.linalg.norm(x, dim=1).view(-1, 1)
            case "none":
                self.orthonormalize = lambda x: x
            case _:
                raise Exception(f"Unknow orthagonalization {orthag}")

        match proj_type:
            case "left":
                self.A = torch.empty(X.shape[0], rank, device=X.device, dtype=X.dtype)
                self.einsum_down = "or,oi->ri"
                self.einsum_up = "ri,or->oi"
                self.einsum_grad = "ri,oi->or"
            case "right":
                self.A = torch.empty(rank, X.shape[1], device=X.device, dtype=X.dtype)
                self.einsum_down = "ri,oi->ro"
                self.einsum_up = "ro,ri->oi"
                self.einsum_grad = "ro,oi->ri"
            case _:
                raise Exception(f"Unknow projection type {proj_type}")
        
        nn.init.orthogonal_(self.A)
        self.fit_projection_(X)
        self.max_steps = 1
        self.dloss_target = None
        self.update_steps = update_steps
        self.step = 0

    def down(self, X):
        return torch.einsum(self.einsum_down, self.A, X)

    def up(self, X):
        return torch.einsum(self.einsum_up, X, self.A)

    def update(self, X):
        if self.step % self.update_steps == 0:
            self.fit_projection_(X, max_steps=self.max_steps, dloss_target=self.dloss_target)
        self.step += 1

    @torch.no_grad()
    def fit_projection_(self, X, lr=1.0, max_steps=100, dloss_target=1e-4, epsilon=1e-6):
        var = X.var().item()
        lr = lr / (var + epsilon)
        if dloss_target is not None:
            dloss_target *= var
            prev_loss = None
    
        grad_scale = 2 / X.numel()
        
        for step in range(max_steps):
            down = self.down(X)
            up = self.up(down)
            error = up - X
    
            # Compute gradient
            dw = torch.einsum(self.einsum_grad, down, error * grad_scale)
    
            # SGD weight update
            self.A -= lr * dw
    
            # Orthonormalize / Normalize
            self.A = self.orthonormalize(self.A)
    
            # Early stopping?
            if dloss_target is not None:
                loss = error.square().mean().item()
                if prev_loss is not None:
                    dloss = loss - prev_loss
                    if dloss <= 0 and dloss >= -dloss_target:
                        #print(f"break at step {step}")
                        break
                prev_loss = loss