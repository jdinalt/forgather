import torch
import torch.nn.functional as F
from torch import nn
import math

class SubspaceProjector:
    def __init__(self, rank, dim, proj_type, update_steps):
        self.rank = rank
        self.proj_type = proj_type
        self.update_steps = update_steps
        self._step = 0
        self.proj_type = proj_type
        self.scale = math.sqrt(dim)/math.sqrt(rank)
        
        match self.proj_type:
            case "left":
                # dim = X.shape[0]
                self.dim = dim
                self.proj_shape = (dim, rank)
                self.einsum_down = "or,oi->ri"
                self.einsum_up = "ri,or->oi"
                self.einsum_grad = "ri,oi->or"
            case "right":
                # dim = X.shape[1]
                self.dim = dim
                self.proj_shape = (rank, dim)
                self.einsum_down = "ri,oi->ro"
                self.einsum_up = "ro,ri->oi"
                self.einsum_grad = "ro,oi->ri"
            case _:
                raise Exception(f"Unknow projection type {self.proj_type}")

    def down(self, x):
        return torch.einsum(self.einsum_down, self._projection_matrix(), x)

    def up(self, x):
        return torch.einsum(self.einsum_up, x, self._projection_matrix())

    def step(self, x):
        if self._step % self.update_steps == 0:
            self._update(x)
        self._step += 1

    def _update(self, x):
        pass

    def _projection_matrix(self):
        return None
        
class OnlinePCAProjector(SubspaceProjector):
    def __init__(self, rank, dim, proj_type, update_steps=10, orthag="none"):
        super().__init__(rank, dim, proj_type, update_steps)
        
        match orthag:
            case "qr":
                match self.proj_type:
                    case "left":
                        self.orthonormalize = lambda x: torch.linalg.qr(x).Q
                    case "right":
                        self.orthonormalize = lambda x: torch.linalg.qr(x.T).Q.T
            case "norm":
                match self.proj_type:
                    case "left":
                        self.orthonormalize = lambda x: x / torch.linalg.norm(x, dim=0)
                    case "right":
                        self.orthonormalize = lambda x: x / torch.linalg.norm(x, dim=1).view(-1, 1)
            case "none":
                self.orthonormalize = lambda x: x
            case _:
                raise Exception(f"Unknow orthagonalization {orthag}")
        self.A = None
    
    def _projection_matrix(self):
        return self.A
    
    def _update(self, x):
        if self.A is None:
            self.A = torch.empty(*self.proj_shape, device=x.device, dtype=x.dtype)
            nn.init.orthogonal_(self.A)
            self._fit_projection(x)
        else:
            self._fit_projection(x, max_steps=1, dloss_target=None)

    @torch.no_grad()
    def _fit_projection(self, X, lr=1.0, max_steps=100, dloss_target=1e-4, epsilon=1e-6):
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

class RandProjector(SubspaceProjector):
    def __init__(self, rank, dim, proj_type, update_steps=10, init="normal", lazy=True, seed=None):
        super().__init__(rank, dim, proj_type, update_steps)
        self.init = init
        self.A = None
        self.gen = None
        self.seed = seed
        self.lazy = lazy

    def _projection_matrix(self):
        if not self.lazy:
            return self.A
        
        self.gen.set_state(self.saved_gen_state)
        m = torch.empty(*self.proj_shape, device=self.device, dtype=self.dtype)
        self._init_matrix(m)
        return m

    def _init_matrix(self, m):
        match(self.init):
            case "normal":
                nn.init.normal_(m, std=(1/math.sqrt(self.dim)), generator=self.gen)
            case "orthogonal":
                nn.init.orthogonal_(m, generator=self.gen)
            case _:
                raise Exception(f"Unknow init method {orthag}")
    
    def _update(self, x):
        if self.gen is None:
            self.gen = torch.Generator(device=x.device)
            if self.seed is not None:
                self.gen.manual_seed(self.seed)

        # Remember these for building the projection
        if self.lazy:
            self.saved_gen_state = self.gen.get_state()
            self.device = x.device
            self.dtype = x.dtype
            return
                
        if self.A is None:
            self.A = torch.empty(*self.proj_shape, device=x.device, dtype=x.dtype)
        self._init_matrix(self.A)
        
        
