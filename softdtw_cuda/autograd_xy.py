from __future__ import annotations

import torch
from torch.autograd import Function

from .cuda.launcher import (
    softdtw_forward_cuda_fused_sqeuclid,
    softdtw_backward_cuda_fused_sqeuclid,   # returns E (exp(logE))
)


class SoftDTWXYAutograd(Function):
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        Y: torch.Tensor,
        gamma: float,
        bandwidth: float | None,
        lens_x: torch.Tensor | None = None,
        lens_y: torch.Tensor | None = None,
    ):
        # Forward CUDA fused: returns out (B,) and R (B,N+2,M+2)
        out, R = softdtw_forward_cuda_fused_sqeuclid(
            X, Y, float(gamma), -1.0 if bandwidth is None else float(bandwidth), lens_x, lens_y
        )

        # Save X,Y for gradient math; save detached R (no graph needed)
        to_save = [X, Y, R.detach()]
        if lens_x is not None:
            to_save.append(lens_x)
        if lens_y is not None:
            to_save.append(lens_y)
        ctx.lens_x_saved = lens_x is not None
        ctx.lens_y_saved = lens_y is not None
        ctx.save_for_backward(*to_save)
        ctx.gamma = float(gamma)

        # Normalize bandwidth semantics: <=0 means disabled
        if bandwidth is None:
            ctx.bandwidth = -1.0
        else:
            bw = float(bandwidth)
            ctx.bandwidth = -1.0 if bw <= 0 else bw

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = list(ctx.saved_tensors)
        X, Y, R = saved[0], saved[1], saved[2]
        rest = saved[3:]
        lens_x = rest.pop(0) if ctx.lens_x_saved else None
        lens_y = rest.pop(0) if ctx.lens_y_saved else None
        gamma = ctx.gamma
        bw = ctx.bandwidth

        # Compute E via fused log-space backward (Numba). Pass detached X/Y to be safe.
        # E is exactly 0 outside each sample's valid region, so the chain
        # rule below yields exactly-zero gradients on padding frames.
        E = softdtw_backward_cuda_fused_sqeuclid(X.detach(), Y.detach(), R, gamma, bw, lens_x, lens_y)  # (B,N,M)

        # Scale by upstream grad (B,) -> (B,1,1)
        g = grad_output.reshape(-1).to(device=X.device, dtype=X.dtype).view(-1, 1, 1)
        E = E * g

        # Reductions for sqeuclidean chain rule
        EX = E.sum(dim=2)  # (B,N)
        EY = E.sum(dim=1)  # (B,M)

        grad_X = 2.0 * (X * EX.unsqueeze(2) - torch.bmm(E, Y))                 # (B,N,D)
        grad_Y = 2.0 * (Y * EY.unsqueeze(2) - torch.bmm(E.transpose(1, 2), X)) # (B,M,D)

        return grad_X, grad_Y, None, None, None, None
