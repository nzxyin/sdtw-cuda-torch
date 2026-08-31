from __future__ import annotations

import torch
from torch.autograd import Function

from .utils.checks import check_D
from .cuda.launcher import softdtw_forward_cuda, softdtw_backward_cuda_log
from .cuda.launcher import softdtw_forward_cpu, softdtw_backward_cpu


class SoftDTWAutograd(Function):
    @staticmethod
    def forward(
        ctx,
        D: torch.Tensor,
        gamma: float,
        bandwidth: float | None,
        lens_x: torch.Tensor | None = None,
        lens_y: torch.Tensor | None = None,
    ):
        check_D(D)
        gamma_f = float(gamma)
        if gamma_f <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma_f}")

        if bandwidth is None:
            bandwidth_f = -1.0
        else:
            bw = float(bandwidth)
            bandwidth_f = -1.0 if bw <= 0 else bw

        if D.is_cuda:
            out, R = softdtw_forward_cuda(D, gamma_f, bandwidth_f, lens_x, lens_y)
        else:
            out, R = softdtw_forward_cpu(D, gamma_f, bandwidth_f, lens_x, lens_y)

        to_save = [D, R.detach()]
        ctx.has_lens = lens_x is not None or lens_y is not None
        if lens_x is not None:
            to_save.append(lens_x)
        if lens_y is not None:
            to_save.append(lens_y)
        ctx.lens_x_saved = lens_x is not None
        ctx.lens_y_saved = lens_y is not None
        ctx.save_for_backward(*to_save)
        ctx.gamma = gamma_f
        ctx.bandwidth = bandwidth_f
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = list(ctx.saved_tensors)
        D, R = saved[0], saved[1]
        rest = saved[2:]
        lens_x = rest.pop(0) if ctx.lens_x_saved else None
        lens_y = rest.pop(0) if ctx.lens_y_saved else None
        gamma_f = ctx.gamma
        bandwidth_f = ctx.bandwidth

        if D.is_cuda:
            E = softdtw_backward_cuda_log(D, R, gamma_f, bandwidth_f, lens_x, lens_y)
        else:
            E = softdtw_backward_cpu(D, R, gamma_f, bandwidth_f, lens_x, lens_y)

        g = grad_output.reshape(-1).to(dtype=E.dtype).view(-1, 1, 1)
        grad_D = g * E
        return grad_D, None, None, None, None
