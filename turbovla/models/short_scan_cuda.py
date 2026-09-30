"""Short BF16 selective scan matching the reference scan's rounding points."""

from __future__ import annotations

import threading

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:  # CPU-only environments can still import the model.
    triton = None


if triton is not None:
    @triton.jit
    def _round_bf16(x):
        return x.to(tl.bfloat16).to(tl.float32)


    @triton.jit
    def _short_scan_kernel(V, DT, A, BP, CP, D, MASK, OUT,
                           T: tl.constexpr, CHANNELS: tl.constexpr,
                           N: tl.constexpr, BN: tl.constexpr):
        batch = tl.program_id(0)
        channel = tl.program_id(1)
        n = tl.arange(0, BN)
        active = n < N
        a = tl.load(A + channel * N + n, active, 0).to(tl.float32)
        skip = tl.load(D + channel).to(tl.float32)
        state = tl.full((BN,), 0, tl.float32)
        for t in range(T):
            valid = tl.load(MASK + batch * T + t)
            value = tl.load(V + (batch * T + t) * CHANNELS + channel).to(tl.float32)
            delta = tl.load(DT + (batch * T + t) * CHANNELS + channel).to(tl.float32)
            b = tl.load(BP + (batch * T + t) * N + n, active, 0).to(tl.float32)
            c = tl.load(CP + (batch * T + t) * N + n, active, 0).to(tl.float32)
            transition = _round_bf16(libdevice.exp(_round_bf16(delta * a)))
            carry = _round_bf16(transition * state)
            injection = _round_bf16(_round_bf16(delta * b) * value)
            candidate = _round_bf16(carry + injection)
            state = tl.where(valid, candidate, state)
            readout = _round_bf16(tl.sum(_round_bf16(state * c), 0))
            skip_value = _round_bf16(skip * value)
            result = _round_bf16(readout + skip_value)
            tl.store(OUT + (batch * T + t) * CHANNELS + channel,
                     tl.where(valid, result, 0))


def _reference_recurrence(value, delta, a, b, c, skip, valid_mask):
    state = value.new_zeros(value.shape[0], value.shape[2], a.shape[1])
    outputs = []
    for t in range(value.shape[1]):
        transition = torch.exp(delta[:, t, :, None] * a[None])
        candidate = transition * state
        candidate = candidate + delta[:, t, :, None] * b[:, t, None, :] * value[:, t, :, None]
        valid = valid_mask[:, t, None, None]
        state = torch.where(valid, candidate, state)
        result = (state * c[:, t, None, :]).sum(-1)
        result = result + skip * value[:, t]
        outputs.append(result * valid_mask[:, t, None].to(value.dtype))
    return torch.stack(outputs, dim=1)


class _CapturedBackward:
    """Replay the exact PyTorch backward without its per-step launch overhead."""

    def __init__(self, inputs):
        self.static = [x.detach().clone().requires_grad_(True) for x in inputs[:6]]
        self.mask = inputs[6].clone()
        self.cotangent = torch.empty_like(inputs[0])

        def differentiate():
            with torch.enable_grad():
                output = _reference_recurrence(*self.static, self.mask)
                return torch.autograd.grad(output, self.static, self.cotangent)

        # Populate allocator caches before capture; the warmup graph is discarded.
        differentiate()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.grads = differentiate()

    def __call__(self, inputs, cotangent):
        for target, source in zip(self.static, inputs[:6]):
            target.copy_(source)
        self.mask.copy_(inputs[6])
        self.cotangent.copy_(cotangent)
        self.graph.replay()
        # The graph owns its output buffers; callers need independent gradients.
        return tuple(grad.clone() for grad in self.grads)


_backward_graphs = {}


def _graph_backward(inputs, cotangent):
    key = (threading.get_ident(), inputs[0].device,
           tuple((x.shape, x.stride(), x.dtype) for x in inputs))
    graph = _backward_graphs.get(key)
    if graph is None:
        graph = _CapturedBackward(inputs)
        _backward_graphs[key] = graph
    return graph(inputs, cotangent)


class _ShortScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, delta, a, b, c, skip, valid_mask):
        if triton is None or not value.is_cuda or value.dtype != torch.bfloat16:
            raise ValueError("short_scan_cuda requires Triton and CUDA BF16 tensors")
        batch, steps, channels = value.shape
        state_size = a.shape[1]
        if steps not in (4, 12) or state_size != 16:
            raise ValueError("short_scan_cuda supports T=4/12 and state_size=16")
        if (delta.shape != value.shape or a.shape != (channels, state_size)
                or b.shape != (batch, steps, state_size)
                or c.shape != (batch, steps, state_size)
                or skip.shape != (channels,)
                or valid_mask.shape != (batch, steps)):
            raise ValueError("short_scan_cuda received inconsistent tensor shapes")
        if any(tensor.dtype != torch.bfloat16 for tensor in (delta, a, b, c, skip)):
            raise ValueError("short_scan_cuda requires BF16 recurrence tensors")
        if valid_mask.dtype != torch.bool:
            raise ValueError("short_scan_cuda requires a boolean validity mask")
        value, delta, a, b, c, skip = (
            tensor.contiguous() for tensor in (value, delta, a, b, c, skip)
        )
        valid_mask = valid_mask.contiguous()
        output = torch.empty_like(value)
        _short_scan_kernel[(batch, channels)](
            value, delta, a, b, c, skip, valid_mask, output,
            steps, channels, state_size, triton.next_power_of_2(state_size),
            num_warps=1,
        )
        ctx.save_for_backward(value, delta, a, b, c, skip, valid_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        grads = _graph_backward(saved, grad_output)
        return (*grads, None)


def short_scan_cuda(value, delta, a, b, c, skip, valid_mask):
    return _ShortScan.apply(value, delta, a, b, c, skip, valid_mask)
