import torch
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    import selective_scan_cuda
except:
    pass

try:
    "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# OCSS: Omnidirectional Continuous Selective Scan (snake Cross Scan)
# Unlike VMamba (row/col scans that jump at line ends), when reaching the end of a
# row/column the next token is the spatially adjacent token in the next row/column:
#   horiz (3x3): 1-2-3-6-5-4-7-8-9 ; reverse: 9-8-7-4-5-6-3-2-1
#   vert  (3x3): 1-4-7-8-5-2-3-6-9 ; reverse: 9-6-3-2-5-8-7-4-1
# Used by SS2D in backbone VSSBlock and neck XSSBlock.

_CONT_SCAN_IDX_CACHE = {}


def _continuous_scan_indices(H: int, W: int, device):
    """Return (idx_h, idx_v): flatten indices for continuous horiz / vert snake scans."""
    key = (H, W, str(device))
    cached = _CONT_SCAN_IDX_CACHE.get(key)
    if cached is not None:
        return cached

    # Horizontal: even rows L→R, odd rows R→L (spatially continuous at row ends)
    idx_h = []
    for i in range(H):
        row = list(range(i * W, (i + 1) * W))
        if i % 2 == 1:
            row.reverse()
        idx_h.extend(row)

    # Vertical: even cols T→B, odd cols B→T (spatially continuous at col ends)
    idx_v = []
    for j in range(W):
        col = [i * W + j for i in range(H)]
        if j % 2 == 1:
            col.reverse()
        idx_v.extend(col)

    idx_h = torch.tensor(idx_h, dtype=torch.long, device=device)
    idx_v = torch.tensor(idx_v, dtype=torch.long, device=device)
    _CONT_SCAN_IDX_CACHE[key] = (idx_h, idx_v)
    return idx_h, idx_v


def _gather_by_index(x_flat: torch.Tensor, index: torch.Tensor):
    """x_flat: (B, C, L), index: (L,) -> (B, C, L) gathered along last dim."""
    B, C, L = x_flat.shape
    return x_flat.gather(-1, index.view(1, 1, L).expand(B, C, L))


def _scatter_add_by_index(src: torch.Tensor, index: torch.Tensor, out: torch.Tensor):
    """Scatter-add src (B, C, L) into out (B, C, L) with flatten index (L,)."""
    B, C, L = src.shape
    out.scatter_add_(-1, index.view(1, 1, L).expand(B, C, L), src)
    return out


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        L = H * W
        idx_h, idx_v = _continuous_scan_indices(H, W, x.device)
        ctx.idx_h = idx_h
        ctx.idx_v = idx_v

        x_flat = x.flatten(2, 3)  # (B, C, L) row-major spatial
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = _gather_by_index(x_flat, idx_h)  # L→R continuous snake
        xs[:, 1] = _gather_by_index(x_flat, idx_v)  # T→B continuous snake
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # R→L / B→T reverses
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # ys: (B, 4, C, L) grads in scan order -> spatial (B, C, H, W)
        B, C, H, W = ctx.shape
        L = H * W
        idx_h, idx_v = ctx.idx_h, ctx.idx_v
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1])  # (B, 2, C, L)
        y = ys.new_zeros(B, C, L)
        _scatter_add_by_index(ys[:, 0], idx_h, y)
        _scatter_add_by_index(ys[:, 1], idx_v, y)
        return y.view(B, C, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        # ys: (B, K, D, H, W) — last two dims are a view of scan-length L=H*W
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        L = H * W
        idx_h, idx_v = _continuous_scan_indices(H, W, ys.device)
        ctx.idx_h = idx_h
        ctx.idx_v = idx_v

        ys = ys.view(B, K, D, L)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1])  # (B, 2, D, L) scan order
        y = ys.new_zeros(B, D, L)
        _scatter_add_by_index(ys[:, 0], idx_h, y)
        _scatter_add_by_index(ys[:, 1], idx_v, y)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # x: (B, D, L) spatial flat -> (B, 4, D, H, W) scan-order grads
        H, W = ctx.shape
        B, C, L = x.shape
        idx_h, idx_v = ctx.idx_h, ctx.idx_v
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = _gather_by_index(x, idx_h)
        xs[:, 1] = _gather_by_index(x, idx_v)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs.view(B, 4, C, H, W)


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default'
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)