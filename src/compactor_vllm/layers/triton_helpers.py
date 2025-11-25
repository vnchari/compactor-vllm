import torch
import triton
import triton.language as tl


@triton.jit
def _masked_index_select_kernel(
    X_ptr,
    IDX_ptr,
    OUT_ptr,
    N,
    stride_xn,
    stride_xh,
    stride_ob,
    stride_oh,
):
    b = tl.program_id(0)  # which output row (0..B-1)
    h = tl.program_id(1)
    idx = tl.load(IDX_ptr + b)  # int32
    valid = (idx >= 0) & (idx < N)
    out_ptrs = OUT_ptr + b * stride_ob + h * stride_oh

    if not valid:
        tl.store(out_ptrs, 0)
    else:
        x_ptrs = X_ptr + idx * stride_xn + h * stride_xh
        vals = tl.load(x_ptrs)
        tl.store(out_ptrs, vals)


def masked_index_select_triton_dim0(
    input: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """
    X:   [N, H] : contiguous in the H dimension
    b_m: [B]  int32/int64 on same device; out-of-range -> zeros)
    Returns: [B, H]
    """
    assert input.ndim == 2 and index.ndim == 1
    N, H = input.shape
    B = index.numel()
    out = torch.empty((B, H), dtype=input.dtype, device=input.device)
    _masked_index_select_kernel[(B, H)](
        input,
        index,
        out,
        N,
        input.stride(0),
        input.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


@triton.jit
def _masked_index_copy_kernel(
    DST_ptr,
    IDX_ptr,
    SRC_ptr,
    N,
    stride_dn,
    stride_dh,
    stride_sb,
    stride_sh,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    idx = tl.load(IDX_ptr + b)
    valid = (idx >= 0) & (idx < N)
    if valid:
        src_ptrs = SRC_ptr + b * stride_sb + h * stride_sh
        dst_ptrs = DST_ptr + idx * stride_dn + h * stride_dh
        tl.store(dst_ptrs, tl.load(src_ptrs))


def masked_index_copy_triton_dim0(
    dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor
):
    """
    In-place: dst.index_copy_(0, index, src) but masked:
      - rows with index[b] < 0 or >= dst.shape[0] are skipped (no write).
    Shapes:
      dst: [N, H]
      src: [B, H]
      index: [B]
    """
    assert dst.ndim == 2 and src.ndim == 2 and index.ndim == 1
    N, H = dst.shape
    B, Hs = src.shape
    assert Hs == H and index.numel() == B
    _masked_index_copy_kernel[(B, H)](
        dst,
        index,
        src,
        N,
        dst.stride(0),
        dst.stride(1),
        src.stride(0),
        src.stride(1),
    )
