import torch
import triton
import triton.language as tl


def scatter_to_page_table(
    add_pages: torch.Tensor,  # [L, H] int32
    new_phys_pages: torch.Tensor,  # [N]
    curr_pages: torch.Tensor,  # [L, H] int32
    page_table: torch.Tensor,  # [L, H, max_pages_per_head] int32, NOT assumed contiguous globally
    max_pages_per_head: int,
):
    """
    Append newly allocated physical pages into a layered page table via Triton.
    For each (layer ``l``, head ``h``):
    Args:
        :param add_pages:
            Tensor of shape ``[L, H]`` (int32) indicating how many pages to
            append for each (layer, head).
        :param new_phys_pages:
            1D tensor of shape ``[N]`` (int32) containing physical page IDs
            for all (layer, head) pairs, concatenated in row-major (L, H)
            order. ``N`` must equal ``add_pages.sum()``.
        :param curr_pages:
            Tensor of shape ``[L, H]`` (int32) with the current logical page
            counts per (layer, head) before this update.
        :param page_table:
            Tensor of shape ``[L, H, max_pages_per_head]`` (int32) holding
            the logical to physical page mapping. The last dimension is
            logically indexed as logical_page ∈ [0, max_pages_per_head).
        :param max_pages_per_head:
            Maximum number of logical pages permitted per (layer, head). The
            kernel skips writes beyond this bound.
    Returns:
        None. The function updates ``page_table`` in-place.
    """
    L, H = add_pages.shape
    if L == 0 or H == 0:
        return
    add_flat = add_pages.to(torch.int32).contiguous().view(-1)
    curr_flat = curr_pages.to(torch.int32).contiguous().view(-1)
    cum_page_heads = torch.empty(L * H + 1, device="cuda", dtype=torch.int32)
    cum_page_heads[0] = 0
    torch.cumsum(add_flat, 0, out=cum_page_heads[1:])
    stride_pl, stride_ph, stride_pp = page_table.stride()
    grid = (L, H)
    _scatter_pages_kernel_lh[grid](
        add_flat,
        cum_page_heads,
        new_phys_pages,
        curr_flat,
        page_table,
        stride_pl,
        stride_ph,
        stride_pp,
        L=L,
        H=H,
        max_pages_per_head=max_pages_per_head,
    )


@triton.jit
def _scatter_pages_kernel_lh(
    add_pages,  # int32 [L*H]
    cum_page_heads,  # int32 [L*H], base offset in flat_new_phys per (l,h)
    flat_new_phys,  # int32 [total_pages]
    curr_pages,  # int32 [L*H], existing logical pages per (l,h)
    page_table_ptr,  # int32* base pointer to page_table
    stride_pl,  # int, stride for layer dim
    stride_ph,  # int, stride for head dim
    stride_pp,  # int, stride for page dim
    L: tl.constexpr,
    H: tl.constexpr,
    max_pages_per_head: tl.constexpr,
):
    layer_idx = tl.program_id(0)
    h = tl.program_id(1)
    if layer_idx >= L or h >= H:
        return

    lh = layer_idx * H + h
    ap = tl.load(add_pages + lh)
    if ap <= 0:
        return

    base = tl.load(cum_page_heads + lh)
    cp = tl.load(curr_pages + lh)

    # Append ap pages: logical pages [cp .. cp+ap)
    for i in tl.range(0, ap):
        phys = tl.load(flat_new_phys + base + i)
        lp = cp + i
        if lp < max_pages_per_head:
            offset = layer_idx * stride_pl + h * stride_ph + lp * stride_pp
            tl.store(page_table_ptr + offset, phys)


# TODO: write reclaim kernel
@triton.jit
def reclaim_page_kernel():
    pass


def reclaim_pages(
    batch_index: int,
    bh_seq_lens: torch.Tensor,
    bh_num_pages: torch.Tensor,
    page_table: torch.Tensor,
):
    pass
