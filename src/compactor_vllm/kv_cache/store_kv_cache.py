import torch
import triton
import triton.language as tl
from compactor_vllm.config.constants import (
    TRITON_RESERVED_BATCH as _TRITON_RESERVED_BATCH,
)


@triton.jit
def _prefill_store_topk_kv_kernel(
    key,
    value,  # [N_total, H, D] (D stride assumed 1)
    batch_mapping,  # [B] int32  (local b -> true batch)
    num_tokens_to_retain,  # [B] int32
    indices_topk,  # [B, MAX_SEL] int32 (across all heads)
    # Lengths & page table:
    bh_lens,  # [B, H] int32 (contiguous)
    page_table,  # [B_total * H * N_LOGICAL_PAGES_MAX] int32 (flattened), read-only
    k_cache,
    v_cache,  # [N_PAGES * PAGE_SIZE, D]
    sk_n,
    sk_h,  # strides for key,value. D stride assumed 1
    sv_n,
    sv_h,
    # Runtime ints
    MAX_SEL,  # num tokens that are ranked in indices for each batch (might be bigger than num_tokens_to_retain)
    HKV: tl.constexpr,
    N_LOGICAL_PAGES_MAX: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_TILE: tl.constexpr,  # how many selected tokens each program processes
    TRITON_RESERVED_BATCH: tl.constexpr,
):
    b_local = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs = tl.arange(0, D)
    # how many tokens we actually keep for this batch
    k_total = tl.load(num_tokens_to_retain + b_local)
    if k_total == 0:
        return
    # map to true batch row in the page table
    b_true = tl.load(batch_mapping + b_local)
    if b_true == TRITON_RESERVED_BATCH:
        return
    base = tile_id * K_TILE
    # process up to K_TILE tokens
    for j in tl.range(0, K_TILE):
        sel_idx = base + j
        if sel_idx < k_total and sel_idx < MAX_SEL:
            # flattened selection: sel = token * H + head
            sel = tl.load(indices_topk + b_local * MAX_SEL + sel_idx)
            tok = sel // HKV
            head = sel - (tok * HKV)
            # atomically reserve one position in (b_local, hed)
            # i.e the KV cache is scrambled when storing
            len_ptr = bh_lens + b_local * HKV + head
            pos = tl.atomic_add(len_ptr, 1)  # old length (int32)
            lp = pos // PAGE_SIZE
            off = pos - lp * PAGE_SIZE
            # translate logical page to physical page
            pt_base = (b_true * HKV + head) * N_LOGICAL_PAGES_MAX
            phys = tl.load(page_table + pt_base + lp).to(tl.int64)
            # destination row and element offset
            dst_row = phys * PAGE_SIZE + off
            dst_off = dst_row * D + offs
            # load one vector from [N_total, H, D]
            k_src = key + tok * sk_n + head * sk_h + offs
            v_src = value + tok * sv_n + head * sv_h + offs
            tl.store(
                k_cache + dst_off,
                tl.load(k_src, cache_modifier=".cv", eviction_policy="evict_first"),
                eviction_policy="evict_first",
            )
            tl.store(
                v_cache + dst_off,
                tl.load(v_src, cache_modifier=".cv", eviction_policy="evict_first"),
                eviction_policy="evict_first",
            )


def prefill_store_topk_kv(
    *,
    new_keys: torch.Tensor,  # [N_total, H, D]
    new_vals: torch.Tensor,  # [N_total, H, D]
    indices_topk: torch.Tensor,  # [B, MAX_SEL] int32 (global flattened token*H + head)
    num_tokens_to_retain: torch.Tensor,  # [B] int32
    page_table: torch.Tensor,  # [B_total, H, N_LOGICAL_PAGES_MAX] int32
    batch_mapping: torch.Tensor,  # [B] int32 (local -> true batch rows)
    bh_lens: torch.Tensor,  # [B, H] int32 (contiguous), UPDATED atomically
    k_cache: torch.Tensor,  # [N_PAGES * PAGE_SIZE, D]
    v_cache: torch.Tensor,  # [N_PAGES * PAGE_SIZE, D]
    PAGE_SIZE: int,
    PAD_TO_PAGE_SIZE: bool = True,
    cu_seqlens_k: torch.Tensor | None = None,
    K_TILE: int = 16,
    TRITON_RESERVED_BATCH: int = None,
):
    assert new_keys.shape == new_vals.shape
    N_total, H, D = new_keys.shape
    B = indices_topk.shape[0]
    assert page_table.shape[1] == H
    assert bh_lens.shape == (B, H)
    assert new_keys.device == k_cache.device == v_cache.device
    assert page_table.is_contiguous(), "page table must be contiguous."
    assert bh_lens.is_contiguous(), "bh_lens must be contiguous."
    assert batch_mapping.is_contiguous(), "batch mapping must be contiguous."
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert new_keys.stride(-1) == 1 and new_vals.stride(-1) == 1, (
        "new_keys/new_vals last dim must be contiguous."
    )
    assert (D & (D - 1)) == 0, "D must be a power of 2"
    page_table = page_table.to(torch.int32)
    bh_lens = bh_lens.to(torch.int32)
    batch_mapping = batch_mapping.to(torch.int32)
    indices_topk = indices_topk.to(torch.int32)
    num_tokens_to_retain = num_tokens_to_retain.to(torch.int32)

    # strides (elements) for [N_total, H, D]
    sk_n, sk_h, _ = new_keys.stride()
    sv_n, sv_h, _ = new_vals.stride()

    # tile second grid dim
    MAX_SEL = indices_topk.shape[-1]
    N_TILES = (MAX_SEL + K_TILE - 1) // K_TILE
    grid = (B, max(1, N_TILES))
    if TRITON_RESERVED_BATCH is None:
        TRITON_RESERVED_BATCH = _TRITON_RESERVED_BATCH
    _prefill_store_topk_kv_kernel[grid](
        key=new_keys,
        value=new_vals,
        batch_mapping=batch_mapping,
        num_tokens_to_retain=num_tokens_to_retain,
        indices_topk=indices_topk,
        bh_lens=bh_lens,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        sk_n=sk_n,
        sk_h=sk_h,
        sv_n=sv_n,
        sv_h=sv_h,
        MAX_SEL=int(MAX_SEL),
        HKV=H,
        N_LOGICAL_PAGES_MAX=page_table.shape[2],
        D=D,
        PAGE_SIZE=PAGE_SIZE,
        K_TILE=K_TILE,
        TRITON_RESERVED_BATCH=TRITON_RESERVED_BATCH,
    )
    if PAD_TO_PAGE_SIZE:
        assert cu_seqlens_k is not None
        assert indices_topk.is_contiguous()
        assert page_table.is_contiguous()
        _prefill_store_topk_pad_kernel[(B, H)](
            key=new_keys,
            value=new_vals,
            batch_mapping=batch_mapping,
            num_tokens_to_retain=num_tokens_to_retain,
            indices=indices_topk,
            local_lens=bh_lens,
            page_table_flat=page_table,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_k=cu_seqlens_k,
            sk_n=sk_n,
            sk_h=sk_h,
            sv_n=sv_n,
            sv_h=sv_h,
            MAX_SEL=int(MAX_SEL),
            H=H,  # type: ignore
            N_LOGICAL_PAGES_MAX=page_table.shape[2],  # type: ignore
            D=D,  # type: ignore
            PAGE_SIZE=PAGE_SIZE,  # type: ignore
            TRITON_RESERVED_BATCH=TRITON_RESERVED_BATCH,
        )


@triton.jit
def _prefill_store_topk_pad_kernel(
    key,  # [N_total, H, D]
    value,  # [N_total, H, D]
    batch_mapping,  # [B] int32  (local b -> true batch)
    num_tokens_to_retain,  # [B] int32
    indices,  # [B, MAX_SEL] int32 (across all heads)
    local_lens,  # [B, H] int32 (contiguous)
    page_table_flat,  # [B_total*H*N_LOGICAL_PAGES_MAX] int32
    k_cache,
    v_cache,  # [N_PAGES*PAGE_SIZE, D]
    cu_seqlens_k,
    sk_n,
    sk_h,
    sv_n,
    sv_h,
    MAX_SEL,
    # Constexprs
    H: tl.constexpr,  # number of KV heads
    N_LOGICAL_PAGES_MAX: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TRITON_RESERVED_BATCH: tl.constexpr,
):
    b_local = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.arange(0, D)
    L = tl.load(local_lens + b_local * H + h)
    modulo_page_size = L - (L // PAGE_SIZE) * PAGE_SIZE
    if modulo_page_size == 0:
        return
    need = PAGE_SIZE - modulo_page_size
    b_true = tl.load(batch_mapping + b_local)
    if b_true == TRITON_RESERVED_BATCH:
        return
    pt_base = (b_true * H + h) * N_LOGICAL_PAGES_MAX
    written_tokens = 0
    idx = tl.load(num_tokens_to_retain + b_local)
    this_batch_ctx_len = tl.load(cu_seqlens_k + b_local + 1) - tl.load(
        cu_seqlens_k + b_local
    )
    max_additional = this_batch_ctx_len - L
    while written_tokens < need and idx < MAX_SEL and written_tokens < max_additional:
        # candidate head
        cand_idx = tl.load(indices + b_local * MAX_SEL + idx)
        cand_h = cand_idx % H
        if cand_h == h:
            tok = cand_idx // H
            pos = L + written_tokens
            lp = pos // PAGE_SIZE
            off = pos - lp * PAGE_SIZE
            phys = tl.load(page_table_flat + pt_base + lp).to(tl.int32)

            dst_row = phys * PAGE_SIZE + off
            dst_off = dst_row.to(tl.int64) * D + offs_d

            k_src = key + tok * sk_n + h * sk_h + offs_d
            v_src = value + tok * sv_n + h * sv_h + offs_d

            tl.store(
                k_cache + dst_off,
                tl.load(k_src),
            )
            tl.store(
                v_cache + dst_off,
                tl.load(v_src),
            )

            written_tokens += 1
        idx += 1
    tl.store(local_lens + b_local * H + h, L + written_tokens)


@triton.jit
def _prefill_store_all_kv_kernel(
    key,
    value,  # [N, H, D] (D contiguous)
    cu_seqlens_k,  # [B + 1] int32
    batch_mapping,  # [B] int32 (local b -> true batch index)
    bh_lens,  # [B * HKV] int32 (UPDATED)
    pt_flat,  # [B_total * HKV * N_LOGICAL_PAGES_MAX] int32 (flattened)
    k_cache,
    v_cache,  # [N_PAGES * PAGE_SIZE, D]
    # source strides (elements)
    sk_n,
    sk_h,
    sv_n,
    sv_h,
    # constexpr
    HKV: tl.constexpr,
    N_LOGICAL_PAGES_MAX: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_TILE: tl.constexpr,  # number of (token, head) pairs processed per program
):
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)

    start = tl.load(cu_seqlens_k + pid_b)
    end = tl.load(cu_seqlens_k + pid_b + 1)
    num_toks_this_batch = end - start
    if num_toks_this_batch <= 0:
        return

    total_elems = num_toks_this_batch * HKV

    # base linear index in (token, head) grid for this program
    base = pid_blk * K_TILE

    offs_d = tl.arange(0, D)

    # Iterate K_TILE elements in this tile
    for i in tl.range(0, K_TILE):
        idx = base + i
        if idx < total_elems:
            # map linear idx -> (t, h)
            t = idx // HKV
            h = idx - t * HKV

            len_idx = pid_b * HKV + h
            L0 = tl.load(bh_lens + len_idx)

            token_idx_in_cache = L0 + t
            lp = token_idx_in_cache // PAGE_SIZE  # logical page
            off_in_pg = token_idx_in_cache - lp * PAGE_SIZE  # pos in page

            # physical page
            b_true = tl.load(batch_mapping + pid_b).to(tl.int32)
            pt_base = (b_true * HKV + h) * N_LOGICAL_PAGES_MAX
            phys = tl.load(pt_flat + pt_base + lp).to(tl.int64)

            row = phys * PAGE_SIZE + off_in_pg
            dst_off = row * D + offs_d

            n_global = (start + t).to(tl.int64)

            # Use strides for non-contiguous [N, H, D] (D stride == 1)
            k_src = key + n_global * sk_n + h * sk_h + offs_d
            v_src = value + n_global * sv_n + h * sv_h + offs_d

            tl.store(k_cache + dst_off, tl.load(k_src))
            tl.store(v_cache + dst_off, tl.load(v_src))


def prefill_store_all_kv(
    *,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,  # [N, H_kv, D]
    cu_seqlens_k: torch.Tensor,  # [B + 1] int32
    max_seqlen_k: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,  # [B_total, H_kv, N_LOGICAL_PAGES_MAX] int32
    bh_lens: torch.Tensor,  # [B, H_kv] int32 (UPDATED)
    batch_mapping: torch.Tensor,  # [B] int32 (local->true)
    PAGE_SIZE: int,
    K_TILE: int = 32,  # how many (token, head) pairs per program
):
    assert new_keys.stride(-1) == 1 and new_values.stride(-1) == 1, (
        "last dim must be contiguous"
    )
    assert page_table.is_contiguous(), "page table must be contiguous"
    assert bh_lens.is_contiguous(), "bh_lens must be contiguous"
    assert batch_mapping.is_contiguous(), "batch mapping must be contiguous"
    assert k_cache.is_contiguous() and v_cache.is_contiguous()

    N, HKV, D = new_keys.shape
    B = batch_mapping.shape[0]
    assert (D & (D - 1)) == 0, "D must be a power of 2"

    sk_n, sk_h, _ = new_keys.stride()
    sv_n, sv_h, _ = new_values.stride()
    n_tiles = (max_seqlen_k * HKV + K_TILE - 1) // K_TILE
    grid = (B, n_tiles)
    _prefill_store_all_kv_kernel[grid](
        new_keys,
        new_values,
        cu_seqlens_k,
        batch_mapping,
        bh_lens,
        page_table,
        k_cache,
        v_cache,
        sk_n=sk_n,
        sk_h=sk_h,
        sv_n=sv_n,
        sv_h=sv_h,
        HKV=HKV,
        N_LOGICAL_PAGES_MAX=page_table.shape[-1],
        D=D,
        PAGE_SIZE=PAGE_SIZE,
        K_TILE=K_TILE,
    )
    bh_lens += cu_seqlens_k.diff()[:, None]


@triton.jit
def _decode_store_kv_kernel(
    key,
    value,
    batch_mapping,  # [B] int32
    bh_lens,  # [B*HKV] int32
    page_table,  # [B_total*HKV*N_LOGICAL_PAGES_MAX]
    k_cache,
    v_cache,  # [N_PAGES*PAGE_SIZE, D]
    sk_b,
    sk_h,
    sv_b,
    sv_h,
    HKV: tl.constexpr,
    N_LOGICAL_PAGES_MAX: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TRITON_RESERVED_BATCH: tl.constexpr,
):
    pid_b = tl.program_id(0)
    h = tl.program_id(1)
    mapped_b = tl.load(batch_mapping + pid_b)
    if mapped_b == TRITON_RESERVED_BATCH:
        return
    offs_d = tl.arange(0, D)

    length = tl.load(bh_lens + pid_b * HKV + h)
    logical_page = length // PAGE_SIZE
    internal_offset = length - logical_page * PAGE_SIZE

    pt_base = (mapped_b * HKV + h) * N_LOGICAL_PAGES_MAX
    physical_page = tl.load(page_table + pt_base + logical_page).to(tl.int64)

    dst_row = physical_page * PAGE_SIZE + internal_offset

    # Source addressing using strides (D stride == 1)
    k_src = key + pid_b * sk_b + h * sk_h + offs_d
    v_src = value + pid_b * sv_b + h * sv_h + offs_d

    dst_off = dst_row * D + offs_d
    tl.store(k_cache + dst_off, tl.load(k_src))
    tl.store(v_cache + dst_off, tl.load(v_src))
    tl.store(bh_lens + pid_b * HKV + h, length + 1)


def decode_store_kv(
    *,
    key: torch.Tensor,  # [B, HKV, D]
    value: torch.Tensor,  # [B, HKV, D]
    batch_mapping: torch.Tensor,  # [B] int32
    bh_lens: torch.Tensor,  # [B, HKV] or flattened [B*HKV] int32
    page_table: torch.Tensor,  # [B_total, HKV, N_LOGICAL_PAGES_MAX] int32
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,  # [N_PAGES*PAGE_SIZE, D]
    PAGE_SIZE: int,
    TRITON_RESERVED_BATCH: int = None,
):
    assert key.shape == value.shape and key.ndim == 3, "key/value must be [B, HKV, D]"
    B, HKV, D = key.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1, (
        "key/value last dim must be contiguous."
    )
    assert page_table.is_contiguous(), "page table must be contiguous."
    assert bh_lens.is_contiguous(), "bh_lens must be contiguous."
    assert batch_mapping.is_contiguous(), "batch mapping must be contiguous."
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert (D & (D - 1)) == 0, "D must be a power of 2"
    sk_b, sk_h, _ = key.stride()
    sv_b, sv_h, _ = value.stride()
    grid = (
        int(batch_mapping.shape[0]),
        HKV,
    )
    _decode_store_kv_kernel[grid](
        key=key,
        value=value,
        batch_mapping=batch_mapping,
        bh_lens=bh_lens,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        sk_b=sk_b,
        sk_h=sk_h,
        sv_b=sv_b,
        sv_h=sv_h,
        HKV=HKV,
        N_LOGICAL_PAGES_MAX=page_table.shape[2],
        D=D,
        PAGE_SIZE=PAGE_SIZE,
        TRITON_RESERVED_BATCH=TRITON_RESERVED_BATCH
        if TRITON_RESERVED_BATCH is not None
        else _TRITON_RESERVED_BATCH,
    )
