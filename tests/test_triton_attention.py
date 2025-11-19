import logging
import math
from dataclasses import dataclass
from typing import List

import pytest
import torch
import triton
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

from compactor_vllm.attention.sparse_decode_kernel import head_sparse_decode_attention
from compactor_vllm.attention.sparse_varlen_kernel import (
    causal_sparse_varlen_with_cache,
)

logger = logging.getLogger(__name__)


@dataclass
class Workload:
    name: str
    batch_size: int
    nq_heads: int
    nk_heads: int
    head_dim: int
    cache_lens: List[int]  # per-sequence cached context length
    append_lens: List[int]  # per-sequence new tokens this step (Q_app, K_app, V_app)


WORKLOADS: List[Workload] = [
    Workload(
        name=f"batch_size={BATCH} kv_cache_len={cache_lens} append_len={append_lens} "
        f"HQ={NQ_HEADS} HKV={NK_HEADS} HEAD_DIM={HEAD_DIM}",
        batch_size=BATCH,
        nq_heads=NQ_HEADS,
        nk_heads=NK_HEADS,
        head_dim=HEAD_DIM,
        cache_lens=[cache_lens] * BATCH,
        append_lens=[append_lens] * BATCH,
    )
    for BATCH in [1, 2, 3, 8]
    for NQ_HEADS in [32]
    for NK_HEADS in [8]
    for HEAD_DIM in [128]
    for cache_lens in [0, 1, 70, 128, 8193]
    for append_lens in [1, 2, 13, 8000]
]

WORKLOADS_DECODE: List[Workload] = [
    Workload(
        name=f"batch_size={BATCH} kv_cache_len={cache_lens}"
        f"HQ={NQ_HEADS} HKV={NK_HEADS} HEAD_DIM={HEAD_DIM}",
        batch_size=BATCH,
        nq_heads=NQ_HEADS,
        nk_heads=NK_HEADS,
        head_dim=HEAD_DIM,
        cache_lens=[cache_lens] * BATCH,
        append_lens=[1] * BATCH,
    )
    for BATCH in [1, 2, 3, 8]
    for NQ_HEADS in [32]
    for NK_HEADS in [8]
    for HEAD_DIM in [128]
    for cache_lens in [1, 2, 70, 128, 8000]
]


def build_paged_cache_from_lengths(
    B,
    H_kv,
    D,
    PAGE_SIZE,
    N_LOGICAL_PAGES_MAX,
    L_cache_per_b,  # int32 [B], per-batch cache length
    device,
    dtype,
):
    """
    Construct:
      - seq_lens_bh[b, h] = L_cache_per_b[b]
      - page_table[b, h, lp] giving physical page ids
      - K_cache, V_cache filled for valid cached tokens

    Physical layout:
      physical_page_id = (b * H_kv + h) * N_LOGICAL_PAGES_MAX + lp
      CACHE_SIZE = num_phys_pages * PAGE_SIZE
    """
    assert L_cache_per_b.shape[0] == B
    max_len = PAGE_SIZE * N_LOGICAL_PAGES_MAX
    assert (L_cache_per_b <= max_len).all()

    seq_lens_bh = torch.empty((B, H_kv), dtype=torch.int32, device=device)
    for b in range(B):
        seq_lens_bh[b, :].fill_(L_cache_per_b[b])

    num_phys_pages = B * H_kv * N_LOGICAL_PAGES_MAX
    CACHE_SIZE = num_phys_pages * PAGE_SIZE

    K_cache = torch.zeros((CACHE_SIZE, D), device=device, dtype=dtype)
    V_cache = torch.zeros((CACHE_SIZE, D), device=device, dtype=dtype)
    page_table = torch.empty(
        (B, H_kv, N_LOGICAL_PAGES_MAX), device=device, dtype=torch.int32
    )

    # assign unique physical pages per (b, h, lp)
    phys_page = 0
    for b in range(B):
        for h in range(H_kv):
            for lp in range(N_LOGICAL_PAGES_MAX):
                page_table[b, h, lp] = phys_page
                phys_page += 1

    # fill cached tokens
    g = torch.Generator(device=device).manual_seed(1234)
    for b in range(B):
        Lc = int(L_cache_per_b[b].item())
        for h in range(H_kv):
            for i in range(Lc):
                lp = i // PAGE_SIZE
                off = i % PAGE_SIZE
                phys = int(page_table[b, h, lp].item())
                idx = phys * PAGE_SIZE + off
                K_cache[idx] = torch.randn(D, device=device, dtype=dtype, generator=g)
                V_cache[idx] = torch.randn(D, device=device, dtype=dtype, generator=g)

    return K_cache, V_cache, page_table, seq_lens_bh, CACHE_SIZE


def materialize_kv_for_flash_mixed(
    K_cache,
    V_cache,
    page_table,
    L_cache_per_b,  # [B]
    k_append_raw,  # [N, H_kv, D]
    v_append_raw,  # [N, H_kv, D]
    cu_seqlens_qk,  # [B+1]
    H_kv,
    PAGE_SIZE,
):
    """
    Build (K_total, V_total, cu_seqlens_k) for flash_attn_varlen_func such that:

      For each batch b:
        seqlen_q[b] = L_app[b] = cu[b+1] - cu[b]
        seqlen_k[b] = L_cache_per_b[b] + L_app[b]
      Keys:
        - first L_cache_per_b[b] positions from paged cache
        - next L_app[b] positions from k_append_raw for that batch
    """
    device = K_cache.device
    dtype = K_cache.dtype
    B = cu_seqlens_qk.numel() - 1
    N, H_kv_raw, D = k_append_raw.shape
    assert H_kv_raw == H_kv

    # appended lengths
    L_app = (cu_seqlens_qk[1:] - cu_seqlens_qk[:-1]).to(torch.int32)  # [B]
    seqlen_k = L_cache_per_b + L_app  # [B]

    cu_seqlens_k = torch.empty(B + 1, device=device, dtype=torch.int32)
    cu_seqlens_k[0] = 0

    total_k = int(seqlen_k.sum().item())
    K_total = torch.empty((total_k, H_kv, D), device=device, dtype=dtype)
    V_total = torch.empty((total_k, H_kv, D), device=device, dtype=dtype)

    for b in range(B):
        offset_k = int(cu_seqlens_k[b].item())
        Lc = int(L_cache_per_b[b].item())
        La = int(L_app[b].item())
        q_start = int(cu_seqlens_qk[b].item())

        # cache segment
        for g in range(H_kv):
            for i in range(Lc):
                lp = i // PAGE_SIZE
                off = i % PAGE_SIZE
                phys = int(page_table[b, g, lp].item())
                idx = phys * PAGE_SIZE + off
                K_total[offset_k + i, g] = K_cache[idx]
                V_total[offset_k + i, g] = V_cache[idx]

        # appended segment
        if k_append_raw.numel() > 0:
            for g in range(H_kv):
                for j in range(La):
                    src = q_start + j
                    dst = offset_k + Lc + j
                    K_total[dst, g] = k_append_raw[src, g]
                    V_total[dst, g] = v_append_raw[src, g]

        cu_seqlens_k[b + 1] = cu_seqlens_k[b] + (Lc + La)

    return K_total, V_total, cu_seqlens_k


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda wl: wl.name)
def test_causal_sparse_varlen_with_cache(workload: Workload):
    dtype = torch.float16
    device = triton.runtime.driver.active.get_active_torch_device()
    DEFAULT_PAGE_SIZE = 256
    N_LOGICAL_PAGES_MAX = 256
    L_cache_per_b = torch.as_tensor(
        workload.cache_lens, device=device, dtype=torch.int32
    )
    K_cache, V_cache, page_table, seq_lens_bh, CACHE_SIZE = (
        build_paged_cache_from_lengths(
            B=workload.batch_size,
            H_kv=workload.nk_heads,
            D=workload.head_dim,
            PAGE_SIZE=DEFAULT_PAGE_SIZE,
            N_LOGICAL_PAGES_MAX=N_LOGICAL_PAGES_MAX,
            L_cache_per_b=L_cache_per_b,
            device=device,
            dtype=dtype,
        )
    )

    assert len(workload.append_lens) == workload.batch_size
    cu = [0]
    for L in workload.append_lens:
        cu.append(cu[-1] + L)
    cu_seqlens_qk = torch.tensor(cu, dtype=torch.int32, device=device)
    N = int(cu_seqlens_qk[-1].item())

    q_raw = torch.randn(
        N, workload.nq_heads, workload.head_dim, device=device, dtype=dtype
    )
    k_append_raw = torch.randn(
        N, workload.nk_heads, workload.head_dim, device=device, dtype=dtype
    )
    v_append_raw = torch.randn_like(k_append_raw)

    batch_mapping = torch.arange(workload.batch_size, device=device, dtype=torch.int32)

    sm_scale = 1.0 / math.sqrt(workload.head_dim)
    K_total, V_total, cu_seqlens_k = materialize_kv_for_flash_mixed(
        K_cache=K_cache,
        V_cache=V_cache,
        page_table=page_table,
        L_cache_per_b=L_cache_per_b,
        k_append_raw=k_append_raw,
        v_append_raw=v_append_raw,
        cu_seqlens_qk=cu_seqlens_qk,
        H_kv=workload.nk_heads,
        PAGE_SIZE=DEFAULT_PAGE_SIZE,
    )

    max_seqlen_q = int((cu_seqlens_qk[1:] - cu_seqlens_qk[:-1]).max().item())
    max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
    max_seqlen_k_triton = seq_lens_bh.max().item()
    out_triton = causal_sparse_varlen_with_cache(
        q=q_raw,
        k_cache=K_cache,
        v_cache=V_cache,
        k=k_append_raw,
        v=v_append_raw,
        seq_lens_bh=seq_lens_bh,
        global_page_table=page_table,
        batch_mapping=batch_mapping,
        cu_seqlens_q=cu_seqlens_qk,
        HKV=workload.nk_heads,
        PAGE_SIZE=DEFAULT_PAGE_SIZE,
        sm_scale=sm_scale,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_cache=max_seqlen_k_triton,
    )
    out_flash = flash_attn_varlen_func(
        q=q_raw,
        k=K_total,
        v=V_total,
        cu_seqlens_q=cu_seqlens_qk,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=True,
    )
    assert torch.allclose(out_triton, out_flash, rtol=1e-6, atol=3e-3)
    max_diff = (out_triton - out_flash).abs().max().item()
    logger.info(
        f"[causal_sparse_varlen_with_cache: {workload.name}]: max abs diff={max_diff: .5f}"
    )


def materialize_kv_cache_for_flash_decode(
    K_cache,
    V_cache,
    page_table,
    L_cache_per_b,  # [B] int32
    H_kv: int,
    PAGE_SIZE: int,
):
    """
    Build (K_flash, V_flash) suitable for flash_attn_with_kvcache, with shape:
        (B, seqlen_cache_max, H_kv, D)

    For each batch b:
      - cache_seqlen[b] = L_cache_per_b[b]
      - K_flash[b, :cache_seqlen[b], g] and V_flash[...] are filled from the paged KV cache.
      - Tokens beyond cache_seqlen[b] (if any) are left as zeros and will be masked out
        by flash_attn_with_kvcache via cache_seqlens.
    """
    device = K_cache.device
    dtype = K_cache.dtype
    B = L_cache_per_b.shape[0]
    D = K_cache.shape[1]

    seqlen_cache_max = int(L_cache_per_b.max().item())
    K_flash = torch.zeros((B, seqlen_cache_max, H_kv, D), device=device, dtype=dtype)
    V_flash = torch.zeros_like(K_flash)

    for b in range(B):
        Lc = int(L_cache_per_b[b].item())
        if Lc == 0:
            continue
        for g in range(H_kv):
            for i in range(Lc):
                lp = i // PAGE_SIZE
                off = i % PAGE_SIZE
                phys = int(page_table[b, g, lp].item())
                idx = phys * PAGE_SIZE + off
                K_flash[b, i, g] = K_cache[idx]
                V_flash[b, i, g] = V_cache[idx]

    return K_flash, V_flash


@pytest.mark.parametrize("workload", WORKLOADS_DECODE, ids=lambda wl: wl.name)
def test_sparse_decode_attention(workload: Workload):
    dtype = torch.float16
    device = triton.runtime.driver.active.get_active_torch_device()
    DEFAULT_PAGE_SIZE = 256
    N_LOGICAL_PAGES_MAX = 256

    # per-sequence cache lengths (all equal for WORKLOADS_DECODE)
    L_cache_per_b = torch.as_tensor(
        workload.cache_lens, device=device, dtype=torch.int32
    )

    # build paged KV cache used by the Triton kernel
    K_cache, V_cache, page_table, seq_lens_bh, CACHE_SIZE = (
        build_paged_cache_from_lengths(
            B=workload.batch_size,
            H_kv=workload.nk_heads,
            D=workload.head_dim,
            PAGE_SIZE=DEFAULT_PAGE_SIZE,
            N_LOGICAL_PAGES_MAX=N_LOGICAL_PAGES_MAX,
            L_cache_per_b=L_cache_per_b,
            device=device,
            dtype=dtype,
        )
    )

    B = workload.batch_size
    HQ = workload.nq_heads
    HKV = workload.nk_heads
    D = workload.head_dim

    # Triton kernel expects q: [B, HQ, D]
    q_triton = torch.randn(B, HQ, D, device=device, dtype=dtype)
    batch_mapping = torch.arange(B, device=device, dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(D)

    out_triton = head_sparse_decode_attention(
        q=q_triton,
        k=K_cache,
        v=V_cache,
        seq_lens_bh=seq_lens_bh,
        global_page_table=page_table,
        batch_mapping=batch_mapping,
        HKV=HKV,
        PAGE_SIZE=DEFAULT_PAGE_SIZE,
        sm_scale=sm_scale,
    )  # [B, HQ, D]

    # materialize contiguous KV cache with shape [B, seqlen_cache_max, HKV, D]
    K_flash, V_flash = materialize_kv_cache_for_flash_decode(
        K_cache=K_cache,
        V_cache=V_cache,
        page_table=page_table,
        L_cache_per_b=L_cache_per_b,
        H_kv=HKV,
        PAGE_SIZE=DEFAULT_PAGE_SIZE,
    )

    # flash_attn_with_kvcache expects q: [B, seqlen_q, HQ, D]
    q_flash = q_triton.unsqueeze(1)  # seqlen_q = 1

    out_flash = flash_attn_with_kvcache(
        q=q_flash,
        k_cache=K_flash,
        v_cache=V_flash,
        cache_seqlens=L_cache_per_b,
        softmax_scale=sm_scale,
        causal=True,
    ).squeeze(1)  # [B, 1, HQ, D]

    assert torch.allclose(out_triton, out_flash, rtol=1e-6, atol=3e-3)
    max_diff = (out_triton - out_flash).abs().max().item()
    logger.info(
        f"[head_sparse_decode_attention: {workload.name}]: max abs diff={max_diff: .5f}"
    )
