import logging
import math

import torch
from compactor_vllm.attention.sparse_varlen_kernel import (
    causal_sparse_varlen_with_cache,
)

logger = logging.getLogger(__name__)


def build_mock_paged_cache_from_lengths(
    L_cache_per_b: torch.Tensor,
    HKV: int,
    D: int,
    PAGE_SIZE: int,
    N_LOGICAL_PAGES_MAX: int,
    device,
    dtype,
):
    B = len(L_cache_per_b)
    max_len = PAGE_SIZE * N_LOGICAL_PAGES_MAX
    assert (L_cache_per_b <= max_len).all()

    seq_lens_bh = torch.empty((B, HKV), dtype=torch.int32, device=device)
    for b in range(B):
        seq_lens_bh[b, :].fill_(L_cache_per_b[b])

    num_phys_pages = B * HKV * N_LOGICAL_PAGES_MAX
    CACHE_SIZE = num_phys_pages * PAGE_SIZE

    K_cache = torch.zeros((CACHE_SIZE, D), device=device, dtype=dtype)
    V_cache = torch.zeros((CACHE_SIZE, D), device=device, dtype=dtype)
    page_table = torch.empty(
        (B, HKV, N_LOGICAL_PAGES_MAX), device=device, dtype=torch.int32
    )

    # assign unique physical pages per (b, h, lp)
    phys_page = 0
    for b in range(B):
        for h in range(HKV):
            for lp in range(N_LOGICAL_PAGES_MAX):
                page_table[b, h, lp] = phys_page
                phys_page += 1

    for b in range(B):
        Lc = int(L_cache_per_b[b].item())
        for h in range(HKV):
            for i in range(Lc):
                lp = i // PAGE_SIZE
                off = i % PAGE_SIZE
                phys = int(page_table[b, h, lp].item())
                idx = phys * PAGE_SIZE + off
                K_cache[idx] = torch.randn(D, device=device, dtype=dtype)
                V_cache[idx] = torch.randn(D, device=device, dtype=dtype)

    return K_cache, V_cache, page_table, seq_lens_bh, CACHE_SIZE


def autotune_causal_sparse_varlen_with_cache(
    *, max_length=16384, HKV=8, HQ=32, D=128, PAGE_SIZE=256, N_LOGICAL_PAGES_MAX=256
):
    import itertools

    import tqdm

    device = "cuda"
    dtype = torch.float16

    B = 4
    assert (D & (D - 1)) == 0
    lengths_to_sweep = [0, 256]
    i = 9
    while (v := (1 << i)) < max_length:
        lengths_to_sweep.append(v)
        i += 1

    combos = list(itertools.product(lengths_to_sweep, repeat=2))
    logger.info(
        "tuning kernels. this may take a few minutes, but only needs to be run once per LLMConfig"
    )
    for cache_l, append_l in tqdm.tqdm(combos):
        if cache_l + append_l == 0:
            continue

        L_cache_per_b = torch.tensor(
            [cache_l] * B,
            device=device,
            dtype=torch.int32,
        )
        assert (L_cache_per_b <= PAGE_SIZE * N_LOGICAL_PAGES_MAX).all()
        K_cache, V_cache, page_table, seq_lens_bh, CACHE_SIZE = (
            build_mock_paged_cache_from_lengths(
                L_cache_per_b=L_cache_per_b,
                HKV=HKV,
                D=D,
                PAGE_SIZE=PAGE_SIZE,
                N_LOGICAL_PAGES_MAX=N_LOGICAL_PAGES_MAX,
                device=device,
                dtype=dtype,
            )
        )

        L_app_list = [append_l] * B
        cu = [0]
        for L in L_app_list:
            cu.append(cu[-1] + L)
        cu_seqlens_qk = torch.tensor(cu, dtype=torch.int32, device=device)
        N = int(cu_seqlens_qk[-1].item())

        max_seqlen_q = int((cu_seqlens_qk[1:] - cu_seqlens_qk[:-1]).max().item())
        max_seqlen_k = seq_lens_bh.max().item()
        q_raw = torch.randn(N, HQ, D, device=device, dtype=dtype)
        k_append_raw = torch.randn(N, HKV, D, device=device, dtype=dtype)
        v_append_raw = torch.randn(N, HKV, D, device=device, dtype=dtype)

        # Identity batch mapping (local batch index == global)
        batch_mapping = torch.arange(B, device=device, dtype=torch.int32)

        sm_scale = 1.0 / math.sqrt(D)

        causal_sparse_varlen_with_cache(
            q=q_raw,
            k_cache=K_cache,
            v_cache=V_cache,
            k=k_append_raw,
            v=v_append_raw,
            seq_lens_bh=seq_lens_bh,
            global_page_table=page_table,
            batch_mapping=batch_mapping,
            cu_seqlens_q=cu_seqlens_qk,
            HKV=HKV,
            PAGE_SIZE=PAGE_SIZE,
            sm_scale=sm_scale,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k_cache=max_seqlen_k,
        )


if __name__ == "__main__":
    autotune_causal_sparse_varlen_with_cache(
        max_length=16384, HKV=2, HQ=8, D=128, PAGE_SIZE=128, N_LOGICAL_PAGES_MAX=256
    )
