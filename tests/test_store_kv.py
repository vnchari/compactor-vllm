import collections
import logging
from dataclasses import dataclass
from typing import List

import pytest
import torch
import triton

from compactor_vllm.compression.common import scores_to_retain_indices
from compactor_vllm.kv_cache.store_kv_cache import prefill_store_topk_kv

logger = logging.getLogger(__name__)


@dataclass
class Workload:
    name: str
    batch_size: int
    nk_heads: int
    head_dim: int
    topk: int  # per-sequence cached context length
    page_size: int
    cache_lens: List[int]  # per-sequence cached context length


WORKLOADS: List[Workload] = [
    Workload(
        name=f"batch_size={BATCH} kv_cache_len={cache_lens} "
        f"TOPK={topk} HKV={NK_HEADS} HEAD_DIM={HEAD_DIM}",
        batch_size=BATCH,
        nk_heads=NK_HEADS,
        head_dim=HEAD_DIM,
        cache_lens=[cache_lens] * BATCH,
        topk=topk,
        page_size=ps,
    )
    for BATCH in [1, 2, 3, 8]
    for topk in [10, 20, 30, 40]
    for NK_HEADS in [2, 4, 8]
    for HEAD_DIM in [32, 64, 128]
    for cache_lens in [10, 20, 30, 70, 1000]
    for ps in [128, 256]
]


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda wl: wl.name)
def test_prefill_store_topk_kv(workload: Workload):
    B = workload.batch_size
    H = workload.nk_heads
    D = workload.head_dim
    TOP_K = workload.topk
    PAGE_SIZE = workload.page_size

    dtype = torch.float16
    device = triton.runtime.driver.active.get_active_torch_device()

    lens = torch.tensor(workload.cache_lens, dtype=torch.int32, device=device)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(lens, dim=0)
    N_total = int(cu[-1].item())

    keys = torch.randn((N_total, H, D), dtype=dtype, device=device)
    vals = torch.randn_like(keys)
    scores_flat = torch.randn((N_total, H), dtype=torch.float32, device=device)

    top_k_eff = max(0, min(TOP_K, int(lens.max().item()) * H))
    max_k_len = cu.diff().max().item()
    indices = scores_to_retain_indices(
        scores_flat, cu, max_k_len, top_k_eff, H
    )  # [B, TOP_K]

    LP = max(1, (top_k_eff + PAGE_SIZE - 1) // PAGE_SIZE)
    N_LOGICAL_PAGES_MAX = LP
    N_PAGES = B * H * LP + 32
    S_LARGE = N_PAGES * PAGE_SIZE
    k_cache = torch.empty((S_LARGE, D), dtype=dtype, device=device)
    v_cache = torch.empty_like(k_cache)

    page_table = torch.empty(
        (B, H, N_LOGICAL_PAGES_MAX), dtype=torch.int32, device=device
    )
    phys = 0
    for b in range(B):
        for h in range(H):
            for lp in range(LP):
                page_table[b, h, lp] = phys
                phys += 1
    assert phys <= N_PAGES, "Not enough physical pages"

    local_lens = torch.zeros((B, H), dtype=torch.int32, device=device)
    batch_mapping = torch.arange(B, dtype=torch.int32, device=device)
    num_to_retain = torch.full((B,), top_k_eff, dtype=torch.int32, device=device)

    prefill_store_topk_kv(
        new_keys=keys,
        new_vals=vals,
        indices_topk=indices,
        num_tokens_to_retain=num_to_retain,
        page_table=page_table,
        batch_mapping=batch_mapping,
        bh_lens=local_lens,
        PAGE_SIZE=PAGE_SIZE,
        k_cache=k_cache,
        v_cache=v_cache,
        PAD_TO_PAGE_SIZE=False,
        TRITON_RESERVED_BATCH=-1,
    )
    torch.cuda.synchronize()

    local_lens_cpu = local_lens.cpu()
    page_table_cpu = page_table.cpu()
    k_cache_cpu = k_cache.cpu()
    v_cache_cpu = v_cache.cpu()
    keys_cpu = keys.cpu()
    vals_cpu = vals.cpu()
    indices_cpu = indices.cpu()

    for b in range(B):
        hed = (indices_cpu[b] % H).numpy()
        counts = collections.Counter(hed.tolist())
        for h in range(H):
            expected = counts.get(h, 0)  # type: ignore
            got = int(local_lens_cpu[b, h].item())
            assert got == expected, (
                f"Length mismatch at (b={b}, h={h}): got {got}, expected {expected}"
            )

    def rows_for_head(b, h, L):
        """Return the list of cache row indices storing the first L logical positions for (b,h)."""
        rows = []
        for pos in range(L):
            lp = pos // PAGE_SIZE
            off = pos % PAGE_SIZE
            phys = int(page_table_cpu[b, h, lp].item())
            rows.append(phys * PAGE_SIZE + off)
        return rows

    for b in range(B):
        # which tokens per head were selected for this batch?
        tok = (indices_cpu[b] // H).numpy()
        hed = (indices_cpu[b] % H).numpy()
        per_head = collections.defaultdict(list)
        for t, h in zip(tok, hed):
            per_head[int(h)].append(int(t))

        for h in range(H):
            L = int(local_lens_cpu[b, h].item())
            if L == 0:
                continue

            # expected vectors (unordered) from source
            toks_h = per_head.get(h, [])
            assert len(toks_h) == L
            expK = keys_cpu[toks_h, h, :].contiguous().view(L, -1)
            expV = vals_cpu[toks_h, h, :].contiguous().view(L, -1)

            # actual vectors read back from cache rows
            rows = rows_for_head(b, h, L)
            actK = k_cache_cpu[rows, :].contiguous().view(L, -1)
            actV = v_cache_cpu[rows, :].contiguous().view(L, -1)

            expK_tuples = [tuple(row) for row in expK.numpy().tolist()]
            actK_tuples = [tuple(row) for row in actK.numpy().tolist()]
            expV_tuples = [tuple(row) for row in expV.numpy().tolist()]
            actV_tuples = [tuple(row) for row in actV.numpy().tolist()]

            assert collections.Counter(expK_tuples) == collections.Counter(
                actK_tuples
            ), f"K content mismatch at (b={b}, h={h})"
            assert collections.Counter(expV_tuples) == collections.Counter(
                actV_tuples
            ), f"V content mismatch at (b={b}, h={h})"
