import math
import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
from dataclasses import dataclass
from typing import List, Optional
from typing import Sequence as Seq

import torch
import triton
import triton.testing as testing
from compactor_vllm.attention.sparse_varlen_kernel import (
    causal_sparse_varlen_with_cache,
)
from flash_attn.flash_attn_interface import flash_attn_varlen_func

DEVICE = triton.runtime.driver.active.get_active_torch_device()

DEFAULT_PAGE_SIZE = 256


@dataclass
class Workload:
    name: str
    batch_size: int
    nq_heads: int
    nk_heads: int
    head_dim: int
    cache_lens: List[int]  # per-sequence cached context length
    append_lens: List[int]  # per-sequence new tokens this step

    @property
    def max_cache_len(self) -> int:
        return max(self.cache_lens) if self.cache_lens else 0

    @property
    def max_append_len(self) -> int:
        return max(self.append_lens) if self.append_lens else 0

    @property
    def total_append_tokens(self) -> int:
        return sum(self.append_lens)

    @property
    def total_kv_tokens(self) -> int:
        return sum(c + a for c, a in zip(self.cache_lens, self.append_lens))


def attention_flops(workload: Workload, include_softmax: bool = True) -> float:
    """
    Estimate FLOPs for scaled dot-product attention for a given workload.

    Counts:
      - QK^T: 2 * Nq * Nk * head_dim per head  (mul + add)
      - AV  : 2 * Nq * Nk * head_dim per head
      - softmax (optional): ~4 ops per [Nq, Nk] entry (exp, adds, div)

    Total FLOPs ~= sum_b  heads * [4 * Nq_b * Nk_b * head_dim + 4 * Nq_b * Nk_b]

    where for each sequence b:
      Nq_b = append_lens[b]
      Nk_b = cache_lens[b] + append_lens[b]
    """
    assert len(workload.cache_lens) == workload.batch_size
    assert len(workload.append_lens) == workload.batch_size

    Hq = workload.nq_heads
    D = workload.head_dim

    total_qk_flops = 0
    total_av_flops = 0
    total_softmax_flops = 0

    for cache_len, append_len in zip(workload.cache_lens, workload.append_lens):
        Nq = append_len
        Nk = cache_len + append_len

        if Nq == 0 or Nk == 0:
            continue

        # QK^T and AV cost per sequence (all query heads)
        qk_flops = 2 * Hq * Nq * Nk * D
        av_flops = 2 * Hq * Nq * Nk * D

        total_qk_flops += qk_flops
        total_av_flops += av_flops

        if include_softmax:
            # Approx: 4 FLOPs per attention scalar per head
            total_softmax_flops += 4 * Hq * Nq * Nk

    total = total_qk_flops + total_av_flops + total_softmax_flops
    return float(total)


def build_cu_seqlens(
    lengths: Seq[int],
    device: torch.device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Prefix sums with leading zero: len = B + 1."""
    out = torch.empty(len(lengths) + 1, device=device, dtype=dtype)
    out[0] = 0
    lens = torch.tensor(lengths, device=device, dtype=dtype)
    torch.cumsum(lens, dim=0, out=out[1:])
    return out


def build_flash_inputs(
    wl: Workload,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Build Q / K / V and cu_seqlens for flash_attn_varlen_func.

    For each sequence b:
      - Q consists only of the appended tokens (length = append_lens[b]).
      - K, V consist of [cache_tokens(b), append_tokens(b)].
    """
    HQ = wl.nq_heads
    HK = wl.nk_heads
    Dh = wl.head_dim

    q_lens = wl.append_lens
    k_lens = [c + a for c, a in zip(wl.cache_lens, wl.append_lens)]

    total_q = sum(q_lens)
    total_k = sum(k_lens)

    q = torch.randn(total_q, HQ, Dh, device=device, dtype=dtype)
    k = torch.randn(total_k, HK, Dh, device=device, dtype=dtype)
    v = torch.randn_like(k)

    cu_seqlens_q = build_cu_seqlens(q_lens, device=device, dtype=torch.int32)
    cu_seqlens_k = build_cu_seqlens(k_lens, device=device, dtype=torch.int32)

    # Per-sequence splits (so we can build paged KV cache)
    cache_tokens = []
    app_tokens = []
    k_offset = 0
    for c_len, a_len in zip(wl.cache_lens, wl.append_lens):
        if c_len > 0:
            cache_tokens.append(k[k_offset : k_offset + c_len])
            k_offset += c_len
        else:
            cache_tokens.append(k[k_offset:k_offset])  # empty slice
        app_tokens.append(k[k_offset : k_offset + a_len])
        k_offset += a_len

    assert k_offset == total_k

    return {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "cache_tokens": cache_tokens,  # list[B], each [L_cache_b, H, Dh]
        "app_tokens": app_tokens,  # list[B], each [L_app_b, H, Dh]
    }


def build_sparse_cache(
    wl: Workload,
    cache_tokens: Seq[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
    page_size: int,
):
    """
    Construct a toy paged KV cache compatible with causal_head_sparse_varlen_with_cache.

    Layout: for each (b, h), we assign a contiguous run of pages; tokens for (b, h)
    are laid out densely across those pages.
    """
    B = wl.batch_size
    HKV = wl.nk_heads  # assume HQ == HKV in this benchmark
    Dh = wl.head_dim

    # seq_lens_bh: [B, HKV], number of cached tokens for each (batch, head)
    seq_lens_bh = torch.zeros(B, HKV, device=device, dtype=torch.int32)
    for b, cache_len in enumerate(wl.cache_lens):
        seq_lens_bh[b, :].fill_(cache_len)

    num_pages_per_b = [
        math.ceil(cache_len / page_size) if cache_len > 0 else 0
        for cache_len in wl.cache_lens
    ]
    max_pages_per_b = max(num_pages_per_b) if num_pages_per_b else 0
    N_LOGICAL_PAGES_MAX = max(max_pages_per_b, 1)  # keep tensor non-empty

    total_pages = sum(npb * HKV for npb in num_pages_per_b)
    cache_size = max(total_pages * page_size, page_size)

    # One big flat cache for K/V; adjust to your actual kernel layout if needed
    k_cache = torch.zeros(cache_size, Dh, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)

    # page_table: [B, HKV, N_LOGICAL_PAGES_MAX]
    page_table = torch.zeros(
        B, HKV, N_LOGICAL_PAGES_MAX, device=device, dtype=torch.int32
    )

    # Fill K/V cache and page_table
    page_counter = 0
    for b in range(B):
        cache_len = wl.cache_lens[b]
        if cache_len == 0:
            continue
        tokens_b = cache_tokens[b]  # [L_cache_b, H, Dh]
        assert tokens_b.shape[0] == cache_len
        num_pages = num_pages_per_b[b]
        for h in range(HKV):
            # assign pages for this (b, h)
            for lp in range(num_pages):
                phys = page_counter
                page_table[b, h, lp] = phys
                page_counter += 1
            # write tokens into those pages
            base_token_idx = 0
            for lp in range(num_pages):
                phys = int(page_table[b, h, lp].item())
                page_start = phys * page_size
                remain = cache_len - base_token_idx
                page_len = min(page_size, remain)
                if page_len <= 0:
                    break
                k_slice = tokens_b[base_token_idx : base_token_idx + page_len, h, :]
                v_slice = k_slice  # timing only; reuse K as V
                k_cache[page_start : page_start + page_len].copy_(k_slice)
                v_cache[page_start : page_start + page_len].copy_(v_slice)
                base_token_idx += page_len

    batch_mapping = torch.arange(B, device=device, dtype=torch.int32)

    return {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "seq_lens_bh": seq_lens_bh,
        "page_table": page_table,
        "batch_mapping": batch_mapping,
        "N_LOGICAL_PAGES_MAX": N_LOGICAL_PAGES_MAX,
        "CACHE_SIZE": cache_size,
    }


def run_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
):
    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
    )


def run_sparse(
    q: torch.Tensor,
    k_app: torch.Tensor,
    v_app: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_bh: torch.Tensor,
    page_table: torch.Tensor,
    batch_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_size: int,
    softmax_scale: float,
):
    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k_cache = seq_lens_bh.max().item()
    HKV = k_app.shape[1]  # assuming Q and KV share head count

    return causal_sparse_varlen_with_cache(
        q,
        k_app,
        v_app,
        k_cache,
        v_cache,
        seq_lens_bh=seq_lens_bh,
        global_page_table=page_table,
        batch_mapping=batch_mapping,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_cache=max_seqlen_k_cache,
        HKV=HKV,
        PAGE_SIZE=page_size,
        sm_scale=softmax_scale,
    )


if __name__ == "__main__":
    BATCH = 8
    NQ_HEADS = 32
    NK_HEADS = 8
    HEAD_DIM = 128

    configs: List[testing.Benchmark] = []

    for Q_LEN in [256, 1024, 4096, 8192, 16384]:
        configs.append(
            testing.Benchmark(
                x_names=["K_LEN"],  # length of appended query tokens
                x_vals=[0, 128, 256, 1024, 4096, 8192, 16384],
                line_arg="backend",
                line_vals=["flash", "sparse_varlen"],
                line_names=["FlashAttention", "SparseVarlen"],
                styles=[("green", "-"), ("red", "-")],
                ylabel="TFLOPS",
                plot_name=f"sparse-vs-flash-attn-3-QLEN={Q_LEN}",
                args={
                    "BATCH": BATCH,
                    "HQ": NQ_HEADS,
                    "HK": NK_HEADS,
                    "HEAD_DIM": HEAD_DIM,
                    "Q_LEN": Q_LEN,
                },
            )
        )

    @testing.perf_report(configs)
    def bench_sparse_vs_flash(
        K_LEN,
        backend: str,
        BATCH: int,
        HQ: int,
        HK: int,
        HEAD_DIM: int,
        Q_LEN: int,
        device: Optional[torch.device] = DEVICE,
    ):
        """
        Returns latency in ms for given backend & workload.
        Called by Triton perf_report for each point in `configs`.
        """
        assert backend in {"flash", "sparse_varlen"}
        device = device or torch.device("cuda")
        dtype = torch.float16

        wl = Workload(
            name="",
            batch_size=BATCH,
            nk_heads=HK,
            nq_heads=HQ,
            head_dim=HEAD_DIM,
            cache_lens=[K_LEN] * BATCH,
            append_lens=[Q_LEN] * BATCH,
        )

        # Build base tensors once; do_bench will re-run only the kernels.
        flash_inputs = build_flash_inputs(wl, dtype=dtype, device=device)
        q = flash_inputs["q"]
        k = flash_inputs["k"]
        v = flash_inputs["v"]
        cu_q = flash_inputs["cu_seqlens_q"]
        cu_k = flash_inputs["cu_seqlens_k"]
        cache_tokens = flash_inputs["cache_tokens"]
        app_tokens = flash_inputs["app_tokens"]

        # Appended tokens concatenated for sparse kernel
        k_app = torch.cat(app_tokens, dim=0)
        v_app = k_app.clone()

        cache = build_sparse_cache(
            wl,
            cache_tokens=cache_tokens,
            dtype=dtype,
            device=device,
            page_size=DEFAULT_PAGE_SIZE,
        )

        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)

        if backend == "flash":

            def fn():
                return run_flash(q, k, v, cu_q, cu_k, softmax_scale)
        else:

            def fn():
                return run_sparse(
                    q,
                    k_app,
                    v_app,
                    cu_q,
                    cache["seq_lens_bh"],
                    cache["page_table"],
                    cache["batch_mapping"],
                    cache["k_cache"],
                    cache["v_cache"],
                    DEFAULT_PAGE_SIZE,
                    softmax_scale,
                )

        ms = testing.do_bench(fn)
        flops = attention_flops(wl, include_softmax=True)
        # flops_per_sec = (flops / 1e12) / (ms / 1e3)
        return (flops / 1e12) / (ms / 1e3)

    bench_sparse_vs_flash.run(save_path=None, print_data=True)
