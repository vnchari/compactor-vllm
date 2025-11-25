import argparse
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
    *,
    max_length: int = 16384,
    HKV: int = 8,
    HQ: int = 32,
    D: int = 128,
    PAGE_SIZE: int = 128,
    device: str = "cuda",
    dtype=torch.float16,
):
    """
    Autotune causal_sparse_varlen_with_cache over a sweep of cache/append lengths.
    """
    import itertools

    import tqdm

    N_LOGICAL_PAGES_MAX = ((max_length + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    B = 4

    # D must be a power of two (kernel requirement).
    assert (D & (D - 1)) == 0

    lengths_to_sweep = [0, 256]
    i = 9
    while (v := (1 << i)) < max_length:
        lengths_to_sweep.append(v)
        i += 1

    combos = list(itertools.product(lengths_to_sweep, repeat=2))
    logger.info(
        "tuning kernels. this may take a few minutes, "
        "but only needs to be run once per LLMConfig"
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autotune Triton kernels. "
                    "Results are cached, so this should only need to be run once per configuration."
                    "This script doesn't need to be run, as the kernels will be autotuned at runtime"
                    "if no cached autotuning data exists. Running this before hand will prevent run-time"
                    "autotuning, which will accelerate compactor-vllm at inference time."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="Maximum total sequence length to consider.",
    )
    parser.add_argument(
        "--HKV",
        type=int,
        default=8,
        help="Number of KV heads.",
    )
    parser.add_argument(
        "--HQ",
        type=int,
        default=32,
        help="Number of query heads.",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=128,
        help="Per-head hidden dimension (must be power of 2).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=128,
        help="Page size (tokens per physical page).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run on (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Dtype for tensors: one of {float16, fp16, bfloat16, bf16, float32, fp32}.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level.",
    )
    return parser.parse_args()


def _resolve_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dtype = _resolve_dtype(args.dtype)
    logger.info(
        "Starting autotune with max_length=%d, HKV=%d, HQ=%d, D=%d, page_size=%d, "
        "device=%s, dtype=%s",
        args.max_length,
        args.HKV,
        args.HQ,
        args.D,
        args.page_size,
        args.device,
        dtype,
    )

    autotune_causal_sparse_varlen_with_cache(
        max_length=args.max_length,
        HKV=args.HKV,
        HQ=args.HQ,
        D=args.D,
        PAGE_SIZE=args.page_size,
        device=args.device,
        dtype=dtype,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    main()
