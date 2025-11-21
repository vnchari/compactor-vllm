import functools
import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def head_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens_bh: torch.Tensor,
    global_page_table: torch.Tensor,
    batch_mapping: torch.Tensor,
    HKV: int,
    PAGE_SIZE: int,
    sm_scale: float = None,
    key_split: int = None,
):
    """
    Decode-time head-sparse attention over a paged KV cache.

    This is a wrapper around the Triton decode kernel used during incremental
    generation. For each batch, we read the cached keys
    and values from a global paged KV buffer, apply causal attention with one
    new query token, and return the attention output.

    The KV cache is stored in a single global K/V tensor of shape
    ``[CACHE_SIZE, D]`` and indexed via a per-layer page table. Each logical
    (batch, kv_head, token_idx) is mapped to a physical row in the cache by:

        1. Looking up the logical page index in ``global_page_table[b, h, lp]``,
        2. Computing ``phys_row = page_id * PAGE_SIZE + (token_idx % PAGE_SIZE)``.

    Grouped-query attention (GQA / MQA) is supported by passing more query
    heads than KV heads (``HQ`` must be a multiple of ``HKV``).

    Args:
        :param q: Query tensor of shape ``[B, HQ, D]`` or `[B, 1, HQ, D]``
            containing the new decode tokens for each sequence in the launch batch.
        :param k: Global key cache of shape ``[CACHE_SIZE, D]``. This is the shared
            backing buffer for all (batch, head) KV pages.
        :param v: Global value cache of shape ``[CACHE_SIZE, D]``.
        :param seq_lens_bh: Tensor of shape ``[B, HKV]`` (int32) giving, for each
            local batch index and KV head, the number of valid cached tokens
            in the paged KV cache.
        :param global_page_table: Tensor of shape
            ``[MAX_NUM_BATCHES, HKV, N_LOGICAL_PAGES_MAX]`` (int32) mapping
            ``(true_batch_idx, kv_head, logical_page)`` to a physical page id
            in the global cache.
        :param batch_mapping: Tensor of shape ``[B]`` (int32) mapping the launch-batch
            index used by this call to the true batch row used to index
            ``global_page_table``.
        :param HKV: Number of KV heads.
        :param PAGE_SIZE: Number of tokens stored per physical KV page.
        :param sm_scale: Optional scaling factor applied to the attention logits
            before softmax. If ``None``, ``1 / sqrt(D)`` is used.
        :param key_split: Optional number of splits along the key sequence length.
            If > 1, the kernel will process the KV sequence in ``key_split``
            chunks to reduce on-chip memory usage. If ``None`` or 0, a
            heuristic is used.

    Returns:
        :return torch.Tensor: Attention output of shape ``[B, HQ, D]`` on the same
        device and dtype as ``q``.
    """

    with torch.cuda.device(q.device):
        if q.ndim != 3:
            assert q.ndim == 4
            B, HQ, S, D = q.shape
            assert S == 1, "head_sparse_decode_attention only supports q_len=1"
            q = q.squeeze(-2)
        elif q.ndim == 3:
            B, HQ, D = q.shape

        CACHE_SIZE = k.shape[0]
        assert PAGE_SIZE % 32 == 0, "PAGE_SIZE must be divisible by 128"
        GROUP_M = HQ // HKV
        assert GROUP_M * HKV == HQ, "HQ must be divisible by H_kv"

        FP8 = hasattr(torch, "float8_e5m2") and q.dtype == torch.float8_e5m2

        seq_lens_bh = seq_lens_bh.to(torch.int32)
        assert B <= 32767, "too many batches"
        assert global_page_table.shape[1] == HKV
        assert q.is_contiguous()
        assert (D & (D - 1)) == 0, "D must be a power of 2"
        N_LOGICAL_PAGES_MAX = global_page_table.shape[-1]

        sm_scale = 1 / math.sqrt(D) if sm_scale is None else sm_scale
        if key_split is None:
            # round max_seq_len to the next power of two to maximize cache hits
            key_split = num_splits_heuristic(
                B * HKV,
                max_seq_len=1 << int(seq_lens_bh.max()).bit_length(),
                num_sms=torch.cuda.get_device_properties(
                    q.device
                ).multi_processor_count,
                max_splits=12,
            )

        triton.set_allocator(
            lambda size, align, _: torch.empty(size, dtype=torch.int8, device=q.device)
        )

        # stage 1 scratch
        mid_o = torch.empty((B, key_split, HQ, D), device=q.device, dtype=q.dtype)
        mid_lse = torch.empty((B, key_split, HQ), device=q.device, dtype=torch.float32)
        # processes all queries for a KV head together
        # pointers are lowercase, CONSTANTS are upper
        grid1 = (B, HKV, key_split)
        _varkv_stage1_groupM[grid1](
            q=q,
            k=k,
            v=v,
            mid_o=mid_o,
            mid_lse=mid_lse,
            page_table_bhl=global_page_table,
            batch_mapping=batch_mapping,
            seq_lens_bh=seq_lens_bh.contiguous(),
            SM_SCALE=sm_scale,
            B=B,
            HKV=HKV,
            HQ=HQ,
            CACHE_SIZE=CACHE_SIZE,
            STRIDE_LBS=mid_lse.stride(0),
            STRIDE_LS=mid_lse.stride(1),
            STRIDE_LH=mid_lse.stride(2),
            N_LOGICAL_PAGES_MAX=N_LOGICAL_PAGES_MAX,
            D=D,
            KEY_SPLIT=key_split,
            GROUP_M=GROUP_M,
            DTYPE=tl.float8e5
            if FP8
            else (tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16),
            PAGE_SIZE=PAGE_SIZE,
        )

        if key_split == 1:
            return mid_o.squeeze(1).contiguous()

        # reduce partial results across splits
        output = torch.empty_like(q)
        grid2 = (B, HQ)
        _varkv_stage2_reduce[grid2](
            mid_o=mid_o,
            mid_lse=mid_lse,
            output=output,
            STRIDE_LBS=mid_lse.stride(0),
            STRIDE_LS=mid_lse.stride(1),
            STRIDE_LH=mid_lse.stride(2),
            STRIDE_OBS=output.stride(0),
            STRIDE_OH=output.stride(1),
            B=B,
            HQ=HQ,
            D=D,  # type: ignore
            KEY_SPLIT=key_split,  # type: ignore
            DTYPE=tl.float8e5
            if FP8
            else (tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16),
        )
        return output


# similar to flash attention split heuristic
@functools.lru_cache(maxsize=128)
def num_splits_heuristic(
    total_mblocks: int,
    max_seq_len: int,
    num_sms: int,
    max_splits: int,
) -> int:
    # If we nearly fill SMs already, prefer 1 split
    if total_mblocks >= 0.8 * num_sms or max_seq_len <= 1024:
        return 1
    eff = []
    max_eff = 0.0
    for s in range(1, min(max_splits, num_sms) + 1):
        if (max_seq_len / s) <= 512:
            break
        n_waves = float(total_mblocks * s) / float(num_sms)
        e = n_waves / math.ceil(n_waves) if n_waves > 0 else 0.0
        eff.append(e)
        max_eff = max(max_eff, e)
    threshold = 0.75 * max_eff  # if not split_min_hit else 0.9 * max_eff
    for i, e in enumerate(eff, start=1):
        if e >= threshold:
            return i
    return 1


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        # zero padded
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


def prune_invalid_configs(configs, _, **kwargs):
    PAGE_SIZE = kwargs["PAGE_SIZE"]
    return [conf for conf in configs if conf.kwargs.get("BLOCK_N", 0) <= PAGE_SIZE]


def _stage1_host_desc_pre_hook(nargs):
    D = nargs["D"]
    BLOCK_N = nargs["BLOCK_N"]
    GROUP_M = nargs["GROUP_M"]
    if isinstance(nargs["q"], TensorDescriptor):
        nargs["q"].block_shape = [GROUP_M, D]
    if isinstance(nargs["k"], TensorDescriptor):  # K is the GLOBAL cache now
        nargs["k"].block_shape = [BLOCK_N, D]
    if isinstance(nargs["v"], TensorDescriptor):  # V is the GLOBAL cache now
        nargs["v"].block_shape = [BLOCK_N, D]
    if isinstance(nargs["mid_o"], TensorDescriptor):
        nargs["mid_o"].block_shape = [GROUP_M, D]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": BLOCK_N, "MIN_BLOCK_KV": MIN_BLOCK_KV, "WARPSPEC": ws},
            num_warps=w,
            num_stages=s,
            pre_hook=_stage1_host_desc_pre_hook,
        )
        for BLOCK_N in [32, 64, 128]
        for MIN_BLOCK_KV in [8]
        for s in [2, 3, 4]
        for w in [4, 8]
        for ws in [True, False]
    ],
    key=[
        "HKV",
        "GROUP_M",
        "D",
        "PAGE_SIZE",  # "B"
    ],
    cache_results=True,
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _varkv_stage1_groupM(
    q,  # [B*H_q, D] via descriptor
    k,
    v,  # GLOBAL caches: [S_LARGE, D] via descriptor
    mid_o,
    mid_lse,
    page_table_bhl,  # int32 [B*H_kv*N_LOGICAL_PAGES_MAX] (flattened)
    batch_mapping,  # int32 [B]  maps local pid_b -> true batch index
    seq_lens_bh,  # int32 [B*H_kv] valid tokens per (b,h)
    SM_SCALE,
    B,
    HKV,
    HQ,
    CACHE_SIZE,  # CACHE_SIZE = N_PAGES * PAGE_SIZE
    STRIDE_LBS,
    STRIDE_LS,
    STRIDE_LH,
    # constexprs
    N_LOGICAL_PAGES_MAX: tl.constexpr,  # page table width per (b,h)
    D: tl.constexpr,
    KEY_SPLIT: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    WARPSPEC: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_kvh = tl.program_id(1)  # kv head
    pid_s = tl.program_id(2)  # split

    # valid length L for this (b,h)
    bh_stride = HKV
    L = tl.load(seq_lens_bh + pid_b * bh_stride + pid_kvh)
    if L == 0:
        return

    tl.assume(L > 0)

    ydim_q = B * HQ
    ydim_mid = B * KEY_SPLIT * HQ
    desc_q = _maybe_make_tensor_desc(
        q, shape=[ydim_q, D], strides=[D, 1], block_shape=[GROUP_M, D]
    )
    desc_k = _maybe_make_tensor_desc(
        k, shape=[CACHE_SIZE, D], strides=[D, 1], block_shape=[BLOCK_N, D]
    )
    desc_v = _maybe_make_tensor_desc(
        v, shape=[CACHE_SIZE, D], strides=[D, 1], block_shape=[BLOCK_N, D]
    )
    desc_mid = _maybe_make_tensor_desc(
        mid_o, shape=[ydim_mid, D], strides=[D, 1], block_shape=[GROUP_M, D]
    )

    # split sizing on logical token axis [0..L)
    base = tl.cdiv(L, KEY_SPLIT)
    per_split_len = tl.cdiv(base, MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_start = pid_s * per_split_len
    split_end = tl.minimum(split_start + per_split_len, L)

    # query heads mapped to this kv head
    base_qh = pid_kvh * GROUP_M
    offs_m = tl.arange(0, GROUP_M)

    # load Q tile [M, D]
    row_base_q = pid_b * HQ + base_qh
    q = desc_q.load([row_base_q, 0])  # [GROUP_M, D]

    # streaming softmax state per query
    e_max = tl.zeros([GROUP_M], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([GROUP_M], dtype=tl.float32)
    acc = tl.zeros([GROUP_M, D], dtype=tl.float32)

    if split_end > split_start:
        # logical pages covering [split_start, split_end)
        lp0 = split_start // PAGE_SIZE
        lp1 = tl.cdiv(split_end, PAGE_SIZE)  # exclusive

        mapped_b = tl.load(batch_mapping + pid_b)
        tl.assume(mapped_b >= 0)
        # page table base for this (b,h)
        pt_stride = N_LOGICAL_PAGES_MAX
        pt_base = (mapped_b * HKV + pid_kvh) * pt_stride

        for lp in tl.range(lp0, lp1):
            phys = tl.load(
                page_table_bhl + pt_base + lp, cache_modifier=".cg"
            )  # physical page id
            # bounds within the logical page
            local_start = tl.where(lp == lp0, split_start - lp * PAGE_SIZE, 0)
            local_end = tl.where(lp == (lp1 - 1), split_end - lp * PAGE_SIZE, PAGE_SIZE)

            page_base = phys * PAGE_SIZE
            page_base = tl.multiple_of(page_base, BLOCK_N)
            for s in tl.range(
                local_start, local_end, BLOCK_N, warp_specialize=WARPSPEC
            ):
                s = tl.multiple_of(s, MIN_BLOCK_KV)

                k_blk = desc_k.load([page_base + s, 0])  # [BLOCK_N, D]
                qk = tl.dot(q, k_blk.T) * SM_SCALE  # [M, BN]

                offs_n = s + tl.arange(0, BLOCK_N)
                mask_n = offs_n < local_end
                qk = tl.where(mask_n[None, :], qk, -float("inf"))

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)  # [M]
                re_scale = tl.exp(e_max - n_e_max)  # [M]
                acc = acc * re_scale[:, None]  # [M, D]

                v_blk = desc_v.load([page_base + s, 0])  # [BLOCK_N, D]
                p = tl.exp(qk - n_e_max[:, None])  # [M, BN]
                acc = tl.dot(p.to(DTYPE), v_blk, acc)

                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

        # write mid outputs [M, D] for this split
        row_mid = pid_b * (KEY_SPLIT * HQ) + pid_s * HQ + base_qh
        tmp = (acc / e_sum[:, None]).to(DTYPE)
        desc_mid.store([row_mid, 0], tmp)

        ml_ptrs = (
            mid_lse
            + pid_b * STRIDE_LBS
            + pid_s * STRIDE_LS
            + (base_qh + offs_m) * STRIDE_LH
        )
        tl.store(ml_ptrs, e_max + tl.log(e_sum))
    else:
        # empty split
        row_mid = pid_b * (KEY_SPLIT * HQ) + pid_s * HQ + base_qh
        zero_md = tl.zeros([GROUP_M, D], dtype=DTYPE)
        desc_mid.store([row_mid, 0], zero_md)
        ml_ptrs = (
            mid_lse
            + pid_b * STRIDE_LBS
            + pid_s * STRIDE_LS
            + (base_qh + offs_m) * STRIDE_LH
        )
        tl.store(ml_ptrs, -float("inf"))


@triton.jit
def _varkv_stage2_reduce(
    mid_o,
    mid_lse,
    output,
    STRIDE_LBS,
    STRIDE_LS,
    STRIDE_LH,
    STRIDE_OBS,
    STRIDE_OH,
    B,
    HQ,
    D: tl.constexpr,
    KEY_SPLIT: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_d = tl.arange(0, D)

    # across split LSE combine
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([D], dtype=tl.float32)

    ydim_mid = B * KEY_SPLIT * HQ
    desc_mid = _maybe_make_tensor_desc(
        mid_o, shape=[ydim_mid, D], strides=[D, 1], block_shape=[1, D]
    )

    for s in tl.range(KEY_SPLIT):
        row_mid = pid_b * (KEY_SPLIT * HQ) + s * HQ + pid_h
        tv = desc_mid.load([row_mid, 0]).reshape([D])  # [D]
        tl_ptr = mid_lse + pid_b * STRIDE_LBS + s * STRIDE_LS + pid_h * STRIDE_LH
        tlogic = tl.load(tl_ptr)

        n_e_max = tl.maximum(e_max, tlogic)
        old_scale = tl.exp(e_max - n_e_max)
        acc = acc * old_scale + tl.exp(tlogic - n_e_max) * tv.to(tl.float32)
        e_sum = e_sum * old_scale + tl.exp(tlogic - n_e_max)
        e_max = n_e_max

    o = (acc / e_sum).to(DTYPE)
    o_ptr = output + pid_b * STRIDE_OBS + pid_h * STRIDE_OH + offs_d
    tl.store(o_ptr, o)
