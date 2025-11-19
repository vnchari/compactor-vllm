import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def causal_sparse_varlen_with_cache(
    q,
    k,
    v,
    k_cache,
    v_cache,
    seq_lens_bh,
    global_page_table,
    batch_mapping,
    cu_seqlens_q,
    max_seqlen_q: int,
    max_seqlen_k_cache: int,
    HKV: int,
    PAGE_SIZE: int,
    sm_scale=None,
):
    """
    Causal prefill attention over a paged KV cache plus a block of newly
    appended tokens in a packed batch format.

    This function wraps the Triton kernel
    ``_causal_head_sparse_varlen_with_cache`` to compute prefill attention for
    a batch of variable-length sequences, where:
      • Past keys/values are stored in a paged global KV cache
        (``k_cache``, ``v_cache``) with a (per-layer) page table.

      • New tokens for this step are given as K/V blocks
        (``k``, ``v``), together with a packed query block ``q``.

      • The result is equivalent to applying causal attention over the
        concatenation of:
            [ cached KV prefix  ||  (K_app, V_app) for this step ]
        for each sequence in the batch.

    Grouped-query attention (GQA / MQA) is supported by allowing more query
    heads than KV heads: ``HQ`` must be divisible by ``HKV``.

    Args:
        :param q:
            Query tensor of shape ``[N, HQ, D]`` (float16 / bfloat16/float32).
            ``N`` is the total number of new tokens across the batch
            (i.e. ``N = sum_b seqlen_q[b]``), packed according to
            ``cu_seqlens_q``. ``HQ`` is the number of query heads, ``D`` the
            head dimension (must be a power of two).
        :param k:
            New key tensor of shape ``[N, HKV, D]`` for the same tokens as
            ``q``. These are the K values appended to the cache for this
            prefill step.
        :param v:
            New value tensor of shape ``[N, HKV, D]`` for the same tokens as
            ``q``.
        :param k_cache:
            Global key cache backing buffer of shape ``[CACHE_SIZE, D]``.
            Keys for all cached tokens and heads are stored here; the mapping
            from (batch, head, token index) to a row in this buffer is
            given by ``global_page_table``.
        :param v_cache:
            Global value cache of shape ``[CACHE_SIZE, D]``. Must have the
            same layout as ``k_cache`` (same ``CACHE_SIZE`` and ``D``).
        :param seq_lens_bh:
            Tensor of shape ``[B, HKV]`` (int32) giving, for each local batch
            index and KV head, the number of cached tokens already present
            in the paged KV cache before this prefill step.
        :param global_page_table:
            Tensor of shape ``[MAX_NUM_BATCHES, HKV, N_LOGICAL_PAGES_MAX]`` (int32)
            mapping ``(true_batch_idx, kv_head, logical_page)`` to a physical
            page id in the global KV cache. A physical page id `p` refers to
            the slice:
                ``k_cache[p * PAGE_SIZE : (p + 1) * PAGE_SIZE]``.
        :param batch_mapping:
            Tensor of shape ``[B]`` (int16 / int32) mapping the local batch
            index used in this kernel launch to the global batch index used
            to index ``global_page_table``. This allows the same global cache
            to be shared across multiple microbatches.
        :param cu_seqlens_q:
            Tensor of shape ``[B + 1]`` (int32) with cumulative sequence
            lengths for the *new* tokens (q/k/v) in packed form. For batch
            element ``b``:
                ``seqlen_q[b] = cu_seqlens_q[b + 1] - cu_seqlens_q[b]``.
            The total number of tokens satisfies
                ``N = cu_seqlens_q[-1]``.
        :param max_seqlen_q:
            Maximum new query sequence length across the batch, i.e.
            ``max_b seqlen_q[b]``.
        :param max_seqlen_k_cache:
            Maximum cached sequence length across (batch, KV head), i.e.
            ``max_{b,h} seq_lens_bh[b, h]``.
        :param HKV:
            Number of KV heads. Must divide ``HQ``.
        :param PAGE_SIZE:
            Number of tokens stored per physical page in the paged KV cache.
            ``CACHE_SIZE`` must be divisible by ``PAGE_SIZE``.
        :param sm_scale:
            Optional scaling factor applied to the attention logits before
            softmax. If ``None``, defaults to ``1.0 / sqrt(D)``.
        :returns torch.Tensor:
            Attention output of shape ``[N, HQ, D]``, with the same dtype and
            device as ``q``. The output is laid out in the same packed
            varlen format as the input queries, i.e. the first
            ``seqlen_q[0]`` rows correspond to batch 0, the next
            ``seqlen_q[1]`` rows to batch 1, etc.
    """
    assert q.ndim == 3, "q should be [N, HQ, D]"
    N, HQ, D = q.shape
    assert (D & (D - 1)) == 0, "D must be power of two"

    B = cu_seqlens_q.numel() - 1
    assert B > 0
    assert HQ % HKV == 0, "Number of query heads must divide number of keys heads"
    H_g = HQ // HKV
    # view Q as [HKV, N, QUERY_GROUP_SIZE, D]
    out = torch.empty_like(q)
    q = q.view(N, HKV, H_g, D).permute(1, 0, 2, 3)
    out = out.view(N, HKV, H_g, D).permute(1, 0, 2, 3)

    # K_app/V_app: [N, HKV, D] -> [HKV, N, D]
    k_app = k.view(N, HKV, D).permute(1, 0, 2)
    v_app = v.view(N, HKV, D).permute(1, 0, 2)

    cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=q.device)
    seq_lens_bh = seq_lens_bh.to(dtype=torch.int32, device=q.device)
    batch_mapping = batch_mapping.to(dtype=torch.int16, device=q.device)

    N_LOGICAL_PAGES_MAX = global_page_table.shape[-1]
    CACHE_SIZE = k_cache.shape[0]
    assert v_cache.shape[0] == CACHE_SIZE
    assert k_cache.shape[1] == D and v_cache.shape[1] == D
    assert PAGE_SIZE > 0 and CACHE_SIZE % PAGE_SIZE == 0

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # strides for Q [G, N, QUERY_GROUP_SIZE, D]
    STRIDE_Q_G, STRIDE_Q_N, STRIDE_Q_H, STRIDE_Q_D = q.stride()
    STRIDE_KC, STRIDE_VC = k_cache.stride(0), v_cache.stride(0)
    # [G, N, D]
    STRIDE_KA_G, STRIDE_KA_N, STRIDE_KA_D = k_app.stride()
    STRIDE_VA_G, STRIDE_VA_N, STRIDE_VA_D = v_app.stride()

    # OUT [G, N, QUERY_GROUP_SIZE, D]
    STRIDE_OUT_G, STRIDE_OUT_N, STRIDE_OUT_H, STRIDE_OUT_D = out.stride()
    # launch grid
    triton.set_allocator(
        lambda size, align, _: torch.empty(size, dtype=torch.int8, device=q.device)
    )
    assert STRIDE_KA_D == STRIDE_VA_D == STRIDE_Q_D == STRIDE_OUT_D == 1, (
        "final dimension must be contiguous"
    )

    def grid(META):
        return HKV, B, triton.cdiv(max_seqlen_q, META["BLOCK_M"])

    AUTOTUNE_MAX_Q_LEN = triton.next_power_of_2(max_seqlen_q)
    AUTOTUNE_MAX_K_LEN = triton.next_power_of_2(max_seqlen_k_cache)
    _causal_head_sparse_varlen_with_cache[grid](
        Q=q,
        K_cache=k_cache,
        V_cache=v_cache,
        K_app=k_app,
        V_app=v_app,
        cu_seqlens_qk=cu_seqlens_q,
        seq_lens_bh=seq_lens_bh.to(torch.int16),
        page_table=global_page_table,
        batch_mapping=batch_mapping,
        OUT=out,
        HKV=HKV,
        QUERY_GROUP_SIZE=H_g,
        PAGE_SIZE=PAGE_SIZE,
        N_LOGICAL_PAGES_MAX=N_LOGICAL_PAGES_MAX,
        STRIDE_Q_G=STRIDE_Q_G,
        STRIDE_Q_N=STRIDE_Q_N,
        STRIDE_Q_H=STRIDE_Q_H,
        STRIDE_KC=STRIDE_KC,
        STRIDE_VC=STRIDE_VC,
        STRIDE_KA_G=STRIDE_KA_G,
        STRIDE_KA_N=STRIDE_KA_N,
        STRIDE_VA_G=STRIDE_VA_G,
        STRIDE_VA_N=STRIDE_VA_N,
        STRIDE_OUT_G=STRIDE_OUT_G,
        STRIDE_OUT_N=STRIDE_OUT_N,
        STRIDE_OUT_H=STRIDE_OUT_H,
        sm_scale=sm_scale,
        D=D,
        AUTOTUNE_MAX_Q_LEN=AUTOTUNE_MAX_Q_LEN,
        AUTOTUNE_MAX_K_LEN=AUTOTUNE_MAX_K_LEN,
    )
    return out.permute(1, 0, 2, 3).view(N, HQ, D)  # already contiguous


autotune_configs_cc9 = [
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_M": 64, "WARPSPEC": True}, num_warps=16, num_stages=3
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_M": 64, "WARPSPEC": True}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_M": 32, "WARPSPEC": True}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_M": 32, "WARPSPEC": True}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_M": 32, "WARPSPEC": False}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_M": 16, "WARPSPEC": True}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_M": 16, "WARPSPEC": True}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_M": 16, "WARPSPEC": False}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_N": 32, "BLOCK_M": 32, "WARPSPEC": True}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_N": 32, "BLOCK_M": 32, "WARPSPEC": False}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_N": 32, "BLOCK_M": 16, "WARPSPEC": False}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_N": 32, "BLOCK_M": 16, "WARPSPEC": False}, num_warps=4, num_stages=4
    ),
]

autotune_configs_cc8 = [
    triton.Config(
        {"BLOCK_N": BN, "BLOCK_M": BM, "WARPSPEC": True}, num_warps=w, num_stages=s
    )
    for BN in [16, 32]
    for BM in [64]
    for w in [4, 8]
    for s in [2, 3]
]


def prune_invalid_configs(configs, _, **kwargs):
    return [
        conf
        for conf in configs
        if not (conf.kwargs.get("BLOCK_N") == 32 and conf.kwargs.get("num_stages") == 4)
    ]


def get_autotune_configs():
    if tl.target_info.cuda_capability_geq(9.0):
        return autotune_configs_cc9
    else:
        return autotune_configs_cc8


@triton.autotune(
    configs=get_autotune_configs(),
    key=[
        "HKV",
        "QUERY_GROUP_SIZE",
        "D",
        "PAGE_SIZE",
        "AUTOTUNE_MAX_K_LEN",
        "AUTOTUNE_MAX_Q_LEN",
    ],
    cache_results=True,
)
@triton.jit
def _causal_head_sparse_varlen_with_cache(
    Q,  # [HKV, N, QUERY_GROUP_SIZE, D] (non-contiguous)
    K_cache,
    V_cache,  # [CACHE_SIZE, D]
    K_app,
    V_app,  # [HKV, N, D]
    cu_seqlens_qk,  # [B+1]
    seq_lens_bh,  # [B, HKV]
    page_table,  # [B_total, HKV, N_LOGICAL_PAGES_MAX]
    batch_mapping,  # [B], maps local b -> global batch index
    OUT,  # [HKV, N, QUERY_GROUP_SIZE, D]
    #
    HKV: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    N_LOGICAL_PAGES_MAX,
    STRIDE_Q_G,
    STRIDE_Q_N,
    STRIDE_Q_H,
    STRIDE_KC,
    STRIDE_VC,
    STRIDE_KA_G,
    STRIDE_KA_N,
    STRIDE_VA_G,
    STRIDE_VA_N,
    STRIDE_OUT_G,
    STRIDE_OUT_N,
    STRIDE_OUT_H,
    sm_scale,
    #
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WARPSPEC: tl.constexpr,
    AUTOTUNE_MAX_Q_LEN: tl.constexpr,  # used for autotune key
    AUTOTUNE_MAX_K_LEN: tl.constexpr,  # used for autotune key
):
    TOTAL_N_QUERIES: tl.constexpr = BLOCK_M * QUERY_GROUP_SIZE
    pid_g = tl.program_id(0)  # kv_head id in [0, HKV)
    pid_b = tl.program_id(1)  # batch id
    pid_m = tl.program_id(2)  # query-tile id within batch

    # batch segment [qb, qe) in N
    off_b = tl.load(cu_seqlens_qk + pid_b)
    off_b1 = tl.load(cu_seqlens_qk + pid_b + 1)
    seq_len_append = off_b1 - off_b

    q_start = off_b + pid_m * BLOCK_M
    q_end = tl.minimum(q_start + BLOCK_M, off_b1)
    # number of queries in this tile for this batch
    M = q_end - q_start
    if M <= 0:
        return

    # cached length for (b, kv_head=pid_g)
    L_cache = tl.load(seq_lens_bh + pid_b * HKV + pid_g)
    # row indices flattened over [QUERY_GROUP_SIZE, M]
    offs_row = tl.arange(0, TOTAL_N_QUERIES)
    row_m = offs_row % BLOCK_M
    row_h = offs_row // BLOCK_M
    # valid rows: only those with row_m < M
    row_mask = row_m < M

    # global query index per row
    q_idx = q_start + row_m
    offs_d = tl.arange(0, D)
    # Q tile: [TOTAL_N_QUERIES, D]
    # Q layout: [HKV, N, QUERY_GROUP_SIZE, D]
    q_ptrs = (
        Q
        + pid_g * STRIDE_Q_G
        + q_idx[:, None] * STRIDE_Q_N
        + row_h[:, None] * STRIDE_Q_H
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    e_max = tl.zeros([TOTAL_N_QUERIES], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([TOTAL_N_QUERIES], dtype=tl.float32)
    acc = tl.zeros([TOTAL_N_QUERIES, D], dtype=tl.float32)

    offs_block_n = tl.arange(0, BLOCK_N)
    qk_scale = sm_scale * 1.44269504

    # 1) attend over cachee K/V
    if L_cache > 0:
        # map local (b) to global batch index
        mapped_b = tl.load(batch_mapping + pid_b)
        pt_base = (mapped_b * HKV + pid_g) * N_LOGICAL_PAGES_MAX
        # iterate logical pages
        num_lp = tl.cdiv(L_cache, PAGE_SIZE)
        for lp in tl.range(0, num_lp):
            # can overflow in 32 bits so upcast
            phys = tl.load(page_table + pt_base + lp).to(tl.int64)
            page_start = phys * PAGE_SIZE
            # how many valid tokens in this page for this (b,g)
            remain = L_cache - lp * PAGE_SIZE
            page_len = tl.minimum(PAGE_SIZE, remain)
            # iterate over this page in BLOCK_N chunks
            for ks in tl.range(0, page_len, BLOCK_N, warp_specialize=WARPSPEC):
                offs_n = ks + offs_block_n
                mask_n = offs_n < page_len

                key_idx = page_start + offs_n
                k_ptrs = K_cache + key_idx[:, None] * STRIDE_KC + offs_d[None, :]

                k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)  # [BN, D]
                qk = tl.dot(q, k.T) * qk_scale  # [TOTAL_N_QUERIES, BN]
                qk = tl.where(row_mask[:, None] & mask_n[None, :], qk, -1.0e6)

                # softmax update
                cur_max = tl.max(qk, 1)
                n_e_max = tl.maximum(e_max, cur_max)
                re_scale = tl.math.exp2(e_max - n_e_max)
                p = tl.math.exp2(qk - n_e_max[:, None])

                v_ptrs = V_cache + key_idx[:, None] * STRIDE_VC + offs_d[None, :]
                v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # [BN, D]

                acc = acc * re_scale[:, None]
                acc = tl.dot(p.to(v.dtype), v, acc)

                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

    # 2) attend over appended K_app/V_app (causal)
    # appended tokens for batch b are in [off_b, off_b1)
    # query tile is [q_start, q_end)
    # for each query at index q_idx, valid appended keys k satisfy off_b <= k <= q_idx
    if q_end > off_b:
        # exactly one appended token
        if seq_len_append == 1:
            ka_ptrs = K_app + pid_g * STRIDE_KA_G + off_b * STRIDE_KA_N + offs_d
            k = tl.load(ka_ptrs)  # [D]
            qk = tl.sum(q * k[None, :], 1) * qk_scale
            qk = tl.where(row_mask, qk, -1.0e6)
            n_e_max = tl.maximum(e_max, qk)
            re_scale = tl.math.exp2(e_max - n_e_max)
            p = tl.math.exp2(qk - n_e_max)
            va_ptrs = V_app + pid_g * STRIDE_VA_G + off_b * STRIDE_VA_N + offs_d
            v = tl.load(va_ptrs)  # [D]
            acc = acc * re_scale[:, None] + p[:, None] * v[None, :]
            e_sum = e_sum * re_scale + p
        else:
            # off-band: k in [off_b, q_start)
            # for all queries t in [q_start, q_end), any k < q_start satisfies k <= t.
            # so no causal mask needed.
            off_band_start = off_b
            off_band_end = q_start

            if off_band_end > off_band_start:
                for ks in tl.range(
                    off_band_start, off_band_end, BLOCK_N, warp_specialize=WARPSPEC
                ):
                    offs_n = ks + offs_block_n
                    mask_n = offs_n < off_band_end

                    ka_ptrs = (
                        K_app
                        + pid_g * STRIDE_KA_G
                        + offs_n[:, None] * STRIDE_KA_N
                        + offs_d[None, :]
                    )
                    k = tl.load(ka_ptrs, mask=mask_n[:, None], other=0.0)

                    qk = tl.dot(q, k.T) * qk_scale
                    qk = tl.where(row_mask[:, None] & mask_n[None, :], qk, -1.0e6)

                    cur_max = tl.max(qk, 1)
                    n_e_max = tl.maximum(e_max, cur_max)

                    re_scale = tl.math.exp2(e_max - n_e_max)
                    p = tl.math.exp2(qk - n_e_max[:, None])

                    va_ptrs = (
                        V_app
                        + pid_g * STRIDE_VA_G
                        + offs_n[:, None] * STRIDE_VA_N
                        + offs_d[None, :]
                    )
                    v = tl.load(va_ptrs, mask=mask_n[:, None], other=0.0)

                    acc = acc * re_scale[:, None]
                    acc = tl.dot(p.to(v.dtype), v, acc)

                    e_sum = e_sum * re_scale + tl.sum(p, 1)
                    e_max = n_e_max

            # on-band remaining k
            on_band_start = tl.maximum(q_start, off_b)
            if on_band_start < q_end:
                for ks in tl.range(
                    on_band_start, q_end, BLOCK_N, warp_specialize=WARPSPEC
                ):
                    offs_n = ks + tl.arange(0, BLOCK_N)
                    mask_n = offs_n < q_end

                    ka_ptrs = (
                        K_app
                        + pid_g * STRIDE_KA_G
                        + offs_n[:, None] * STRIDE_KA_N
                        + offs_d[None, :]
                    )

                    k = tl.load(ka_ptrs, mask=mask_n[:, None], other=0.0)

                    qk = tl.dot(q, k.T) * qk_scale

                    caus_mask = offs_n[None, :] <= q_idx[:, None]
                    full_mask = row_mask[:, None] & mask_n[None, :] & caus_mask

                    qk = tl.where(full_mask, qk, -1.0e6)

                    cur_max = tl.max(qk, 1)
                    n_e_max = tl.maximum(e_max, cur_max)
                    re_scale = tl.math.exp2(e_max - n_e_max)
                    p = tl.math.exp2(qk - n_e_max[:, None])

                    va_ptrs = (
                        V_app
                        + pid_g * STRIDE_VA_G
                        + offs_n[:, None] * STRIDE_VA_N
                        + offs_d[None, :]
                    )
                    v = tl.load(va_ptrs, mask=mask_n[:, None], other=0.0)

                    acc = acc * re_scale[:, None]
                    acc = tl.dot(p.to(v.dtype), v, acc)

                    e_sum = e_sum * re_scale + tl.sum(p, 1)
                    e_max = n_e_max

    # 3) write outputs
    o = (acc / e_sum[:, None]).to(q.dtype)
    out_ptrs = (
        OUT
        + pid_g * STRIDE_OUT_G
        + q_idx[:, None] * STRIDE_OUT_N
        + row_h[:, None] * STRIDE_OUT_H
        + offs_d[None, :]
    )
    tl.store(out_ptrs, o, mask=row_mask[:, None])
