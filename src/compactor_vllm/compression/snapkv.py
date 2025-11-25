import math
from typing import Optional

import torch
import triton
from triton import language as tl

from compactor_vllm.compression.common import BaseCompressionMethod
from compactor_vllm.utils.helpers import maybe_execute_in_stream


class SnapKVCompression(BaseCompressionMethod):
    @staticmethod
    def pre_rope_scoring(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context
    ) -> Optional[torch.Tensor]:
        return None

    @staticmethod
    def post_rope_scoring(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pre_rope_scores: torch.Tensor,
        context,
    ) -> Optional[torch.Tensor]:
        scores = maybe_execute_in_stream(
            query_aware_key_scores,
            q,
            k,
            context.cu_seqlens_q,
            context.cu_seqlens_k,
            w=32,
            STORE_STREAM=context.STORE_STREAM,
        )
        return scores


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_Q": bq, "BLOCK_K": bk}, num_warps=num_warps, num_stages=num_stages
        )
        for bq in [32, 64]
        for bk in [32, 64]
        for num_warps in [4, 8]
        for num_stages in [3, 4]
    ],
    key=["QUERY_GROUP_SIZE", "D", "ROWS_MAX"],
    cache_results=True,
)
@triton.jit
def _lse_and_store_logits_kernel(
    Q,
    K,
    cu_q,
    cu_k,
    w_b,  # int32 pointers
    out_m,
    out_S,  # [B, Hk, ROWS_MAX] float32
    LOGITS,  # [Nk, Hk, ROWS_MAX] float32
    sm_scale,  # float
    QUERY_GROUP_SIZE: tl.constexpr,
    D: tl.constexpr,
    STRIDE_Q_NQ,
    STRIDE_Q_HQ,
    STRIDE_K_NK,
    STRIDE_K_HK,
    STRIDE_M_B,
    STRIDE_M_H,
    STRIDE_M_R,
    STRIDE_S_B,
    STRIDE_S_H,
    STRIDE_S_R,
    STRIDE_LG_NK,
    STRIDE_LG_HK,
    STRIDE_LG_R,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS_MAX,
):
    # program ids
    b = tl.program_id(0)
    hk = tl.program_id(1)
    rid = tl.program_id(2)  # row-tile id
    # batch segment bounds
    q_end = tl.load(cu_q + b + 1)
    k_beg = tl.load(cu_k + b)
    k_end = tl.load(cu_k + b + 1)
    win = tl.load(w_b + b)

    q_win_beg = q_end - win
    k_eff_end = k_end - win
    if (win <= 0) or (k_eff_end <= k_beg):
        return

    # rows for this (b,hk)
    rows_b = win * QUERY_GROUP_SIZE
    row0 = rid * BLOCK_Q
    if row0 >= rows_b:
        return

    # exp(x) = exp2(x * 1/ln2)
    qk_scale = sm_scale * 1.4426950408889634

    offs_qrow = row0 + tl.arange(0, BLOCK_Q)
    row_mask = offs_qrow < rows_b

    # map row -> (q_idx, hq_local)
    hq_local = offs_qrow % QUERY_GROUP_SIZE
    q_off = offs_qrow // QUERY_GROUP_SIZE
    q_idx = q_win_beg + q_off
    hq_glob = hk * QUERY_GROUP_SIZE + hq_local

    offs_d = tl.arange(0, D)

    q_ptrs = (
        Q
        + q_idx[:, None] * STRIDE_Q_NQ
        + hq_glob[:, None] * STRIDE_Q_HQ
        + offs_d[None, :]
    )
    q_rows = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)
    m = tl.zeros([BLOCK_Q], dtype=tl.float32) + (-float("inf"))
    S = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for ks in tl.range(k_beg, k_eff_end, BLOCK_K):
        nk = ks + tl.arange(0, BLOCK_K)
        kmask = nk < k_eff_end

        k_ptrs = K + nk[:, None] * STRIDE_K_NK + hk * STRIDE_K_HK + offs_d[None, :]
        k_blk = tl.load(k_ptrs, mask=kmask[:, None], other=0.0)  # [BK, D]

        s = tl.dot(q_rows, k_blk.T) * qk_scale  # [BQ, BK]
        s = tl.where(kmask[None, :], s, -float("inf"))

        # store into LOGITS[nk, hk, row]  -> [BK, BQ]
        log_ptrs = (
            LOGITS
            + nk[:, None] * STRIDE_LG_NK
            + hk * STRIDE_LG_HK
            + (row0 + tl.arange(0, BLOCK_Q))[None, :] * STRIDE_LG_R
        )
        tl.store(log_ptrs, s.T, mask=kmask[:, None] & row_mask[None, :])

        # log2 streaming LSE update
        cur_max = tl.max(s, 1)  # [BQ]
        n_m = tl.maximum(m, cur_max)
        rescale = tl.math.exp2(m - n_m)
        S = S * rescale + tl.sum(tl.math.exp2(s - n_m[:, None]), 1)
        m = n_m

    # store m,S for these rows
    m_base = out_m + b * STRIDE_M_B + hk * STRIDE_M_H + row0 * STRIDE_M_R
    S_base = out_S + b * STRIDE_S_B + hk * STRIDE_S_H + row0 * STRIDE_S_R
    tl.store(m_base + tl.arange(0, BLOCK_Q) * STRIDE_M_R, m, mask=row_mask)
    tl.store(S_base + tl.arange(0, BLOCK_Q) * STRIDE_S_R, S, mask=row_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": bq, "BLOCK_K": bk})
        for bq in [16, 32, 64]
        for bk in [32, 64, 128]
    ],
    key=["HK", "HQ"],
    cache_results=True,
)
@triton.jit
def _scores_from_logits_kernel(
    cu_k,
    w_b,
    in_m,
    in_S,  # [B, Hk, ROWS_MAX] f32
    LOGITS,  # [Nk, Hk, ROWS_MAX] f32, base-2 logits
    OUT,  # [Nk, Hk] f32
    #
    QUERY_GROUP_SIZE: tl.constexpr,
    STRIDE_M_B,
    STRIDE_M_H,
    STRIDE_M_R,
    STRIDE_S_B,
    STRIDE_S_H,
    STRIDE_S_R,
    STRIDE_LG_NK,
    STRIDE_LG_HK,
    STRIDE_LG_R,
    STRIDE_OUT_NK,
    STRIDE_OUT_HK,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    #
    DO_POOL: tl.constexpr,  # set True to enable in-place avg pool
    KPOOL: tl.constexpr,  # kernel size for avg pool (stride=1)
):
    b = tl.program_id(0)
    hk = tl.program_id(1)

    k_beg = tl.load(cu_k + b)
    k_end = tl.load(cu_k + b + 1)
    win = tl.load(w_b + b)

    k_eff_end = k_end - win
    if (win <= 0) or (k_eff_end <= k_beg):
        return

    rows_b = win * QUERY_GROUP_SIZE

    # === scores over computed region ===
    for ks in tl.range(k_beg, k_eff_end, BLOCK_K):
        nk = ks + tl.arange(0, BLOCK_K)
        kmask = nk < k_eff_end

        scores = tl.zeros([BLOCK_K], dtype=tl.float32)

        for row0 in tl.range(0, rows_b, BLOCK_Q):
            r_idx = row0 + tl.arange(0, BLOCK_Q)
            rmask = r_idx < rows_b

            # load m, S for rows
            m_ptr = in_m + b * STRIDE_M_B + hk * STRIDE_M_H + row0 * STRIDE_M_R
            S_ptr = in_S + b * STRIDE_S_B + hk * STRIDE_S_H + row0 * STRIDE_S_R
            m = tl.load(
                m_ptr + tl.arange(0, BLOCK_Q) * STRIDE_M_R,
                mask=rmask,
                other=-float("inf"),
            )
            S = tl.load(
                S_ptr + tl.arange(0, BLOCK_Q) * STRIDE_S_R, mask=rmask, other=0.0
            )

            valid_row = S > 0
            m = tl.where(valid_row, m, 0.0)
            S = tl.where(valid_row, S, 1.0)

            # load stored logits^T: [BK, BQ]
            log_ptrs = (
                LOGITS
                + nk[:, None] * STRIDE_LG_NK
                + hk * STRIDE_LG_HK
                + (row0 + tl.arange(0, BLOCK_Q))[None, :] * STRIDE_LG_R
            )
            s_T = tl.load(
                log_ptrs, mask=kmask[:, None] & rmask[None, :], other=-float("inf")
            )  # [BK, BQ]

            # probs^T = exp2(s_T - m) / S, sum over rows
            probs_T = tl.math.exp2(s_T - m[None, :]) / S[None, :]
            probs_T = tl.where(valid_row[None, :], probs_T, 0.0)

            scores += tl.sum(probs_T, 1)  # [BK]

        if DO_POOL and (KPOOL > 1):
            i = tl.arange(0, BLOCK_K)[:, None]
            j = tl.arange(0, BLOCK_K)[None, :]
            band = (j <= i) & ((i - j) < KPOOL)
            band = band & kmask[None, :]
            # sum within band
            sums = tl.sum(tl.where(band, scores[None, :], 0.0), 1)  # [BK]
            denom = tl.sum(band, 1).to(tl.float32)  # [BK]
            denom = tl.where(denom > 0, denom, 1.0)
            scores = sums / denom

        out_ptrs = OUT + nk * STRIDE_OUT_NK + hk * STRIDE_OUT_HK
        tl.store(out_ptrs, scores, mask=kmask)

    pad_beg = k_eff_end
    pad_end = k_end
    if pad_end > pad_beg:
        for ks in tl.range(pad_beg, pad_end, BLOCK_K):
            nk = ks + tl.arange(0, BLOCK_K)
            kmask = nk < pad_end
            out_ptrs = OUT + nk * STRIDE_OUT_NK + hk * STRIDE_OUT_HK
            tl.store(
                out_ptrs, tl.full([BLOCK_K], float("inf"), dtype=tl.float32), mask=kmask
            )


@triton.autotune(
    configs=[triton.Config({"BLOCK_K": bk}) for bk in [32, 64, 128]],
    key=["HK"],
    cache_results=True,
)
@triton.jit
def _zscore_per_batch_epilogue(
    OUT,  # [Nk, Hk], float32
    cu_k,
    w_b,  # [B+1], [B] int32
    STRIDE_OUT_NK,
    STRIDE_OUT_HK,
    HK: tl.constexpr,  # Hk
    EPS: tl.constexpr,  # e.g., 1e-12
    BLOCK_K: tl.constexpr,  # e.g., 128
):
    b = tl.program_id(0)

    k_beg = tl.load(cu_k + b)
    k_end = tl.load(cu_k + b + 1)
    win = tl.load(w_b + b)

    k_eff_end = k_end - win
    if k_eff_end <= k_beg:
        return

    sumv = tl.zeros([], dtype=tl.float32)
    sumsq = tl.zeros([], dtype=tl.float32)
    count = ((k_eff_end - k_beg) * HK).to(tl.float32)

    for ks in tl.range(k_beg, k_eff_end, BLOCK_K):
        nk = ks + tl.arange(0, BLOCK_K)
        kmask = nk < k_eff_end
        for h in tl.range(0, HK):
            ptrs = OUT + nk * STRIDE_OUT_NK + h * STRIDE_OUT_HK
            vals = tl.load(ptrs, mask=kmask, other=0.0).to(tl.float32)
            sumv += tl.sum(vals, 0)
            sumsq += tl.sum(vals * vals, 0)

    mean = sumv / count
    var = tl.maximum(sumsq / count - mean * mean, 0.0)
    invstd = 1.0 / tl.sqrt(var + EPS)

    for ks in tl.range(k_beg, k_eff_end, BLOCK_K):
        nk = ks + tl.arange(0, BLOCK_K)
        kmask = nk < k_eff_end
        for h in tl.range(0, HK):
            ptrs = OUT + nk * STRIDE_OUT_NK + h * STRIDE_OUT_HK
            vals = tl.load(ptrs, mask=kmask, other=0.0).to(tl.float32)
            vals = (vals - mean) * invstd
            tl.store(ptrs, vals, mask=kmask)


def query_aware_key_scores(
    q: torch.Tensor,  # [N_q, Hq, D]
    k: torch.Tensor,  # [N_k, Hk, D]
    cu_seqlens_q: torch.Tensor,  # [B+1], int32
    cu_seqlens_k: torch.Tensor,  # [B+1], int32
    w: torch.Tensor | int,  # [B], int32
    sm_scale: float = None,  # defaults to 1/sqrt(D)
    *,
    accum_scores: torch.Tensor = None,
    accum_blending: float = None,
    normalize: bool = False,
) -> Optional[torch.Tensor]:
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "last dim must be contiguous"
    device = q.device
    N_q, Hq, D = q.shape
    N_k, Hk, Dk = k.shape
    assert (Hq % Hk) == 0, "Hq must be a multiple of Hk"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    B = cu_seqlens_q.numel() - 1
    assert B == cu_seqlens_k.numel() - 1

    G = Hq // Hk
    if type(w) is int:
        max_w = w
        w = torch.full((B,), fill_value=w, device=device, dtype=torch.int32)
    else:
        max_w = int(w.max().item())
        assert w.numel() == B
    ROWS_MAX = max_w * G
    if ROWS_MAX == 0:
        return torch.zeros((N_k, Hk), dtype=torch.float32, device=device)

    out = torch.empty((N_k, Hk), dtype=torch.float32, device=device)
    m_scratch = torch.empty((B, Hk, ROWS_MAX), dtype=torch.float32, device=device)
    S_scratch = torch.empty((B, Hk, ROWS_MAX), dtype=torch.float32, device=device)
    logits_buf = torch.empty((N_k, Hk, ROWS_MAX), dtype=torch.float32, device=device)

    # strides
    STRIDE_Q_NQ, STRIDE_Q_HQ, _ = q.stride()
    STRIDE_K_NK, STRIDE_K_HK, _ = k.stride()
    STRIDE_M_B, STRIDE_M_H, STRIDE_M_R = m_scratch.stride()
    STRIDE_S_B, STRIDE_S_H, STRIDE_S_R = S_scratch.stride()
    STRIDE_LG_NK, STRIDE_LG_HK, STRIDE_LG_R = logits_buf.stride()
    STRIDE_OUT_NK, STRIDE_OUT_HK = out.stride()

    def grid(META):
        return B, Hk, triton.cdiv(ROWS_MAX, META["BLOCK_Q"])

    _lse_and_store_logits_kernel[grid](
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        w,
        m_scratch,
        S_scratch,
        logits_buf,
        sm_scale,
        QUERY_GROUP_SIZE=Hq // Hk,
        D=D,
        STRIDE_Q_NQ=STRIDE_Q_NQ,
        STRIDE_Q_HQ=STRIDE_Q_HQ,
        STRIDE_K_NK=STRIDE_K_NK,
        STRIDE_K_HK=STRIDE_K_HK,
        STRIDE_M_B=STRIDE_M_B,
        STRIDE_M_H=STRIDE_M_H,
        STRIDE_M_R=STRIDE_M_R,
        STRIDE_S_B=STRIDE_S_B,
        STRIDE_S_H=STRIDE_S_H,
        STRIDE_S_R=STRIDE_S_R,
        STRIDE_LG_NK=STRIDE_LG_NK,
        STRIDE_LG_HK=STRIDE_LG_HK,
        STRIDE_LG_R=STRIDE_LG_R,
        ROWS_MAX=ROWS_MAX,
    )

    _scores_from_logits_kernel[(B, Hk)](
        cu_seqlens_k,
        w,
        m_scratch,
        S_scratch,
        logits_buf,
        out,
        QUERY_GROUP_SIZE=Hq // Hk,
        STRIDE_M_B=STRIDE_M_B,
        STRIDE_M_H=STRIDE_M_H,
        STRIDE_M_R=STRIDE_M_R,
        STRIDE_S_B=STRIDE_S_B,
        STRIDE_S_H=STRIDE_S_H,
        STRIDE_S_R=STRIDE_S_R,
        STRIDE_LG_NK=STRIDE_LG_NK,
        STRIDE_LG_HK=STRIDE_LG_HK,
        STRIDE_LG_R=STRIDE_LG_R,
        STRIDE_OUT_NK=STRIDE_OUT_NK,
        STRIDE_OUT_HK=STRIDE_OUT_HK,
        DO_POOL=True,
        KPOOL=5,
    )
    if normalize:
        _zscore_per_batch_epilogue[(B,)](
            out,
            cu_seqlens_k,
            w,
            STRIDE_OUT_NK,
            STRIDE_OUT_HK,
            HK=Hk,
            EPS=1e-12,
        )
    if accum_scores is not None:
        if accum_blending is not None:
            accum_scores.mul_(accum_blending)
        accum_scores.add_(out)
        return accum_scores
    else:
        return out
