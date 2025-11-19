import logging
import math
from typing import List, Optional

import torch
import triton
from tqdm.contrib.logging import logging_redirect_tqdm
from triton import language as tl

from compactor_vllm.compression.common import BaseCompressionMethod
from compactor_vllm.utils.helpers import maybe_execute_in_stream

logger = logging.getLogger(__name__)


class CompactorCompression(BaseCompressionMethod):
    chunk_size: int = 128

    @staticmethod
    def pre_rope_scoring(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context
    ) -> Optional[torch.Tensor]:
        compression_context = context.compression_context
        scores = maybe_execute_in_stream(
            approximate_leverage_scores,
            k,
            compression_context.context_lens,
            compression_context.PHI,
            normalize=True,
            chunk_size=compression_context.compression_chunk_size,
            STORE_STREAM=context.STORE_STREAM,
        )
        return scores

    @staticmethod
    def post_rope_scoring(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pre_rope_scores: torch.Tensor,
        context,
    ) -> Optional[torch.Tensor]:
        compression_context = context.compression_context
        return maybe_execute_in_stream(
            non_causal_attn_scores,
            q,
            k,
            v,
            context.cu_seqlens_q,
            context.max_seqlen_q,
            chunk_size=CompactorCompression.chunk_size,
            sm_scale=1.0,
            normalize=True,
            accum_scores=pre_rope_scores,
            context_lens=compression_context.context_lens,
            protected_first_tokens=compression_context.protected_first_tokens,
            protected_last_tokens=compression_context.protected_last_tokens,
            accum_blending=0.5,
        )


def split_into_chunks(xs, chunk_size):
    """
    Convert a list of sequence lengths into a sequence of coalesced chunk lengths.

    Given an iterable of per-sequence context lengths ``xs`` and a target ``chunk_size``,
    this helper produces two parallel lists:

      * ``coalesced_chunks`` – lengths of contiguous segments in the
        **concatenated** sequence space, where each segment corresponds either
        to a full chunk of size ``chunk_size`` or to a residual "epilogue"
        tail shorter than ``chunk_size``.

      * ``chunks`` – the actual chunk sizes used within each original sequence.
        For a length ``n``, we produce ``n // chunk_size`` entries of
        ``chunk_size`` (the "prologue") and at most one final entry equal to
        ``n % chunk_size`` (the "epilogue").

    ``chunks`` reflects how each input length is decomposed into
    fixed-size (plus optional tail) processing blocks, while
    ``coalesced_chunks`` describes those same blocks after concatenating consecutive
    chunks of size ``chunk_size``. together

    Example:
        xs = [257, 127], chunk_size = 128
        coalesced_chunks = [256, 1, 127]
        chunks           = [128, 128, 1, 127]

    Args:
        :param xs:
            Iterable of non-negative integers
        :param chunk_size:
            Target chunk size

    Returns:
        :return Tuple[List[int], List[int]]:
            ``(coalesced_chunks, chunks)`` as described above.
    """
    coalesced_chunks, chunks = [], []
    for n in xs:
        nchunks = n // chunk_size
        prologue = nchunks * chunk_size
        epilogue = n - prologue
        if prologue > 0:
            coalesced_chunks.append(prologue)
            chunks.extend([chunk_size] * nchunks)
        if epilogue > 0:
            coalesced_chunks.append(epilogue)
            chunks.append(epilogue)
    return coalesced_chunks, chunks


def approximate_leverage_scores(
    key_states: torch.Tensor,  # [N, H, D]
    context_lens: List[int],  # [B]
    PHI: torch.Tensor,  # [D, k]
    regularizer: float = 5e-3,
    normalize: bool = False,
    chunk_size: int = 512,
) -> torch.Tensor:  # returns [N, H]
    """
    Approximate leverage scores for keys via randomized sketching.

    This implements a randomized approximation to per-token leverage scores for
    the key matrix, as described in Compactor: Calibrated Query-Agnostic KV Cache
    Compression with Approximate Leverage Scores (https://arxiv.org/abs/2507.08143).
    Args:
        :param key_states:
            Tensor of shape ``[N, H, D]`` containing pre-RoPE key states for
            all tokens across the batch, packed along the sequence dimension.
            ``N = sum(context_lens)``.
        :param context_lens:
            List of per-sequence context lengths, length ``B``.
        :param PHI:
            Random projection matrix of shape ``[D, k]`` used to sketch the
            keys into a lower-dimensional subspace (k < D).
        :param regularizer:
            Small positive scalar added to the diagonal of each Gram matrix
            before SVD to improve numerical stability. Defaults to ``1e-2``.
        :param normalize:
            If True, apply per-sequence z-score normalization to the scores
            across all heads and tokens in a batch.
        :param chunk_size:
            Target chunk size along the sequence dimension. If > 0, the
            concatenated sequence is split into chunks of at most this size
            before forming Gram matrices and SVD. If ≤ 0, the entire sequence
            for each context is treated as a single chunk.
    Returns:
        :return torch.Tensor:
            Approximate leverage scores of shape ``[N, H]``, where each row
            corresponds to a token and each column to a head.
    """
    if chunk_size > 0:
        coalesced_chunk_lens, chunks_lens = split_into_chunks(context_lens, chunk_size)
    else:
        coalesced_chunk_lens, chunks_lens = context_lens, context_lens
    chunk_lens_cuda = torch.tensor([0] + chunks_lens).cuda(non_blocking=True)
    X = torch.matmul(key_states.transpose(0, 1), PHI)
    H, N, k = X.shape
    chunks = torch.split(X, coalesced_chunk_lens, dim=-2)
    gram_matrices = []
    for i, L in enumerate(coalesced_chunk_lens):
        chunk = chunks[i]
        if chunk_size <= 0 or L % chunk_size != 0:
            chunk.sub_(chunk.mean(dim=-2, keepdim=True))
            g = torch.matmul(chunk.transpose(-1, -2), chunk)  # [H, k, k]
            g = g.unsqueeze(1)
        else:
            chunk = chunk.view(H, -1, chunk_size, k)  # [H, num_chunks, chunk_size, k]
            chunk.sub_(chunk.mean(dim=-2, keepdim=True))
            g = torch.matmul(chunk.transpose(-1, -2), chunk)  # [H, num_chunks, k, k]
        gram_matrices.append(g)
    G = torch.cat(gram_matrices, dim=1).to(torch.float32)
    diag = G.diagonal(dim1=-2, dim2=-1)
    diag.add_(regularizer)
    try:
        V, S, Vt = torch.linalg.svd(G, full_matrices=False, driver="gesvda")
    except RuntimeError:
        try:
            diag = G.diagonal(dim1=-2, dim2=-1)
            diag.add_(regularizer * 10)
            V, S, Vt = torch.linalg.svd(G, full_matrices=False, driver="gesvda")
        except RuntimeError:
            with logging_redirect_tqdm():
                logger.warning(
                    "GESVDA failed, falling back to QR decomposition, which will be MUCH slower. "
                    "Try increasing chunk_size if this issue persists."
                )
            # this is over 50 times slower than using GESVDA
            return _approximate_leverage_scores_qr_fallback(
                X=X,
                chunks_lens=chunks_lens,
                chunk_lens_cuda=chunk_lens_cuda,
                normalize=normalize,
                chunk_size=chunk_size,
            )
    SV = (V * S.rsqrt().unsqueeze(-2)).to(X.dtype)
    start = 0
    all_scores = []
    for i, L in enumerate(coalesced_chunk_lens):
        chunk = chunks[i]
        if chunk_size <= 0 or L % chunk_size != 0:
            num_chunks = 1
            sv = SV[:, start]
        else:
            num_chunks = L // chunk_size
            chunk = chunk.view(H, -1, chunk_size, k)  # [H, NC, CS]
            sv = SV[:, start : start + num_chunks]
        U = torch.matmul(chunk, sv)
        scores = (U * U).sum(dim=-1).clamp_min_(0.0).view(H, -1)
        all_scores.append(scores.transpose(-1, -2))
        start += num_chunks

    scores = torch.cat(all_scores, dim=0)
    if normalize:
        grid = (len(chunks_lens),)
        cu_k = chunk_lens_cuda.cumsum(dim=0)
        _zscore_per_batch_epilogue_no_window[grid](
            scores, cu_k, scores.stride(0), scores.stride(1), H
        )
    return scores


@triton.autotune(
    configs=[triton.Config({"BLOCK_K": bk}) for bk in [32, 64, 128]],
    key=["HK"],
    cache_results=True,
)
@triton.jit
def _zscore_per_batch_epilogue_no_window(
    OUT,  # [Nk, Hk], float32
    cu_k,  # [B+1] int32
    STRIDE_OUT_NK,
    STRIDE_OUT_HK,
    HK: tl.constexpr,  # Hk
    BLOCK_K: tl.constexpr,  # e.g., 128
):
    b = tl.program_id(0)

    k_beg = tl.load(cu_k + b)
    k_end = tl.load(cu_k + b + 1)
    if k_end <= k_beg:
        return

    sumv = tl.zeros([], dtype=tl.float32)
    sumsq = tl.zeros([], dtype=tl.float32)
    count = ((k_end - k_beg) * HK).to(tl.float32)

    for ks in tl.range(k_beg, k_end, BLOCK_K):
        nk = ks + tl.arange(0, BLOCK_K)
        kmask = nk < k_end
        for h in tl.range(0, HK):
            ptrs = OUT + nk * STRIDE_OUT_NK + h * STRIDE_OUT_HK
            vals = tl.load(ptrs, mask=kmask, other=0.0).to(tl.float32)
            sumv += tl.sum(vals, 0)
            sumsq += tl.sum(vals * vals, 0)

    mean = sumv / count
    var = tl.maximum(sumsq / count - mean * mean, 0.0)
    invstd = 1.0 / tl.sqrt(var)

    for ks in tl.range(k_beg, k_end, BLOCK_K):
        nk = ks + tl.arange(0, BLOCK_K)
        kmask = nk < k_end
        for h in tl.range(0, HK):
            ptrs = OUT + nk * STRIDE_OUT_NK + h * STRIDE_OUT_HK
            vals = tl.load(ptrs, mask=kmask, other=0.0).to(tl.float32)
            vals = (vals - mean) * invstd
            tl.store(ptrs, vals, mask=kmask)


def _approximate_leverage_scores_qr_fallback(
    X: torch.Tensor,  # [H, N, k], already sketched (KΦ) and centered in-place
    chunks_lens: List[int],  # [num_chunks]
    chunk_lens_cuda: torch.Tensor,  # [num_chunks + 1] (prefix base)
    normalize: bool,
    chunk_size: int,
) -> torch.Tensor:
    H, N, k = X.shape
    device, dtype = X.device, X.dtype
    offsets: List[int] = []
    offset = 0
    for L in chunks_lens:
        offsets.append(offset)
        offset += L
    if offset != N:
        raise RuntimeError(
            f"QR fallback: sum(chunks_lens)={offset} does not match N={N}"
        )

    blocks = torch.split(X, chunks_lens, dim=-2)
    scores = torch.empty(N, H, device=device, dtype=dtype)
    if chunk_size > 0:
        full_indices = [i for i, L in enumerate(chunks_lens) if L == chunk_size]
        epi_indices = [i for i, L in enumerate(chunks_lens) if L != chunk_size]

        if full_indices:
            # stack full chunks
            full_blocks = torch.stack(
                [blocks[i] for i in full_indices], dim=0
            )  # [M, H, CS, k]
            M, Hf, Lf, kf = full_blocks.shape
            assert Lf == chunk_size

            # merge (M, H) into a single batch dim for torch.linalg.q
            full_blocks_2d = full_blocks.view(M * Hf, Lf, kf).to(torch.float32)

            U_full, _ = torch.linalg.qr(full_blocks_2d, mode="reduced")
            U_full = U_full.to(dtype)
            scores_full = (U_full * U_full).sum(dim=-1).clamp_min(0.0)  # [M * Hf, Lf]
            scores_full = scores_full.view(M, Hf, Lf).transpose(-1, -2)  # [M, H, CS]
            for m, chunk_idx in enumerate(full_indices):
                start = offsets[chunk_idx]
                Lc = chunks_lens[chunk_idx]
                scores[start : start + Lc].copy_(scores_full[m])
    else:
        epi_indices = list(range(len(chunks_lens)))

    for chunk_idx in epi_indices:
        block = blocks[chunk_idx]
        _, Lc, _ = block.shape
        if Lc == 0:
            continue
        U_epi, _ = torch.linalg.qr(block.to(torch.float32), mode="reduced")
        scores_epi = (U_epi * U_epi).sum(dim=-1).to(dtype)  # [H, Lc]
        start = offsets[chunk_idx]
        scores[start : start + Lc] = scores_epi.transpose(0, 1)  # [Lc, H]

    if normalize:
        grid = (len(chunks_lens),)
        cu_k = chunk_lens_cuda.cumsum(dim=0)
        _zscore_per_batch_epilogue_no_window[grid](
            scores, cu_k, scores.stride(0), scores.stride(1), H
        )
    return scores


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_K": BK, "WARPSPEC": False}, num_warps=w, num_stages=s
        )
        for BM in [64]
        for BK in [64]
        for w in [4]
        for s in [2]
    ],
    key=[
        "QUERY_GROUP_SIZE",
        "D",
        "CHUNK_SIZE",
    ],
    cache_results=True,
)
@triton.jit
def _non_causal_attn_kernel(
    Q,
    K,
    V,
    accum_scores,
    cu_seqlens_qk,
    #
    STRIDE_Q_G,
    STRIDE_Q_N,
    STRIDE_Q_H,
    STRIDE_Q_D,
    STRIDE_K_G,
    STRIDE_K_N,
    STRIDE_K_D,
    STRIDE_V_G,
    STRIDE_V_N,
    STRIDE_V_D,
    STRIDE_OUT_N,
    STRIDE_OUT_H,
    sm_scale,
    #
    CHUNK_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
    WARPSPEC: tl.constexpr,
):
    TOTAL_QUERIES_PER_BLOCK: tl.constexpr = BLOCK_M * QUERY_GROUP_SIZE
    INVERSE_CHUNK: tl.constexpr = 1.0 / CHUNK_SIZE
    pid_g = tl.program_id(0)  # KV head in [0, HKV)
    pid_b = tl.program_id(1)  # batch id
    pid_m = tl.program_id(2)  # chunk id within batch

    off_b = tl.load(cu_seqlens_qk + pid_b)
    off_b1 = tl.load(cu_seqlens_qk + pid_b + 1)

    chunk_start = off_b + pid_m * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, off_b1)
    M = chunk_end - chunk_start
    if M <= 0:
        return

    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_K)

    # Flattened query rows inside a [BLOCK_M, QUERY_GROUP_SIZE] tile
    offs_q = tl.arange(0, TOTAL_QUERIES_PER_BLOCK)
    row_m = offs_q % BLOCK_M  # token offset in this tile
    row_h = offs_q // BLOCK_M  # query-group index

    qk_scale = sm_scale * 1.44269504  # convert to log2-domain
    NEG_INF = -1.0e9

    # Iterate over query tiles within this chunk
    for qs in tl.range(chunk_start, chunk_end, BLOCK_M):
        # Global query indices for rows in this tile
        q_idx = qs + row_m  # [TOTAL_QUERIES_PER_BLOCK]
        q_mask = q_idx < chunk_end  # mask for valid rows in this tile

        # Load Q tile: [TOTAL_QUERIES_PER_BLOCK, D]
        q_ptrs = (
            Q
            + pid_g * STRIDE_Q_G
            + q_idx[:, None] * STRIDE_Q_N
            + row_h[:, None] * STRIDE_Q_H
            + offs_d[None, :] * STRIDE_Q_D
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

        # ---- Pass 1: per-row max and denominator over all keys in this chunk ----
        row_max = tl.full([TOTAL_QUERIES_PER_BLOCK], NEG_INF, tl.float32)
        row_sum = tl.zeros([TOTAL_QUERIES_PER_BLOCK], dtype=tl.float32)

        for ks in tl.range(chunk_start, chunk_end, BLOCK_K, warp_specialize=WARPSPEC):
            k_idx = ks + offs_k  # [BLOCK_K]
            k_mask = k_idx < chunk_end  # which keys are valid in this tile

            k_ptrs = (
                K
                + pid_g * STRIDE_K_G
                + k_idx[:, None] * STRIDE_K_N
                + offs_d[None, :] * STRIDE_K_D
            )
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_K, D]

            # logits: [TOTAL_QUERIES_PER_BLOCK, BLOCK_K]
            qk = tl.dot(q, k.T) * qk_scale
            qk = tl.where(q_mask[:, None] & k_mask[None, :], qk, NEG_INF)

            cur_max = tl.max(qk, 1)
            new_max = tl.maximum(row_max, cur_max)

            # rescale previous sum to new_max (base 2)
            rescale = tl.math.exp2(row_max - new_max)
            p = tl.math.exp2(qk - new_max[:, None])

            row_sum = row_sum * rescale + tl.sum(p, 1)
            row_max = new_max

        # Avoid division by zero for inactive rows
        denom = tl.where(q_mask, row_sum, 1.0)

        for ks in tl.range(chunk_start, chunk_end, BLOCK_K, warp_specialize=WARPSPEC):
            k_idx = ks + offs_k
            k_mask = k_idx < chunk_end

            k_ptrs = (
                K
                + pid_g * STRIDE_K_G
                + k_idx[:, None] * STRIDE_K_N
                + offs_d[None, :] * STRIDE_K_D
            )
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

            qk = tl.dot(q, k.T) * qk_scale
            qk = tl.where(q_mask[:, None] & k_mask[None, :], qk, NEG_INF)

            # p has shape [TOTAL_QUERIES_PER_BLOCK, BLOCK_K]
            p = tl.math.exp2(qk - row_max[:, None]) / denom[:, None]
            # zero-out invalid rows / columns
            p = tl.where(
                q_mask[:, None], p, INVERSE_CHUNK
            )  # preserve attention mass in shorter chunks

            contrib = tl.sum(p, 0)  # [BLOCK_K], sum over queries & query-groups

            out_ptrs = accum_scores + k_idx * STRIDE_OUT_N + pid_g * STRIDE_OUT_H
            old = tl.load(out_ptrs, mask=k_mask, other=0.0)
            new = old + contrib.to(old.dtype)
            tl.store(out_ptrs, new, mask=k_mask)


def non_causal_attn_scores(
    q: torch.Tensor,  # [N, HQ, D]
    k: torch.Tensor,  # [N, HKV, D]
    v: torch.Tensor,  # [N, HKV, D]
    cu_seqlens_qk: torch.Tensor,  # [B + 1]
    max_seqlen_qk: int,
    chunk_size: int,
    sm_scale: float = None,
    normalize: bool = True,
    context_lens: List[int] = None,
    protected_first_tokens: List[int] = None,
    protected_last_tokens: List[int] = None,
    *,
    accum_scores: torch.Tensor = None,  # [N, HKV] (float32)
    accum_blending: float = None,
) -> torch.Tensor:
    """
    :param q: Tensor of shape ``[N, H, D]`` containing post-rope queries
    :param k: Tensor of shape ``[N, H, D]`` containing post-rope keys
    :param v: Tensor of shape ``[N, H, D]`` containing values
    :param cu_seqlens_qk Tensor of shape ``[B + 1]`` demarcating batch boundaries
    :param max_seqlen_qk int containing the maximum sequence length
    :param chunk_size: int specifying the size of the chunk to perform non-causal attention over
    :param sm_scale: float specifying the scaling factor applied to attention scores (1/sqrt(D) if None)
    :param normalize: bool specifying whether to z-score normalize final attention scores
    :param context_lens: List[int] specifying the context lengths. CPU version of cu_seqlens_qk.diff(0)
    :param protected_first_tokens: List[int] specifying how many tokens should be protected at the
            start of each sequence
    :param protected_last_tokens: List[int] specifying how many tokens should be protected at the
            end of each sequence
    :param accum_scores: Tensor of shape ``[N, H]`` containing key scores that should be accumulated into
    :param accum_blending float specifying the scaling of ``accum_scores`` prior to adding the new
        non-causal attention scores. Final output is equivalent to return out + accum_blending * accum_scores
    """
    assert q.ndim == 3 and k.ndim == 3
    assert q.shape[0] == k.shape[0] and q.shape[-1] == k.shape[-1]
    N, HQ, D = q.shape
    HKV = k.shape[1]
    assert HQ % HKV == 0, "Number of query heads must divide number of KV heads"
    assert (D & (D - 1)) == 0, "D must be a power of two"

    B = cu_seqlens_qk.numel() - 1
    H_g = HQ // HKV  # query-group size per KV head

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    out = torch.zeros(N, HKV, device=q.device, dtype=torch.float32)
    q = q.view(N, HKV, H_g, D).permute(1, 0, 2, 3)
    k = k.view(N, HKV, D).permute(1, 0, 2)
    # v = v.view(N, HKV, D).permute(1, 0, 2)

    if cu_seqlens_qk.device != q.device:
        cu_seqlens_qk = cu_seqlens_qk.to(device=q.device)
    cu_seqlens_qk = cu_seqlens_qk.to(torch.int32)

    STRIDE_Q_G, STRIDE_Q_N, STRIDE_Q_H, STRIDE_Q_D = q.stride()
    STRIDE_K_G, STRIDE_K_N, STRIDE_K_D = k.stride()
    STRIDE_V_G, STRIDE_V_N, STRIDE_V_D = v.stride()
    STRIDE_OUT_N, STRIDE_OUT_H = out.stride()

    assert STRIDE_Q_D == 1 and STRIDE_K_D == 1, "last dim must be contiguous"

    def grid(_):
        return (
            HKV,
            B,
            triton.cdiv(max_seqlen_qk, chunk_size),
        )

    _non_causal_attn_kernel[grid](
        q,
        k,
        v,
        out,
        cu_seqlens_qk,
        STRIDE_Q_G,
        STRIDE_Q_N,
        STRIDE_Q_H,
        STRIDE_Q_D,
        STRIDE_K_G,
        STRIDE_K_N,
        STRIDE_K_D,
        STRIDE_V_G,
        STRIDE_V_N,
        STRIDE_V_D,
        STRIDE_OUT_N,
        STRIDE_OUT_H,
        sm_scale,
        CHUNK_SIZE=chunk_size,
        QUERY_GROUP_SIZE=H_g,
        D=D,
    )
    if normalize:
        grid = (B,)
        _zscore_per_batch_epilogue_no_window[grid](
            out, cu_seqlens_qk, out.stride(0), out.stride(1), HKV
        )
    if accum_scores is not None:
        if accum_blending is not None:
            out += accum_scores * accum_blending
        else:
            out += accum_scores
    if protected_first_tokens is not None or protected_last_tokens is not None:
        start = 0
        for first, last, L in zip(
            protected_first_tokens, protected_last_tokens, context_lens
        ):
            out[start : start + first].fill_(torch.inf)
            out[start + L - last : start + L].fill_(torch.inf)
            start += L
    return out
