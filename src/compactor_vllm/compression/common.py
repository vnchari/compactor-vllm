from abc import ABC, abstractmethod
from typing import Optional

import torch

from compactor_vllm.kv_cache.store_kv_cache import prefill_store_topk_kv


class BaseCompressionMethod(ABC):
    """
    Abstract interface for KV cache compression methods.

    A compression method is implemented as a pair of optional scoring phases
    that run before and after rotary position embedding (RoPE) is applied:

      1. ``pre_rope_scoring`` operates on pre-RoPE Q/K.

      2. ``post_rope_scoring`` operates on post-RoPE Q/K and can either:
         - refine / reweight the pre-RoPE scores, or
         - compute potentially position-aware.

    Concrete subclasses are expected to implement both
    static methods and return a single tensor of scores (or ``None`` if the
    phase is a no-op), which the caller can then feed into the shared
    “scores → top-k indices → KV extraction” pipeline.
    """

    @staticmethod
    @abstractmethod
    def pre_rope_scoring(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> Optional[torch.Tensor]:
        """
        Compute per-token importance scores from pre-RoPE queries/keys.

        Args:
            :param q:
                Pre-RoPE query tensor. Shape ``[total_tokens, HQ, D]```.
            :param k:
                Pre-RoPE key tensor. Shape ``[total_tokens, HKV, D]```.
            :param v:
                Value tensor. Shape ``[total_tokens, HKV, D]```
            :param context:
                compactor_vllm.utils.context.Context object carrying additional metadata,
                such as batch mappings or temporary buffers

        Returns:
            :return Optional[torch.Tensor]:
                A tensor of scores (e.g. per-token, per-head importance values)
                to be passed to ``post_rope_scoring`` or directly into the
                top-k selection step. If this phase is a no-op, implementations
                should return ``None``. Shape ``[total_tokens, HKV]```.
        """
        pass

    @staticmethod
    @abstractmethod
    def post_rope_scoring(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pre_rope_scores: Optional[torch.Tensor],
        context,
    ) -> Optional[torch.Tensor]:
        """
        Compute or refine importance scores from post-RoPE queries/keys.

        This method is called after rotary embeddings have been applied. It can
        optionally use both the post-RoPE Q/K and any scores produced by
        ``pre_rope_scoring`` to produce final scores used for token selection.

        Common patterns include:
          * Using ``pre_rope_scores`` as a base signal and applying a
            position-aware correction.
          * Only computing scores that depend on absolute or relative positions.
          * Simply passing through ``pre_rope_scores`` unchanged.

        Args:
            :param q:
                Post-RoPE query tensor. Shape ``[total_tokens, HQ, D]```.
            :param k:
                Post-RoPE key tensor. Shape ``[total_tokens, HKV, D]```.
            :param pre_rope_scores:
                Optional scores returned by ``pre_rope_scoring``. May be
                ``None`` if the pre-RoPE phase returned None.
            :param v:
                Value tensor. Shape ``[total_tokens, HKV, D]```
            :param context:
                compactor_vllm.utils.context.Context object carrying additional metadata,
                such as batch mappings or temporary buffers
        Returns:
            :return Optional[torch.Tensor]:
                Final importance scores to be consumed by the compression
                pipeline (for top-k token selection). If this phase is a
                no-op, implementations may return ``pre_rope_scores``. If
                None is returned, no compression will be applied.
        """
        pass


class NoCompression(BaseCompressionMethod):
    """
    Trivial compression method that disables KV cache compression.
    """

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
        return pre_rope_scores


def extract_and_store_top_kv(
    scores: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_k_len: int,
    top_k: int,
    H: int,
    new_keys: torch.Tensor,  # [N_total, H, D]
    new_vals: torch.Tensor,  # [N_total, H, D]
    num_tokens_to_retain: torch.Tensor,  # [B] int32
    page_table: torch.Tensor,  # [B_total, H, N_LOGICAL_PAGES_MAX] int32
    batch_mapping: torch.Tensor,  # [B] int32 (local -> true batch rows)
    bh_lens: torch.Tensor,  # [B, H] int32 (contiguous), UPDATED atomically
    k_cache: torch.Tensor,  # [N_PAGES * PAGE_SIZE, D]
    v_cache: torch.Tensor,  # [N_PAGES * PAGE_SIZE, D]
    PAGE_SIZE: int,
    PAD_TO_PAGE_SIZE: bool = True,
    K_TILE: int = 16,
    padding: float = -float("inf"),
):
    """helper method to extract and store top-k indices into KV cache (so they can be executed in a single stream)"""
    indices_topk = scores_to_retain_indices(
        scores,
        cu_seqlens_k=cu_seqlens_k,
        max_k_len=max_k_len,
        top_k=top_k,
        H=H,
        padding=padding,
    )
    prefill_store_topk_kv(
        new_keys=new_keys,
        new_vals=new_vals,
        indices_topk=indices_topk,
        num_tokens_to_retain=num_tokens_to_retain,
        page_table=page_table,
        batch_mapping=batch_mapping,
        bh_lens=bh_lens,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_k=cu_seqlens_k,
        PAGE_SIZE=PAGE_SIZE,
        PAD_TO_PAGE_SIZE=PAD_TO_PAGE_SIZE,
        K_TILE=K_TILE,
    )


def scores_to_retain_indices(
    scores: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_k_len: int,
    top_k: int,
    H: int,
    padding: float = -float("inf"),
) -> torch.Tensor:
    """
    Select global top-k token–head indices per sequence from packed scores.

    This helper takes per-token, per-head scores in packed varlen form and
    returns, for each batch element, the indices of the top-k (token, head)
    pairs in the flattened global layout.
    Inputs are assumed to follow the usual packed varlen convention:
      • ``scores`` is laid out as ``[N_total, H]``, where:
          ``N_total = sum_b seqlen_k[b]``
        and ``HKV`` is the number of KV heads.

      • ``cu_seqlens_k`` is ``[B + 1]`` (int32), giving cumulative lengths
        for the keys per batch:
            ``seqlen_k[b] = cu_seqlens_k[b + 1] - cu_seqlens_k[b]``.

      • ``max_k_len`` is an upper bound on ``seqlen_k[b]`` across the batch.

    The function pads each sequence to length ``max_k_len`` with ``padding``
    (default: ``-inf``), flattens the per-sequence scores into shape
    ``[B, max_k_len * H]``, and runs a per-batch top-k. The returned indices
    are shifted so that they directly index into the flattened global
    score layout of shape ``[N_total * H]``:
        global_index = (token_global_offset * H) + head_index

    Args:
        :param scores:
            Tensor of shape ``[N_total, HKV]`` containing scores for each
            (token, head) pair in packed varlen format.
        :param cu_seqlens_k:
            Tensor of shape ``[B + 1]`` (int32) with cumulative key sequence
            lengths for each batch element. The total number of tokens
            satisfies ``N_total = cu_seqlens_k[-1]``.
        :param max_k_len:
            Maximum key sequence length across the batch (i.e.
            ``max_b seqlen_k[b]``). Used to allocate the padded buffer.
        :param top_k:
            Number of (token, head) entries to retain **per batch element**.
            If ``top_k > max_k_len * HKV``, it is clamped to ``max_k_len * HKV``.
        :param H:
            Number of key heads; must match ``scores.shape[1]``.
        :param padding:
            Padding value used when extending sequences shorter than
            ``max_k_len``. Defaults to ``-inf``, so that padded positions are
            never selected in the top-k.

    Returns:
        :return torch.Tensor:
            Tensor of shape ``[B, k_eff]`` (int64) where
            ``k_eff = min(top_k, max_k_len * H)``. Each entry is a global
            index into the flattened score array of shape ``[N_total * H]``
            (i.e. scores viewed as ``scores.view(-1)``),
    """
    # idea: pad and then select top-k.
    B, device = cu_seqlens_k.numel() - 1, scores.device
    padded = torch.full(
        (B, max_k_len, H), fill_value=padding, dtype=scores.dtype, device=device
    )
    for b in range(B):
        s, e = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
        padded[b, : e - s, :].copy_(scores[s:e, :])
    flat = padded.view(B, max_k_len * H)
    idx = torch.topk(
        flat, k=min(top_k, max_k_len * H), dim=1, largest=True, sorted=True
    ).indices
    return idx + (cu_seqlens_k[:-1] * H).unsqueeze(-1)
