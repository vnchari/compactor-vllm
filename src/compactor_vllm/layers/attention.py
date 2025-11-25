from typing import Optional

import torch
from compactor_vllm.attention.sparse_decode_kernel import head_sparse_decode_attention
from compactor_vllm.attention.sparse_varlen_kernel import (
    causal_sparse_varlen_with_cache,
)
from compactor_vllm.compression.common import extract_and_store_top_kv
from compactor_vllm.config.engine_config import AttentionBackend
from compactor_vllm.kv_cache.store_kv_cache import decode_store_kv, prefill_store_all_kv
from compactor_vllm.utils.context import Context, get_context
from compactor_vllm.utils.helpers import maybe_execute_in_stream
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch import nn


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim = head_dim
        self.scale: float = scale
        self.num_kv_heads = int(num_kv_heads)

        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.page_table: Optional[torch.Tensor] = None
        self.bh_seq_lens: Optional[torch.Tensor] = None
        self.page_size: Optional[int] = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ):
        context: Context = get_context()
        batch_mapping = context.batch_mapping
        seq_lens = (
            None
            if self.bh_seq_lens is None
            else self.bh_seq_lens.index_select(0, batch_mapping).contiguous()
        )
        if context.is_prefill:
            seq_lens_copy = seq_lens.clone() if seq_lens is not None else None
            if (
                self.k_cache is not None
                and context.do_compression
                and scores is not None
            ):
                compression_context = context.compression_context
                assert scores is not None
                assert compression_context is not None
                maybe_execute_in_stream(
                    extract_and_store_top_kv,
                    scores=scores,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_k_len=context.max_seqlen_k,
                    top_k=compression_context.max_tokens_to_retain,
                    H=int(self.num_kv_heads),
                    new_keys=k,
                    new_vals=v,
                    num_tokens_to_retain=compression_context.batch_tokens_to_retain,
                    page_table=self.page_table,
                    batch_mapping=batch_mapping,
                    bh_lens=seq_lens,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    PAGE_SIZE=self.page_size,
                    PAD_TO_PAGE_SIZE=True,
                    STORE_STREAM=context.STORE_STREAM,
                )
            elif self.k_cache is not None:
                maybe_execute_in_stream(
                    prefill_store_all_kv,
                    new_keys=k,
                    new_values=v,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_k=context.max_seqlen_k,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    page_table=self.page_table,
                    bh_lens=seq_lens,
                    batch_mapping=batch_mapping,
                    PAGE_SIZE=self.page_size,
                    STORE_STREAM=context.STORE_STREAM,
                )

            if context.attention_backend == AttentionBackend.FLASH_ATTENTION:
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                )
            elif context.attention_backend == AttentionBackend.COMPACTOR_TRITON:
                o = causal_sparse_varlen_with_cache(
                    q,
                    k,
                    v,
                    self.k_cache,
                    self.v_cache,
                    seq_lens_bh=seq_lens_copy,
                    global_page_table=self.page_table,
                    batch_mapping=batch_mapping,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k_cache=context.max_bh_len,
                    HKV=int(self.num_kv_heads),
                    PAGE_SIZE=self.page_size,
                    sm_scale=self.scale,
                )
            else:
                raise NotImplementedError
        else:
            assert self.k_cache is not None, "KV Cache must be initialized for decoding"
            decode_store_kv(
                key=k,
                value=v,
                batch_mapping=batch_mapping,
                bh_lens=seq_lens,
                page_table=self.page_table,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                PAGE_SIZE=self.page_size,
            )

            o = head_sparse_decode_attention(
                q,
                self.k_cache,
                self.v_cache,
                seq_lens,
                self.page_table,
                batch_mapping,
                int(self.num_kv_heads),
                self.page_size,
                self.scale,
                key_split=context.key_split,
            )
        if self.bh_seq_lens is not None:
            longbm = batch_mapping.to(torch.long)
            maybe_execute_in_stream(
                self.bh_seq_lens.index_copy_,
                0,
                longbm,
                seq_lens,
                STORE_STREAM=context.STORE_STREAM if context.is_prefill else None,
            )
        return o
