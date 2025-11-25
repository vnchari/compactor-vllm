from dataclasses import dataclass
from typing import List

import torch
from compactor_vllm.compression import CompressionMethod
from compactor_vllm.config.engine_config import AttentionBackend


@dataclass
class CompressionContext:
    compression_method: CompressionMethod = CompressionMethod.COMPACTOR

    compression_chunk_size: int = -1
    batch_tokens_to_retain: torch.Tensor | None = None
    max_tokens_to_retain: int = 0
    context_lens: List[int] | None = None
    PHI: torch.Tensor | None = None

    protected_first_tokens: List[int] | None = None
    protected_last_tokens: List[int] | None = None


@dataclass
class Context:
    is_prefill: bool = False
    do_compression: bool = False

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    batch_mapping: torch.Tensor | None = None
    max_bh_len: int = 0

    compression_context: CompressionContext | None = None
    STORE_STREAM: torch.cuda.Stream | None = None

    key_split: int | None = None
    attention_backend: AttentionBackend = AttentionBackend.COMPACTOR_TRITON


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    *,
    is_prefill,
    do_compression=False,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    batch_mapping=None,
    max_bh_len=0,
    compression_context: CompressionContext = None,
    STORE_STREAM=None,
    key_split=None,
    attention_backend=AttentionBackend.COMPACTOR_TRITON,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        do_compression,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        batch_mapping,
        max_bh_len,
        compression_context,
        STORE_STREAM,
        key_split,
        attention_backend,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
