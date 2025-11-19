import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from transformers import AutoConfig


class AttentionBackend(Enum):
    FLASH_ATTENTION = auto()
    COMPACTOR_TRITON = auto()


@dataclass
class LLMConfig:
    """Configuration for the :class:`LLM` engine.
    Parameters
    ----------
    model : str
        Hugging Face model identifier (e.g. ``"meta-llama/Meta-Llama-3-8B"``) or
        a local model name that can be resolved by
        :func:`transformers.AutoConfig.from_pretrained`.
    path : str, optional
        Local directory containing the model weights. If ``None``, the engine
        will attempt to resolve a local snapshot for ``model`` using
        :func:`huggingface_hub.snapshot_download`.
    max_num_seqs : int, default 256
        Upper bound on the number of concurrent batches that the scheduler and
        KV-cache manager are allowed to handle. This affects the size of the
        page table and some internal buffers.
    max_model_len : int, default 40960
        Maximum context length (in tokens) that the engine will allocate KV cache
        and CUDA graphs for. During initialization this value is clamped to
        ``hf_config.max_position_embeddings`` for the chosen model.
    gpu_memory_utilization : float, default 0.9
        Fraction of the total GPU memory that may be used for KV cache and model
        activations. Values should be in ``(0, 1]``. If this budget is too small,
        the KV-cache manager may raise an error at warmup time due
        to insufficient memory.
    tensor_parallel_size : int, default 1
        Number of tensor-parallel workers to shard the model
        across. Must be between 1 and 8, and must evenly divide the model's
        number of key/value heads.
    enforce_eager : bool, default False
        If ``True``, disable CUDA graph capture and always run the model in
        eager mode during decoding. This reduces throughput. When ``False``,
        the engine will capture and reuse CUDA graphs for supported
        batch sizes and sequence lengths.
    hf_config : transformers.AutoConfig, optional
        Pre-loaded Hugging Face configuration for the model. If ``None``,
         it will then be populated automatically based on ``model``.
    eos : int, default -1
        Token id to treat as end-of-sequence during generation. If left at
        ``-1``, the :class:`LLM` constructor will set this to the tokenizer's
        ``eos_token_id``.
    kvcache_page_size : int, default 128
        Number of tokens stored in a single KV-cache page. Smaller pages improve
        allocation flexibility but increase page-table overhead; larger pages
        reduce overhead but have coarser granularity.
    leverage_sketch_size : int, default 48
        Sketch dimension used by the Compactor leverage-score estimator.
    attention_backend : AttentionBackend, default AttentionBackend.COMPACTOR_TRITON
        Attention implementation to use. ``COMPACTOR_TRITON`` selects the custom
        Triton kernels used by Compactor; ``FLASH_ATTENTION`` selects the
        FlashAttention3 varlen backend. The COMPACTOR_TRITON tends to be faster
        for longer sequence lengths, while FA3 is faster at shorter lengths.
    """

    model: str
    path: Optional[str] = None
    nccl_port: Optional[int] = 1218
    max_num_seqs: int = 256
    max_model_len: int = 40960
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_page_size: int = 128
    leverage_sketch_size: int = 48
    attention_backend: AttentionBackend = AttentionBackend.COMPACTOR_TRITON
    show_progress_bar: bool = True

    def __post_init__(self):
        if self.path is not None and not os.path.isdir(self.path):
            raise NotADirectoryError(f"Engine config dir {self.path} does not exist")
        if self.tensor_parallel_size <= 0 or self.tensor_parallel_size > 8:
            assert 1 <= self.tensor_parallel_size <= 8
            raise ValueError("tensor_parallel_size must be >= 1 and <= 8")
        if self.hf_config is None:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
